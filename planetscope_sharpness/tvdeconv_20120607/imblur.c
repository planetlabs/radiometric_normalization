/**
 * @file imblur.c
 * @brief Image blurring command line program
 * @author Pascal Getreuer <getreuer@gmail.com>
 * 
 * 
 * Copyright (c) 2010-2012, Pascal Getreuer
 * All rights reserved.
 * 
 * This program is free software: you can use, modify and/or
 * redistribute it under the terms of the simplified BSD License. You
 * should have received a copy of this license along with this program.
 * If not, see <http://www.opensource.org/licenses/bsd-license.html>.
 */
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <fftw3.h>
#include "cliio.h"
#include "kernels.h"
#include "randmt.h"

#ifndef MIN
#define MIN(a,b)  (((a) <= (b)) ? (a) : (b))
#endif

#define _CONCAT(A,B)    A ## B

#ifdef NUM_SINGLE
#define FFT(S)          _CONCAT(fftwf_,S)
#else
#define FFT(S)          _CONCAT(fftw_,S)
#endif


/** @brief Complex value type */
typedef num numcomplex[2];

typedef enum {NOISE_GAUSSIAN, NOISE_LAPLACE, NOISE_POISSON} noisetype;

/** @brief Program parameters struct */
typedef struct
{
    /** @brief Input file name */
    const char *InputFile;
    /** @brief Output file name */
    const char *OutputFile;
    /** @brief Quality for saving JPEG images (0 to 100) */
    int JpegQuality;
    
    /** @brief Noise type */
    noisetype NoiseType;
    /** @brief Noise standard deviation */
    num Sigma;
    /** @brief Blur kernel */
    image Kernel;
} programparams;


/** @brief Print program information and usage message */
static void PrintHelpMessage()
{    
    puts("Image blurring utility, P. Getreuer 2011-2012\n\n"
    "Usage: imblur [param:value ...] input output\n\n"
    "where \"input\" and \"output\" are " 
    READIMAGE_FORMATS_SUPPORTED " files.\n");
    puts("Parameters");
    puts("  K:<kernel>             blur kernel for deconvolution");
    puts("      K:disk:<radius>         filled disk kernel");
    puts("      K:gaussian:<sigma>      Gaussian kernel");
    puts("      K:<file>                read kernel from text or image file");
    puts("  noise:<model>:<sigma>  simulate noise with standard deviation sigma");
    puts("      noise:gaussian:<sigma>  additive white Gaussian noise");
    puts("      noise:laplace:<sigma>   Laplace noise");
    puts("      noise:poisson:<sigma>   Poisson noise");
    puts("  f:<file>               input file (alternative syntax)");
    puts("  u:<file>               output file (alternative syntax)");
#ifdef USE_LIBJPEG
    puts("  jpegquality:<number>   quality for saving JPEG images (0 to 100)");
#endif
    puts("\nExample: \n"
    "   imblur noise:gaussian:5 K:disk:2 input.bmp blurry.bmp\n");
}

int ParseParams(programparams *Params, int argc, const char *argv[]);


/**
 * @brief Boundary handling function for whole-sample symmetric extension
 * @param Data pointer to a contiguous 2D array in row-major order
 * @param Width,Height size of the data
 * @param x,y sampling location
 * @return extrapolated value
 */
static num WSymExtension(const num *Data, int Width, int Height, 
    int x, int y)
{
    while(1)
    {
        if(x < 0)
            x = -1 - x;
        else if(x >= Width)
            x = (2*Width - 1) - x;
        else
            break;
    }
    
    while(1)
    {
        if(y < 0)
            y = -1 - y;
        else if(y >= Height)
            y = (2*Height - 1) - y;
        else
            break;
    }
    
    return Data[x + ((long)Width)*y];
}


/** 
 * @brief Pad an image and compute Fourier transform
 * @param PadTemp temporary buffer
 * @param PadWidth, PadHeight dimensions of the padded image
 * @param Src the source image
 * @param SrcWidth, SrcHeight dimensions of Src
 * @param IsKernelFlag nonzero if Src is a convolution kernel
 * @return the Fourier transform of the padded image, or NULL on failure
 */
numcomplex *ComputePaddedDFT(num *PadTemp, int PadWidth, int PadHeight,
    const num *Src, int SrcWidth, int SrcHeight,
    int IsKernelFlag)
{
    const int PadXOffset = (PadWidth - SrcWidth)/2;
    const int PadYOffset = (PadHeight - SrcHeight)/2;    
    const long TransNumPixels = ((long)(PadWidth/2 + 1))*((long)PadHeight);
    numcomplex *Dest = NULL;
    FFT(plan) Plan = NULL;
    int x, y;
    
    /* Allocate memory */
    if(!(Dest = (numcomplex *)FFT(malloc)(sizeof(numcomplex)*TransNumPixels)))
    {
        fprintf(stderr, "Memory allocation failed.\n");
        return NULL;
    }
    
    /* Create plan for DFT transform of PadTemp */
    if(!(Plan = FFT(plan_dft_r2c_2d)(PadHeight, PadWidth, PadTemp, 
        Dest, FFTW_ESTIMATE | FFTW_DESTROY_INPUT)))
    {
        fprintf(stderr, "FFTW plan creation failed.\n");
        FFT(free)(Dest);
        return NULL;
    }
    
    if(IsKernelFlag)    /* Do zero-padding extrapolation for the kernel */
    {
        const int SrcHalfWidth = SrcWidth/2;
        const int SrcHalfHeight = SrcHeight/2;
        
        for(y = 0; y < SrcWidth; y++)
            for(x = 0; x < SrcHeight; x++)
                PadTemp[x + ((long)PadWidth)*y] = 0;
        
        for(y = 0; y < SrcHeight; y++)
            for(x = 0; x < SrcWidth; x++)
                PadTemp[((x < SrcHalfWidth) ? 
                    (PadWidth - SrcHalfWidth + x) : (x - SrcHalfWidth))
                    + ((long)PadWidth)*((y < SrcHalfHeight) ? 
                    (PadHeight - SrcHalfHeight + y) : (y - SrcHalfHeight))]
                    = Src[x + SrcWidth*y];
    }
    else    /* Whole-sample symmetric extrapolation for the image */
        for(y = 0; y < PadHeight; y++)
            for(x = 0; x < PadWidth; x++)
                PadTemp[x + ((long)PadWidth)*y] 
                    = WSymExtension(Src, SrcWidth, SrcHeight, 
                        x - PadXOffset, y - PadYOffset);
    
    FFT(execute)(Plan);
    FFT(destroy_plan)(Plan);
    return Dest;
}


/** 
 * @brief Invert Fourier transform and trim padding
 * @param Dest destination buffer
 * @param DestWidth, DestHeight dimensions of Dest
 * @param PadTemp temporary buffer
 * @param PadWidth, PadHeight dimensions of the padded image
 * @param Src the transformed image
 * @return 1 on succes, 0 on failure
 */
int ComputePaddedIDFT(num *Dest, int DestWidth, int DestHeight,
    num *PadTemp, int PadWidth, int PadHeight, numcomplex *Src)
{    
    const int PadXOffset = (PadWidth - DestWidth)/2;
    const int PadYOffset = (PadHeight - DestHeight)/2;    
    FFT(plan) Plan = NULL;
    int x, y;
    
    /* Create plan for DFT transform of PadTemp */
    if(!(Plan = FFT(plan_dft_c2r_2d)(PadHeight, PadWidth,
        Src, PadTemp, FFTW_ESTIMATE | FFTW_DESTROY_INPUT)))
    {
        fprintf(stderr, "FFTW plan creation failed.\n");
        return 0;
    }

    FFT(execute)(Plan);
    FFT(destroy_plan)(Plan);
    
    PadTemp += PadXOffset + PadWidth*PadYOffset;
    
    for(y = 0; y < DestHeight; y++, Dest += DestWidth, PadTemp += PadWidth)
        for(x = 0; x < DestWidth; x++)
            Dest[x] = PadTemp[x];
            
    return 1;
}


/**
 * @brief Convolve an image with a kernel
 * @param Image the image data in row-major planar order
 * @param ImageWidth, ImageHeight, NumChannels the image dimensions
 * @param Kernel the convolution kernel
 * @param KernelWidth, KernelHeight the kernel dimensions
 * @return 1 on success, 0 on failure
 */
int BlurImage(num *Image, int ImageWidth, int ImageHeight, int NumChannels,
    const num *Kernel, int KernelWidth, int KernelHeight)
{
    const int ImageNumPixels = ImageWidth*ImageHeight;
    num *PadTemp = NULL;
    numcomplex *ImageFourier = NULL;
    numcomplex *KernelFourier = NULL;    
    numcomplex G;
    num Temp;
    int PadWidth, PadHeight, PadNumPixels;
    int TransWidth;
    int i, x, y, Channel, Success = 0;
    
    
    if(!Image || !Kernel || !ImageWidth || !ImageHeight)
        return 0;
    else if(KernelWidth*KernelHeight <= 1)
        return 1;
    
    /* Determine padded size of the image */
    PadWidth = 2*ImageWidth;
    PadHeight = 2*ImageHeight;    
    PadNumPixels = PadWidth*PadHeight;
    
    while(PadWidth < KernelWidth)
        PadWidth += 2*ImageWidth;
    while(PadHeight < KernelHeight)
        PadHeight += 2*KernelHeight;

    /* Determine size of Fourier transform array */
    TransWidth = PadWidth/2 + 1;
    
    /* Allocate memory */
    if(!(PadTemp = (num *)FFT(malloc)(sizeof(num)*PadNumPixels)))
    {
        fprintf(stderr, "Memory allocation failed.\n");
        goto Catch;
    }
    
    /* Compute Fourier transform of Kernel */
    if(!(KernelFourier = ComputePaddedDFT(PadTemp, PadWidth, PadHeight,
        Kernel, KernelWidth, KernelHeight, 1)))
        goto Catch;
    
    for(Channel = 0; Channel < NumChannels; Channel++)
    {
        /* Compute Fourier transform of image channel */
        if(!(ImageFourier = ComputePaddedDFT(PadTemp, PadWidth, PadHeight,
            Image + Channel*ImageNumPixels, 
            ImageWidth, ImageHeight, 0)))
            goto Catch;
    
        for(y = i = 0; y < PadHeight; y++)
            for(x = 0; x < TransWidth; x++, i++)
            {
                G[0] = KernelFourier[i][0]/PadNumPixels;
                G[1] = KernelFourier[i][1]/PadNumPixels;
                
                /* Compute the filtered frequency */
                Temp = G[0]*ImageFourier[i][0] - G[1]*ImageFourier[i][1];
                ImageFourier[i][1] = G[0]*ImageFourier[i][1] 
                    + G[1]*ImageFourier[i][0];
                ImageFourier[i][0] = Temp;
            }
        
        if(!ComputePaddedIDFT(Image + Channel*ImageNumPixels, 
            ImageWidth, ImageHeight,
            PadTemp, PadWidth, PadHeight, ImageFourier))
            goto Catch;
    }
    
    Success = 1;
Catch:
    if(KernelFourier)
        FFT(free)(KernelFourier);
    if(ImageFourier)
        FFT(free)(ImageFourier);
    if(PadTemp)
        FFT(free)(PadTemp);
    return Success; 
}


/** 
 * @brief Simulate noise of a specified type and standard deviation 
 * @param Data the image data upon which noise is simulated
 * @param NumEl number of elements in Data
 * @param NoiseType type of noise (Gaussian, Laplace, or Poisson)
 * @param Sigma standard deviation of the noise
 */
void GenerateNoise(num *Data, long NumEl, noisetype NoiseType, num Sigma)
{
    long i;
    
    switch(NoiseType)
    {
    case NOISE_GAUSSIAN:
        for(i = 0; i < NumEl; i++)
            Data[i] += (num)(Sigma * rand_normal());
        break;
    case NOISE_LAPLACE:
        {
            const num Mu = (num)(M_1_SQRT2 * Sigma);

            for(i = 0; i < NumEl; i++)
                Data[i] += (num)(rand_exp(Mu) 
                    * ((rand_unif() < 0.5) ? -1 : 1));
        }
        break;
    case NOISE_POISSON:
        {
            double a, Mean = 0;
            
            for(i = 0; i < NumEl; i++)
                Mean += Data[i];
            
            Mean /= NumEl;
            a = Sigma * Sigma / ((Mean > 0) ? Mean : (0.5/255));
            
            for(i = 0; i < NumEl; i++)
                Data[i] = (num)(rand_poisson(Data[i] / a) * a);
        }
        break;
    }        
}


int main(int argc, char **argv)
{
    programparams Params;
    image f = NullImage;
    int Status = 1;
    
    if(!ParseParams(&Params, argc, (const char **)argv))
        goto Catch;
    
    /* Initialize random number generator */
    init_randmt_auto();
    
    /* Read the input image */
    if(!ReadImageObj(&f, Params.InputFile))
        goto Catch;
       
    /* Perform blurring */
    if(!BlurImage(f.Data, f.Width, f.Height, f.NumChannels,
        Params.Kernel.Data, Params.Kernel.Width, Params.Kernel.Height))
        goto Catch;
    
    if(Params.Sigma != 0)
        GenerateNoise(f.Data, 
            (((long)f.Width)*((long)f.Height))*f.NumChannels,
            Params.NoiseType, Params.Sigma);
    
    /* Write output */
    if(!WriteImageObj(f, Params.OutputFile, Params.JpegQuality))    
        fprintf(stderr, "Error writing to \"%s\".\n", Params.OutputFile);
    
    Status = 0;
Catch:
    FreeImageObj(f); 
    FreeImageObj(Params.Kernel);
    return Status;
}


/** @brief Parse noise command line argument */
static int ReadNoise(programparams *Params, const char *String)
{
    const char *ColonPtr;    
    int Length;
    char NoiseName[32];
    
    if(!(ColonPtr = strchr(String, ':')) 
        || (Length = (int)(ColonPtr - String)) > 9)
        return 0;
    
    strncpy(NoiseName, String, Length);
    NoiseName[Length] = '\0';
    Params->Sigma = (num)atof(ColonPtr + 1);
    Params->Sigma /= 255;
    
    if(Params->Sigma < 0)
    {
        fputs("Noise standard deviation must be nonnegative.\n", stderr);
        return 0;
    }
    
    if(!strcmp(NoiseName, "Gaussian") || !strcmp(NoiseName, "gaussian"))
        Params->NoiseType = NOISE_GAUSSIAN;
    else if(!strcmp(NoiseName, "Laplace") || !strcmp(NoiseName, "laplace"))
        Params->NoiseType = NOISE_LAPLACE;
    else if(!strcmp(NoiseName, "Poisson") || !strcmp(NoiseName, "poisson"))
        Params->NoiseType = NOISE_POISSON;
    else
    {
        fprintf(stderr, "Unknown noise model, \"%s\".\n", NoiseName);
        return 0;
    }
    
    return 1;
}


/** @brief Parse command line arguments */
int ParseParams(programparams *Params, int argc, const char *argv[])
{
    static const char *DefaultOutputFile = (char *)"out.bmp";
    const char *Param, *Value;
    num NumValue;
    char TokenBuf[256];
    int k, kread, Skip;
    
        
    /* Set parameter defaults */
    Params->InputFile = NULL;
    Params->OutputFile = DefaultOutputFile;
    Params->JpegQuality = 85;
    
    Params->NoiseType = NOISE_GAUSSIAN;
    Params->Sigma = 0;
    Params->Kernel = NullImage;
        
    if(argc < 2)
    {
        PrintHelpMessage();
        return 0;
    }    
    
    k = 1;
    
    while(k < argc)
    {
        Skip = (argv[k][0] == '-') ? 1 : 0;        
        kread = CliParseArglist(&Param, &Value, TokenBuf, sizeof(TokenBuf),
            k, &argv[k][Skip], argc, argv, ":");        
       
        if(!Param)
        {
            if(!Params->InputFile)
                Param = (char *)"f";
            else
                Param = (char *)"u";
        }
        
        if(Param[0] == '-')     /* Argument begins with two dashes "--" */
        {
            PrintHelpMessage();
            return 0;
        }

        if(!strcmp(Param, "f") || !strcmp(Param, "input"))
        {
            if(!Value)
            {
                fprintf(stderr, "Expected a value for option %s.\n", Param);
                return 0;
            }
            Params->InputFile = Value;
        }
        else if(!strcmp(Param, "u") || !strcmp(Param, "output"))
        {
            if(!Value)
            {
                fprintf(stderr, "Expected a value for option %s.\n", Param);
                return 0;
            }
            Params->OutputFile = Value;
        }
        else if(!strcmp(Param, "K"))
        {
            if(!Value)
            {
                fprintf(stderr, "Expected a value for option %s.\n", Param);
                return 0;
            }
            else if(!ReadKernel(&Params->Kernel, Value))
                return 0;
        }
        else if(!strcmp(Param, "noise"))
        {
            if(!Value)
            {
                fprintf(stderr, "Expected a value for option %s.\n", Param);
                return 0;
            }
            if(!ReadNoise(Params, Value))
                return 0;
        }
        else if(!strcmp(Param, "jpegquality"))
        {
            if(!CliGetNum(&NumValue, Value, Param))
                return 0;
            else if(NumValue < 0 || 100 < NumValue)
            {
                fprintf(stderr, "JPEG quality must be between 0 and 100.\n");
                return 0;
            } 
            else
                Params->JpegQuality = (int)NumValue;
        }
        else if(Skip)
        {
            fprintf(stderr, "Unknown option \"%s\".\n", Param);
            return 0;
        }
        else
        {
            if(!Params->InputFile)
                Params->InputFile = argv[k];
            else
                Params->OutputFile = argv[k];
            
            kread = k;
        }
        
        k = kread + 1;
    }
    
    if(!Params->Kernel.Data && !ReadKernel(&Params->Kernel, "disk:0"))
        return 0;
    
    if(!Params->InputFile)
    {
        PrintHelpMessage();
        return 0;
    }

    return 1;
}
