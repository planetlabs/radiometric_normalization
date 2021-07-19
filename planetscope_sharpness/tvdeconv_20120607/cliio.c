/**
 * @file cliio.c
 * @brief Utilities for creating a command line interface
 * @author Pascal Getreuer <getreuer@gmail.com>
 */
#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "cliio.h"


const image NullImage = {NULL, 0, 0, 0};


int AllocImageObj(image *f, int Width, int Height, int NumChannels)
{
    if(!f)
        return 0;
    
    if(!(f->Data = (num *)malloc(sizeof(num)*
        (((size_t)Width)*((size_t)Height))*NumChannels)))
    {
        *f = NullImage;
        return 0;
    }    
    
    f->Width = Width;
    f->Height = Height;
    f->NumChannels = NumChannels;
    return 1;
}


void FreeImageObj(image f)
{
    if(f.Data)
        free(f.Data);
}


int ReadImageObj(image *f, const char *FileName)
{
    if(!f || !(f->Data = (num *)ReadImage(&f->Width, &f->Height, FileName,
         IMAGEIO_NUM | IMAGEIO_RGB | IMAGEIO_PLANAR)))
    {
        *f = NullImage;
        return 0;
    }
    
    f->NumChannels = (IsGrayscale(f->Data, f->Width, f->Height)) ? 1:3;
    return 1;
}


int ReadImageObjGrayscale(image *f, const char *FileName)
{
    if(!f || !(f->Data = (num *)ReadImage(&f->Width, &f->Height, FileName,
         IMAGEIO_NUM | IMAGEIO_GRAYSCALE | IMAGEIO_PLANAR)))
    {
        *f = NullImage;
        return 0;
    }
    
    f->NumChannels = 1;
    return 1;
}


int WriteImageObj(image f, const char *FileName, int JpegQuality)
{
    if(!f.Data || !FileName)
        return 0;
    
    switch(f.NumChannels)
    {
    case 1:
        return WriteImage(f.Data, f.Width, f.Height, FileName,
            IMAGEIO_NUM | IMAGEIO_GRAYSCALE | IMAGEIO_PLANAR, JpegQuality);
    case 3:
        return WriteImage(f.Data, f.Width, f.Height, FileName,
            IMAGEIO_NUM | IMAGEIO_RGB | IMAGEIO_PLANAR, JpegQuality);
    case 4:
        return WriteImage(f.Data, f.Width, f.Height, FileName,
            IMAGEIO_NUM | IMAGEIO_RGBA | IMAGEIO_PLANAR, JpegQuality);
    default:
        return 0;
    }
}


/** @brief Check whether all three channels in a color image are the same */
int IsGrayscale(num *Data, int Width, int Height)
{    
    const long NumPixels = ((long)Width) * ((long)Height);
    const num *Red = Data;
    const num *Green = Data + NumPixels;
    const num *Blue = Data + 2*NumPixels;
    long n;
    
    for(n = 0; n < NumPixels; n++)
        if(Red[n] != Green[n] || Red[n] != Blue[n])
            return 0;
    
    return 1;
}


/** 
 * @brief Extract a token from a null-terminated string 
 * @param Token destination buffer (with space for at least MaxLength+1 chars)
 * @param Start pointer to the source string
 * @param MaxLength maximum length of the token, not including null terminator
 * @param Delim delimiter characters
 * @return length of the token (greater than MaxLength indicates truncation).
 */
int GetStrToken(char *Token, const char *Start, int MaxLength, const char *Delim)
{
    int NumChars = strcspn(Start, Delim);
    int NumCopy = (NumChars <= MaxLength) ? NumChars : MaxLength;
    
    strncpy(Token, Start, NumCopy);
    Token[NumCopy] = '\0';
    return NumChars;
}


/** 
 * @brief Read a floating-point number from a string
 * @param Num is a pointer to where to store the result
 * @param String is a pointer to the source string to parse
 * @return the number of characters read (possibly 0).
 * 
 * The function strtod does not seem to work on some systems.  We use
 * the following routine instead.
 * 
 * The routine reads as many characters as possible to form a valid 
 * floating-point number in decimal or scientific notation.  A return
 * value of zero indicates that no valid number was found.
 */
int ParseDouble(double *Num, const char *String)
{
    double Accum = 0, Div = 1, Exponent = 0;
    int i = 0, Sign = 1, ExponentSign = 1;
    char c;
    
    
    if(!Num || !String)
        return 0;
    
    while(isspace(String[i]))   /* Eat leading whitespace */
        i++;
    
    if(String[i] == '-')        /* Read sign */
    {
        Sign = -1;
        i++;
    }
    else if(String[i] == '+')
        i++;
    
    /* Read one or more digits appearing left of the decimal point */
    if(isdigit(c = String[i]))
        Accum = c - '0';
    else
        return 0;               /* First character is not a digit */
    
    while(isdigit(c = String[++i]))
        Accum = 10*Accum + (c - '0');
        
    if(c == '.')                /* There is a decimal point */
    {        
        /* Read zero or more digits appearing right of the decimal point */
        while(isdigit(c = String[++i]))
        {
            Div *= 10;
            Accum += (c - '0')/Div;
        }
    }
    
    if(c == 'e' || c == 'E')    /* There is an exponent */
    {
        i++;
        
        if(String[i] == '-')      /* Read exponent sign */
        {
            ExponentSign = -1;
            i++;
        }
        else if(String[i] == '+')
            i++;
        
        /* Read digits in the exponent */
        if(isdigit(c = String[i]))
        {
            Exponent = c - '0';
            
            while(isdigit(c = String[++i]))
                Exponent = 10*Exponent + (c - '0');
            
            Exponent *= ExponentSign;
            Accum = Accum * pow(10, Exponent);
        }
    }
    
    Accum *= Sign;
    *Num = Accum;
    return i;
}


/**
 * @brief Parse an arg list for param-value pairs
 * @param Param where to store the parameter pointer
 * @param Value where to store the value pointer
 * @param TokenBuf token buffer with space for at least MaxLength+1 chars
 * @param MaxLength token buffer size
 * @param k starting arg
 * @param Start starting character position in argv[k]
 * @param argc, argv the arg list
 * @param Delimiters characters that delimit parameters from values
 * @return index of the arg containing the value
 * 
 * For example, with Delimiters = ":", the routine parses arg lists of
 * the form 
 *    {"param1:value1", "param2:value2", "param3:value3"}.
 * It is flexible to allow argument breaks around the delimiter, including
 * any of the following syntaxes:
 *    {"param", "value"},
 *    {"param:", "value"},
 *    {"param", ":", "value"},
 *    {"param", ":", "", "value"}.
 * 
 * The routine can be used as
@code
    char TokenBuf[256];
    int k = 1;
    while(k < argc)
    {
        kread = CliParseArglist(&Param, &Value, TokenBuf, sizeof(TokenBuf),
            k, argv[k], argc, argv, ":");
        printf("Read parameter %s = value %s\n", Param, Value);
        k = kread + 1;
    }
@endcode
 */
int CliParseArglist(const char **Param, const char **Value, 
    char *TokenBuf, int MaxLength, int k, const char *Start, 
    int argc, const char *argv[], const char *Delimiters)
{
    int TokLen, kread = k;
    
    *Param = *Value = NULL;
    TokLen = GetStrToken(TokenBuf, Start, MaxLength, Delimiters);
    
    if(TokLen > MaxLength) /* Token is too long */
        *Value = Start;
    else
    {
        *Param = TokenBuf;
        
        /* Check for a non-null character after token (a delimiter) 
            followed by at least one more non-null character. */
        if(Start[TokLen] && Start[TokLen + 1])
            *Value = &Start[TokLen + 1];
        else    /* Otherwise, scan ahead for the value */
            for(kread = k + 1; kread < argc; kread++)
                if(!argv[kread][0])             /* Null arg */
                    continue;
                else if(!strchr(Delimiters, argv[kread][0]))
                {
                    *Value = &argv[kread][0];
                    break;
                }
                else if(argv[kread][1])
                {
                    *Value = &argv[kread][1];    /* Skip first character */
                    break;
                }
    }
    
    return kread;
}


/** @brief Strictly read a num value */
int CliGetNum(num *Value, const char *String, const char *Param)
{
    double DoubleValue;
    int i;
    
    if(!Value)
    {
        fprintf(stderr, "Null pointer.\n");
        return 0;
    }
    else if(!String)
    {
        fprintf(stderr, "Expected a number for %s.\n", Param);
        return 0;
    }
        
    i = ParseDouble(&DoubleValue, String);
    
    if(!i || String[i])   /* No number read, or ends with non-null character */
    {
        *Value = 0;
        fprintf(stderr, "Invalid syntax \"%s\".\n", String);
        return 0;
    }    
    else
    {
        *Value = (num)DoubleValue;
        return 1;
    }
}


/** @brief Read a matrix from a text file */
int ReadMatrixFromTextFile(image *f, const char *FileName)
{
    FILE *File;
    num *Dest = NULL;
    double Value;
    long DestNumEl = 0, DestCapacity = 64;
    int c, Line = 1, Col = 0, NumRows = 0, NumCols = 0;
    
    if(!f)
        return 0;
    
    *f = NullImage;
    
    if(!(File = fopen(FileName, "rt")))
    {
        fprintf(stderr, "Error reading \"%s\":\n", FileName);
        fprintf(stderr, "Unable to open file.\n");
        return 0;
    }
    
    /* Allocate an initial destination buffer, it will be resized as needed. */
    if(!(Dest = (num *)malloc(sizeof(num)*DestCapacity)))
        goto Catch;
    
    while(1)
    {
        /* Eat whitespace */
        do
        {
            c = getc(File);
            
            if(c == '\n' || c == '\r' || !isspace(c))
                break;
            else if(ferror(File))
            {
                fprintf(stderr, "Error reading \"%s\".\n", FileName);
                goto Catch;
            }
        }while(!feof(File));
        
        if(c == '#')
        {
            /* Found a comment, ignore the rest of the line. */
            do
            {
                c = getc(File);
                
                if(c == '\n' || c == '\r')
                    break;
                else if(ferror(File))
                {
                    fprintf(stderr, "Error reading \"%s\".\n", FileName);
                    goto Catch;
                }
            }while(!feof(File));
        }
        
        if(c == EOF || c == '\n' || c == '\r')
        {
            if(Col) /* End of a non-empty line */
            {
                if(!NumCols)
                    NumCols = Col;
                else if(NumCols != Col)
                {
                    fprintf(stderr, 
                        "Error reading \"%s\" on line %d:\n"
                        "Rows must have a consistent number of elements.\n", 
                        FileName, Line);
                    goto Catch;
                }
                
                NumRows++;
                Col = 0;
            }
            
            if(c == EOF)
                break;
            
            Line++;
        }
        else
        {
            /* There should be a number, try to read it. */
            ungetc(c, File);
        
            if(fscanf(File, "%lg", &Value) != 1)
            {
                fprintf(stderr, 
                    "Error reading \"%s\" on line %d:\nInvalid number.\n",
                    FileName, Line);
                goto Catch;     /* Failed to parse number */
            }
            
            /* Put Value into Dest */
            if(DestNumEl == DestCapacity)
            {
                /* Increase Dest capacity by 10% */
                DestCapacity += DestCapacity/10 + 1;
                
                if(!(Dest = (num *)realloc(Dest, sizeof(num)*DestCapacity)))
                {
                    fprintf(stderr, "Memory allocation failed.\n");
                    goto Catch;
                }
            }
            
            Dest[DestNumEl++] = (num)Value;
            Col++;
        }   
    }
    
    fclose(File);
    f->Data = Dest;
    f->Width = NumCols;
    f->Height = NumRows;
    f->NumChannels = 1;
    return 1;
Catch:    
    fclose(File);    
    if(Dest)
        free(Dest);
    return 0;
}


/** @brief Read a matrix from a text or image file */
int ReadMatrixFromFile(image *f, const char *FileName, 
    int (*RescaleFun)(image *f))
{
    char Type[8];
    
    if(!f)
        return 0;
    
    /* If the file is not a known image type, attempt to read it as
       a text file. */
    if(!IdentifyImageType(Type, FileName))
        return ReadMatrixFromTextFile(f, FileName);
    
    /* The file appears to be an image type, attempt to read it. */
    if(!ReadImageObjGrayscale(f, FileName))
    {
        fprintf(stderr, "Error reading \"%s\".\n", FileName);
        return 0;
    }
    
    if(RescaleFun && !RescaleFun(f))
    {
        FreeImageObj(*f);
        *f = NullImage;
        return 0;
    }
        
    return 1;
}
