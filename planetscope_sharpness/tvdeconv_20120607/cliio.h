/**
 * @file cliio.h
 * @brief Utilities for creating a command line interface
 * @author Pascal Getreuer <getreuer@gmail.com>
 */
#ifndef _CLIIO_H_
#define _CLIIO_H_

#include "num.h"
#include "imageio.h"

#ifdef NUM_SINGLE
#define IMAGEIO_NUM           (IMAGEIO_SINGLE)
#else
#define IMAGEIO_NUM           (IMAGEIO_DOUBLE)
#endif

/** @brief struct representing an image */
typedef struct
{
    /** @brief Float image data */
    num *Data;
    /** @brief Image width */
    int Width;
    /** @brief Image height */
    int Height;
    /** @brief Number of channels */
    int NumChannels;
} image;


int AllocImageObj(image *f, int Width, int Height, int NumChannels);
void FreeImageObj(image f);
int ReadImageObj(image *f, const char *FileName);
int ReadImageObjGrayscale(image *f, const char *FileName);
int WriteImageObj(image f, const char *FileName, int JpegQuality);

int IsGrayscale(num *Data, int Width, int Height);
int GetStrToken(char *Token, const char *Start, int MaxLength, const char *Delim);
int ParseDouble(double *Num, const char *String);
int CliParseArglist(const char **Param, const char **Value, 
    char *TokenBuf, int MaxLength, int k, const char *Start, 
    int argc, const char *argv[], const char *Delimiters);
int CliGetNum(num *Value, const char *String, const char *Param);
int ReadMatrixFromTextFile(image *f, const char *FileName);
int ReadMatrixFromFile(image *f, const char *FileName, 
    int (*RescaleFun)(image *f));

extern const image NullImage;

#endif /* _CLIIO_H_ */
