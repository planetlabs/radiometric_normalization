/**
 * @file imageio.h
 * @brief Implements ReadImage and WriteImage functions
 * @author Pascal Getreuer <getreuer@gmail.com>
 * 
 * 
 * Copyright (c) 2010-2012, Pascal Getreuer
 * All rights reserved.
 * 
 * This program is free software: you can use, modify and/or 
 * redistribute it under the terms of the simplified BSD License. You 
 * should have received a copy of this license along this program. If 
 * not, see <http://www.opensource.org/licenses/bsd-license.html>.
 */

#ifndef _IMAGEIO_H_
#define _IMAGEIO_H_

#include <stdio.h>
#include "basic.h"

/** @brief Limit on the maximum allowed image width or height (security). */
#define MAX_IMAGE_SIZE 10000


#ifndef DOXYGEN

/* Build string macros listing the supported formats */
#ifdef USE_LIBJPEG
#define SUPPORTEDSTRING_JPEG	"/JPEG"
#else
#define SUPPORTEDSTRING_JPEG	""
#endif
#ifdef USE_LIBPNG
#define SUPPORTEDSTRING_PNG		"/PNG"
#else
#define SUPPORTEDSTRING_PNG		""
#endif
#ifdef USE_LIBTIFF
#define SUPPORTEDSTRING_TIFF	"/TIFF"
#else
#define SUPPORTEDSTRING_TIFF	""
#endif

/* Definitions for specifying image formats */
#define IMAGEIO_U8            0x0000
#define IMAGEIO_SINGLE        0x0001
#define IMAGEIO_FLOAT         IMAGEIO_SINGLE
#define IMAGEIO_DOUBLE        0x0002
#define IMAGEIO_STRIP_ALPHA   0x0010
#define IMAGEIO_BGRFLIP       0x0020
#define IMAGEIO_AFLIP         0x0040
#define IMAGEIO_GRAYSCALE     0x0080
#define IMAGEIO_GRAY          IMAGEIO_GRAYSCALE
#define IMAGEIO_PLANAR        0x0100
#define IMAGEIO_COLUMNMAJOR   0x0200
#define IMAGEIO_RGB           (IMAGEIO_STRIP_ALPHA)
#define IMAGEIO_BGR           (IMAGEIO_STRIP_ALPHA | IMAGEIO_BGRFLIP)
#define IMAGEIO_RGBA          0x0000
#define IMAGEIO_BGRA          (IMAGEIO_BGRFLIP)
#define IMAGEIO_ARGB          (IMAGEIO_AFLIP)
#define IMAGEIO_ABGR          (IMAGEIO_BGRFLIP | IMAGEIO_AFLIP)

#endif /* DOXYGEN */


/** 
 * @brief String macro listing supported formats for \c ReadImage
 * 
 * This macro can be used for example as
@code
    printf("Supported formats for reading: " READIMAGE_FORMATS_SUPPORTED ".\n");
@endcode
 */
#define READIMAGE_FORMATS_SUPPORTED	\
    "BMP" SUPPORTEDSTRING_JPEG SUPPORTEDSTRING_PNG SUPPORTEDSTRING_TIFF
    
/** @brief String macro listing supported formats for \c WriteImage */
#define WRITEIMAGE_FORMATS_SUPPORTED	\
    "BMP" SUPPORTEDSTRING_JPEG SUPPORTEDSTRING_PNG SUPPORTEDSTRING_TIFF

#ifndef _CRT_SECURE_NO_WARNINGS
/** @brief Avoid MSVC warnings on using fopen */
#define _CRT_SECURE_NO_WARNINGS
#endif

int IdentifyImageType(char *Type, const char *FileName);

void *ReadImage(int *Width, int *Height, 
    const char *FileName, unsigned Format);

int WriteImage(void *Image, int Width, int Height, 
    const char *FileName, unsigned Format, int Quality);
    
#endif /* _IMAGEIO_H_ */
