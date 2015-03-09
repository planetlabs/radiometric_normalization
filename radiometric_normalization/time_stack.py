'''
Copyright 2015 Planet Labs, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''

''' Creates the time stacks for further analysis
'''

import numpy
import logging

from osgeo import gdal

from radiometric_normalization import gimage


def generate(image_paths, output_path, method='identity'):
    '''Synthesizes a time stack image set into a single reference image.

    All images in time stack must:
        - contain the same number of bands
        - have the same band order
        - have the same size
        - have the same nodata value

    The output file is only supposed to be used internally so options to change
    the nodata value and the datatype are not exposed.
    - The output nodata value is 2 ** 15 - 1
    - The output image data type is uint16
    - Since the output file is only meant to be used internally, the
        geographic information isn't carried over to the output file.

    Input:
        image_paths (list of str): A list of paths for input time stack images
        method (str): Time stack analysis method [identity]
        output_path (str): A path to write the file to
    '''

    output_nodata = 2 ** 15 - 1
    output_datatype = numpy.uint16

    all_bands = _read_in_bands(image_paths)
    if method is 'identity':
        output_bands, mask = _mean_with_uniform_weight(
            all_bands, output_nodata, output_datatype)

    output_gimage = gimage.GImage(output_bands, mask, {})
    gimage.save(output_gimage, output_path, nodata=output_nodata)

    return output_path


def _read_in_bands(image_paths):
    ''' Reads in a list of image paths and outputs a list of numpy arrays
    representing the bands.

    Input:
        image_paths (list of str): A list of the input file paths

    Output:
        all_bands (list of list of arrays): A list of each band,
            each band consisting of a list of images each entry being a
            masked array of the image data
    '''

    all_images, nodata = _read_in_gdal_images(image_paths)
    all_bands = _organize_images_to_bands(all_images, nodata)

    return all_bands


def _read_in_gdal_images(image_paths):
    ''' Uses GDAL to read in the image as a numpy array and check them.

    Input:
        image_paths (list of str): A list of the input file paths

    Output:
        all_images (list of arrays of numbers): A list of each image,
            each image consisting of a stacked array of the bands
        nodata (number): The nodata value for this raster set
    '''

    # Find some constants
    test_image_ds = gdal.Open(image_paths[0])
    no_bands = test_image_ds.RasterCount
    cols = test_image_ds.RasterXSize
    rows = test_image_ds.RasterYSize
    nodata = test_image_ds.GetRasterBand(1).GetNoDataValue()

    # Read in each image as a numpy array
    all_images = []
    for image_path in image_paths:
        image_ds = gdal.Open(image_path)

        logging.info('Processing ' + image_path + '...')
        assert image_ds.RasterCount == no_bands, \
            '{} has a different number of bands'.format(image_path)
        assert image_ds.RasterXSize == cols and image_ds.RasterYSize == rows, \
            '{} has a different size'.format(image_path)
        assert image_ds.GetRasterBand(1).GetNoDataValue() == nodata, \
            '{} has a different nodata value'.format(image_path)
        alpha_band = image_ds.GetRasterBand(image_ds.RasterCount)
        assert alpha_band.GetColorInterpretation() != gdal.GCI_AlphaBand, \
            '{} has an alpha band'.format(image_path)

        all_images.append(image_ds.ReadAsArray())

    return all_images, nodata


def _organize_images_to_bands(all_images, nodata):
    ''' Organizes the arrays of images by band

    Input:
        all_images (list of arrays of numbers): A list of each image,
            each image consisting of a stacked array of the bands

    Output:
        all_bands (list of list of arrays): A list of each band,
            each band consisting of a list of images each entry being a
            masked array of the image data
    '''

    no_images = len(all_images)
    no_bands = all_images[0].shape[0]
    all_bands = []
    for band in range(no_bands):
        one_band = \
            [numpy.ma.masked_equal(all_images[i][band, :, :], nodata)
             for i in range(no_images)]
        all_bands.append(one_band)

    return all_bands


def _mean_with_uniform_weight(all_bands, output_nodata, output_datatype):
    ''' Calculates the reference from a list of the bands

    Input:
        all_bands (list of list of arrays): A list of each band,
            each band consisting of a list of images each entry being a
            masked array of the image data
        output_nodata (number): The no data value for the output image
        output_datatype (numpy datatype): Data type for the output image

    Output:
        output_bands (list of arrays of numbers): A list of arrays, each
            representing one band of the time stack
        mask (array of numbers): An array representing the weight of the
            time stack
    '''

    logging.info('Time stack analysis is using: Mean with uniform weight.')

    rows, cols = all_bands[0][0].shape
    all_bands_mask = numpy.zeros((rows, cols))  # 0 for valid; 1 for nodata
    output_bands = []
    for band in all_bands:
        masked_mean = numpy.ma.mean(band, axis=0)
        band_mean = masked_mean.data
        band_mask = masked_mean.mask
        band_mean[numpy.nonzero(band_mask)] = output_nodata
        all_bands_mask = numpy.logical_or(band_mask, all_bands_mask)
        output_bands.append(band_mean.astype(output_datatype))

    output_mask = numpy.logical_not(all_bands_mask).astype(output_datatype) * \
        numpy.iinfo(output_datatype).max

    return output_bands, output_mask
