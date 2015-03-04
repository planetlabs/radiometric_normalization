''' Creates the time stacks for further analysis
'''

import numpy
import logging

from osgeo import gdal, gdal_array


def generate(image_paths, method='identity', output_path='time_stack.tif'):
    ''' This takes a list of image paths and creates a time stack image.

    It assumes that all images
        - contain the same number of bands
        - the bands are in the same band order
        - all the images are of the same size
        - all the images have the same nodata value

    The output file is only supposed to be used internally so options to change
    the nodata value and the datatype are not exposed.
    - The output nodata value is 2 ** 15 - 1
    - The output image data type is uint16
    - Since the output file is only meant to be used internally, the
        geographic information isn't carried over to the output file.

    Input:
        image_paths (list of str): A list of the input file paths
        method (str): Time stack creation method [identity]
        output_path (str): A path to write the file to
    '''

    output_nodata = 2 ** 15 - 1
    output_datatype = numpy.uint16

    all_bands = _read_in_bands(image_paths)
    if method is 'identity':
        output_bands, mask = _mean_with_uniform_weight(
            all_bands, output_nodata, output_datatype)
    _write_out_bands(output_bands, mask, output_path, output_nodata)

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
    all_bands = _organise_images_to_bands(all_images, nodata)

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
            '%s has a different number of bands' % image_path
        assert image_ds.RasterXSize == cols and image_ds.RasterYSize == rows, \
            '%s has a different size' % image_path
        assert image_ds.GetRasterBand(1).GetNoDataValue() == nodata, \
            '%s has a different nodata value' % image_path
        alpha_band = image_ds.GetRasterBand(image_ds.RasterCount)
        assert alpha_band.GetColorInterpretation() != gdal.GCI_AlphaBand, \
            '%s has an alpha band' % image_path

        all_images.append(image_ds.ReadAsArray())

    return all_images, nodata


def _organise_images_to_bands(all_images, nodata):
    ''' Organises the arrays of images by band

    Input:
        all_images (list of arrays of numbers): A list of each image,
            each image consisting of a stacked array of the bands

    Output:
        all_bands (list of list of arrays): A list of each band,
            each band consisting of a list of images each entry being a
            masked array of the image data
    '''

    no_images = len(all_images)
    no_bands, _, _ = all_images[0].shape
    all_bands = []
    for band in range(no_bands):
        one_band = \
            [numpy.ma.masked_equal(all_images[image][band, :, :], nodata)
             for image in range(no_images)]
        all_bands.append(one_band)

    return all_bands


def _mean_with_uniform_weight(all_bands, output_nodata, output_datatype):
    ''' Calculates the time stack from a list of the bands

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


def _write_out_bands(output_bands, mask,
                     output_path, output_nodata):
    ''' Writes out the time stack to a GeoTIFF

    Input:
        output_bands (list of arrays of numbers): A list of arrays, each
            representing one band of the time stack
        mask (array of numbers): An array representing the weight of the
            time stack
        output_path (str): The path to save the image at
        output_nodata (number): The no data value for the output image
    '''

    no_bands = len(output_bands)
    rows, cols = output_bands[0].shape

    # Set up output file
    options = ['ALPHA=YES']
    datatype = gdal_array.NumericTypeCodeToGDALTypeCode(
        output_bands[0].dtype.type)
    gdal_ds = gdal.GetDriverByName('GTIFF').Create(
        output_path, cols, rows, no_bands + 1, datatype,
        options=options)

    # Write output file data
    for i in range(no_bands):
        gdal_array.BandWriteArray(
            gdal_ds.GetRasterBand(i + 1),
            output_bands[i])
    gdal_ds.GetRasterBand(1).SetNoDataValue(output_nodata)

    # Set the alpha band
    alpha_band = gdal_ds.GetRasterBand(gdal_ds.RasterCount)
    gdal_array.BandWriteArray(alpha_band, mask)
    logging.info('Successfully wrote output file as: ' + output_path)
