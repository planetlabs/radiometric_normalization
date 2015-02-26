''' Creates the time stacks for further analysis
'''

import numpy
import logging

from osgeo import gdal, gdal_array


def generate(image_paths, method='identity',
             output_path='time_stack.tif', output_nodata=2 ** 15 - 1):
    ''' This takes a list of image paths and creates a time stack image.

    It assumes that all images
        - contain the same number of bands
        - the bands are in the same band order
        - all the images are of the same size
        - all the images have the same nodata value

    The output nodata value is 2 ** 15 - 1

    Since the output file is only meant to be used internally, the
    geographic information isn't carried over to the output file.

    Input:
        image_paths (list of str): A list of the input file paths
        method (str): Time stack creation method [identity]
        output_path (str): A path to write the file to
        output_nodata (number): The no data value for the output image
    '''

    all_bands, nodata = _read_in_bands(image_paths)
    output_bands, mask = _calculate_value_and_weight(
        all_bands, nodata, method, output_nodata)
    _write_out_bands(output_bands, mask, output_path, output_nodata)

    return output_path


def _read_in_bands(image_paths):
    ''' Reads in a list of image paths and outputs a list of numpy arrays
    representing the bands.

    Input:
        image_paths (list of str): A list of the input file paths

    Output:
        all_bands (list of arrays of numbers): A list of each band,
            each band consisting of a stacked array of the images
        nodata (number): The nodata value for this raster set
    '''

    all_images, nodata = _read_in_gdal_images(image_paths)
    all_bands = _organise_images_to_bands(all_images)

    return all_bands, nodata


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

        all_images.append(image_ds.ReadAsArray())

    return all_images, nodata


def _organise_images_to_bands(all_images):
    ''' Organises the arrays of images by band

    Input:
        all_images (list of arrays of numbers): A list of each image,
            each image consisting of a stacked array of the bands

    Output:
        all_bands (list of arrays of numbers): A list of each band,
            each band consisting of a stacked array of the images
    '''

    no_images = len(all_images)
    no_bands, rows, cols = all_images[0].shape
    all_bands = []
    for band in range(no_bands):
        one_band = numpy.zeros((no_images, rows, cols),
                               dtype=all_images[0].dtype)
        for image in range(no_images):
            one_band[image, :, :] = all_images[image][band, :, :]
        all_bands.append(one_band)

    return all_bands


def _calculate_value_and_weight(all_bands, nodata,
                                method, output_nodata):
    ''' Calculates the time stack from a list of the bands

    Input:
        all_bands (list of arrays of numbers): A list of each band,
            each band consisting of a stacked array of the images
        nodata (number): The nodata value for this raster set
        method (str): Change between different methods of calculating the
            time stack
        output_nodata (number): The no data value for the output image

    Output:
        output_bands (list of arrays of numbers): A list of arrays, each
            representing one band of the time stack
        mask (array of numbers): An array representing the weight of the
            time stack
    '''

    if method is 'identity':
        logging.info('Identity method chosen.')

        _, rows, cols = all_bands[0].shape
        no_bands = len(all_bands)
        output_bands = numpy.zeros((no_bands, rows, cols),
                                   dtype=all_bands[0].dtype)
        for band in range(no_bands):
            for row in range(rows):
                for col in range(cols):
                    valid_pixels = all_bands[band][numpy.nonzero(
                        all_bands[band][:, row, col] != nodata), row, col]
                    if len(valid_pixels) == 0:
                        output_bands[band, row, col] = output_nodata
                    else:
                        output_bands[band, row, col] = numpy.mean(valid_pixels)

        mask = numpy.ones((rows, cols), dtype=all_bands[0].dtype)*65535

    return output_bands, mask


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

    no_bands, rows, cols = output_bands.shape

    # Set up output file
    options = ['ALPHA=YES']
    if no_bands == 3:
        options.append('PHOTOMETRIC=RGB')
    datatype = gdal_array.NumericTypeCodeToGDALTypeCode(
        output_bands.dtype.type)
    gdal_ds = gdal.GetDriverByName('GTIFF').Create(
        output_path, cols, rows, no_bands + 1, datatype,
        options=options)

    # Write output file data
    for i in range(no_bands):
        gdal_array.BandWriteArray(
            gdal_ds.GetRasterBand(i + 1),
            output_bands[i, :, :])
    gdal_ds.GetRasterBand(1).SetNoDataValue(output_nodata)

    # Set the alpha band
    # To conform to 16 bit TIFF alpha expectations transform
    # alpha to 16bit.
    alpha_band = gdal_ds.GetRasterBand(gdal_ds.RasterCount)
    if alpha_band.DataType == gdal.GDT_UInt16:
        mask = ((mask.astype(numpy.uint32) * 65535) / 255).astype(
            numpy.uint16)
    gdal_array.BandWriteArray(alpha_band, mask)
    logging.info('Successfully wrote output file as: ' + output_path)
