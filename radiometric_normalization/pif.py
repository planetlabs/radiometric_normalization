''' Creates a list of pseudo invariant features
'''

import numpy

from osgeo import gdal


def generate(candidate_path, time_stack_path='time_stack.tif',
             method='identity'):
    ''' Generates psuedo invariant features as a list of pixel pairs

    Input:
        time_stack_path (str): A path to the time stack image
        method (str): Which psuedo invariant feature generation method to use

    Output:
        pixel_pairs (list of pixel pair dict): Pixel pairs are
            {'coordinates', (int, int),
             'weighting', float,
             'reference', list of numbers,
             'candidate', list of numbers}
    '''
    time_stack_ds = gdal.Open(time_stack_path)
    no_bands_reference = time_stack_ds.RasterCount
    time_stack_full = time_stack_ds.ReadAsArray()
    time_stack_ds.RasterXSize
    time_stack = time_stack_full[0:no_bands_reference - 1, :, :]
    weighting = time_stack_full[no_bands_reference - 1, :, :]

    candidate_ds = gdal.Open(candidate_path)
    no_bands_candidate = candidate_ds.RasterCount
    candidate_full = candidate_ds.ReadAsArray()
    candidate = candidate_full[0:no_bands_candidate - 1, :, :]
    candidate_mask = candidate_full[no_bands_candidate - 1, :, :]

    assert no_bands_reference == no_bands_candidate, \
        '%s and %s have different number of bands' % \
        (candidate_path, time_stack_path)
    assert candidate_ds.RasterXSize == time_stack_ds.RasterXSize and \
        candidate_ds.RasterYSize == time_stack_ds.RasterYSize, \
        '%s and %s have different sizes' % \
        (candidate_path, time_stack_path)

    if method is 'identity':
        pixel_pairs = _filter_nodata(time_stack, weighting,
                                     candidate, candidate_mask)

    return pixel_pairs


def _filter_nodata(time_stack, weighting, candidate, candidate_mask):
    ''' Creates the pixel pairs from the time stack.

    Input:
        time_stack (array of numbers): The time stack of shape:
            (no_bands, rows, cols)
        weighting (array of numbers): The weightings of the time stack
            (0 is nodata)
        time_stack (array of numbers): The candidate image of shape:
            (no_bands, rows, cols)
        candidate_mask (array of numbers): The nodata mask of the candidate
            image (0 is nodata)

    Output:
        pixel_pairs (list of pixel pair dict): Pixel pairs are
            {'coordinates', (int, int),
             'weighting', float,
             'reference', list of numbers,
             'candidate', list of numbers}
    '''
    all_mask = numpy.logical_not(numpy.logical_or(
        weighting == 0, candidate_mask == 0))

    valid_pixels = numpy.nonzero(all_mask)

    pixel_pairs = []
    for pixel in range(len(valid_pixels[0])):
        row = valid_pixels[0][pixel]
        col = valid_pixels[1][pixel]
        pixel_dict = {'coordinates': (row, col),
                      'weighting': weighting[row, col],
                      'reference': time_stack[:, row, col],
                      'candidate': candidate[:, row, col]}
        pixel_pairs.append(pixel_dict)

    return pixel_pairs
