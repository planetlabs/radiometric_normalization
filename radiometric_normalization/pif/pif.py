import numpy

from osgeo import gdal


def generate(candidate_path, time_stack_path='time_stack.tif', method='identity'):
    ''' Generates psuedo invariant features as a list of pixel pairs

    Input:
        time_stack_path (str): A path to the time stack image
        method (str): Which psuedo invariant feature generation method to use

    Output:
        pixel_pairs (list of pixel pair dict): Pixel pairs are
            {'coordinates', (int, int), 'weighting', float,
             'reference_bands', list of doubles,
             'candidate bands', list of doubles}

    '''

    time_stack_ds = gdal.Open(time_stack_path)
    nodata = time_stack_ds.GetRasterBand(1).GetNoDataValue()
    no_bands_reference = time_stack_ds.RasterCount
    time_stack_full = time_stack_ds.ReadAsArray()
    time_stack = time_stack_full[0:no_bands-1, :, :]
    weighting = time_stack_full[no_bands-1, :, :]

    candidate_ds = gdal.Open(candidate_path)
    no_bands_candidate = candidate_ds.RasterCount
    candidate_full = candidate_ds.ReadAsArray()
    candidate = candidate_full[0:no_bands-1, :, :]
    mask_candidate = candidate_full[no_bands-1, :, :]

    pixel_pairs = _create_pif_dict(time_stack, weighting, nodata,
                                   candidate, mask_candidate, method)

    return pixel_pairs


def _create_pif_dict(time_stack, weighting, nodata,
                     candidate, mask_candidate, method):
    ''' Creates the pixel pairs from the time stack.

    Input:
        time_stack (array of numbers): The time stack of
            shape: (no_bands, rows, cols)
        weighting (array of numbers): The weightings
        nodata (number): The no data value
        method (str): The PIF generation method
    Output:
        pixel_pairs (list of pixel pair dict): Pixel pairs are
            {'coordinates', (int, int), 'weighting', float,
             'reference_bands', list of doubles,
             'candidate bands', list of doubles}
    '''

    if method is 'identity':
        pixel_pairs = []
        no_bands, rows, cols = time_stack.shape
        for row in range(rows):
            for col in range(cols):
                if mask_candidate[row, col] != 0:
                    time_stack_bands = [
                        time_stack[band, row, col] for band in range(no_bands)]
                    candidate_bands = [
                        candidate[band, row, col] for band in range(no_bands)]
                    if len(numpy.nonzero(time_stack_bands == nodata)) == 0:
                        pixel_dict = {'coordinates', (row, col),
                                      'weighting', weighting[row, col],
                                      'reference_bands', time_stack_bands
                                      'candidate_bands', candidate_bands}
                        pixel_pairs.append(pixel_dict)

    return pixel_pairs
