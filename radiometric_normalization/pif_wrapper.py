import numpy

from osgeo import gdal

from radiometric_normalization import gimage
from radiometric_normalization import pif


def generate(candidate_path, reference_path, method='filter_nodata',
             method_options=None):
    ''' Generates psuedo invariant features as a list of pixel pairs
    Input:
        candidate_path (str): Path to the candidate image
        reference_path (str): Path to the reference image
        method (str): Which psuedo invariant feature generation method to use
        method_options: A passthrough argument for any specific options for the
            method chosen:
                - Not applicable for 'filter_nodata'
                - The width of the filter for 'filter_PCA'
    Output:
        pif_weight (numpy uint16 array): A numpy array in the same coordinate
            system of the candidate/reference image with a weight for how
            a PIF the pixel is (0 for not a PIF)
    '''
    if method == 'filter_nodata':
        reference_gimg = gimage.load(reference_path)
        candidate_gimg = gimage.load(candidate_path)
        pif_mask = pif.generate_alpha_band_pifs(
            candidate_gimg.alpha, reference_gimg.alpha)
    elif method == 'filter_PCA':
        reference_ds = gdal.Open(reference_path)
        candidate_ds = gdal.Open(candidate_path)
        c_alpha, c_band_count = gimage._read_alpha_and_band_count(candidate_ds)
        r_alpha, r_band_count = gimage._read_alpha_and_band_count(reference_ds)

        assert r_band_count == c_band_count
        assert r_alpha.shape == c_alpha.shape

        pif_mask = numpy.ones(c_alpha.shape, dtype=numpy.bool)
        for band_no in range(1, c_band_count + 1):
            candidate_band = gimage._read_single_band(candidate_ds, band_no)
            reference_band = gimage._read_single_band(reference_ds, band_no)
            pif_band_mask = pif.generate_pca_pifs(
                candidate_band, reference_band,
                c_alpha, r_alpha, method_options)
            pif_mask = numpy.logical_and(pif_mask, pif_band_mask)
    else:
        raise NotImplementedError("Only 'filter_nodata' and 'PCA_filtering' "
                                  "methods are implemented.")

    return pif_mask
