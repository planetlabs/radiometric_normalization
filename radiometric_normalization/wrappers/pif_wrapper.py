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
import logging
import numpy

from osgeo import gdal

from radiometric_normalization import gimage
from radiometric_normalization import pif, pif_modified


def generate(candidate_path, reference_path,
             method='filter_alpha', method_options=None,
             last_band_alpha=False):
    ''' Generates psuedo invariant features as a mask

    :param str candidate_path: Path to the candidate image
    :param str reference_path: Path to the reference image
    :param str method: Which psuedo invariant feature generation method to use
    :param object method_options: A passthrough argument for any specific
        options for the method chosen:
            - Not applicable for 'filter_alpha'
            - The width of the filter for 'filter_PCA'

    :returns: A boolean array in the same coordinate system of the
        candidate/reference image (True for the PIF)
    '''
    if method == 'filter_alpha':
        _, c_alpha, c_band_count = _open_image_and_get_info(
            candidate_path, last_band_alpha)
        _, r_alpha, r_band_count = _open_image_and_get_info(
            reference_path, last_band_alpha)

        _assert_consistent(c_alpha, r_alpha, c_band_count, r_band_count)
        combined_alpha = numpy.logical_and(c_alpha, r_alpha)

        pif_mask = pif.generate_alpha_band_pifs(combined_alpha)
    elif method == 'filter_PCA':
        c_ds, c_alpha, c_band_count = _open_image_and_get_info(
            candidate_path, last_band_alpha)
        r_ds, r_alpha, r_band_count = _open_image_and_get_info(
            reference_path, last_band_alpha)

        _assert_consistent(c_alpha, r_alpha, c_band_count, r_band_count)
        combined_alpha = numpy.logical_and(c_alpha, r_alpha)

        if method_options:
            parameters = method_options
        else:
            parameters = pif.DEFAULT_PCA_OPTIONS

        pif_mask = numpy.ones(c_alpha.shape, dtype=numpy.bool)
        for band_no in range(1, c_band_count + 1):
            logging.info('PIF: Band {}'.format(band_no))
            c_band = gimage.read_single_band(c_ds, band_no)
            r_band = gimage.read_single_band(r_ds, band_no)
            pif_band_mask = pif_modified.generate_pca_pifs(
                c_band, r_band, combined_alpha)
            pif_mask = numpy.logical_and(pif_mask, pif_band_mask)

        no_total_pixels = c_alpha.size
        no_valid_pixels = len(numpy.nonzero(pif_mask)[0])
        valid_percent = 100.0 * no_valid_pixels / no_total_pixels
        logging.info(
            'PIF: Found {} final pifs out of {} pixels ({}%) for all '
            'bands'.format(no_valid_pixels, no_total_pixels, valid_percent))
    elif method == 'filter_robust':
        c_ds, c_alpha, c_band_count = _open_image_and_get_info(
            candidate_path, last_band_alpha)
        r_ds, r_alpha, r_band_count = _open_image_and_get_info(
            reference_path, last_band_alpha)

        _assert_consistent(c_alpha, r_alpha, c_band_count, r_band_count)
        combined_alpha = numpy.logical_and(c_alpha, r_alpha)

        if method_options:
            parameters = method_options
        else:
            parameters = pif.DEFAULT_ROBUST_OPTIONS

        pif_mask = numpy.ones(c_alpha.shape, dtype=numpy.bool)
        for band_no in range(1, c_band_count + 1):
            logging.info('PIF: Band {}'.format(band_no))
            c_band = gimage.read_single_band(c_ds, band_no)
            r_band = gimage.read_single_band(r_ds, band_no)
            pif_band_mask = pif.generate_robust_pifs(
                c_band, r_band, combined_alpha, parameters)
            pif_mask = numpy.logical_and(pif_mask, pif_band_mask)

        no_total_pixels = c_alpha.size
        no_valid_pixels = len(numpy.nonzero(pif_mask)[0])
        valid_percent = 100.0 * no_valid_pixels / no_total_pixels
        logging.info(
            'PIF: Found {} final pifs out of {} pixels ({}%) for all '
            'bands'.format(no_valid_pixels, no_total_pixels, valid_percent))
    else:
        raise NotImplementedError('Only "filter_alpha", "filter_PCA" and '
                                  '"filter_robust" methods are implemented.')

    return pif_mask


def _open_image_and_get_info(path, last_band_alpha):
    gdal_ds = gdal.Open(path)
    alpha_band, band_count = gimage.read_alpha_and_band_count(
        gdal_ds, last_band_alpha=last_band_alpha)
    return gdal_ds, alpha_band, band_count


def _assert_consistent(c_alpha, r_alpha, c_band_count, r_band_count):
    assert r_band_count == c_band_count
    assert r_alpha.shape == c_alpha.shape
