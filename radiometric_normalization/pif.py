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
from collections import namedtuple

from radiometric_normalization import pca_filter
from radiometric_normalization import robust
from radiometric_normalization import filtering


pca_options = namedtuple('pca_options', 'threshold')
DEFAULT_PCA_OPTIONS = pca_options(threshold=30)

robust_options = namedtuple('robust_options', 'threshold')
DEFAULT_ROBUST_OPTIONS = robust_options(threshold=1000)


def generate_alpha_band_pifs(combined_alpha):
    ''' Creates the pseudo-invariant features from the reference and candidate
    alpha masks (filtering out pixels where either the candidate or reference
    is masked, i.e. the alpha value is False).

    :param array combined_alpha: A 2D array representing the alpha mask of the
                                 valid pixels in both the candidate array and
                                 reference array

    :returns: A 2-D boolean array representing pseudo invariant features
    '''
    logging.info('Pseudo invariant feature generation is using: Filtering '
                 'using the alpha mask.')

    # Only analyse valid pixels
    valid_pixels = numpy.nonzero(combined_alpha)

    pif_mask = numpy.zeros(combined_alpha.shape, dtype=numpy.bool)
    pif_mask[valid_pixels] = True

    if logging.getLogger().getEffectiveLevel() <= logging.INFO:
        pif_pixels = numpy.nonzero(pif_mask)
        no_pif_pixels = len(pif_pixels[0])
        no_total_pixels = combined_alpha.size
        valid_percent = 100.0 * no_pif_pixels / no_total_pixels
        logging.info(
            'PIF Info: Found {} final PIFs out of {} pixels ({}%)'.format(
                no_pif_pixels, no_total_pixels, valid_percent))

    return pif_mask


def generate_robust_pifs(candidate_band, reference_band, combined_alpha,
                         parameters=DEFAULT_ROBUST_OPTIONS):
    ''' Performs a robust fit to the valid pixels and filters according
    to the distance from the fit line.

    :param array candidate_band: A 2D array representing the image data of the
                                 candidate band
    :param array reference_band: A 2D array representing the image data of the
                                 reference image
    :param array combined_alpha: A 2D array representing the alpha mask of the
                                 valid pixels in both the candidate array and
                                 reference array
    :param robust_options parameters: Method specific parameters. Currently:
        threshold (float): Representing the distance from the fit line
                           to look for PIF pixels

    :returns: A 2-D boolean array representing pseudo invariant features
    '''
    logging.info('Pseudo invariant feature generation is using: Filtering '
                 'using a robust fit.')

    # Only analyse valid pixels
    valid_pixels = numpy.nonzero(combined_alpha)

    # Robust fit
    gain, offset = robust.fit(
        candidate_band[valid_pixels], reference_band[valid_pixels])

    # Filter using the robust fit
    pif_mask = filtering.filter_by_residuals_from_line(
        candidate_band, reference_band, combined_alpha,
        threshold=parameters.threshold, line_gain=gain, line_offset=offset)

    if logging.getLogger().getEffectiveLevel() <= logging.INFO:
        _info_logging(candidate_band, reference_band,
                     valid_pixels, numpy.nonzero(pif_mask))

    return pif_mask


def generate_pca_pifs(candidate_band, reference_band, combined_alpha,
                      parameters=DEFAULT_PCA_OPTIONS):
    ''' Performs PCA analysis on the valid pixels and filters according
    to the distance from the principle eigenvector.

    :param array candidate_band: A 2D array representing the image data of the
                                 candidate band
    :param array reference_band: A 2D array representing the image data of the
                                 reference image
    :param array combined_alpha: A 2D array representing the alpha mask of the
                                 valid pixels in both the candidate array and
                                 reference array
    :param pca_options parameters: Method specific parameters. Currently:
        threshold (float): Representing the width of the PCA filter

    :returns: A 2-D boolean array representing pseudo invariant features
    '''
    logging.info('Pseudo invariant feature generation is using: Filtering '
                 'using PCA.')

    # Find PIFs
    pif_mask = pca_filter.get_pif_mask(
        candidate_band, reference_band, combined_alpha, parameters)

    if logging.getLogger().getEffectiveLevel() <= logging.INFO:
        _info_logging(candidate_band, reference_band,
                      numpy.nonzero(combined_alpha),
                      numpy.nonzero(pif_mask))

    return pif_mask


def _info_logging(c_band, r_band, valid_pixels, pif_pixels):
    ''' Optional logging information
    '''
    logging.info('PIF Info: Original corrcoef = {}'.format(
        numpy.corrcoef(c_band[valid_pixels], r_band[valid_pixels])[0, 1]))

    if pif_pixels[0] != [] and pif_pixels[1] != []:
        logging.info('PIF Info: Filtered corrcoef = {}'.format(
            numpy.corrcoef(c_band[pif_pixels], r_band[pif_pixels])[0, 1]))

        no_pif_pixels = len(pif_pixels[0])
        no_total_pixels = c_band.size
        valid_percent = 100.0 * no_pif_pixels / no_total_pixels
        logging.info(
            'PIF Info: Found {} final PIFs out of {} pixels ({}%)'.format(
                no_pif_pixels, no_total_pixels, valid_percent))
    else:
        logging.info('PIF Info: No PIF pixels found.')
