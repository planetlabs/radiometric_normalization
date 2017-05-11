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
import itertools
import logging
import numpy
from collections import namedtuple
from sklearn.decomposition import PCA


pca_options = namedtuple('pca_options', 'limit')
DEFAULT_PCA_OPTIONS = pca_options(limit=30)


def generate_alpha_band_pifs(candidate_alpha, reference_alpha):
    ''' Creates the pseudo-invariant features from the reference and candidate
    alpha masks (filtering out pixels where either the candidate or reference
    is masked, i.e. the alpha value is False).

    :param array candidate_alpha: A 2D array representing the alpha mask of the
                                  candidate image
    :param array reference_alpha: A 2D array representing the alpha mask of the
                                  reference image

    :returns: A 2-D boolean array representing pseudo invariant features
    '''
    logging.info('Pseudo invariant feature generation is using: Filtering '
                 'using the alpha mask.')
    combined_alpha = numpy.logical_and(candidate_alpha, reference_alpha)
    valid_pixels = numpy.nonzero(combined_alpha)

    if logging.getLogger().getEffectiveLevel() <= logging.INFO:
        no_total_pixels = candidate_alpha.size
        no_valid_pixels = len(valid_pixels[0])
        valid_percent = 100.0 * no_valid_pixels / no_total_pixels
        logging.info(
            'PCA Info: Found {} final pifs out of {} pixels ({}%)'.format(
                no_valid_pixels, no_total_pixels, valid_percent))

    pif_mask = numpy.zeros(candidate_alpha.shape, dtype=numpy.bool)
    pif_mask[valid_pixels] = True

    return pif_mask


def generate_pca_pifs(candidate_band, candidate_alpha,
                      reference_band, reference_alpha,
                      parameters=DEFAULT_PCA_OPTIONS):
    ''' Performs PCA analysis on the valid pixels and filters according
    to the distance from the principle eigenvector.

    :param array candidate_band: A 2D array representing the image data of the
                                 candidate band
    :param array candidate_alpha: A 2D array representing the alpha mask of the
                                  candidate image
    :param array reference_band: A 2D array representing the image data of the
                                  reference image
    :param array reference_alpha: A 2D array representing the alpha mask of the
                                  reference image
    :param pca_options parameters: Method specific parameters. Currently:
        limit (float): Representing the width of the PCA filter

    :returns: A 2-D boolean array representing pseudo invariant features
    '''
    logging.info('Pseudo invariant feature generation is using: Filtering '
                 'using PCA.')

    # Only analyse valid pixels
    combined_alpha = numpy.logical_and(candidate_alpha, reference_alpha)
    valid_pixels = numpy.nonzero(combined_alpha)

    # Find PIFs
    passed_pixels = _pca_fit_and_filter_valid_pixels(
        candidate_band[valid_pixels], reference_band[valid_pixels],
        parameters)
    pif_mask = _create_pif_mask(passed_pixels, combined_alpha)

    if logging.getLogger().getEffectiveLevel() <= logging.INFO:
        _pca_logging(candidate_band, reference_band,
                     valid_pixels, numpy.nonzero(pif_mask))

    return pif_mask


def _pca_logging(c_band, r_band, valid_pixels, pif_pixels):
    ''' Optional logging information
    '''
    logging.info('PCA Info: Original corrcoef = {}'.format(
        numpy.corrcoef(c_band[valid_pixels], r_band[valid_pixels])[0, 1]))

    if pif_pixels[0] != [] and pif_pixels[1] != []:
        logging.info('PCA Info: Filtered corrcoef = {}'.format(
            numpy.corrcoef(c_band[pif_pixels], r_band[pif_pixels])[0, 1]))

        no_pif_pixels = len(pif_pixels[0])
        no_total_pixels = c_band.size
        valid_percent = 100.0 * no_pif_pixels / no_total_pixels
        logging.info(
            'PCA Info: Found {} final PIFs out of {} pixels ({}%)'.format(
                no_pif_pixels, no_total_pixels, valid_percent))
    else:
        logging.info('PCA Info: No PIF pixels found.')


def _pca_fit_and_filter_valid_pixels(candidate_pixels, reference_pixels,
                                     parameters):
    ''' Performs PCA analysis, on the valid pixels and filters according
    to the distance from the principle eigenvector, for a single band.
    '''
    fitted_pca = _pca_fit_single_band(candidate_pixels, reference_pixels)
    passed_pixels = _pca_filter_single_band(
        fitted_pca, candidate_pixels, reference_pixels, parameters.limit)

    return passed_pixels


def _pca_fit_single_band(cand_valid, ref_valid):
    ''' Uses SK Learn PCA module to do PCA fit
    '''

    X = _numpy_array_from_2arrays(cand_valid, ref_valid)

    # SK Learn PCA
    pca = PCA(n_components=2)

    # Fit the points
    pca.fit(X)

    return pca


def _numpy_array_from_2arrays(array1, array2, dtype=numpy.uint16):
    '''Efficiently combine two 1-D arrays into a single 2-D array.

    Avoids large memory usage by creating the array using
    ``numpy.fromiter`` and then reshaping a view of the resulting
    record array.  This does the equivalent of:

        numpy.array(zip(array1, array2))

    but avoids holding a potentially large number of tuples in memory.

    >>> a = numpy.array([1, 2, 3])
    >>> b = numpy.array([4, 5, 6])
    >>> X = _numpy_array_from_2arrays(a, b)
    >>> X
    array([[1, 4],
           [2, 5],
           [3, 6]], dtype=uint16)

    :param array array1: A 1-D numpy array.
    :param array array2: A second 1-D numpy array.
    :param data-type dtype: Data type for array elements
        (must be same for both arrays)

    :returns: A 2-D numpy array combining the two input arrays
    '''

    array_dtype = [('x', dtype), ('y', dtype)]

    return numpy.fromiter(itertools.izip(array1, array2), dtype=array_dtype) \
                .view(dtype=dtype) \
                .reshape((-1, 2))


def _pca_filter_single_band(pca, cand_valid, ref_valid, limit):
    ''' Uses SciKit Learn PCA module to transform the data and filter
    '''

    X = _numpy_array_from_2arrays(cand_valid, ref_valid)
    X_trans = pca.transform(X)

    # This is ok, only a single tuple and a 1-D array is being created
    _, ref_valid_trans_list = zip(*X_trans)
    ref_valid_trans = numpy.array(ref_valid_trans_list)

    # Filter
    pixels_pass_filter = numpy.nonzero(numpy.logical_and(
        ref_valid_trans >= (limit * -1), ref_valid_trans <= limit))[0]

    return pixels_pass_filter


def _create_pif_mask(passed_pixels, alpha):
    ''' Converts the list of filtered pixels to a PIF array
    '''

    # PIF images
    pif = numpy.zeros(alpha.shape, dtype=numpy.bool)
    pif_vec = pif.ravel()
    alpha_vec = alpha.ravel()
    valid_pixels = numpy.nonzero(alpha_vec)[0]
    pif_vec[valid_pixels[passed_pixels]] = 1
    pif = numpy.reshape(pif_vec, alpha.shape)

    return pif
