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
    gimages by filtering out pixels where either the candidate or mask alpha
    value is zero (masked)
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

    pif_weight = numpy.zeros(candidate_alpha.shape, dtype=numpy.bool)
    pif_weight[valid_pixels] = True

    return pif_weight


def generate_pca_pifs(candidate_band, candidate_alpha,
                      reference_band, reference_alpha,
                      parameters=None):
    ''' Performs PCA analysis, on the valid pixels and filters according
    to the distance from the principle eigenvector.
    '''
    logging.info('Pseudo invariant feature generation is using: Filtering '
                 'using PCA.')

    if not parameters:
        parameters = DEFAULT_PCA_OPTIONS

    # Only analyse valid pixels
    combined_alpha = numpy.logical_and(candidate_alpha, reference_alpha)
    valid_pixels = numpy.nonzero(combined_alpha)

    # Find PIFs
    passed_pixels = _pca_fit_and_filter_valid_pixels(
        candidate_band[valid_pixels], reference_band[valid_pixels],
        parameters)

    if logging.getLogger().getEffectiveLevel() <= logging.INFO:
        no_pif_pixels = len(passed_pixels)
        no_total_pixels = candidate_band.size
        valid_percent = 100.0 * no_pif_pixels / no_total_pixels
        logging.info(
            'PCA Info: Found {} final pifs out of {} pixels ({}%)'.format(
                no_pif_pixels, no_total_pixels, valid_percent))

    return _create_pif_weights(passed_pixels, combined_alpha)


def _pca_fit_and_filter_valid_pixels(candidate_pixels, reference_pixels,
                                     parameters):
    ''' Performs PCA analysis, on the valid pixels and filters according
    to the distance from the principle eigenvector, for a single band.
    '''
    fitted_pca = _pca_fit_single_band(candidate_pixels, reference_pixels)
    passed_pixels = _pca_filter_single_band(
        fitted_pca, candidate_pixels, reference_pixels, parameters.limit)

    if logging.getLogger().getEffectiveLevel() <= logging.INFO:
        logging.info('PCA Info: Original corrcoef = {}'.format(
            numpy.corrcoef(candidate_pixels, reference_pixels)[0, 1]))

        for component_no in xrange(len(fitted_pca.components_)):
            logging.info(
                'PCA Info: Component {} is {} with an explained '
                'variance of {}'.format(
                    component_no,
                    fitted_pca.components_[component_no],
                    fitted_pca.explained_variance_ratio_[component_no]))

        candidate_pifs = candidate_pixels[passed_pixels]
        reference_pifs = reference_pixels[passed_pixels]
        logging.info('PCA Info: Filtered corrcoef = {}'.format(
            numpy.corrcoef(candidate_pifs, reference_pifs)[0, 1]))

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

    :param array1: A 1-D numpy array.

    :param array2: A second 1-D numpy array.

    :param dtype: Data type for array elements (must be same for both arrays)

    :returns: A 2-D numpy array.
    '''

    array_dtype = [('x', dtype), ('y', dtype)]

    return numpy.fromiter(itertools.izip(array1, array2), dtype=array_dtype) \
                .view(dtype=dtype) \
                .reshape((-1, 2))


def _pca_filter_single_band(pca, cand_valid, ref_valid, limit):
    ''' Uses SK Learn PCA module to transform the data and filter
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


def _create_pif_weights(passed_pixels, alpha):
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
