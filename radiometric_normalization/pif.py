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
from sklearn.decomposition import PCA

from radiometric_normalization import gimage


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

    reference_img = gimage.load(reference_path)
    candidate_img = gimage.load(candidate_path)

    if method == 'filter_nodata':
        pif_weight = _filter_zero_alpha_pifs(reference_img, candidate_img)
    if method == 'filter_PCA':
        pif_weight = _filter_PCA_pifs(reference_img, candidate_img,
                                      numpy.uint16(method_options))
    else:
        raise NotImplementedError("Only 'filter_nodata' and 'PCA_filtering' "
                                  "methods are implemented.")

    return pif_weight, reference_img, candidate_img


def _filter_zero_alpha_pifs(reference_gimage, candidate_gimage):
    ''' Creates the pseudo-invariant features from the reference and candidate
    gimages by filtering out pixels where either the candidate or mask alpha
    value is zero (masked)
    '''

    logging.info('Pseudo invariant feature generation is using: Filtering '
                 'using the alpha mask.')

    gimage.check_comparable([reference_gimage, candidate_gimage])

    all_mask = numpy.logical_not(numpy.logical_or(
        reference_gimage.alpha == 0, candidate_gimage.alpha == 0))

    valid_pixels = numpy.nonzero(all_mask)

    no_total_pixels = reference_gimage.bands[0].size
    no_valid_pixels = len(valid_pixels[0])
    valid_percent = 100.0 * no_valid_pixels / no_total_pixels
    logging.info('Found {} pifs out of {} pixels ({}%)'.format(
        no_valid_pixels, no_total_pixels, valid_percent))

    pif_weight = numpy.zeros(reference_gimage.bands[0].shape,
                             dtype=numpy.uint16)
    pif_weight[valid_pixels] = reference_gimage.alpha[valid_pixels]

    return pif_weight


def _filter_PCA_pifs(reference_gimage, candidate_gimage, lim):
    ''' Performs PCA analysis, on the valid pixels and filters according
    to the distance from the principle eigenvector.
    '''

    logging.info('Pseudo invariant feature generation is using: Filtering '
                 'using PCA.')

    gimage.check_comparable([reference_gimage, candidate_gimage])

    array_shape = candidate_gimage.bands[0].shape

    # Only analyse valid pixels
    alpha = numpy.ones(array_shape)
    alpha[numpy.nonzero(candidate_gimage.alpha == 0)] = 0
    alpha[numpy.nonzero(reference_gimage.alpha == 0)] = 0
    alpha_vec = alpha.ravel()
    valid_pixels_list = numpy.nonzero(alpha_vec != 0)

    PIF_all = numpy.ones(array_shape)

    # Per band analysis
    for band_no in xrange(len(candidate_gimage.bands)):

        logging.info('PCA Info: Band {}'.format(band_no))

        PIF_band = _PCA_fit_and_filter_single_band(
            candidate_gimage.bands[band_no].ravel(),
            reference_gimage.bands[band_no].ravel(),
            valid_pixels_list, array_shape, lim)
        PIF_all = numpy.logical_and(PIF_band, PIF_all)

    if logging.getLogger().getEffectiveLevel() <= logging.INFO:
        for band_no in xrange(len(candidate_gimage.bands)):
            cand_combo_filtered = candidate_gimage.bands[band_no][
                numpy.nonzero(PIF_all != 0)].ravel()
            ref_combo_filtered = reference_gimage.bands[band_no][
                numpy.nonzero(PIF_all != 0)].ravel()

            logging.info('PCA Info: Filtered combo corrcoef (band {}) ='
                         ' {}'.format(band_no, numpy.corrcoef(
                            cand_combo_filtered, ref_combo_filtered)[0, 1]))

    no_valid_pixels = len(numpy.nonzero(PIF_all.ravel() != 0)[0])
    no_total_pixels = candidate_gimage.bands[0].size
    valid_percent = 100.0 * no_valid_pixels / no_total_pixels
    logging.info('PCA Info: Found {} final pifs out of {} pixels ({}%)'.format(
        no_valid_pixels, no_total_pixels, valid_percent))

    return 65535 * PIF_all.astype(numpy.uint16)


def _PCA_fit_and_filter_single_band(candidate_band, reference_band,
                                    valid_pixels_list, array_shape, lim):
    ''' Performs PCA analysis, on the valid pixels and filters according
    to the distance from the principle eigenvector, for a single band.
    '''

    cand_valid = candidate_band[valid_pixels_list]
    ref_valid = reference_band[valid_pixels_list]

    if logging.getLogger().getEffectiveLevel() <= logging.INFO:
        logging.info('PCA Info: Original corrcoef = {}'.format(
            numpy.corrcoef(cand_valid, ref_valid)[0, 1]))

    fitted_pca = _PCA_fit_single_band(cand_valid, ref_valid)

    if logging.getLogger().getEffectiveLevel() <= logging.INFO:
        for component_no in xrange(len(fitted_pca.components_)):
            logging.info('PCA Info: Component {} is {} with an explained '
                         'variance of {}'.format(
                            component_no,
                            fitted_pca.components_[component_no],
                            fitted_pca.explained_variance_ratio_[component_no])
                         )

    def split_into_batches(array, no_per_batch):
        return [array[i:i + no_per_batch]
                for i in xrange(0, len(array), no_per_batch)]

    no_per_batch = 10000
    cand_valid_batches = split_into_batches(cand_valid, no_per_batch)
    ref_valid_batches = split_into_batches(ref_valid, no_per_batch)

    logging.info('PCA Info: Dataset split into {} batches of {} '
                 'each'.format(len(cand_valid_batches),
                               no_per_batch))

    passed_pixels = [[]]
    for batch_no in xrange(len(cand_valid_batches)):
        passed_pixels_batch = _PCA_filter_single_band(
            fitted_pca, cand_valid_batches[batch_no],
            ref_valid_batches[batch_no], lim)
        passed_pixels_batch = [
            no_per_batch * batch_no + passed_pixels_batch[0]]
        passed_pixels = numpy.concatenate(
            (passed_pixels, passed_pixels_batch), axis=1)

    no_valid_pixels = len(passed_pixels[0])
    no_total_pixels = candidate_band.size
    valid_percent = 100.0 * no_valid_pixels / no_total_pixels
    logging.info('Found {} single band pifs out of {} pixels ({}%)'.format(
        no_valid_pixels, no_total_pixels, valid_percent))

    return _PCA_PIF_single_band(passed_pixels,
                                valid_pixels_list,
                                array_shape)


def _PCA_fit_single_band(cand_valid, ref_valid):
    ''' Uses SK Learn PCA module to do PCA fit
    '''

    # SK Learn PCA
    X = numpy.array(zip(cand_valid, ref_valid))
    pca = PCA(n_components=2)

    # Fit the points
    pca.fit(X)

    return pca


def _PCA_filter_single_band(pca, cand_valid, ref_valid, lim):
    ''' Uses SK Learn PCA module to transform the data and filter
    '''

    X = numpy.array(zip(cand_valid, ref_valid))
    X_trans = pca.transform(X)
    _, ref_valid_trans_list = zip(*X_trans)
    ref_valid_trans = numpy.array(ref_valid_trans_list)

    # Filter
    pixels_pass_filter = numpy.nonzero(numpy.logical_and(
        ref_valid_trans >= (lim * -1), ref_valid_trans <= lim))

    return pixels_pass_filter


def _PCA_PIF_single_band(pixels_pass_filter, valid_pixels, array_shape):
    ''' Converts the list of filtered pixels to a PIF array
    '''

    # PIF images
    PIF = numpy.zeros(array_shape)
    PIF_vec = PIF.ravel()
    for filtered_pixel in pixels_pass_filter[0]:
        PIF_vec[valid_pixels[0][filtered_pixel]] = 1
    PIF = numpy.reshape(PIF_vec, array_shape)

    return PIF
