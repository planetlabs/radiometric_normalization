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
from osgeo import gdal

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

    if method == 'filter_nodata':
        reference_gimg = gimage.load(reference_path)
        candidate_gimg = gimage.load(candidate_path)
        pif_weight = _filter_zero_alpha_pifs(candidate_gimg, reference_gimg)
    elif method == 'filter_PCA':
        pif_weight = _generate_PCA_pifs(candidate_path, reference_path,
                                        method_options)
        reference_gimg = gimage.load(reference_path)
        candidate_gimg = gimage.load(candidate_path)
    else:
        raise NotImplementedError("Only 'filter_nodata' and 'PCA_filtering' "
                                  "methods are implemented.")

    return pif_weight, reference_gimg, candidate_gimg


def _filter_zero_alpha_pifs(candidate_gimage, reference_gimage):
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


def _generate_PCA_pifs(candidate_path, reference_path, method_options):
    if method_options is None:
        lim = 10
        no_per_batch = None
    elif not isinstance(method_options, (list, tuple)):
        lim = numpy.uint16(method_options)
        no_per_batch = None
    elif len(method_options) >= 2:
        lim = numpy.uint16(method_options[0])
        no_per_batch = numpy.uint16(method_options[1])
    else:
        raise Exception('Unknown method options for PCA PIF '
                        'filtering: {}'.format(method_options))

    return _filter_PCA_pifs(candidate_path, reference_path,
                            lim, no_per_batch)


def _filter_PCA_pifs(candidate_path, reference_path, lim, no_per_batch):
    ''' Performs PCA analysis, on the valid pixels and filters according
    to the distance from the principle eigenvector.
    '''
    reference_ds = gdal.Open(reference_path)
    candidate_ds = gdal.Open(candidate_path)

    logging.info('Pseudo invariant feature generation is using: Filtering '
                 'using PCA.')

    c_alpha, c_band_count = gimage._read_alpha_and_band_count(candidate_ds)
    r_alpha, r_band_count = gimage._read_alpha_and_band_count(reference_ds)

    # Check comparable
    assert r_band_count == c_band_count
    assert r_alpha.shape == c_alpha.shape

    array_shape = c_alpha.shape

    # Only analyse valid pixels
    alpha = numpy.ones(array_shape)
    alpha[numpy.nonzero(c_alpha == 0)] = 0
    alpha[numpy.nonzero(r_alpha == 0)] = 0
    alpha_vec = alpha.ravel()
    valid_pixels_list = numpy.nonzero(alpha_vec != 0)

    PIF_all = numpy.ones(array_shape)

    # Per band analysis
    for band_no in xrange(c_band_count):

        logging.info('PCA Info: Band {}'.format(band_no))

        PIF_band = _PCA_fit_and_filter_single_band(
            gimage._read_single_band(candidate_ds, band_no+1).ravel(),
            gimage._read_single_band(reference_ds, band_no+1).ravel(),
            valid_pixels_list, array_shape, lim, no_per_batch)
        PIF_all = numpy.logical_and(PIF_band, PIF_all)

    if logging.getLogger().getEffectiveLevel() <= logging.INFO:
        for band_no in xrange(c_band_count):
            cand_combo_filtered = gimage._read_single_band(
                candidate_ds, band_no + 1)[
                    numpy.nonzero(PIF_all != 0)].ravel()
            ref_combo_filtered = gimage._read_single_band(
                reference_ds, band_no + 1)[
                    numpy.nonzero(PIF_all != 0)].ravel()

            logging.info('PCA Info: Filtered combo corrcoef (band {}) ='
                         ' {}'.format(band_no, numpy.corrcoef(
                            cand_combo_filtered, ref_combo_filtered)[0, 1]))

    no_valid_pixels = len(numpy.nonzero(PIF_all.ravel() != 0)[0])
    no_total_pixels = c_alpha.size
    valid_percent = 100.0 * no_valid_pixels / no_total_pixels
    logging.info('PCA Info: Found {} final pifs out of {} pixels ({}%)'.format(
        no_valid_pixels, no_total_pixels, valid_percent))

    return 65535 * PIF_all.astype(numpy.uint16)


def _PCA_fit_and_filter_single_band(candidate_band, reference_band,
                                    valid_pixels_list, array_shape, lim,
                                    no_per_batch):
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

    if no_per_batch is None:
        passed_pixels = _PCA_filter_single_band(
            fitted_pca, cand_valid,
            ref_valid, lim)
    else:
        def split_into_batches(array, no_per_batch):
            return [array[i:i + no_per_batch]
                    for i in xrange(0, len(array), no_per_batch)]

        cand_valid_batches = split_into_batches(cand_valid, no_per_batch)
        ref_valid_batches = split_into_batches(ref_valid, no_per_batch)

        logging.info('PCA Info: Dataset split into {} batches of {} '
                     'each'.format(len(cand_valid_batches),
                                   no_per_batch))

        passed_pixels = [[]]

        batch_iterator = numpy.nditer(numpy.array(cand_valid_batches),
                                      flags=['f_index', 'refs_ok'])
        while not batch_iterator.finished:
            batch_no = batch_iterator.index
            if batch_no < len(cand_valid_batches):
                passed_pixels_batch = _PCA_filter_single_band(
                    fitted_pca, cand_valid_batches[batch_no],
                    ref_valid_batches[batch_no], lim)
                passed_pixels_batch = [
                    no_per_batch * batch_no + passed_pixels_batch[0]]
                passed_pixels = numpy.concatenate(
                    (passed_pixels, passed_pixels_batch), axis=1)
            batch_iterator.iternext()

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
