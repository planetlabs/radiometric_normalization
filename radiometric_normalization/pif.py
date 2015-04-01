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

from radiometric_normalization import gimage


def generate(candidate_path, reference_path, method='filter_nodata'):
    ''' Generates psuedo invariant features as a list of pixel pairs

    Input:
        candidate_path (str): Path to the candidate image
        reference_path (str): Path to the reference image
        method (str): Which psuedo invariant feature generation method to use

    Output:
        pif_weight (numpy uint16 array): A numpy array in the same coordinate
            system of the candidate/reference image with a weight for how
            a PIF the pixel is (0 for not a PIF)
    '''

    reference_img = gimage.load(reference_path)
    candidate_img = gimage.load(candidate_path)

    if method == 'filter_nodata':
        pif_weight = _filter_zero_alpha_pifs(reference_img, candidate_img)
    else:
        raise NotImplementedError("Only 'filter_nodata' method is implemented.")

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
    valid_percent = 100 * no_valid_pixels / no_total_pixels
    logging.info('Found {} pifs out of {} pixels ({}%)'.format(
        no_valid_pixels, no_total_pixels, valid_percent))

    pif_weight = numpy.zeros(reference_gimage.bands[0].shape,
                             dtype=numpy.uint16)
    pif_weight[valid_pixels] = reference_gimage.alpha[valid_pixels]

    return pif_weight
