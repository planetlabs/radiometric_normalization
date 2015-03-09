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
import numpy

from radiometric_normalization import gimage


def generate(candidate_path, reference_path, method='identity'):
    ''' Generates psuedo invariant features as a list of pixel pairs

    Input:
        candidate_path (str): Path to the candidate image
        reference_path (str): Path to the reference image
        method (str): Which psuedo invariant feature generation method to use

    Output:
        pixel_pairs (list of pixel pair dict): Pixel pairs are
            {'coordinates', (int, int),
             'weighting', float,
             'reference', list of numbers,
             'candidate', list of numbers}
    '''
    reference_img, candidate_img = _load_gimages(
        reference_path, candidate_path)

    if method == 'identity':
        pixel_pairs = _filter_zero_alpha_pifs(reference_img, candidate_img)
    else:
        raise NotImplementedError("Only 'identity' method is implemented.")

    return pixel_pairs


def _load_gimages(reference_path, candidate_path):
    reference_img = gimage.load(reference_path)
    candidate_img = gimage.load(candidate_path)

    assert len(reference_img.bands) == len(candidate_img.bands), \
        '{} and {} have different number of bands'.format(
            candidate_path, reference_path)
    assert reference_img.bands[0].shape == reference_img.bands[0].shape, \
        '{} and {} have different shapes'.format(
            candidate_path, reference_path)
    return (reference_img, candidate_img)


def _filter_zero_alpha_pifs(reference_gimage, candidate_gimage):
    ''' Creates the pseudo-invariant features from the reference and candidate
    gimages by filtering out pixels where either the candidate or mask alpha
    value is zero (masked)'''
    all_mask = numpy.logical_not(numpy.logical_or(
        reference_gimage.alpha == 0, candidate_gimage.alpha == 0))

    valid_pixels = numpy.nonzero(all_mask)

    def per_band_values(gimg, row, col):
        return [band[row, col] for band in gimg.bands]

    pixel_pairs = []
    for pixel in range(len(valid_pixels[0])):
        row = valid_pixels[0][pixel]
        col = valid_pixels[1][pixel]
        pixel_dict = {
            'coordinates': (row, col),
            'weighting': reference_gimage.alpha[row, col],
            'reference': per_band_values(reference_gimage, row, col),
            'candidate': per_band_values(candidate_gimage, row, col)}
        pixel_pairs.append(pixel_dict)

    return pixel_pairs
