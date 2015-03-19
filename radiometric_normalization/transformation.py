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
from collections import namedtuple

import numpy

from radiometric_normalization import gimage


PIFSet = namedtuple('PIFSet', 'reference, candidate, weight')

# gain and offset are floats
LinearTransformation = namedtuple('LinearTransformation', 'gain, offset')


def generate(pif_weights, reference_gimg, candidate_gimg,
             method='linear_relationship'):
    '''Calculates the linear transformations (scale, offset) that
    normalizes the candidate image to the reference image based on
    on pseudo-invariant feature (pif) weights. Pixels where pif_weights
    is non-zero are pifs, and the value gives the strength of the pif.

    :param pif_weights: array of pif strength for each pixel (numpy array)
    :param reference_img: a gimage representing the reference image
    :param candidate_img: a gimage representing the candidate image
    :param method: transformation generation method
    :param output transformations: list of LinearTransformations of length
    equal to number of entries in input pif 'reference' and 'candidate'
    '''
    pif_set = pifs_to_pifset(pif_weights, reference_gimg, candidate_gimg)
    transform_fcn = get_transform_function(method)
    return transform_fcn(pif_set)


def get_transform_function(method):
    if method == 'linear_relationship':
        fcn = linear_relationship
    else:
        raise Exception('Unrecognized transformation method')
    return fcn


def pifs_to_pifset(pif_weights, reference_gimg, candidate_gimg):
    '''
    Creates a PIFSet, where weights, reference values, and candidate
    values of pifs are combined into separate numpy arrays.
    '''
    logging.debug('converting pifs for processing')

    valid_pixels = numpy.nonzero(pif_weights)
    logging.info('{} pifs out of {} pixels ({}%)'.format(
        valid_pixels[0].size, pif_weights.size,
        int(100 * float(valid_pixels[0].size) / pif_weights.size)))

    weight = numpy.array(pif_weights[valid_pixels])
    logging.debug('weight shape: {}'.format(weight.shape))

    r_values = numpy.dstack([band[valid_pixels]for band in
                            reference_gimg.bands])[0, :]
    logging.debug('reference values shape: {}'.format(r_values.shape))

    c_values = numpy.dstack([band[valid_pixels]for band in
                            candidate_gimg.bands])[0, :]
    logging.debug('candidate values shape: {}'.format(c_values.shape))

    return PIFSet(r_values, c_values, weight)


def linear_relationship(pif_set):
    logging.info('Calculating linear relationship transformations')

    c_means = numpy.mean(pif_set.candidate, axis=0)
    r_means = numpy.mean(pif_set.reference, axis=0)
    logging.info('Means: candidate - {}, reference {}'.format(
        c_means, r_means))

    c_stds = numpy.std(pif_set.candidate, axis=0)
    r_stds = numpy.std(pif_set.reference, axis=0)
    logging.info('Stddev: candidate - {}, reference {}'.format(
        c_stds, r_stds))

    def calculate_gain(c_std, r_std):
        # if c_std is zero it is a constant image so default gain to 1
        if c_std == 0:
            return 1
        return float(r_std) / c_std

    gains = [calculate_gain(c_std, r_std)
             for (c_std, r_std) in zip(c_stds, r_stds)]
    offsets = r_means - gains * c_means

    transformations = []
    for (gain, offset) in zip(gains, offsets):
        logging.info("Transformation: gain {}, offset {}".format(gain, offset))
        transformations.append(LinearTransformation(gain, offset))

    return transformations


def apply(input_gimage, transformations):
    '''Applies a set of linear transformations to a gimage

    :param input_gimage: gimage transformations are applied to
    :param transformations: set of LinearTransformations (length equal to the
        number of bands in input_gimage) to apply
    :param output: gimage that represents input_gimage with
        transformations applied
    '''
    logging.info('Applying linear transformations to gimage')

    def apply_lut(band, lut):
        'Changes band intensity values based on intensity look up table (lut)'
        if lut.dtype != band.dtype:
            raise Exception(
                "Band ({}) and lut ({}) must be the same data type.").format(
                band.dtype, lut.dtype)
        return numpy.take(lut, band, mode='clip')

    assert len(input_gimage.bands) == len(transformations)

    output_bands = []
    for input_band, lt in zip(input_gimage.bands, transformations):
        lut = linear_transformation_to_lut(lt)
        output_bands.append(apply_lut(input_band, lut))

    return gimage.GImage(
        output_bands, input_gimage.alpha, input_gimage.metadata)


def linear_transformation_to_lut(linear_transformation, max_value=None):
    logging.debug('Creating lut from linear transformation')
    dtype = numpy.uint16

    min_value = 0
    if max_value is None:
        max_value = numpy.iinfo(dtype).max

    def gain_offset_to_lut(gain, offset):
        logging.info("calculating lut values for gain {} and offset {}"
                     .format(gain, offset))
        lut = numpy.arange(min_value, max_value + 1, dtype=numpy.float)
        return gain * lut + offset

    lut = gain_offset_to_lut(linear_transformation.gain,
                             linear_transformation.offset)

    logging.info("clipping lut to [{},{}]".format(min_value, max_value))
    numpy.clip(lut, min_value, max_value, lut)

    return lut.astype(dtype)
