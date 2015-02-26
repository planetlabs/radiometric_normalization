import logging
from collections import namedtuple

import numpy


PIFSet = namedtuple('PIFSet', 'reference, candidate, weight')

LinearTransformation = namedtuple('LinearTransformation', 'gain, offset')


def generate(pifs, method='linear_relationship'):
    '''Calculates the look-up table that applies a radiometric transformation
    to the candidate image based on pseudo-invariant features (pifs). Each
    pif is a dict with fields 'coordinates', 'reference',
    'candidate', and 'weight'.

    'reference' and 'candidate' are tuples with length equal to the number of
    radiometric bands
    'weight' is a float
    'coordinates' is a tuple of length 2

    :param pifs: list of pifs (dicts)
    :param method: transformation generation method
    :param output transformations: list of LinearTransformations of length
    equal to number of entries in input pif 'reference' and 'candidate'
    '''
    pif_set = pifs_to_pifset(pifs)
    transform_fcn = get_transform_function(method)
    return transform_fcn(pif_set)


def transformations_to_luts(transformations):
    luts = [linear_transformation_to_lut(lt) for lt in transformations]
    return luts


def get_transform_function(method):
    if method == 'linear_relationship':
        fcn = linear_relationship
    else:
        raise Exception('Unrecognized transformation method')
    return fcn


def pifs_to_pifset(pifs):
    '''
    Creates a PIFSet, where weights, reference values, and candidate
    values of pifs are combined into separate numpy arrays.
    '''
    weight = numpy.array([pif['weighting'] for pif in pifs])
    r_values = numpy.array([pif['reference'] for pif in pifs],
                           dtype=numpy.uint16)
    c_values = numpy.array([pif['candidate'] for pif in pifs],
                           dtype=numpy.uint16)
    return PIFSet(r_values, c_values, weight)


def linear_relationship(pif_set):
    c_means = numpy.mean(pif_set.candidate, axis=0)
    r_means = numpy.mean(pif_set.reference, axis=0)
    c_stds = numpy.std(pif_set.candidate, axis=0)
    r_stds = numpy.std(pif_set.reference, axis=0)

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


def linear_transformation_to_lut(linear_transformation, max_value=None):
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
