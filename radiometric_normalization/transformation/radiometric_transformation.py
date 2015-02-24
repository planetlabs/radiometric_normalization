import logging
from collections import namedtuple

import numpy

# pif:
# coordinates (int, int)
# weighting (float, range 0-1)
# ref_values (tuple, type == 16 bit unsigned int, length = number of bands)
# candidate_values (tuple, type == 16 bit unsigned int, len = number of bands)}
PIF = namedtuple('PIF', 'coordinates, weighting, ref_values, candidate_values')

PIFSet = namedtuple('PIFSet', 'reference, candidate, weight')

LinearTransformation = namedtuple('LinearTransformation', 'gain, offset')


def calculate_lut(pifs):
    pif_set = pifs_to_pifset(pifs)
    transformations = linear_relationship(pif_set)

    luts = [linear_transformation_to_lut(lt) for lt in transformations]
    return luts


def pifs_to_pifset(pifs):
    weight = numpy.array([pif['weighting'] for pif in pifs])

    r_values = numpy.array([pif['ref_values'] for pif in pifs],
                           dtype=numpy.uint16)
    c_values = numpy.array([pif['candidate_values'] for pif in pifs],
                           dtype=numpy.uint16)

    return PIFSet(r_values, c_values, weight)


def linear_relationship(pif_set):
    c_means = numpy.mean(pif_set.candidate, axis=0)
    r_means = numpy.mean(pif_set.reference, axis=0)
    c_stds = numpy.std(pif_set.candidate, axis=0)
    r_stds = numpy.std(pif_set.reference, axis=0)

    def calculate_gain(c_std, r_std):
        if c_std == 0:
            return 0
        return float(r_std) / c_std

    gains = [calculate_gain(c_std, r_std)
             for (c_std, r_std) in zip(c_stds, r_stds)]
    offsets = r_means - gains * c_means

    transformations = [LinearTransformation(gain, offset)
                       for (gain, offset) in zip(gains, offsets)]

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
