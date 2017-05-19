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


def apply(input_band, transformation, method='lut'):
    '''Applies a linear transformation to an array. Wrapper function around
    the two methods.

    :param array input_band: A 2D array representing the image data of the
        a single band
    :param LinearTransformation transformation: A LinearTransformation
        (gain and offset)

    :returns: A 2D array of of the input_band with the transformation applied
    '''
    if method == 'direct':
        return apply_directly(input_band, transformation)
    else:
        return apply_using_lut(input_band, transformation)


def apply_directly(input_band, transformation):
    '''Applies a linear transformation to an array directly. It will output
    an array of float values. This is not very memory efficient.

    :param array input_band: A 2D array representing the image data of the
        a single band
    :param LinearTransformation transformation: A LinearTransformation
        (gain and offset)

    :returns: A 2D array of of the input_band with the transformation applied
    '''
    logging.info('Normalize: Applying linear transformation to band (float)')
    gain = transformation.gain
    offset = transformation.offset
    return input_band.astype('float') * gain + offset


def apply_using_lut(input_band, transformation):
    '''Applies a linear transformation to an array using a look up table.
    This creates a uint16 array as the output and clips the output band
    to the range of a uint16.

    :param array input_band: A 2D array representing the image data of the
        a single band
    :param LinearTransformation transformation: A LinearTransformation
        (gain and offset)

    :returns: A 2D array of of the input_band with the transformation applied
    '''
    logging.info('Normalize: Applying linear transformation to band (uint16)')

    def _apply_lut(band, lut):
        '''Changes band intensity values based on intensity look up table (lut)
        '''
        if lut.dtype != band.dtype:
            raise Exception(
                'Band ({}) and lut ({}) must be the same data type.').format(
                band.dtype, lut.dtype)
        return numpy.take(lut, band, mode='clip')

    lut = _linear_transformation_to_lut(transformation)
    return _apply_lut(input_band, lut)


def _linear_transformation_to_lut(linear_transformation,
                                  max_value=None, dtype=numpy.uint16):
    min_value = 0
    if max_value is None:
        max_value = numpy.iinfo(dtype).max

    def gain_offset_to_lut(gain, offset):
        logging.debug(
            'Normalize: Calculating lut values for gain '
            '{} and offset {}'.format(gain, offset))
        lut = numpy.arange(min_value, max_value + 1, dtype=numpy.float)
        return gain * lut + offset

    lut = gain_offset_to_lut(linear_transformation.gain,
                             linear_transformation.offset)

    logging.debug('Normalize: Clipping lut from [{}, {}] to [{},{}]'.format(
        min(lut), max(lut), min_value, max_value))
    numpy.clip(lut, min_value, max_value, lut)

    return lut.astype(dtype)
