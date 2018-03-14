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
import logging
from collections import namedtuple
from scipy.stats import linregress

from radiometric_normalization import robust


# Gain and offset are floats
LinearTransformation = namedtuple('LinearTransformation', 'gain, offset')


def generate_linear_relationship(candidate_band, reference_band, pif_mask):
    ''' Performs PCA analysis on the valid pixels and filters according
    to the distance from the principle eigenvector.

    :param array candidate_band: A 2D array representing the image data of the
                                 candidate band
    :param array reference_band: A 2D array representing the image data of the
                                  reference image
    :param array pif_mask: A 2D array representing the PIF pixels in the images

    :returns: A LinearTransformation object (gain and offset)
    '''
    candidate_pifs = candidate_band[numpy.nonzero(pif_mask)]
    reference_pifs = reference_band[numpy.nonzero(pif_mask)]

    return generate_linear_relationship_pixel_list(
        candidate_pifs, reference_pifs)


def generate_linear_relationship_pixel_list(candidate_pifs, reference_pifs):
    ''' Performs PCA analysis on the valid pixels and filters according
    to the distance from the principle eigenvector.

    :param list candidate_pifs: A list of candidate PIF data
    :param list reference_pifs: A list of coincident reference PIF data

    :returns: A LinearTransformation object (gain and offset)
    '''
    logging.info('Transformation: Calculating linear relationship '
                 'transformations')

    c_mean = numpy.mean(candidate_pifs)
    r_mean = numpy.mean(reference_pifs)
    logging.debug('Means: candidate - {}, reference {}'.format(c_mean, r_mean))

    c_std = numpy.std(candidate_pifs)
    r_std = numpy.std(reference_pifs)
    logging.debug('Stddev: candidate - {}, reference {}'.format(c_std, r_std))

    def calculate_gain(c_std, r_std):
        # if c_std is zero it is a constant image so default gain to 1
        if c_std == 0:
            return 1
        return float(r_std) / c_std

    gain = calculate_gain(c_std, r_std)
    offset = r_mean - gain * c_mean

    logging.info("Transformation: gain {}, offset {}".format(gain, offset))

    return LinearTransformation(gain, offset)


def generate_ols_regression(candidate_band, reference_band, pif_mask):
    ''' Performs PCA analysis on the valid pixels and filters according
    to the distance from the principle eigenvector.

    :param array candidate_band: A 2D array representing the image data of the
                                 candidate band
    :param array reference_band: A 2D array representing the image data of the
                                  reference image
    :param array pif_mask: A 2D array representing the PIF pixels in the images

    :returns: A LinearTransformation object (gain and offset)
    '''
    candidate_pifs = candidate_band[numpy.nonzero(pif_mask)]
    reference_pifs = reference_band[numpy.nonzero(pif_mask)]

    return generate_ols_regression_pixel_list(candidate_pifs, reference_pifs)


def generate_ols_regression_pixel_list(candidate_pifs, reference_pifs):
    ''' Performs PCA analysis on the valid pixels and filters according
    to the distance from the principle eigenvector.

    :param list candidate_pifs: A list of candidate PIF data
    :param list reference_pifs: A list of coincident reference PIF data

    :returns: A LinearTransformation object (gain and offset)
    '''
    logging.info('Transformation: Calculating ordinary least squares '
                 'regression transformations')

    gain, offset, r_value, p_value, std_err = linregress(
        candidate_pifs, reference_pifs)
    logging.debug(
        'Fit statistics: r_value = {}, p_value = {}, std_err = {}'.format(
            r_value, p_value, std_err))
    logging.info("Transformation: gain {}, offset {}".format(gain, offset))

    return LinearTransformation(gain, offset)


def generate_robust_fit(candidate_band, reference_band, pif_mask):
    ''' Performs a robust fit on the valid pixels.

    :param array candidate_band: A 2D array representing the image data of the
                                 candidate band
    :param array reference_band: A 2D array representing the image data of the
                                  reference image
    :param array pif_mask: A 2D array representing the PIF pixels in the images

    :returns: A LinearTransformation object (gain and offset)
    '''
    candidate_pifs = candidate_band[numpy.nonzero(pif_mask)]
    reference_pifs = reference_band[numpy.nonzero(pif_mask)]

    return generate_robust_fit_pixel_list(candidate_pifs, reference_pifs)


def generate_robust_fit_pixel_list(candidate_pifs, reference_pifs):
    ''' Performs a robust fit on the valid pixels.

    :param list candidate_pifs: A list of candidate PIF data
    :param list reference_pifs: A list of coincident reference PIF data

    :returns: A LinearTransformation object (gain and offset)
    '''
    logging.info('Transformation: Calculating robust fit '
                 'transformations')

    gain, offset = robust.fit(candidate_pifs, reference_pifs)
    logging.info("Transformation: gain {}, offset {}".format(gain, offset))

    return LinearTransformation(gain, offset)
