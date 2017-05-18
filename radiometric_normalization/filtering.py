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


def _get_residual(data_1, data_2, line_gain, line_offset):
    return numpy.absolute(line_gain * data_1 - data_2 + line_offset) / \
        numpy.sqrt(1 + line_gain * line_gain)


def combine_alphas(list_of_alphas):
    return numpy.logical_and.reduce(list_of_alphas)


def calculate_residuals_from_line(candidate_band, reference_band,
                                  combined_alpha, line_gain=1.0,
                                  line_offset=0.0):
    ''' Calculates the residuals from a line.

    :param array candidate_band: A 2D array representing the image data of the
                                 candidate band
    :param array reference_band: A 2D array representing the image data of the
                                 reference image
    :param array combined_alpha: A 2D array representing the alpha mask of the
                                 valid pixels in both the candidate array and
                                 reference array
    :param float line_gain: The gradient of the line
    :param float line_offset: The intercept of the line

    :returns: A 2-D array of floats with the residuals of each valid pixel
             (otherwise 0)
    '''
    logging.info('PIF Filtering: Calculating residuals from line: y = '
                 '{} * x + {}'.format(line_gain, line_offset))

    valid_pixels = numpy.nonzero(combined_alpha)

    residuals = _get_residual(
        candidate_band[valid_pixels], reference_band[valid_pixels],
        line_gain, line_offset)

    output = numpy.zeros(candidate_band.shape)
    output[valid_pixels] = residuals
    return output


def filter_by_residuals_from_line(candidate_band, reference_band,
                                  combined_alpha, threshold=1000,
                                  line_gain=1.0, line_offset=0.0):
    ''' Filters pixels by their residuals from a line

    :param array candidate_band: A 2D array representing the image data of the
                                 candidate band
    :param array reference_band: A 2D array representing the image data of the
                                 reference image
    :param array combined_alpha: A 2D array representing the alpha mask of the
                                 valid pixels in both the candidate array and
                                 reference array
    :param float threshold: The distance from the line within which to include
                            pixels
    :param float line_gain: The gradient of the line
    :param float line_offset: The intercept of the line

    :returns: A 2-D array of boolean mask representing each valid pixel (True)
    '''
    logging.info('PIF Filtering: Filtering from line: y = '
                 '{} * x + {}'.format(line_gain, line_offset))

    residual_band = calculate_residuals_from_line(
        candidate_band, reference_band, combined_alpha, line_gain, line_offset)

    mask = numpy.zeros(candidate_band.shape, dtype=numpy.bool)
    mask[numpy.nonzero(residual_band < threshold)] = 1
    mask[numpy.nonzero(numpy.logical_not(combined_alpha))] = 0
    return mask


def filter_by_histogram(candidate_band, reference_band,
                        combined_alpha, threshold=0.1,
                        number_of_valid_bins=None, rough_search=False,
                        number_of_total_bins_in_one_axis=10):
    ''' Filters pixels using a 2D histogram of common values.

    There is one mutually exclusive option:
    - either a number_of_valid_bins is specified, which represents how many of
      the most popular bins to select
    - or a threshold is specified, which specifies a minimum population above
      which a bin is selected

    :param array candidate_band: A 2D array representing the image data of the
                                 candidate band
    :param array reference_band: A 2D array representing the image data of the
                                 reference image
    :param array combined_alpha: A 2D array representing the alpha mask of the
                                 valid pixels in both the candidate array and
                                 reference array
    :param float threshold: A threshold on the population of a bin. Above this
                            number, a bin is selected. This number is a
                            fraction of the maximum bin (i.e. a value of 0.1
                            will mean that the bin will need to have more than
                            10% than the most populous bin)
    :param int number_of_valid_bins: The total number of bins to select. The
                                     bins are arranged in descending order of
                                     population. This number specifies how
                                     many of the top bins are selected (i.e.
                                     a value of 3 will mean that the top three
                                     bins are selected). If this is specified
                                     then the threshold parameter is ignored
    :param boolean rough_search: If this is true, the bins will specify a
                                 maximum and minimum range within which to
                                 select pixels (i.e. one superset bin will be
                                 created from the bins that are selected).
                                 This should speed up the computation but be
                                 less exact
    :param int number_of_total_bins_in_one_axis: This number controls the total
                                                 number of bins. This number
                                                 represents the total number of
                                                 bins in one axis. Both axes
                                                 have the same number (i.e. a
                                                 value of 10 will mean that
                                                 there are 100 bins in total)

    :returns: A 2-D array of boolean mask representing each valid pixel (True)
    '''
    logging.info('PIF Filtering: Filtering by histogram.')

    valid_pixels = numpy.nonzero(combined_alpha)
    candidate_data = candidate_band[valid_pixels]
    reference_data = reference_band[valid_pixels]

    H, candidate_bins, reference_bins = numpy.histogram2d(
        candidate_data, reference_data, bins=number_of_total_bins_in_one_axis)

    def check_in_valid_bin(c_data_point, r_data_point):
        for valid_cand_bin, valid_ref_bin in zip(
          passed_bins[0], passed_bins[1]):
            if c_data_point >= candidate_bins[valid_cand_bin] and \
               c_data_point <= candidate_bins[valid_cand_bin + 1] and \
               r_data_point >= reference_bins[valid_ref_bin] and \
               r_data_point <= reference_bins[valid_ref_bin + 1]:
                return True
        return False

    def get_valid_range():
        c_min = min([candidate_bins[v] for v in passed_bins[0]])
        c_max = min([candidate_bins[v + 1] for v in passed_bins[0]])
        r_min = min([reference_bins[v] for v in passed_bins[1]])
        r_max = min([reference_bins[v + 1] for v in passed_bins[1]])
        return c_min, c_max, r_min, r_max

    def check_in_valid_range(c_data_point, r_data_point):
        if c_data_point >= c_min and \
           c_data_point <= c_max and \
           r_data_point >= r_min and \
           r_data_point <= r_max:
            return True
        return False

    if number_of_valid_bins:
        logging.info('Filtering by number of histogram bins.')
        passed_bins = numpy.unravel_index(
          numpy.argsort(H.ravel())[-number_of_valid_bins:], H.shape)
    else:
        logging.info('Filtering by threshold.')
        H_max = float(max(H.flatten()))
        passed_bins = numpy.nonzero(H / H_max > threshold)
        logging.info(
            '{} bins out of {} passed'.format(
                len(passed_bins[0]), len(H.flatten())))

    if rough_search:
        logging.info('Rough filtering only')
        c_min, c_max, r_min, r_max = get_valid_range()
        logging.info(
            'Valid range: Candidate = ({}, {}), Reference = ({}, {})'.format(
                c_min, c_max, r_min, r_max))
        passed_pixels = numpy.nonzero(
            [check_in_valid_range(c, r) for c, r in zip(
              candidate_data, reference_data)])
    else:
        logging.info('Exact filtering by bins')
        passed_pixels = numpy.nonzero(
            [check_in_valid_bin(c, r) for c, r in zip(
                candidate_data, reference_data)])
    logging.info(
        'Valid data = {} out of {} ({}%)'.format(
            len(passed_pixels[0]),
            len(valid_pixels[0]),
            100.0 * len(passed_pixels[0]) / len(valid_pixels[0])))

    mask_pixels = (valid_pixels[0][passed_pixels[0]],
                   valid_pixels[1][passed_pixels[0]])
    mask = numpy.zeros(candidate_band.shape, dtype=numpy.bool)
    mask[mask_pixels] = 1
    return mask
