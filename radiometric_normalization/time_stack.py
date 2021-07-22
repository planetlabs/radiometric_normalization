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

from radiometric_normalization import gimage


def generate(image_paths, output_path,
             method='mean_with_uniform_weight',
             image_nodata=None):
    '''Synthesizes a time stack image set into a single reference image.

    All images in time stack must:
        - contain the same number of bands
        - have the same band order
        - have the same size
        - have the same nodata value
        - have the same geographic metadata

    The output file is only supposed to be used internally so options to change
    the nodata value and the datatype are not exposed.
    - The output image data type is uint16
    - The output nodata values are indicated by a 0 in the alpha band

    Input:
        image_paths (list of str): A list of paths for input time stack images
        output_path (str): A path to write the file to
        method (str): Time stack analysis method [Identity]
        image_nodata (int): [Optional] Manually provide a no data value
    '''

    output_datatype = numpy.uint16

    if method == 'mean_with_uniform_weight':
        output_gimage = mean_with_uniform_weight(
            image_paths, output_datatype, image_nodata)
    else:
        raise NotImplementedError("Only 'mean_with_uniform_weight'"
                                  "method is implemented")

    gimage.save(output_gimage, output_path, compress=False)


def _sum_masked_array_list(sum_masked_arrays,
                           frequency_arrays,
                           new_masked_arrays):
    ''' Calculates the sum of two lists of masked arrays

    Input:
        sum_masked_arrays (list of masked arrays): A list of masked
            arrays (one for each band)
        frequency_arrays (list of arrays of ints): Keeping track of
            how many times each pixel is summed (one for each band)
        new_masked_arrays (list of masked arrays): A list of masked
            arrays (one for each band)
    Output:
        sum_masked_array (masked array): A list of masked
            arrays (one for each band)
    '''
    no_bands = len(sum_masked_arrays)
    for band_index in xrange(no_bands):
        sum_masked_arrays[band_index] = \
            numpy.ma.sum([sum_masked_arrays[band_index],
                          new_masked_arrays[band_index]],
                         axis=0)
        frequency_arrays[band_index] = frequency_arrays[band_index] + \
            numpy.logical_not(new_masked_arrays[band_index].mask).astype('int')

    return sum_masked_arrays, frequency_arrays


def _masked_arrays_from_gimg(input_gimg, working_datatype):
    '''A gimage as input and outputs a masked array of output_datatype.

    Input:
        input_gimg (a gimage): A gimage to convert
        working_datatype (numpy datatype): The datatype in use for
            the calculations

    Output:
        all_bands (list of numpy arrays): A list of each band of the gimage
            as a masked array
    '''
    no_bands = len(input_gimg.bands)
    all_bands_masked_array_list = []
    for band_index in xrange(no_bands):
        one_band_masked_array = \
            numpy.ma.masked_array(
                input_gimg.bands[band_index].astype(working_datatype),
                input_gimg.alpha == 0)
        all_bands_masked_array_list.append(one_band_masked_array)

    return all_bands_masked_array_list


def _uniform_weight_alpha(sum_masked_arrays, output_datatype):
    '''Calculates the cumulative mask of a list of masked array

    Input:
        sum_masked_arrays (list of numpy masked arrays): The list of
            masked arrays to find the cumulative mask of, each element
            represents one band.
            (sums_masked_array.mask has a 1 for a no data pixel and
            a 0 otherwise)
        output_datatype (numpy datatype): The output datatype

    Output:
        output_alpha (numpy uint16 array): The output mask
            (0 for a no data pixel, uint16 max value otherwise)
    '''

    output_alpha = numpy.ones(sum_masked_arrays[0].shape)
    for band_sum_masked_array in sum_masked_arrays:
        output_alpha[numpy.nonzero(band_sum_masked_array.mask == 1)] = 0

    output_alpha = output_alpha.astype(output_datatype) * \
        numpy.iinfo(output_datatype).max

    return output_alpha


def _mean_from_sum(sum_masked_arrays,
                   frequency_arrays,
                   output_datatype):
    ''' Calculates the mean from the summation of all the images

    Input:
        sum_masked_arrays (list of numpy masked arrays): The list of
            masked arrays to find the mean of, each element of the list
            represents one band.
            (sums_masked_array.mask has a 1 for a no data pixel and
            a 0 otherwise)
        frequency_arrays (numpy array of ints): The number of times each
            pixel has been summed
        output_datatype (numpy data type): The output datatype

    Output:
        output_mean(list of numpy arrays): A list of the means of the
            images. Each element of the list represents one band.
    '''
    no_bands = len(sum_masked_arrays)
    output_mean = []
    for band_index in xrange(no_bands):
        output_band = numpy.zeros(frequency_arrays[band_index].shape)
        good_indices = numpy.nonzero(frequency_arrays[band_index] != 0)
        output_band[good_indices] = \
            sum_masked_arrays[band_index].data[good_indices] / \
            frequency_arrays[band_index][good_indices]
        output_mean.append(output_band.astype(output_datatype))

    return output_mean


def mean_with_uniform_weight(image_paths, output_datatype, image_nodata):
    ''' Calculates the reference image as the mean of each band with uniform
    weighting (zero for nodata pixels, 2 ** 16 - 1 for valid pixels)

    The input are a set of uint16 geotiffs, the output is a uint16 geotiff but
    inbetween we use numpy double masked arrays so that we can safely take the
    summation of all the values without reaching the maximum value.

    This function is written so that it should only load two gimages into
    memory at any one time (to save memory when analysing lists of > 100
    images)

    Input:
        image_paths (list of strings): A list of image paths for each image
        output_datatype (numpy datatype): Data type for the output image

    Output:
        output_gimage (gimage): The mean for each band and the weighting in a
            gimage data format
    '''

    logging.info('Time stack analysis is using: Mean with uniform weight.')

    working_datatype = numpy.double
    no_images = len(image_paths)
    first_gimg = gimage.load(image_paths[0], image_nodata)

    sum_masked_arrays = _masked_arrays_from_gimg(first_gimg,
                                                 working_datatype)

    no_bands = len(sum_masked_arrays)
    frequency_arrays = \
        [numpy.logical_not(sum_masked_arrays[band_index].mask).astype('int')
         for band_index in xrange(no_bands)]

    for image_index in xrange(1, no_images):
        new_gimg = gimage.load(image_paths[image_index])
        gimage.check_comparable([first_gimg, new_gimg], check_metadata=True)

        new_masked_arrays = _masked_arrays_from_gimg(new_gimg,
                                                     working_datatype)
        sum_masked_arrays, frequency_arrays = _sum_masked_array_list(
            sum_masked_arrays, frequency_arrays, new_masked_arrays)

    output_alpha = _uniform_weight_alpha(sum_masked_arrays, output_datatype)
    output_bands = _mean_from_sum(sum_masked_arrays, frequency_arrays,
                                  output_datatype)

    output_gimage = gimage.GImage(output_bands, output_alpha,
                                  first_gimg.metadata)

    return output_gimage
