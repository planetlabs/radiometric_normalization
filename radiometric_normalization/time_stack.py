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

''' Creates the time stacks for further analysis
'''

import numpy
import logging

from radiometric_normalization import gimage


def generate(image_paths, output_path, method='mean_with_uniform_weight', image_nodata=None):
    '''Synthesizes a time stack image set into a single reference image.

    All images in time stack must:
        - contain the same number of bands
        - have the same band order
        - have the same size
        - have the same nodata value

    The output file is only supposed to be used internally so options to change
    the nodata value and the datatype are not exposed.
    - The output nodata value is 2 ** 15 - 1
    - The output image data type is uint16
    - Since the output file is only meant to be used internally, the
        geographic information isn't carried over to the output file.

    Input:
        image_paths (list of str): A list of paths for input time stack images
        method (str): Time stack analysis method [identity]
        output_path (str): A path to write the file to
    '''

    output_datatype = numpy.uint16

    all_gimages = [gimage.load(image_path, image_nodata)
                   for image_path in image_paths]

    if method == 'mean_with_uniform_weight':
        output_gimage = mean_with_uniform_weight(
            all_gimages, output_datatype)
    else:
        raise NotImplementedError("Only 'mean_with_uniform_weight' method is implemented")

    gimage.save(output_gimage, output_path)


def _mean_one_band(all_gimages, band_index, output_datatype):
    ''' Calculates the reference as the mean of each band with uniform
    weighting (zero for nodata pixels, 2 ** 16 - 1 for valid pixels)

    Input:
        all_gimages (list of gimages): A list of all the gimage
        band_index (int): Which band of the gimage to operate on
        output_datatype (numpy datatype): Data type for the output image

    Output:
        band_mean (numpy array): Output array representing the mean for
            the specified
        band_mask (numpy binary array): Mask representing the no data
            values for this band (0 is a no data pixel, 1 is an active pixel)
    '''

    no_images = len(all_gimages)
    one_band = \
        [numpy.ma.masked_array(all_gimages[i].bands[band_index],
                               all_gimages[i].alpha == 0)
         for i in range(no_images)]

    masked_mean = numpy.ma.mean(one_band, axis=0)
    band_mean = masked_mean.data.astype(output_datatype)
    band_mask = numpy.logical_not(masked_mean.mask)

    return band_mean, band_mask


def _uniform_weight_alpha(all_masks, output_datatype):
    '''Uses the gimages.mask entry to make a cumulative mask
    of all the gimages in the list

    Input:
        all_gimages (list of gimages): The list of gimages to
            find the cumulative mask of.

    Output:
        output_alpha (numpy uint16 array): The output mask.
    '''

    output_alpha = numpy.ones(all_masks[0].shape)
    for mask in all_masks:
        output_alpha[numpy.nonzero(mask == 0)] = 0

    output_alpha = output_alpha.astype(output_datatype) * \
        numpy.iinfo(output_datatype).max

    return output_alpha


def mean_with_uniform_weight(all_gimages, output_datatype):
    ''' Calculates the reference as the mean of each band with uniform
    weighting (zero for nodata pixels, 2 ** 16 - 1 for valid pixels)

    Input:
        all_bands (list of list of arrays): A list of each band,
            each band consisting of a list of images each entry being a
            masked array of the image data
        output_nodata (number): The no data value for the output image
        output_datatype (numpy datatype): Data type for the output image

    Output:
        output_gimage (gimage): The mean for each band and the mask in a
            gimage data format
    '''

    logging.info('Time stack analysis is using: Mean with uniform weight.')
    gimage.check_comparable(all_gimages, check_metadata=True)

    no_bands = len(all_gimages[0].bands)
    all_means = []
    all_masks = []
    for band in range(no_bands):
        band_mean, band_mask = _mean_one_band(
            all_gimages, band, output_datatype)
        all_means.append(band_mean)
        all_masks.append(band_mask)

    output_alpha = _uniform_weight_alpha(all_masks, output_datatype)

    output_gimage = gimage.GImage(all_means, output_alpha,
                                  all_gimages[0].metadata)

    return output_gimage
