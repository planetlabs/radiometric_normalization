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
import unittest
import numpy
import os

from radiometric_normalization import time_stack, gimage


class Tests(unittest.TestCase):
    def test_mean_with_uniform_weight(self):
        # Three images of three bands of 2 by 2 pixels
        gimage_one = gimage.GImage(
            [numpy.array([[4, 1],
                          [2, 5]], dtype='uint16'),
             numpy.array([[4, 1],
                          [2, 5]], dtype='uint16'),
             numpy.array([[7, 8],
                          [6, 3]], dtype='uint16')],
            numpy.array([[65535, 0], [65535, 65535]], dtype='uint16'), {})

        gimage_two = gimage.GImage(
            [numpy.array([[9, 9],
                          [5, 1]], dtype='uint16'),
             numpy.array([[2, 7],
                          [7, 3]], dtype='uint16'),
             numpy.array([[2, 6],
                          [7, 2]], dtype='uint16')],
            numpy.array([[0, 0], [65535, 65535]], dtype='uint16'), {})

        gimage_three = gimage.GImage(
            [numpy.array([[4, 7],
                          [5, 3]], dtype='uint16'),
             numpy.array([[1, 2],
                          [5, 1]], dtype='uint16'),
             numpy.array([[1, 6],
                          [3, 2]], dtype='uint16')],
            numpy.array([[65535, 0], [65535, 0]], dtype='uint16'), {})

        gimage.save(gimage_one, 'gimage_one.tif')
        gimage.save(gimage_two, 'gimage_two.tif')
        gimage.save(gimage_three, 'gimage_three.tif')
        image_paths = ['gimage_one.tif', 'gimage_two.tif', 'gimage_three.tif']

        golden_output = gimage.GImage(
            [numpy.array([[4, 0],
                          [4, 3]], dtype='uint16'),
             numpy.array([[2, 0],
                          [4, 4]], dtype='uint16'),
             numpy.array([[4, 0],
                          [5, 2]], dtype='uint16')],
            numpy.array([[65535, 0],
                         [65535, 65535]], dtype='uint16'),
            {'geotransform': (0.0, 1.0, 0.0, 0.0, 0.0, 1.0)})

        output_gimage = time_stack.mean_with_uniform_weight(image_paths,
                                                            numpy.uint16,
                                                            None)

        for band in range(3):
            numpy.testing.assert_array_equal(output_gimage.bands[band],
                                             golden_output.bands[band])
        numpy.testing.assert_array_equal(output_gimage.alpha,
                                         golden_output.alpha)
        self.assertEqual(output_gimage.metadata,
                         golden_output.metadata)

        for image in image_paths:
            os.unlink(image)

    def test__sum_masked_array_list(self):
        # Two masked_arrays with two bands of 2 by 2 pixels
        band_one_1 = numpy.ma.masked_array(
            numpy.array([[1, 2], [5, 1]],
                        dtype='uint16'),
            mask=numpy.array([[False, True],
                              [False, False]],
                             dtype='bool'))

        band_two_1 = numpy.ma.masked_array(
            numpy.array([[9, 2], [3, 6]],
                        dtype='uint16'),
            mask=numpy.array([[False, False],
                              [False, False]],
                             dtype='bool'))

        band_one_2 = numpy.ma.masked_array(
            numpy.array([[6, 3], [7, 8]],
                        dtype='uint16'),
            mask=numpy.array([[True, True],
                              [False, False]],
                             dtype='bool'))

        band_two_2 = numpy.ma.masked_array(
            numpy.array([[8, 9], [3, 4]],
                        dtype='uint16'),
            mask=numpy.array([[True, True],
                              [False, True]],
                             dtype='bool'))

        sum_masked_array = [band_one_1, band_two_1]
        new_masked_array = [band_one_2, band_two_2]
        frequency_array = [numpy.array([[1, 0], [1, 1]]),
                           numpy.array([[1, 1], [1, 1]])]

        golden_band_one = numpy.ma.masked_array(
            numpy.array([[1, 0], [12, 9]],
                        dtype='uint16'),
            mask=numpy.array([[False, True],
                              [False, False]],
                             dtype='bool'))

        golden_band_two = numpy.ma.masked_array(
            numpy.array([[9, 2], [6, 6]],
                        dtype='uint16'),
            mask=numpy.array([[False, False],
                              [False, False]],
                             dtype='bool'))

        golden_sum_masked_array = [golden_band_one, golden_band_two]
        golden_frequency_array = [numpy.array([[1, 0], [2, 2]]),
                                  numpy.array([[1, 1], [2, 1]])]

        sum_masked_array_output, frequency_array_output = \
            time_stack._sum_masked_array_list(sum_masked_array,
                                              frequency_array,
                                              new_masked_array)

        numpy.testing.assert_array_equal(sum_masked_array_output[0],
                                         golden_sum_masked_array[0])
        numpy.testing.assert_array_equal(sum_masked_array_output[1],
                                         golden_sum_masked_array[1])
        numpy.testing.assert_array_equal(frequency_array_output,
                                         golden_frequency_array)

    def test__masked_arrays_from_gimg(self):
        # Gimages of four bands of 3 by 3 pixels
        test_gimage = gimage.GImage(
            [numpy.array([[5, 9, 6],
                          [1, 2, 7],
                          [6, 2, 3]], dtype='uint16'),
             numpy.array([[3, 1, 8],
                          [2, 3, 9],
                          [1, 7, 1]], dtype='uint16'),
             numpy.array([[6, 6, 3],
                          [7, 2, 7],
                          [8, 2, 2]], dtype='uint16')],
            numpy.array([[65535, 0, 65535],
                         [65535, 65535, 65535],
                         [0, 65535, 65535]], dtype='uint16'), {})

        golden_masked_array_list = \
            [numpy.ma.masked_array(numpy.array([[5, 9, 6],
                                                [1, 2, 7],
                                                [6, 2, 3]], dtype='double'),
                                   mask=numpy.array([[False, True, False],
                                                     [False, False, False],
                                                     [True, False, False]],
                                                    dtype='bool')),
             numpy.ma.masked_array(numpy.array([[3, 1, 8],
                                                [2, 3, 9],
                                                [1, 7, 1]], dtype='double'),
                                   mask=numpy.array([[False, True, False],
                                                     [False, False, False],
                                                     [True, False, False]],
                                                    dtype='bool')),
             numpy.ma.masked_array(numpy.array([[6, 6, 3],
                                                [7, 2, 7],
                                                [8, 2, 2]], dtype='double'),
                                   mask=numpy.array([[False, True, False],
                                                     [False, False, False],
                                                     [True, False, False]],
                                                    dtype='bool'))]

        masked_array_list = time_stack._masked_arrays_from_gimg(test_gimage,
                                                                numpy.double)

        numpy.testing.assert_array_equal(masked_array_list[0],
                                         golden_masked_array_list[0])
        numpy.testing.assert_array_equal(masked_array_list[1],
                                         golden_masked_array_list[1])
        numpy.testing.assert_array_equal(masked_array_list[2],
                                         golden_masked_array_list[2])

    def test__mean_from_sum(self):
        # A masked array with four bands of 3 by 3 pixels
        band_one = numpy.ma.masked_array(
            numpy.array([[1, 2, 5],
                         [7, 3, 4],
                         [8, 6, 2]],
                        dtype='double'),
            mask=numpy.array([[False, False, False],
                              [False, False, True],
                              [False, True, False]],
                             dtype='bool'))

        band_two = numpy.ma.masked_array(
            numpy.array([[6, 8, 9],
                         [3, 1, 8],
                         [2, 3, 6]],
                        dtype='double'),
            mask=numpy.array([[False, False, False],
                              [False, True, False],
                              [False, True, False]],
                             dtype='bool'))

        band_three = numpy.ma.masked_array(
            numpy.array([[3, 5, 2],
                         [9, 6, 3],
                         [7, 4, 6]],
                        dtype='double'),
            mask=numpy.array([[True, False, False],
                              [False, False, False],
                              [False, True, False]],
                             dtype='bool'))

        band_four = numpy.ma.masked_array(
            numpy.array([[7, 4, 3],
                         [2, 6, 5],
                         [1, 2, 8]],
                        dtype='double'),
            mask=numpy.array([[False, True, False],
                              [False, True, True],
                              [False, True, False]],
                             dtype='bool'))

        sum_masked_array = [band_one, band_two, band_three, band_four]

        frequency_array = [numpy.array([[1, 1, 4], [2, 4, 1], [2, 0, 1]]),
                           numpy.array([[2, 3, 1], [2, 2, 2], [5, 0, 2]]),
                           numpy.array([[2, 1, 2], [1, 5, 1], [3, 0, 2]]),
                           numpy.array([[3, 2, 3], [1, 7, 2], [4, 0, 3]])]

        golden_mean = [numpy.array([[1, 2, 1], [3, 0, 4], [4, 0, 2]]),
                       numpy.array([[3, 2, 9], [1, 0, 4], [0, 0, 3]]),
                       numpy.array([[1, 5, 1], [9, 1, 3], [2, 0, 3]]),
                       numpy.array([[2, 2, 1], [2, 0, 2], [0, 0, 2]])]

        output_mean = time_stack._mean_from_sum(
            sum_masked_array, frequency_array, numpy.uint16)

        numpy.testing.assert_array_equal(
            output_mean, golden_mean)

    def test__uniform_weight_alpha(self):
        # Five images of 2 by 2 pixels
        band_one = numpy.ma.masked_array(
            numpy.array([[0, 0], [0, 0]],
                        dtype='double'),
            mask=numpy.array([[False, True],
                              [False, False]],
                             dtype='bool'))

        band_two = numpy.ma.masked_array(
            numpy.array([[0, 0], [0, 0]],
                        dtype='double'),
            mask=numpy.array([[False, False],
                              [False, False]],
                             dtype='bool'))

        band_three = numpy.ma.masked_array(
            numpy.array([[0, 0], [0, 0]],
                        dtype='double'),
            mask=numpy.array([[True, True],
                              [False, False]],
                             dtype='bool'))

        band_four = numpy.ma.masked_array(
            numpy.array([[0, 0], [0, 0]],
                        dtype='double'),
            mask=numpy.array([[True, True],
                              [False, True]],
                             dtype='bool'))

        band_five = numpy.ma.masked_array(
            numpy.array([[0, 0], [0, 0]],
                        dtype='double'),
            mask=numpy.array([[False, True],
                              [False, True]],
                             dtype='bool'))

        sum_masked_array = [band_one, band_two, band_three,
                            band_four, band_five]

        golden_alpha = numpy.array([[0, 0],
                                    [65535, 0]], dtype='uint16')

        output_alpha = time_stack._uniform_weight_alpha(sum_masked_array,
                                                        numpy.uint16)

        numpy.testing.assert_array_equal(output_alpha,
                                         golden_alpha)


if __name__ == '__main__':
    unittest.main()
