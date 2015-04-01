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

        all_gimages = [gimage_one, gimage_two, gimage_three]

        golden_output = gimage.GImage(
            [numpy.array([[4, 0],
                          [4, 3]], dtype='uint16'),
             numpy.array([[2, 0],
                          [4, 4]], dtype='uint16'),
             numpy.array([[4, 0],
                          [5, 2]], dtype='uint16')],
            numpy.array([[65535, 0],
                         [65535, 65535]], dtype='uint16'), {})

        output_gimage = time_stack.mean_with_uniform_weight(all_gimages,
                                                             numpy.uint16)

        for band in range(3):
            self.assertEqual(output_gimage.bands[band].all(),
                             golden_output.bands[band].all())
        self.assertEqual(output_gimage.alpha.all(),
                         golden_output.alpha.all())
        self.assertEqual(output_gimage.metadata,
                         golden_output.metadata)

    def test__mean_one_band(self):
        # Four images of two bands of 2 by 2 pixels
        gimage_one = gimage.GImage(
            [numpy.array([[0, 0],
                          [0, 0]], dtype='uint16'),
             numpy.array([[1, 7],
                          [2, 6]], dtype='uint16')],
            numpy.array([[0, 65535], [65535, 0]], dtype='uint16'), {})

        gimage_two = gimage.GImage(
            [numpy.array([[0, 0],
                          [0, 0]], dtype='uint16'),
             numpy.array([[9, 2],
                          [2, 8]], dtype='uint16')],
            numpy.array([[0, 65535], [65535, 65535]], dtype='uint16'), {})

        gimage_three = gimage.GImage(
            [numpy.array([[0, 0],
                          [0, 0]], dtype='uint16'),
             numpy.array([[6, 8],
                          [4, 9]], dtype='uint16')],
            numpy.array([[0, 65535], [65535, 0]], dtype='uint16'), {})

        gimage_four = gimage.GImage(
            [numpy.array([[0, 0],
                          [0, 0]], dtype='uint16'),
             numpy.array([[8, 4],
                          [6, 2]], dtype='uint16')],
            numpy.array([[0, 0], [65535, 65535]], dtype='uint16'), {})

        all_gimages = [gimage_one, gimage_two, gimage_three, gimage_four]

        golden_mean = numpy.array([[0, 5],
                                   [3, 5]], dtype='uint16')
        golden_mask = numpy.array([[False, True],
                                   [True, True]], dtype='bool')

        band_mean, band_mask = time_stack._mean_one_band(all_gimages, 1,
                                                         numpy.uint16)

        self.assertEqual(band_mean.all(),
                         golden_mean.all())
        self.assertEqual(band_mask.all(),
                         golden_mask.all())

    def test__uniform_weight_alpha(self):
        # Six images of 2 by 2 pixels
        all_masks = [numpy.array([[False, True],
                                  [False, False]], dtype='bool'),
                     numpy.array([[False, False],
                                  [False, False]], dtype='bool'),
                     numpy.array([[True, True],
                                  [False, False]], dtype='bool'),
                     numpy.array([[True, True],
                                  [False, True]], dtype='bool'),
                     numpy.array([[False, True],
                                  [False, True]], dtype='bool')]

        golden_alpha = numpy.array([[False, True],
                                    [True, False]], dtype='uint16')

        output_alpha = time_stack._uniform_weight_alpha(all_masks,
                                                        numpy.uint16)

        self.assertEqual(output_alpha.all(),
                         golden_alpha.all())


if __name__ == '__main__':
    unittest.main()
