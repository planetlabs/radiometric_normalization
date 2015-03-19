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

from radiometric_normalization import gimage, transformation


class Tests(unittest.TestCase):
    def test_get_transform_function(self):
        self.assertEqual(
            transformation.get_transform_function('linear_relationship'),
            transformation.linear_relationship)
        self.assertRaises(Exception,
                          transformation.get_transform_function,
                          'unknown')

    def test_pifs_to_pifset(self):
        testing_pif_weights = numpy.array([[1, 2], [0, 0]], dtype=numpy.uint16)

        r_band0 = numpy.array([[1, 3], [0, 0]], dtype=numpy.uint16)
        r_band1 = numpy.array([[2, 3], [0, 0]], dtype=numpy.uint16)
        r_band2 = numpy.array([[3, 3], [0, 0]], dtype=numpy.uint16)
        r_band3 = numpy.array([[4, 3], [0, 0]], dtype=numpy.uint16)
        r_alpha = numpy.array([[97, 125], [0, 0]], dtype=numpy.uint16)
        reference_img = gimage.GImage(
            [r_band0, r_band1, r_band2, r_band3], r_alpha, {})

        c_band0 = numpy.array([[2, 4], [0, 0]], dtype=numpy.uint16)
        c_band1 = numpy.array([[3, 4], [0, 0]], dtype=numpy.uint16)
        c_band2 = numpy.array([[4, 4], [0, 0]], dtype=numpy.uint16)
        c_band3 = numpy.array([[5, 4], [0, 0]], dtype=numpy.uint16)
        c_alpha = numpy.array([[1, 1], [0, 0]], dtype=numpy.uint16)
        candidate_img = gimage.GImage(
            [c_band0, c_band1, c_band2, c_band3], c_alpha, {})

        pifset = transformation.pifs_to_pifset(
            testing_pif_weights, reference_img, candidate_img)

        expected_reference = numpy.array(
            [[1, 2, 3, 4], [3, 3, 3, 3]])
        numpy.testing.assert_array_equal(pifset.reference, expected_reference)

        expected_candidate = numpy.array(
            [[2, 3, 4, 5], [4, 4, 4, 4]])
        numpy.testing.assert_array_equal(pifset.candidate, expected_candidate)

        expected_weight = numpy.array([1, 2])
        numpy.testing.assert_array_equal(pifset.weight, expected_weight)

    def test_linear_relationship(self):
        test_candidate = numpy.array(
            [[1, 2, 3, 4],
             [1, 3, 4, 4]])
        test_reference = numpy.array(
            [[2, 4, 4, 3],
             [2, 6, 5, 3]])
        test_weight = numpy.array([0.5, 0.75])

        test_pifset = transformation.PIFSet(test_reference,
                                            test_candidate,
                                            test_weight)

        transformations = transformation.linear_relationship(test_pifset)

        gains = [tf.gain for tf in transformations]
        expected_gains = [1, 2, 1, 1]
        self.assertEqual(gains, expected_gains)

        offsets = [tf.offset for tf in transformations]
        expected_offsets = [1, 0, 1, -1]
        self.assertEqual(offsets, expected_offsets)

    def test_apply(self):
        test_band = numpy.array([[0, 100], [1000, 65535]], dtype=numpy.uint16)
        test_alpha = 2 ** 16 - 1 * numpy.ones(
            test_band.shape, dtype=numpy.uint16)
        test_input_gimg = gimage.GImage(
            [test_band, test_band, test_band, test_band], test_alpha, {})
        test_transformations = [
            transformation.LinearTransformation(0.5, 0),
            transformation.LinearTransformation(1, 500),
            transformation.LinearTransformation(1, -500),
            transformation.LinearTransformation(5, 0)]

        output_gimg = transformation.apply(
            test_input_gimg, test_transformations)

        expected_band1 = numpy.array(
            [[0, 50], [500, 32767]], dtype=numpy.uint16)
        numpy.testing.assert_array_equal(output_gimg.bands[0], expected_band1)

        expected_band2 = numpy.array(
            [[500, 600], [1500, 65535]], dtype=numpy.uint16)
        numpy.testing.assert_array_equal(output_gimg.bands[1], expected_band2)

        expected_band3 = numpy.array(
            [[0, 0], [500, 65035]], dtype=numpy.uint16)
        numpy.testing.assert_array_equal(output_gimg.bands[2], expected_band3)

        expected_band4 = numpy.array(
            [[0, 500], [5000, 65535]], dtype=numpy.uint16)
        numpy.testing.assert_array_equal(output_gimg.bands[3], expected_band4)

    def test_linear_transformation_to_lut(self):
        test_linear_transform = \
            transformation.LinearTransformation(gain=1, offset=2)

        lut = transformation.linear_transformation_to_lut(
            test_linear_transform)

        expected_values = list(range(2, 65536)) + 2 * [65535]
        expected_lut = numpy.array(expected_values)
        numpy.testing.assert_array_equal(lut, expected_lut)

if __name__ == '__main__':
    unittest.main()
