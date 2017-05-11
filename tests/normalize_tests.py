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

from radiometric_normalization import normalize
from radiometric_normalization.transformation import LinearTransformation


class Tests(unittest.TestCase):
    def test_apply(self):
        test_band = numpy.array([[0, 100], [1000, 65535]], dtype=numpy.uint16)

        test_transformation1 = LinearTransformation(0.5, 0)
        output_band1 = normalize.apply(test_band, test_transformation1)
        expected_band1 = numpy.array(
            [[0, 50], [500, 32767]], dtype=numpy.uint16)
        numpy.testing.assert_array_equal(output_band1, expected_band1)

        test_transformation2 = LinearTransformation(1, 500)
        output_band2 = normalize.apply(test_band, test_transformation2)
        expected_band2 = numpy.array(
            [[500, 600], [1500, 65535]], dtype=numpy.uint16)
        numpy.testing.assert_array_equal(output_band2, expected_band2)

        test_transformation3 = LinearTransformation(1, -500)
        output_band3 = normalize.apply(test_band, test_transformation3)
        expected_band3 = numpy.array(
            [[0, 0], [500, 65035]], dtype=numpy.uint16)
        numpy.testing.assert_array_equal(output_band3, expected_band3)

        test_transformation4 = LinearTransformation(1, -500)
        output_band4 = normalize.apply(test_band, test_transformation4)
        expected_band4 = numpy.array(
            [[0, 0], [500, 65035]], dtype=numpy.uint16)
        numpy.testing.assert_array_equal(output_band4, expected_band4)

    def test_linear_transformation_to_lut(self):
        test_linear_transform = LinearTransformation(gain=1, offset=2)

        lut = normalize._linear_transformation_to_lut(
            test_linear_transform)

        expected_values = list(range(2, 65536)) + 2 * [65535]
        expected_lut = numpy.array(expected_values)
        numpy.testing.assert_array_equal(lut, expected_lut)


if __name__ == '__main__':
    unittest.main()
