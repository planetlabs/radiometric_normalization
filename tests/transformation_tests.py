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

from radiometric_normalization import transformation


class Tests(unittest.TestCase):
    def test_get_transform_function(self):
        self.assertEqual(
            transformation.get_transform_function('linear_relationship'),
            transformation.linear_relationship)
        self.assertRaises(Exception,
                          transformation.get_transform_function,
                          'unknown')

    def test_pifs_to_pifset(self):
        pif0 = {'coordinates': (0, 0),
                'weighting': 0.5,
                'reference': (1, 2, 3, 4),
                'candidate': (2, 3, 4, 5)}

        pif1 = {'coordinates': (0, 1),
                'weighting': 0.75,
                'reference': (3, 3, 3, 3),
                'candidate': (4, 4, 4, 4)}
        test_pifs = [pif0, pif1]

        pifset = transformation.pifs_to_pifset(test_pifs)

        expected_reference = numpy.array(
            [[1, 2, 3, 4], [3, 3, 3, 3]])
        numpy.testing.assert_array_equal(pifset.reference, expected_reference)

        expected_candidate = numpy.array(
            [[2, 3, 4, 5], [4, 4, 4, 4]])
        numpy.testing.assert_array_equal(pifset.candidate, expected_candidate)

        expected_weight = numpy.array([0.5, 0.75])
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
