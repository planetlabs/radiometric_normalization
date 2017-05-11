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
    def test_generate_linear_relationship(self):
        test_candidate = numpy.array(
            [[1, 2, 3, 4],
             [1, 3, 4, 4]])
        test_reference = numpy.array(
            [[2, 4, 4, 3],
             [2, 6, 8, 8]])
        test_pifs = numpy.array([[1, 1, 0, 0],
                                 [0, 0, 1, 1]], dtype=numpy.bool)

        transform = transformation.generate_linear_relationship(test_reference, test_candidate, test_pifs)

        expected_gain = 0.5
        self.assertEqual(transform.gain, expected_gain)

        expected_offset = 0
        self.assertEqual(transform.offset, expected_offset)

if __name__ == '__main__':
    unittest.main()
