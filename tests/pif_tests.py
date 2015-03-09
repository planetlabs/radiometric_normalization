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

from radiometric_normalization import gimage, pif


class Tests(unittest.TestCase):
    def test__filter_zero_alpha_pifs(self):
        # Pixels at [0, 1], [1, 0], and [1, 1] are not masked
        r_band0 = numpy.array([[5, 1], [1, 2]], dtype='uint16')
        r_band1 = numpy.array([[3, 7], [8, 6]], dtype='uint16')
        r_band2 = numpy.array([[2, 9], [3, 1]], dtype='uint16')
        r_alpha = numpy.array([[0, 65535], [65535, 65535]], dtype='uint16')
        reference = gimage.GImage([r_band0, r_band1, r_band2], r_alpha, {})

        # Pixels at [0, 0] and [1, 1] are not masked
        c_band0 = numpy.array([[8, 6], [3, 7]], dtype='uint16')
        c_band1 = numpy.array([[9, 1], [4, 4]], dtype='uint16')
        c_band2 = numpy.array([[3, 3], [5, 1]], dtype='uint16')
        c_alpha = numpy.array([[65535, 65535], [0, 65535]], dtype='uint16')
        candidate = gimage.GImage([c_band0, c_band1, c_band2], c_alpha, {})

        golden_output_list = [{'coordinates': (0, 1),
                               'weighting': 65535,
                               'reference': [1, 7, 9],
                               'candidate': [6, 1, 3]},
                              {'coordinates': (1, 1),
                               'weighting': 65535,
                               'reference': [2, 6, 1],
                               'candidate': [7, 4, 1]}]

        test_output_list = pif._filter_zero_alpha_pifs(reference, candidate)
        self.assertEqual(len(test_output_list), 2)
        assert 1 == 2
        for test_pif, golden_pif in zip(test_output_list, golden_output_list):
            self.assertEqual(test_pif['coordinates'],
                             golden_pif['coordinates'])
            self.assertEqual(test_pif['weighting'],
                             golden_pif['weighting'])
            self.assertEqual(test_pif['reference'],
                             golden_pif['reference'])
            self.assertEqual(test_pif['candidate'],
                             golden_pif['candidate'])


if __name__ == '__main__':
    unittest.main()
