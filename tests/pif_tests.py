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
    def test__filter_nodata_pifs(self):
        nodata = 2 ** 15 - 1

        # Three bands of 2 by 2 pixels
        r_bands = numpy.array([numpy.array([[nodata, 1],
                                               [nodata, 2]],
                                              dtype='uint16'),
                                  numpy.array([[nodata, 7],
                                               [8, 6]],
                                              dtype='uint16'),
                                  numpy.array([[2, 9],
                                               [nodata, 1]],
                                              dtype='uint16')])
        r_mask = numpy.array([[0, 65535],
                                 [0, 65535]],
                                dtype='uint16')
        reference = gimage.GImage(r_bands, r_mask, {})

        c_bands = numpy.array([numpy.array([[8, 6],
                                              [nodata, 7]],
                                             dtype='uint16'),
                                 numpy.array([[9, 1],
                                              [nodata, 4]],
                                             dtype='uint16'),
                                 numpy.array([[3, 3],
                                              [nodata, 1]],
                                             dtype='uint16')])
        c_mask = numpy.array([[65535, 65535],
                                      [0, 65535]],
                                     dtype='uint16')
        candidate = gimage.GImage(c_bands, c_mask, {})

        golden_output_list = [{'coordinates': (0, 1),
                               'weighting': 65535,
                               'reference': numpy.array([1, 7, 9]),
                               'candidate': numpy.array([6, 1, 3])},
                              {'coordinates': (1, 1),
                               'weighting': 65535,
                               'reference': numpy.array([2, 6, 1]),
                               'candidate': numpy.array([7, 4, 1])}]

        output_list = pif._filter_nodata_pifs(reference, candidate)

        for pixel in range(len(output_list)):
            self.assertEqual(output_list[pixel]['coordinates'],
                             golden_output_list[pixel]['coordinates'])
            self.assertEqual(output_list[pixel]['weighting'],
                             golden_output_list[pixel]['weighting'])
            self.assertEqual(output_list[pixel]['reference'].all(),
                             golden_output_list[pixel]['reference'].all())
            self.assertEqual(output_list[pixel]['candidate'].all(),
                             golden_output_list[pixel]['candidate'].all())


if __name__ == '__main__':
    unittest.main()
