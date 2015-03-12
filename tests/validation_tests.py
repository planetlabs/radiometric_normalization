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

from radiometric_normalization import gimage, validation


class Tests(unittest.TestCase):
    def test_sum_of_rmse(self):
        mask1 = [[0, 1], [0, 0]]
        bands1 = numpy.resize(range(8), (2, 2, 2))
        image1 = gimage.GImage(bands1, mask1, {})

        mask2 = [[0, 0], [1, 0]]
        bands2 = numpy.resize(range(8, 16), (2, 2, 2))
        image2 = gimage.GImage(bands2, mask2, {})
        result = validation.sum_of_rmse(image1, image2)

        expected = 16
        assert result == expected


if __name__ == '__main__':
    unittest.main()
