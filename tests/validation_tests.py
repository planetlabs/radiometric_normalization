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
import os

import numpy
from osgeo import gdal, gdal_array

from radiometric_normalization.validation import validation


class Tests(unittest.TestCase):
    def test_sum_of_rmse(self):
        mask1 = [[0, 1], [0, 0]]
        bands1 = numpy.resize(range(8), (2, 2, 2))
        image1 = validation.Image(bands1, mask1)

        mask2 = [[0, 0], [1, 0]]
        bands2 = numpy.resize(range(8, 16), (2, 2, 2))
        image2 = validation.Image(bands2, mask2)
        result = validation.sum_of_rmse(image1, image2)

        expected = 16
        assert result == expected

    def test_read_image(self):
        band = numpy.array([[0, 1], [2, 3]], dtype=numpy.uint16)
        mask = numpy.array([[0, 1], [0, 1]], dtype=numpy.uint16)

        test_photometric_image = 'test_alpha_image.tif'
        photometric_ds = gdal.GetDriverByName('GTiff').Create(
            test_photometric_image, 2, 2, 4, gdal.GDT_UInt16,
            options=['PHOTOMETRIC=RGB', 'ALPHA=YES'])
        gdal_array.BandWriteArray(photometric_ds.GetRasterBand(1), band)
        gdal_array.BandWriteArray(photometric_ds.GetRasterBand(2), band)
        gdal_array.BandWriteArray(photometric_ds.GetRasterBand(3), band)
        gdal_array.BandWriteArray(photometric_ds.GetRasterBand(4),
                                  65535 * mask)
        photometric_ds = None

        image = validation.read_image(test_photometric_image)

        assert len(image.bands) == 3, len(image.bands)
        numpy.testing.assert_array_equal(image.bands[0], band)
        numpy.testing.assert_array_equal(image.bands[1], band)
        numpy.testing.assert_array_equal(image.bands[2], band)
        numpy.testing.assert_array_equal(image.mask, mask)
        os.unlink(test_photometric_image)


if __name__ == '__main__':
    unittest.main()
