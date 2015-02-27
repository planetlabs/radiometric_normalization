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

from radiometric_normalization import gimage


class Tests(unittest.TestCase):
    def setUp(self):
        self.band = numpy.array([[0, 1], [2, 3]], dtype=numpy.uint16)
        self.mask = 65535 * numpy.array([[0, 1], [0, 1]], dtype=numpy.uint16)
        self.metadata = {'geotransform': (-1.0, 2.0, 0.0, 1.0, 0.0, -1.0)}

        self.test_photometric_alpha_image = 'test_photometric_alpha_image.tif'
        test_ds = gdal.GetDriverByName('GTiff').Create(
            self.test_photometric_alpha_image, 2, 2, 4, gdal.GDT_UInt16,
            options=['PHOTOMETRIC=RGB', 'ALPHA=YES'])
        gdal_array.BandWriteArray(test_ds.GetRasterBand(1), self.band)
        gdal_array.BandWriteArray(test_ds.GetRasterBand(2), self.band)
        gdal_array.BandWriteArray(test_ds.GetRasterBand(3), self.band)
        gdal_array.BandWriteArray(test_ds.GetRasterBand(4), self.mask)
        test_ds.SetGeoTransform(self.metadata['geotransform'])

    def tearDown(self):
        os.unlink(self.test_photometric_alpha_image)

    def test__read_metadata(self):
        gdal_ds = gdal.Open(self.test_photometric_alpha_image)
        test_metadata = gimage._read_metadata(gdal_ds)
        self.assertEqual(test_metadata, self.metadata)

    def test__read_bands(self):
        gdal_ds = gdal.Open(self.test_photometric_alpha_image)
        bands = gimage._read_bands(gdal_ds, 3)
        numpy.testing.assert_array_equal(bands[0], self.band)
        numpy.testing.assert_array_equal(bands[1], self.band)
        numpy.testing.assert_array_equal(bands[2], self.band)

    def test__read_alpha_and_band_count(self):
        gdal_ds = gdal.Open(self.test_photometric_alpha_image)
        alpha, band_count = gimage._read_alpha_and_band_count(gdal_ds)

        self.assertEqual(band_count, 3)
        numpy.testing.assert_array_equal(alpha, self.mask)

    def test_save(self):
        output_file = 'test_file.tif'

        test_gimage = gimage.GImage([self.band], self.mask, self.metadata)
        gimage.save(test_gimage, output_file)

        test_ds = gdal.Open(output_file)

        saved_number_of_bands = test_ds.RasterCount
        self.assertEquals(saved_number_of_bands, 2)

        saved_band = test_ds.GetRasterBand(1).ReadAsArray()
        numpy.testing.assert_array_equal(saved_band, self.band)

        saved_alpha = test_ds.GetRasterBand(2).ReadAsArray()
        numpy.testing.assert_array_equal(saved_alpha, self.mask)

        os.unlink(output_file)


if __name__ == '__main__':
    unittest.main()
