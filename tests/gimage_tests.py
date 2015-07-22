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

    def test__nodata_to_mask(self):
        test_band = numpy.array([[0, 1, 2], [1, 2, 3]], dtype=numpy.uint16)
        test_mask = gimage._nodata_to_mask([test_band], 3)

        expected_mask = numpy.array([[1, 1, 1], [1, 1, 0]], dtype=numpy.uint16)
        numpy.testing.assert_array_equal(test_mask, expected_mask)

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

    def test_create_ds(self):
        output_file = 'test_create_ds.tif'
        test_band = numpy.array([[0, 1, 2], [2, 3, 4]], dtype=numpy.uint16)
        test_gimage = gimage.GImage([test_band], self.mask, self.metadata)
        test_compress = False
        test_ds = gimage.create_ds(test_gimage, output_file,
                                   compress=test_compress)

        self.assertEqual(test_ds.RasterCount, 2)
        self.assertEqual(test_ds.RasterXSize, 3)
        self.assertEqual(test_ds.RasterYSize, 2)

        os.unlink(output_file)

    def test_save_with_compress(self):
        output_file = 'test_save_with_compress.tif'
        test_band = numpy.array([[5, 2, 2], [1, 6, 8]], dtype=numpy.uint16)
        test_alpha = numpy.array([[0, 0, 0], [65535, 65535, 65535]],
                                 dtype=numpy.uint16)
        test_gimage = gimage.GImage([test_band, test_band, test_band],
                                    test_alpha, self.metadata)
        gimage.save(test_gimage, output_file, compress=True)

        result_gimg = gimage.load(output_file)
        numpy.testing.assert_array_equal(result_gimg.bands[0], test_band)
        numpy.testing.assert_array_equal(result_gimg.bands[1], test_band)
        numpy.testing.assert_array_equal(result_gimg.bands[2], test_band)
        numpy.testing.assert_array_equal(result_gimg.alpha, test_alpha)
        self.assertEqual(result_gimg.metadata, self.metadata)

        os.unlink(output_file)

    def test_save_to_ds(self):
        output_file = 'test_save_to_ds.tif'

        test_band = numpy.array([[0, 1], [2, 3]], dtype=numpy.uint16)
        test_gimage = gimage.GImage([test_band], self.mask, self.metadata)
        output_ds = gdal.GetDriverByName('GTiff').Create(
            output_file, 2, 2, 2, gdal.GDT_UInt16,
            options=['ALPHA=YES'])
        gimage.save_to_ds(test_gimage, output_ds, nodata=3)

        # Required for gdal to write to file
        output_ds = None

        test_ds = gdal.Open(output_file)

        saved_number_of_bands = test_ds.RasterCount
        self.assertEquals(saved_number_of_bands, 2)

        saved_band = test_ds.GetRasterBand(1).ReadAsArray()
        numpy.testing.assert_array_equal(saved_band, self.band)

        saved_nodata = test_ds.GetRasterBand(1).GetNoDataValue()
        self.assertEqual(saved_nodata, 3)

        saved_alpha = test_ds.GetRasterBand(2).ReadAsArray()
        numpy.testing.assert_array_equal(saved_alpha, self.mask)

        os.unlink(output_file)

    def test_check_comparable(self):
        band1 = numpy.ones([2, 2])
        metadata = {'dummy_key': 'dummy_var'}
        one_band_gimage = gimage.GImage([band1], None, None)

        two_band_gimage = gimage.GImage([band1, band1], None, None)
        self.assertRaises(
            Exception,
            gimage.check_comparable,
            [one_band_gimage, two_band_gimage])

        one_band_gimage_with_metadata = gimage.GImage(
            [band1], None, metadata)
        self.assertRaises(
            Exception,
            gimage.check_comparable,
            [one_band_gimage, one_band_gimage_with_metadata],
            check_metadata=True)

    def test_check_equal(self):
        # Standard image
        gimage_one = gimage.GImage(
            [numpy.array([[4, 1],
                          [2, 5]], dtype='uint16'),
             numpy.array([[4, 1],
                          [2, 5]], dtype='uint16'),
             numpy.array([[7, 8],
                          [6, 3]], dtype='uint16')],
            numpy.array([[65535, 0], [65535, 65535]], dtype='uint16'),
            {'dummy_key': 'dummy_var'})

        # Different band data
        gimage_two = gimage.GImage(
            [numpy.array([[4, 1],
                          [2, 5]], dtype='uint16'),
             numpy.array([[4, 1],
                          [2, 5]], dtype='uint16'),
             numpy.array([[7, 8],
                          [9, 3]], dtype='uint16')],
            numpy.array([[65535, 0], [65535, 65535]], dtype='uint16'),
            {'dummy_key': 'dummy_var'})

        # Different alpha
        gimage_three = gimage.GImage(
            [numpy.array([[4, 1],
                          [2, 5]], dtype='uint16'),
             numpy.array([[4, 1],
                          [2, 5]], dtype='uint16'),
             numpy.array([[7, 8],
                          [6, 3]], dtype='uint16')],
            numpy.array([[65535, 0], [0, 65535]], dtype='uint16'),
            {'dummy_key': 'dummy_var'})

        # Not comparable
        gimage_four = gimage.GImage(
            [numpy.array([[4, 1],
                          [2, 5]], dtype='uint16')],
            numpy.array([[65535, 0], [65535, 65535]], dtype='uint16'),
            {'dummy_key': 'dummy_var'})

        # Different metadata
        gimage_five = gimage.GImage(
            [numpy.array([[4, 1],
                          [2, 5]], dtype='uint16'),
             numpy.array([[4, 1],
                          [2, 5]], dtype='uint16'),
             numpy.array([[7, 8],
                          [6, 3]], dtype='uint16')],
            numpy.array([[65535, 0], [65535, 65535]], dtype='uint16'),
            {'dummy_key': 'dummy_var',
             'different_key': 'different_var'})

        # All images are equal
        gimage.check_equal([gimage_one, gimage_one, gimage_one])

        # One image different band data
        self.assertRaises(Exception,
                          gimage.check_equal,
                          [gimage_one, gimage_one, gimage_two])

        # One image different alpha
        self.assertRaises(Exception,
                          gimage.check_equal,
                          [gimage_one, gimage_one, gimage_three])

        # One image different not comparable
        self.assertRaises(Exception,
                          gimage.check_equal,
                          [gimage_one, gimage_one, gimage_four])

        # One image different metadata
        self.assertRaises(Exception,
                          gimage.check_equal,
                          [gimage_one, gimage_one, gimage_five],
                          check_metadata=True)

if __name__ == '__main__':
    unittest.main()
