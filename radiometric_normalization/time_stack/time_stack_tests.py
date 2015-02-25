import unittest
import numpy
import os

from osgeo import gdal, gdal_array

import time_stack


class Tests(unittest.TestCase):

    def test_organise_images_to_bands(self):
        # Seven images of five bands of 10 by 11 pixels
        all_images = []
        for i in range(7):
            image = numpy.random.randint(
                2 ** 15, size=(5, 10, 11)).astype('uint16')
            all_images.append(image)

        all_bands = time_stack._organise_images_to_bands(all_images)
        no_bands = len(all_bands)
        no_images, rows, cols = all_bands[0].shape

        self.assertEqual(no_bands, 5)
        self.assertEqual(no_images, 7)
        self.assertEqual(rows, 10)
        self.assertEqual(cols, 11)
        for image in range(no_images):
            for band in range(no_bands):
                for row in range(rows):
                    for col in range(cols):
                        self.assertEqual(all_bands[band][image, row, col],
                                         all_images[image][band, row, col])

    def test_calculate_value_and_weight(self):
        nodata = 2 ** 15 - 1

        # Three images of three bands of 4 by 4 pixels
        all_bands = [numpy.array([[[13129, 29127, nodata, 30488],
                                  [18871, 23944,  3159,  5326],
                                  [17308, 31695,  8241, 29321],
                                  [22494, 31603, 17346,  7585]],
                                 [[22429, 14056, 22361, 14604],
                                  [23402,  9437, 27504, 19035],
                                  [5796, 30905, 21621, 27360],
                                  [20913, 12752, nodata,  8800]],
                                 [[10614,  7821, 26941,  1982],
                                  [19362, 15763,   844, 22554],
                                  [12080,  7927, 26943, 11538],
                                  [7156, 22539, 26198, 19891]]],
                                 dtype='uint16'),
                     numpy.array([[[11137, 12202, 28848, 14805],
                                   [22342, 21157, 18536, 14560],
                                   [nodata, 24355, 21868, 24596],
                                   [19672, 11785, 16609,  1104]],
                                 [[12004,  2218, 21771, 12724],
                                  [22708, 23334, 22187,  3413],
                                  [20650, 11131, nodata, 23231],
                                  [21749,   964, 16115, 19726]],
                                 [[32438, 10125, 15599,  5735],
                                  [10204,  1896,  5504, 26481],
                                  [20889, 27315, 18096, 20800],
                                  [29071,  8727,  7379, 29277]]],
                                 dtype='uint16'),
                     numpy.array([[[7035,  1890, 12231,  1044],
                                   [7945, 22298, 30919,  8235],
                                   [4825, 12739,  8933, 30074],
                                   [6126, 31793, 23123, 32570]],
                                  [[nodata,  8523, 28318,  6490],
                                   [6542, 17906, 12142, 28902],
                                   [4499, 26054,  1878, 19190],
                                   [5892, 23848, 24436, 17464]],
                                  [[13583,  4791, 12165,  2426],
                                   [1964, 21908, 21485, 11603],
                                   [19385, 13181,  nodata,  9779],
                                   [9163,   398, 24851, 23158]]],
                                 dtype='uint16')]

        golden_output = numpy.array([[[15390, 17001, 24651, 15691],
                                      [20545, 16381, 10502, 15638],
                                      [11728, 23509, 18935, 22739],
                                      [16854, 22298, 21772, 12092]],
                                     [[18526,  8181, 22072, 11088],
                                      [18418, 15462, 15409, 14818],
                                      [20769, 20933, 19982, 22875],
                                      [23497,  7158, 13367, 16702]],
                                     [[10309,  5068, 17571,  3320],
                                      [5483, 20704, 21515, 16246],
                                      [9569, 17324,  5405, 19681],
                                      [7060, 18679, 24136, 24397]]],
                                    dtype='uint16')
        golden_mask = numpy.ones((4, 4), dtype='uint16')

        output_bands, mask = time_stack._calculate_value_and_weight(all_bands,
                                                                    nodata,
                                                                    'identity',
                                                                    nodata)

        self.assertEqual(output_bands.all(), golden_output.all())
        self.assertEqual(mask.all(), golden_mask.all())

    def test_write_out_bands(self):
        # An image with eight bands of 102 by 99 pixels
        # and an alpha band
        image = numpy.random.randint(
            2 ** 15, size=(8, 102, 99)).astype('uint16')
        mask = numpy.ones((102, 99), dtype='uint16')
        output_path = 'test.tif'
        output_nodata = 0
        time_stack._write_out_bands(image, mask,
                                    output_path, output_nodata)

        # Check it
        image_ds = gdal.Open('test.tif')
        self.assertEqual(image_ds.RasterCount, 9)
        self.assertEqual(image_ds.RasterXSize, 99)
        self.assertEqual(image_ds.RasterYSize, 102)
        self.assertEqual(image_ds.GetRasterBand(1).GetNoDataValue(), 0)

        os.unlink('test.tif')
