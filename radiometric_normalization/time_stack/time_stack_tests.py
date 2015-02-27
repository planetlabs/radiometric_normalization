import unittest
import numpy
import os

from osgeo import gdal, gdal_array

import time_stack


class Tests(unittest.TestCase):

    def test_organise_images_to_bands(self):
        # Seven images of five bands of 10 by 11 pixels
        all_images = []
        nodata = 2 ** 15 - 1
        for i in range(7):
            image = numpy.random.randint(
                2 ** 15, size=(5, 10, 11)).astype('uint16')
            all_images.append(image)

        all_bands = time_stack._organise_images_to_bands(all_images, nodata)
        no_bands = len(all_bands)
        no_images = len(all_bands[0])
        rows, cols = all_bands[0][0].shape

        self.assertEqual(no_bands, 5)
        self.assertEqual(no_images, 7)
        self.assertEqual(rows, 10)
        self.assertEqual(cols, 11)
        for image in range(no_images):
            for band in range(no_bands):
                for row in range(rows):
                    for col in range(cols):
                        self.assertEqual(all_bands[band][image].data[row, col],
                                         all_images[image][band, row, col])

    def test_mean_with_uniform_weight(self):
        # Three images of three bands of 2 by 2 pixels
        nodata = 2 ** 15 - 1

        all_bands = [[numpy.ma.array(numpy.array([[4, 1],
                                                  [2, 5]]),
                                     mask=[[0, 1], [0, 0]],
                                     dtype='uint16'),
                      numpy.ma.array(numpy.array([[3, 3],
                                                  [3, 1]]),
                                     mask=[[0, 1], [1, 0]],
                                     dtype='uint16'),
                      numpy.ma.array(numpy.array([[2, 9],
                                                  [10, 8]]),
                                     mask=[[0, 1], [0, 0]],
                                     dtype='uint16')],
                     [numpy.ma.array(numpy.array([[7, 8],
                                                  [6, 3]]),
                                     mask=[[0, 1], [1, 0]],
                                     dtype='uint16'),
                      numpy.ma.array(numpy.array([[5, 1],
                                                  [7, 9]]),
                                     mask=[[0, 1], [0, 1]],
                                     dtype='uint16'),
                      numpy.ma.array(numpy.array([[10, 7],
                                                  [1, 2]]),
                                     mask=[[0, 1], [0, 0]],
                                     dtype='uint16')],
                     [numpy.ma.array(numpy.array([[3, 2],
                                                  [7, 4]]),
                                     mask=[[0, 1], [0, 0]],
                                     dtype='uint16'),
                      numpy.ma.array(numpy.array([[6, 5],
                                                  [1, 2]]),
                                     mask=[[0, 1], [0, 0]],
                                     dtype='uint16'),
                      numpy.ma.array(numpy.array([[7, 6],
                                                  [4, 9]]),
                                     mask=[[0, 1], [0, 0]],
                                     dtype='uint16')]]

        golden_output = [numpy.array([[3, nodata],
                                      [6, 4]],
                                     dtype='uint16'),
                         numpy.array([[7, nodata],
                                      [4, 2]],
                                     dtype='uint16'),
                         numpy.array([[5, nodata],
                                      [4, 5]],
                                     dtype='uint16')]

        golden_mask = numpy.array([[65535, 0],
                                   [65535, 65535]],
                                  dtype='uint16')

        output_bands, mask = time_stack._mean_with_uniform_weight(all_bands,
                                                                  nodata,
                                                                  numpy.uint16)

        for band in range(3):
            self.assertEqual(output_bands[band].all(),
                             golden_output[band].all())
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


if __name__ == '__main__':
    unittest.main()
