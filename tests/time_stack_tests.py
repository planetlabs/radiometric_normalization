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

from radiometric_normalization import time_stack


class Tests(unittest.TestCase):
    def test_organize_images_to_bands(self):
        # Seven images of five bands of 10 by 11 pixels
        all_images = []
        nodata = 2 ** 15 - 1
        for i in range(7):
            image = numpy.random.randint(
                2 ** 15, size=(5, 10, 11)).astype('uint16')
            all_images.append(image)

        all_bands = time_stack._organize_images_to_bands(all_images, nodata)
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


if __name__ == '__main__':
    unittest.main()
