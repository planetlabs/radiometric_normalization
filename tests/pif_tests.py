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

        # Pixels at [0, 0], [0, 1] and [1, 1] are not masked
        c_band0 = numpy.array([[8, 6], [3, 7]], dtype='uint16')
        c_band1 = numpy.array([[9, 1], [4, 4]], dtype='uint16')
        c_band2 = numpy.array([[3, 3], [5, 1]], dtype='uint16')
        c_alpha = numpy.array([[65535, 65535], [0, 65535]], dtype='uint16')
        candidate = gimage.GImage([c_band0, c_band1, c_band2], c_alpha, {})

        golden_pif_weight = numpy.array([[0, 65535],
                                        [0, 65535]], dtype='uint16')

        test_pif_weight = pif._filter_zero_alpha_pifs(candidate, reference)

        numpy.testing.assert_array_equal(test_pif_weight,
                                         golden_pif_weight)

    def test__PCA_fit_single_band(self):
        test_pca = pif._PCA_fit_single_band(numpy.array([1, 2, 3, 4, 5]),
                                            numpy.array([1, 2, 3, 4, 5]))
        sqrt_0_5 = numpy.sqrt(0.5)
        numpy.testing.assert_array_almost_equal(
            test_pca.components_, numpy.array([[sqrt_0_5, sqrt_0_5],
                                               [-1 * sqrt_0_5, sqrt_0_5]]))
        self.assertAlmostEqual(test_pca.explained_variance_[1], 0)
        self.assertTrue(
            test_pca.explained_variance_[0] > test_pca.explained_variance_[1])
        test_pca = pif._PCA_fit_single_band([1.00001, 1, 1, 1, 1],
                                            [0, 0, 0, 0, 0])
        numpy.testing.assert_array_almost_equal(
            test_pca.components_, numpy.array([[-1, 0],
                                               [0, 1]]))

    def test__PCA_filter_single_band(self):
        test_pca = pif._PCA_fit_single_band([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
        passed_pixels = pif._PCA_filter_single_band(
            test_pca, [1.1, 2.5, 3, 10, 5], [1.1, 2.5, 2.5, 4, 5], 1)
        numpy.testing.assert_array_equal(passed_pixels,
                                         numpy.array([[0, 1, 2, 4]]))

    def test__PCA_PIF_single_band(self):
        golden_pif_weight = numpy.array(
            [[1, 0, 0, 1, 1],
             [0, 0, 0, 1, 1],
             [1, 1, 1, 1, 1],
             [0, 0, 0, 0, 0],
             [1, 1, 0, 0, 0]])
        array_shape = (5, 5)
        valid_pixels = numpy.array([[0, 1, 2, 3, 4, 5, 6, 8, 9,
                                     10, 11, 12, 13, 14, 15, 20, 21]])
        passed_pixels = numpy.array([[0, 3, 4, 7, 8, 9, 10, 11,
                                      12, 13, 15, 16]])
        pif_weight = pif._PCA_PIF_single_band(passed_pixels, valid_pixels,
                                              array_shape)
        numpy.testing.assert_array_equal(pif_weight, golden_pif_weight)

    def test__PCA_fit_and_filter_single_band(self):
        ref_band = numpy.array([10, 20, 30,
                                40, 50, 60,
                                70, 80, 90])
        cand_band = numpy.array([11, 19, 29,
                                 100, 50, 70,
                                 71, 79, 90])
        alpha_vec = numpy.array([65535, 65535, 65535,
                                 0, 65535, 65535,
                                 65535, 65535, 65535])
        valid_pixels = numpy.nonzero(alpha_vec != 0)
        array_shape = (3, 3)

        # Standard test
        pif_weight = pif._PCA_fit_and_filter_single_band(cand_band, ref_band,
                                                         valid_pixels,
                                                         array_shape,
                                                         5, None)
        golden_pif_weight = numpy.array([[1, 1, 1],
                                         [0, 1, 0],
                                         [1, 1, 1]])
        numpy.testing.assert_array_equal(pif_weight, golden_pif_weight)

        # Make a very tight range
        pif_weight = pif._PCA_fit_and_filter_single_band(cand_band, ref_band,
                                                         valid_pixels,
                                                         array_shape,
                                                         0.001, None)
        golden_pif_weight = numpy.array([[0, 0, 0],
                                         [0, 0, 0],
                                         [0, 0, 0]])
        numpy.testing.assert_array_equal(pif_weight, golden_pif_weight)

        # Make a very wide range
        pif_weight = pif._PCA_fit_and_filter_single_band(cand_band, ref_band,
                                                         valid_pixels,
                                                         array_shape,
                                                         1000, None)
        golden_pif_weight = numpy.array([[1, 1, 1],
                                         [0, 1, 1],
                                         [1, 1, 1]])
        numpy.testing.assert_array_equal(pif_weight, golden_pif_weight)

        # Test batches small
        pif_weight = pif._PCA_fit_and_filter_single_band(cand_band, ref_band,
                                                         valid_pixels,
                                                         array_shape,
                                                         5, 1)
        golden_pif_weight = numpy.array([[1, 1, 1],
                                         [0, 1, 0],
                                         [1, 1, 1]])
        numpy.testing.assert_array_equal(pif_weight, golden_pif_weight)

        # Test batches even
        pif_weight = pif._PCA_fit_and_filter_single_band(cand_band, ref_band,
                                                         valid_pixels,
                                                         array_shape,
                                                         5, 2)
        numpy.testing.assert_array_equal(pif_weight, golden_pif_weight)

        # Test batches large
        pif_weight = pif._PCA_fit_and_filter_single_band(cand_band, ref_band,
                                                         valid_pixels,
                                                         array_shape,
                                                         5, 10000)
        numpy.testing.assert_array_equal(pif_weight, golden_pif_weight)


if __name__ == '__main__':
    unittest.main()
