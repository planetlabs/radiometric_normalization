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

from radiometric_normalization import pif
from radiometric_normalization.pif import pca_options


class Tests(unittest.TestCase):
    def test_generate_alpha_band_pifs(self):
        # Pixels at [0, 1], [1, 0], and [1, 1] are not masked
        r_alpha = numpy.array([[0, 1], [1, 1]], dtype=numpy.bool)

        # Pixels at [0, 0], [0, 1] and [1, 1] are not masked
        c_alpha = numpy.array([[1, 1], [0, 1]], dtype=numpy.bool)

        golden_pif_mask = numpy.array([[0, 1],
                                       [0, 1]], dtype=numpy.bool)

        pif_mask = pif.generate_alpha_band_pifs(c_alpha, r_alpha)

        numpy.testing.assert_array_equal(pif_mask,
                                         golden_pif_mask)

    def test__PCA_fit_single_band(self):
        test_pca = pif._pca_fit_single_band(numpy.array([1, 2, 3, 4, 5]),
                                            numpy.array([1, 2, 3, 4, 5]))
        sqrt_0_5 = numpy.sqrt(0.5)
        numpy.testing.assert_array_almost_equal(
            test_pca.components_, numpy.array([[-1 * sqrt_0_5, -1 * sqrt_0_5],
                                               [-1 * sqrt_0_5, sqrt_0_5]]))
        self.assertAlmostEqual(test_pca.explained_variance_[1], 0)
        self.assertTrue(
            test_pca.explained_variance_[0] > test_pca.explained_variance_[1])
        test_pca = pif._pca_fit_single_band(
            [100001, 100000, 100000, 100000, 100000],
            [0, 0, 0, 0, 0])
        numpy.testing.assert_array_almost_equal(
            test_pca.components_, numpy.array([[1, 0],
                                               [0, 1]]))

    def test__PCA_filter_single_band(self):
        test_pca = pif._pca_fit_single_band([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
        passed_pixels = pif._pca_filter_single_band(
            test_pca, [1.1, 2.5, 3, 10, 5], [1.1, 2.5, 2.5, 4, 5], 1)
        numpy.testing.assert_array_equal(passed_pixels,
                                         numpy.array([0, 1, 2, 4]))

    def test__PCA_PIF_single_band(self):
        golden_pif_weight = numpy.array(
            [[1, 0, 0, 1, 1],
             [0, 0, 0, 1, 1],
             [1, 1, 1, 1, 1],
             [0, 0, 0, 0, 0],
             [1, 1, 0, 0, 0]])
        array_shape = (5, 5)
        valid_pixels = numpy.array([0, 1, 2, 3, 4, 5, 6, 8, 9,
                                    10, 11, 12, 13, 14, 15, 20, 21])
        passed_pixels = numpy.array([0, 3, 4, 7, 8, 9, 10, 11,
                                     12, 13, 15, 16])
        alpha_vec = numpy.zeros(array_shape).ravel()
        alpha_vec[valid_pixels] = 1
        alpha = numpy.reshape(alpha_vec, array_shape)
        pif_weight = pif._create_pif_mask(passed_pixels, alpha)
        numpy.testing.assert_array_equal(pif_weight, golden_pif_weight)

    def test_generate_pca_pifs(self):
        ref_band = numpy.array([[10, 20, 30],
                                [40, 50, 60],
                                [70, 80, 90]], dtype=numpy.uint16)
        cand_band = numpy.array([[11, 19, 29],
                                 [100, 50, 70],
                                 [71, 79, 90]], dtype=numpy.uint16)
        alpha = numpy.array([[1, 1, 1],
                             [0, 1, 1],
                             [1, 1, 1]], dtype=numpy.bool)

        # Standard test
        pif_weight = pif.generate_pca_pifs(cand_band, alpha,
                                           ref_band, alpha,
                                           pca_options(limit=5))
        golden_pif_weight = numpy.array([[1, 1, 1],
                                         [0, 1, 0],
                                         [1, 1, 1]], dtype=numpy.bool)
        numpy.testing.assert_array_equal(pif_weight, golden_pif_weight)

        # Make a very tight range
        pif_weight = pif.generate_pca_pifs(cand_band, alpha,
                                           ref_band, alpha,
                                           pca_options(limit=0.001))
        golden_pif_weight = numpy.array([[0, 0, 0],
                                         [0, 0, 0],
                                         [0, 0, 0]])
        numpy.testing.assert_array_equal(pif_weight, golden_pif_weight)

        # Make a very wide range
        pif_weight = pif.generate_pca_pifs(cand_band, alpha,
                                           ref_band, alpha,
                                           pca_options(limit=1000))
        golden_pif_weight = numpy.array([[1, 1, 1],
                                         [0, 1, 1],
                                         [1, 1, 1]])
        numpy.testing.assert_array_equal(pif_weight, golden_pif_weight)


if __name__ == '__main__':
    unittest.main()
