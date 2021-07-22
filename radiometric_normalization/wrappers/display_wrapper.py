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
import numpy
from osgeo import gdal

from radiometric_normalization import display
from radiometric_normalization import gimage


def create_pixel_plots(candidate_path, reference_path, base_name,
                       last_band_alpha=False, limits=None, custom_alpha=None):
    c_ds, c_alpha, c_band_count = _open_image_and_get_info(
        candidate_path, last_band_alpha)
    r_ds, r_alpha, r_band_count = _open_image_and_get_info(
        reference_path, last_band_alpha)

    _assert_consistent(c_alpha, r_alpha, c_band_count, r_band_count)

    if custom_alpha != None:
        combined_alpha = custom_alpha
    else:
        combined_alpha = numpy.logical_and(c_alpha, r_alpha)
    valid_pixels = numpy.nonzero(combined_alpha)

    for band_no in range(1, c_band_count + 1):
        c_band = gimage.read_single_band(c_ds, band_no)
        r_band = gimage.read_single_band(r_ds, band_no)
        file_name = '{}_{}.png'.format(base_name, band_no)
        display.plot_pixels(file_name, c_band[valid_pixels],
                            r_band[valid_pixels], limits)


def create_all_bands_histograms(candidate_path, reference_path, base_name,
                                last_band_alpha=False,
                                color_order=['b', 'g', 'r', 'y'],
                                x_limits=None, y_limits=None):
    c_gimg = gimage.load(candidate_path, last_band_alpha=last_band_alpha)
    r_gimg = gimage.load(reference_path, last_band_alpha=last_band_alpha)

    gimage.check_comparable([c_gimg, r_gimg])

    combined_alpha = numpy.logical_and(c_gimg.alpha, r_gimg.alpha)
    valid_pixels = numpy.nonzero(combined_alpha)

    file_name = '{}_histograms.png'.format(base_name)
    display.plot_histograms(
        file_name,
        [c_band[valid_pixels] for c_band in c_gimg.bands],
        [r_band[valid_pixels] for r_band in r_gimg.bands],
        color_order, x_limits, y_limits)


def _open_image_and_get_info(path, last_band_alpha):
    gdal_ds = gdal.Open(path)
    alpha_band, band_count = gimage.read_alpha_and_band_count(
        gdal_ds, last_band_alpha=last_band_alpha)
    return gdal_ds, alpha_band, band_count


def _assert_consistent(c_alpha, r_alpha, c_band_count, r_band_count):
    assert r_band_count == c_band_count
    assert r_alpha.shape == c_alpha.shape
