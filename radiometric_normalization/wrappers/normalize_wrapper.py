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
from osgeo import gdal

from radiometric_normalization import gimage
from radiometric_normalization import normalize


def generate(image_path, per_band_transformation, last_band_alpha=False):
    '''Applies a set of linear transformations to a gimage

    :param str image_path: The path to an image
    :param list per_band_transformation: A list of of LinearTransformations
        (length equal to the number of bands in the image)
    :param output: A gimage that represents input_gimage with transformations
        applied
    '''
    img_ds, img_alpha, band_count = _open_image_and_get_info(
        image_path, last_band_alpha)
    img_metadata = gimage.read_metadata(img_ds)

    _assert_consistent(band_count, per_band_transformation)

    output_bands = []
    for band_no, transformation in zip(
        range(1, band_count + 1), per_band_transformation):
        band = gimage.read_single_band(img_ds, band_no)
        output_bands.append(normalize.apply(band, transformation))

    return gimage.GImage(output_bands, img_alpha, img_metadata)


def _open_image_and_get_info(path, last_band_alpha):
    gdal_ds = gdal.Open(path)
    alpha_band, band_count = gimage.read_alpha_and_band_count(
        gdal_ds, last_band_alpha=last_band_alpha)
    return gdal_ds, alpha_band, band_count


def _assert_consistent(band_count, per_band_transformation):
    assert band_count == len(per_band_transformation)
