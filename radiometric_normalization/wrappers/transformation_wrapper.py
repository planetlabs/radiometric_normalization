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
from radiometric_normalization import transformation


def generate(candidate_path, reference_path, pif_mask,
             method='linear_relationship', last_band_alpha=False):
    ''' Calculates the transformations between the PIF pixels of the candidate
    image and PIF pixels of the reference image.

    :param str candidate_path: Path to the candidate image
    :param str reference_path: Path to the reference image
    :param array pif_mask: A boolean array in the same coordinate system of the
        candidate/reference image (True for the PIF)
    :param str method: Which method to find the transformation

    :returns: A list of linear transformations (one for each band)
    '''
    if method == 'linear_relationship':
        c_ds, c_alpha, c_band_count = _open_image_and_get_info(
            candidate_path, last_band_alpha)
        r_ds, r_alpha, r_band_count = _open_image_and_get_info(
            reference_path, last_band_alpha)

        _assert_consistent(c_alpha, r_alpha, c_band_count, r_band_count)

        transformations = []
        for band_no in range(1, c_band_count + 1):
            c_band = gimage.read_single_band(c_ds, band_no)
            r_band = gimage.read_single_band(r_ds, band_no)
            transformations.append(
                transformation.generate_linear_relationship(
                    c_band, r_band, pif_mask))
    else:
        raise NotImplementedError('Only "linear_relationship" '
                                  'method is implemented.')

    return transformations


def _open_image_and_get_info(path, last_band_alpha):
    gdal_ds = gdal.Open(path)
    alpha_band, band_count = gimage.read_alpha_and_band_count(
        gdal_ds, last_band_alpha=last_band_alpha)
    return gdal_ds, alpha_band, band_count


def _assert_consistent(c_alpha, r_alpha, c_band_count, r_band_count):
    assert r_band_count == c_band_count
    assert r_alpha.shape == c_alpha.shape
