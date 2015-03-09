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

from radiometric_normalization import \
    time_stack, pif, transformation, gimage


def generate_luts(candidate_path, reference_paths, config=None):
    if config is None:
        config = {'time_stack_method': 'identity',
                  'pif_method': 'identity',
                  'transformation_method': 'linear_relationship'}

    reference_image = time_stack.generate(
        reference_paths,
        method=config['time_stack_method'])

    pifs = pif.generate(candidate_path,
                        reference_path=reference_image,
                        method=config['pif_method'])
    transformations = transformation.generate(
        pifs, method=config['transformation_method'])
    luts = transformation.transformations_to_luts(transformations)
    return luts


def apply_luts(input_path, luts, output_path):
    def apply_lut(band, lut):
        'Changes band intensity values based on intensity look up table (lut)'
        if lut.dtype != band.dtype:
            raise Exception(
                "Band ({}) and lut ({}) must be the same data type.").format(
                band.dtype, lut.dtype)
        return numpy.take(lut, band, mode='clip')

    img = gimage.load(input_path)
    for i in range(len(img.bands)):
        img.bands[i] = apply_lut(img.bands[i], luts[i])
    img.save(output_path)
