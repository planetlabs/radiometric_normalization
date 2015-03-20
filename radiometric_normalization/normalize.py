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
from radiometric_normalization import \
    time_stack, pif, transformation, gimage


def generate_transforms(candidate_path, reference_paths, config=None):
    if config is None:
        config = {'time_stack_method': 'identity',
                  'pif_method': 'identity',
                  'transformation_method': 'linear_relationship'}

    reference_image = time_stack.generate(
        reference_paths,
        method=config['time_stack_method'])

    pif_weight, reference_img, candidate_img = pif.generate(
        candidate_path, reference_path=reference_image,
        method=config['pif_method'])
    transformations = transformation.generate(
        pif_weight, reference_img, candidate_img,
        method=config['transformation_method'])
    return transformations


def apply_transforms(input_path, transformations, output_path):
    gimg = gimage.load(input_path)
    out_gimg = transformation.apply(gimg, transformations)
    gimage.save(out_gimg, output_path)
