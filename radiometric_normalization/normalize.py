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


def generate_transformations(candidate_path, reference_paths, config=None):
    if config is None:
        config = {'time_stack_method': 'identity',
                  'pif_method': 'identity',
                  'transformation_method': 'linear_relationship'}

    # Ensure all entries are there for partial configs
    if 'time_stack_method' not in config:
        config['time_stack_method'] = 'identity'
    if 'pif_method' not in config:
        config['pif_method'] = 'identity'
    if 'transformation_method' not in config:
        config['transformation_method'] = 'linear_relationship'

    if config['time_stack_method'] != 'skip':
        reference_image = time_stack.generate(
            reference_paths, 'time_stack.tif',
            method=config['time_stack_method'])
    else:
        # Assumes that the reference_paths is a pre-made time stack or
        # another compatible image
        # i.e. it is a single file name NOT a list of file names
        reference_image = reference_paths

    if config['pif_method'] != 'skip':
        pif_weight, reference_gimg, candidate_gimg = pif.generate(
            candidate_path, reference_path=reference_image,
            method=config['pif_method'])
    else:
        # Assumes that the reference_paths is an image with the pif strength
        # weightings as the alpha band
        reference_gimg = gimage.load(reference_image)
        candidate_gimg = gimage.load(candidate_path)
        pif_weight = reference_gimg.alpha

    if config['transformation_method'] != 'skip':
        transformations = transformation.generate(
            pif_weight, reference_gimg, candidate_gimg,
            method=config['transformation_method'])
    else:
        # Nothing really makes sense here, so just output an identity
        # transformation and ignore all inputs
        no_bands = len(reference_gimg.bands)
        transformations = [transformation.LinearTransformation(1.0, 0.0) for
                           band_count in xrange(no_bands)]

    return transformations


def apply_transformations(input_path, transformations, output_path):
    gimg = gimage.load(input_path)
    out_gimg = transformation.apply(gimg, transformations)
    gimage.save(out_gimg, output_path)
