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
    transformation, gimage


def apply_transformations(input_path, transformations, output_path):
    ''' This wrapper function applies the transformations
    derived by the library onto two files.
    '''

    gimg = gimage.load(input_path)
    out_gimg = transformation.apply(gimg, transformations)
    gimage.save(out_gimg, output_path)
