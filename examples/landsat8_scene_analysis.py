import numpy

from radiometric_normalization import \
    time_stack, pif, transformation, gimage, normalize

import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

candidate_path = 'planet_common_test.radiometric_normalization_20150115_043519_080c_r.tif'
output_path = 'radiometric_normalization_reference_test.LC81360442013319LGN00_all_crop.tif'
pif_weight, reference_img, candidate_img = pif.generate(candidate_path, reference_path=output_path, method='identity')
transformations = transformation.generate(pif_weight, reference_img, candidate_img, method='linear_relationship')
normalize.apply_transform(candidate_path, transformations, 'output_landsat8_scene_analysis.tif')
