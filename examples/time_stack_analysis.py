import numpy

from radiometric_normalization import \
    time_stack, pif, transformation, gimage, normalize

import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

ref_paths = [
'radiometric_normalization_reference_test.LC81350432014251LGN00_all_crop.TIF',
'radiometric_normalization_reference_test.LC81360432014258LGN00_all_crop.TIF',
'radiometric_normalization_reference_test.LC81360442013319LGN00_all_crop.TIF'
]

output_path = 'time_stack_output.tif'
reference_image = time_stack.generate(ref_paths, output_path=output_path, method='identity', image_nodata=0)

candidate_path = 'planet_common_test.radiometric_normalization_20150115_043519_080c_r.tif'
pif_weight, reference_img, candidate_img = pif.generate(candidate_path, reference_path=output_path, method='identity')
transformations = transformation.generate(pif_weight, reference_img, candidate_img, method='linear_relationship')
normalize.apply_transform(candidate_path, transformations, 'output_time_stack_analysis.tif')
