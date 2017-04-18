import numpy

from radiometric_normalization import \
    time_stack, pif, transformation, gimage, normalize

import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

candidate_path = 'planet_common_test.radiometric_normalization_20150115_043519_080c_r.tif'
reference_path = 'landsat8_rgb_mosaic.tif'
result_path = 'output_landsat8_rgb_mosaic_analysis.tif'
pif_weight, reference_gimg, candidate_gimg = pif.generate(candidate_path, reference_path=reference_path, method='identity')
transformations = transformation.generate(pif_weight, reference_gimg, candidate_gimg, method='linear_relationship')
normalize.apply_transformations(candidate_path, transformations, result_path)
result_gimg = gimage.load(result_path)

# Comparing the standard deviations
result_std = [numpy.std(result_gimg.bands[0][numpy.nonzero(result_gimg.alpha != 0)]),
              numpy.std(result_gimg.bands[1][numpy.nonzero(result_gimg.alpha != 0)]),
              numpy.std(result_gimg.bands[2][numpy.nonzero(result_gimg.alpha != 0)])]
reference_std = [numpy.std(reference_gimg.bands[0][numpy.nonzero(pif_weight != 0)]),
                 numpy.std(reference_gimg.bands[1][numpy.nonzero(pif_weight != 0)]),
                 numpy.std(reference_gimg.bands[2][numpy.nonzero(pif_weight != 0)])]
candidate_std = [numpy.std(candidate_gimg.bands[0][numpy.nonzero(pif_weight != 0)]),
                 numpy.std(candidate_gimg.bands[1][numpy.nonzero(pif_weight != 0)]),
                 numpy.std(candidate_gimg.bands[2][numpy.nonzero(pif_weight != 0)])]

# Comparing the means
result_mean = [numpy.mean(result_gimg.bands[0][numpy.nonzero(result_gimg.alpha != 0)]),
               numpy.mean(result_gimg.bands[1][numpy.nonzero(result_gimg.alpha != 0)]),
               numpy.mean(result_gimg.bands[2][numpy.nonzero(result_gimg.alpha != 0)])]
reference_mean = [numpy.mean(reference_gimg.bands[0][numpy.nonzero(pif_weight != 0)]),
                  numpy.mean(reference_gimg.bands[1][numpy.nonzero(pif_weight != 0)]),
                  numpy.mean(reference_gimg.bands[2][numpy.nonzero(pif_weight != 0)])]
candidate_mean = [numpy.mean(candidate_gimg.bands[0][numpy.nonzero(pif_weight != 0)]),
                  numpy.mean(candidate_gimg.bands[1][numpy.nonzero(pif_weight != 0)]),
                  numpy.mean(candidate_gimg.bands[2][numpy.nonzero(pif_weight != 0)])]

print '*** SUMMARY ***'

print 'STANDARD DEVIATIONS'
for i in range(3):
    print 'BAND ' + str(i) + ' STD: Candidate (' + str(candidate_std[i]) + ') * Gain (' + str(transformations[i].gain) + ') = ' + str(transformations[i].gain*candidate_std[i])
    print '            Reference = ' + str(reference_std[i])
    print '            Result = ' + str(result_std[i])

print 'MEANS'
for i in range(3):
    print 'BAND ' + str(i) + ' MEAN: Candidate (' + str(candidate_mean[i]) + ') * Gain (' + str(transformations[i].gain) + ')  + Offset (' + str(transformations[i].offset) + ') = ' + str(transformations[i].gain*candidate_mean[i] + transformations[i].offset)
    print '             Reference = ' + str(reference_mean[i])
    print '             Result = ' + str(result_mean[i])
