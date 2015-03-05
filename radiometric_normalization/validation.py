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
import logging
import numpy


def sum_of_rmse(image1, image2):
    assert len(image1.bands) == len(image2.bands)

    def rmse(band1, band2, mask1, mask2):
        b1 = numpy.ma.array(band1, mask=mask1)
        b2 = numpy.ma.array(band2, mask=mask2)
        return numpy.sqrt(numpy.mean((b1 - b2) ** 2))

    rmses = [rmse(band1, band2, image1.alpha, image2.alpha)
             for (band1, band2) in zip(image1.bands, image2.bands)]

    logging.info("Root mean square errors: {}".format(rmses))
    return sum(rmses)
