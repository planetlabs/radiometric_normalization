import logging
import numpy

from radiometric_normalization import gimage


def get_score(image_path1, image_path2, method='rmse'):
    '''
    support method configuration
    '''
    image1 = gimage.load(image_path1)
    image2 = gimage.load(image_path2)

    if method == 'rmse':
        score = sum_of_rmse(image1, image2)
    else:
        raise Exception("Unrecognized method specified: {}".format(method))
    return score


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
