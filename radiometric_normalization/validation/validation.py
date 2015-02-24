import logging
import numpy
from osgeo import gdal_array

from collections import namedtuple

Image = namedtuple('Image', 'bands, mask')


def get_score(image_path1, image_path2):
    image1 = read_image(image_path1)
    image2 = read_image(image_path2)
    score = sum_of_rmse(image1, image2)
    return score


def read_image(image_path):
    array = gdal_array.LoadFile(image_path)

    bands = [array[i, :] for i in range(array.shape[0] - 1)]

    def make_mask(array_band):
        mask = numpy.zeros(array_band.shape, dtype=numpy.uint8)
        mask[array_band > 0] = 1
        return mask

    mask = make_mask(array[-1, :])

    return Image(bands, mask)


def sum_of_rmse(image1, image2):
    assert len(image1.bands) == len(image2.bands)

    def rmse(band1, band2, mask1, mask2):
        b1 = numpy.ma.array(band1, mask=mask1)
        b2 = numpy.ma.array(band2, mask=mask2)
        return numpy.sqrt(numpy.mean((b1 - b2) ** 2))

    rmses = [rmse(band1, band2, image1.mask, image2.mask)
             for (band1, band2) in zip(image1.bands, image2.bands)]

    logging.info("Root mean square errors: {}".format(rmses))
    return sum(rmses)
