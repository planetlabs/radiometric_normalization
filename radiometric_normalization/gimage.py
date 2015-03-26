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
from collections import namedtuple

import numpy
from osgeo import gdal, gdal_array

# In-memory geospatial image
# Bands: list of uint16 numpy arrays, each holding a band of data
# Alpha: uint16 numpy array holding the alpha information
# Metadata: dict containing georeferencing information
# (geotransform, projection, rpc)
GImage = namedtuple('GImage', 'bands, alpha, metadata')


def save(gimage, filename, nodata=None):
    gdal_ds = create_ds(gimage, filename)
    save_to_ds(gimage, gdal_ds, nodata)


def create_ds(gimage, filename):
    # Alpha is saved as the last band
    band_count = len(gimage.bands) + 1
    options = ['ALPHA=YES']

    if band_count == 4:
        options.append('PHOTOMETRIC=RGB')

    datatype = gdal.GDT_UInt16
    ysize, xsize = gimage.bands[0].shape
    gdal_ds = gdal.GetDriverByName('GTIFF').Create(
        filename, xsize, ysize, band_count, datatype,
        options=options)
    return gdal_ds


def save_to_ds(gimage, gdal_ds, nodata=None):
    assert gdal_ds.RasterCount == len(gimage.bands) + 1
    assert gdal_ds.RasterXSize == gimage.bands[0].shape[1]
    assert gdal_ds.RasterYSize == gimage.bands[0].shape[0]

    for i, band in enumerate(gimage.bands):
        gdal_array.BandWriteArray(
            gdal_ds.GetRasterBand(i + 1), band)
        if nodata is not None:
            gdal_ds.GetRasterBand(i + 1).SetNoDataValue(nodata)

    alpha_band = gdal_ds.GetRasterBand(gdal_ds.RasterCount)
    gdal_array.BandWriteArray(alpha_band, gimage.alpha)

    # Save georeferencing information
    if 'projection' in gimage.metadata.keys():
        gdal_ds.SetProjection(gimage.metadata['projection'])
    if 'geotransform' in gimage.metadata.keys():
        gdal_ds.SetGeoTransform(gimage.metadata['geotransform'])
    if 'rpc' in gimage.metadata.keys():
        gdal_ds.SetMetadata(gimage.metadata['rpc'], 'RPC')


def load(filename, nodata=None):
    logging.debug("Loading {} as GImage.".format(filename))
    gdal_ds = gdal.Open(filename)
    if gdal_ds is None:
        raise Exception('Unable to open file "{}" with gdal.Open()'.format(
            filename))

    alpha, band_count = _read_alpha_and_band_count(gdal_ds)
    bands = _read_bands(gdal_ds, band_count)
    metadata = _read_metadata(gdal_ds)

    if nodata is not None:
        alpha = alpha * _nodata_to_mask(bands, nodata)
    return GImage(bands, alpha, metadata)


def _read_metadata(gdal_ds):
    metadata = {}

    default_geotransform = (-1.0, 1.0, 0.0, 1.0, 0.0, -1.0)
    geotransform = gdal_ds.GetGeoTransform()
    if geotransform == default_geotransform:
        logging.info("Raster has default geotransform, not storing.")
    else:
        metadata['geotransform'] = geotransform

    projection = gdal_ds.GetProjection()
    if projection == '':
        logging.info("Raster has no projection information, not storing.")
    else:
        metadata['projection'] = gdal_ds.GetProjection()

    rpc = gdal_ds.GetMetadata('RPC')
    if rpc == {}:
        logging.info("Raster has no rpc information, not storing.")
    else:
        metadata['rpc'] = rpc
    return metadata


def _read_bands(gdal_ds, band_count):
    bands = []
    for band_n in range(1, band_count + 1):
        band = gdal_ds.GetRasterBand(band_n)
        array = band.ReadAsArray()
        if array is None:
            raise Exception(
                'GDAL error occured : {}'.format(gdal.GetLastErrorMsg()))
        bands.append(array.astype(numpy.uint16))
    return bands


def _read_alpha_and_band_count(gdal_ds):
    logging.debug("Loading alpha. Initial band count: {}".format(
        gdal_ds.RasterCount))
    last_band = gdal_ds.GetRasterBand(gdal_ds.RasterCount)
    if last_band.GetColorInterpretation() == gdal.GCI_AlphaBand:
        alpha = last_band.ReadAsArray()
        logging.debug("Alpha band found, reducing band count")
        band_count = gdal_ds.RasterCount - 1
    else:
        alpha = 65535 * numpy.ones(
            (gdal_ds.RasterYSize, gdal_ds.RasterXSize),
            dtype=numpy.uint16)
        band_count = gdal_ds.RasterCount
    return alpha, band_count


def _nodata_to_mask(bands, nodata):
    alpha = numpy.ones(bands[0].shape, dtype=numpy.uint16)
    for band in bands:
        alpha[band == nodata] = 0
    return alpha


def check_comparable(gimages, check_metadata=False):
    '''Checks that the gimages have the same number of bands, band dimensions,
    and, optionally, geospatial metadata'''

    no_bands = len(gimages[0].bands)
    band_shape = gimages[0].bands[0].shape
    metadata = gimages[0].metadata

    logging.debug('Initial image - band number, band shape: {}, {}'.format(
        no_bands, band_shape))
    logging.debug('Initial image metadata: '.format(metadata))

    for i, image in enumerate(gimages[1:]):
        if len(image.bands) != no_bands:
            raise Exception(
                'Image {} has a different number of bands: '
                '{} (initial: {})'.format(i + 1, len(image.bands), no_bands))

        if image.bands[0].shape != band_shape:
            raise Exception(
                'Image {} has a different band shape: {} (initial: {})'.format(
                    i + 1, image.bands[0].shape, band_shape))

        if check_metadata and image.metadata != metadata:
            raise Exception(
                'Image {} has different geographic metadata: {} '
                '(initial: {})'.format(i + 1, image.metadata, metadata))
