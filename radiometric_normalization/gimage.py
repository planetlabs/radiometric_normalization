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


def save(gimage, filename):

    band_count = len(gimage.bands) + 1
    options = ['ALPHA=YES']

    if band_count == 4:
        options.append('PHOTOMETRIC=RGB')

    datatype = gdal_array.NumericTypeCodeToGDALTypeCode(
        gimage.bands[0].dtype.type)
    ysize, xsize = gimage.bands[0].shape
    gdal_ds = gdal.GetDriverByName('GTIFF').Create(
        filename, xsize, ysize, band_count, datatype,
        options=options)

    for i in range(len(gimage.bands)):
        gdal_array.BandWriteArray(
            gdal_ds.GetRasterBand(i + 1),
            gimage.bands[i])

    alpha_band = gdal_ds.GetRasterBand(gdal_ds.RasterCount)
    gdal_array.BandWriteArray(alpha_band, gimage.alpha)

    # Save georeferencing information
    if 'projection' in gimage.metadata.keys():
        gdal_ds.SetProjection(gimage.metadata['projection'])
    if 'geotransform' in gimage.metadata.keys():
        gdal_ds.SetGeoTransform(gimage.metadata['geotransform'])
    if 'rpc' in gimage.metadata.keys():
        gdal_ds.SetMetadata(gimage.metadata['rpc'], 'RPC')


def load(filename):
    gdal_ds = gdal.Open(filename)
    if gdal_ds is None:
        raise Exception('Unable to open file "{}" with gdal.Open()'.format(
            filename))

    alpha, band_count = _read_alpha_and_band_count(gdal_ds)
    bands = _read_bands(gdal_ds, band_count)
    metadata = _read_metadata(gdal_ds)
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
    last_band = gdal_ds.GetRasterBand(gdal_ds.RasterCount)
    if last_band.GetColorInterpretation() == gdal.GCI_AlphaBand:
        alpha = last_band.ReadAsArray()
        band_count = gdal_ds.RasterCount - 1
    else:
        alpha = 65535 * numpy.ones(
            (gdal_ds.RasterXSize, gdal_ds.RasterYSize),
            dtype=numpy.uint16)
        band_count = gdal_ds.RasterCount
    return alpha, band_count
