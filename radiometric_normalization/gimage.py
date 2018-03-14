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

from collections import namedtuple
from osgeo import gdal, gdal_array

'''
A wrapper for a geospatial image

- Bands: A list of uint16 numpy arrays, each holding a band of data
- Alpha: A boolean numpy array holding the alpha information
         - False is a no data pixel
         - True is a valid pixel
- Metadata: A dict containing georeferencing information
            - geotransform, projection and rpc
'''
GImage = namedtuple('GImage', 'bands, alpha, metadata')


def save(gimage, filename, nodata=None, compress=True):
    band_count = len(gimage.bands) + 1
    ysize, xsize = gimage.bands[0].shape
    gdal_ds = create_ds(filename, xsize, ysize, band_count, compress)
    _save_to_ds(gimage, gdal_ds, nodata)


def create_ds(file_name, xsize, ysize, band_count, compress=True):
    options = ['PHOTOMETRIC=RGB']
    if compress:
        options.append('COMPRESS=DEFLATE')
        options.append('PREDICTOR=2')

    datatype = gdal.GDT_UInt16
    gdal_ds = gdal.GetDriverByName('GTIFF').Create(
        file_name, xsize, ysize, band_count, datatype,
        options=options)
    return gdal_ds


def _save_to_ds(gimage, gdal_ds, nodata=None):
    assert gdal_ds.RasterCount == len(gimage.bands) + 1
    assert gdal_ds.RasterXSize == gimage.bands[0].shape[1]
    assert gdal_ds.RasterYSize == gimage.bands[0].shape[0]

    # Image bands
    for i, band in enumerate(gimage.bands):
        save_band(gdal_ds, band, i + 1, nodata)
    save_alpha_band(gdal_ds, gimage.alpha)
    save_metadata(gdal_ds, gimage.metadata)


def save_band(gdal_ds, band_array, band_no, nodata=None):
    gdal_band = gdal_ds.GetRasterBand(band_no)
    gdal_array.BandWriteArray(gdal_band, band_array)
    if nodata is not None:
        gdal_band.SetNoDataValue(nodata)


def save_alpha_band(gdal_ds, alpha_array):
    alpha_band = gdal_ds.GetRasterBand(gdal_ds.RasterCount)
    alpha_band.SetColorInterpretation(gdal.GCI_AlphaBand)
    gdal_array.BandWriteArray(alpha_band,
                              alpha_array.astype(numpy.uint16) * 255)


def save_metadata(gdal_ds, metadata):
    # Save georeferencing information
    if 'projection' in metadata.keys():
        gdal_ds.SetProjection(metadata['projection'])
    if 'geotransform' in metadata.keys():
        gdal_ds.SetGeoTransform(metadata['geotransform'])
    if 'rpc' in metadata.keys():
        gdal_ds.SetMetadata(metadata['rpc'], 'RPC')


def load(filename, nodata=None, last_band_alpha=False):
    logging.info('GImage: Loading {} as GImage'.format(filename))
    gdal_ds = gdal.Open(filename)
    if gdal_ds is None:
        raise Exception('Unable to open file "{}" with gdal.Open()'.format(
            filename))

    alpha, band_count = read_alpha_and_band_count(gdal_ds, last_band_alpha)
    bands = _read_all_bands(gdal_ds, band_count)
    metadata = read_metadata(gdal_ds)

    if nodata:
        alpha = alpha * _nodata_to_mask(bands, nodata)
    return GImage(bands, alpha, metadata)


def read_metadata(gdal_ds):
    metadata = {}

    default_geotransform = (-1.0, 1.0, 0.0, 1.0, 0.0, -1.0)
    geotransform = gdal_ds.GetGeoTransform()
    if geotransform == default_geotransform:
        logging.debug('GImage: Raster has default geotransform, not storing')
    else:
        metadata['geotransform'] = geotransform

    projection = gdal_ds.GetProjection()
    if projection == '':
        logging.debug(
            'GImage: Raster has no projection information, not storing')
    else:
        metadata['projection'] = gdal_ds.GetProjection()

    rpc = gdal_ds.GetMetadata('RPC')
    if rpc == {}:
        logging.debug('GImage: Raster has no rpc information, not storing')
    else:
        metadata['rpc'] = rpc
    return metadata


def _read_all_bands(gdal_ds, band_count):
    bands = []
    for band_n in range(1, band_count + 1):
        bands.append(read_single_band(gdal_ds, band_n))
    return bands


def read_single_band(gdal_ds, band_no):
    ''' band_no is gdal style band numbering, i.e. from 1 onwards not 0 indexed
    '''
    band = gdal_ds.GetRasterBand(band_no)
    array = band.ReadAsArray()
    if array is None:
        raise Exception(
            'GDAL error occured : {}'.format(gdal.GetLastErrorMsg()))
    return array.astype(numpy.uint16)


def read_alpha_and_band_count(gdal_ds, last_band_alpha=False):
    logging.info('GImage: Initial band count: {}'.format(
        gdal_ds.RasterCount))
    last_band = gdal_ds.GetRasterBand(gdal_ds.RasterCount)
    if last_band.GetColorInterpretation() == gdal.GCI_AlphaBand:
        logging.info('GImage: Alpha band found, reducing band count')
        alpha = last_band.ReadAsArray().astype(numpy.bool)
        band_count = gdal_ds.RasterCount - 1
    elif last_band_alpha:
        logging.info(
            'GImage: Forcing last band to be an alpha band, reducing band '
            'count')
        alpha = last_band.ReadAsArray().astype(numpy.bool)
        band_count = gdal_ds.RasterCount - 1
    else:
        logging.info('GImage: No alpha band found')
        alpha = numpy.ones(
            (gdal_ds.RasterYSize, gdal_ds.RasterXSize),
            dtype=numpy.bool)
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

    logging.debug('GImage: Initial image - band number, band shape: '
                  '{}, {}'.format(no_bands, band_shape))
    logging.debug('GImage: Initial image metadata: '.format(metadata))

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


def check_equal(gimages, check_metadata=False):
    '''Checks that a list of gimages are equivalent'''

    check_comparable(gimages, check_metadata)

    first_gimg = gimages[0]
    for i, image in enumerate(gimages[1:]):
        numpy.testing.assert_equal(first_gimg.bands, image.bands,
                                   err_msg='Image {} has different band data'
                                   ' to the first image'.format(i))

        numpy.testing.assert_equal(first_gimg.alpha, image.alpha,
                                   err_msg='Image {} has different alpha data'
                                   ' to the first image'.format(i))
