import json
import logging
import os
import subprocess
import sys
import rasterio

from pathlib import Path
import numpy as np
from osgeo import gdal

from radiometric_normalization.wrappers import display_wrapper
from radiometric_normalization.wrappers import pif_wrapper
from radiometric_normalization.wrappers import transformation_wrapper
from radiometric_normalization.wrappers import normalize_wrapper
from radiometric_normalization import gimage
from radiometric_normalization import pif

def compute_score(kernel_filepath):
    
    kernel = rasterio.open(kernel_filepath).read(1)
    
    kernel = kernel/(np.sum(np.abs(kernel)))
    
    score = np.linalg.norm(kernel)
    
    return score

def perform_data_process(image_path, ref_image_path=None, out_directory=None, out_path=None, deblur=False, product='TOA', source='PLANET', band_count=None):
    """
    Definition: Function to perform normalization and deblurring of image_path wrt referenced image
    image_path: Image to be processed
    ref_image_path: Reference image path for processing [optional]
    out_directory: directory path for processed image output [optional]
    out_path: complete file path for the output image [optional]
    deblur: bool flag for performing deblurring
    """
    try:
        temp_folder = os.path.join(os.path.dirname(__file__), "temp")
        kernel_folder = os.path.join(os.path.dirname(__file__), "kernel")
        region_reference_folder = os.path.dirname(__file__)
    except:
        temp_folder = os.path.join(".","temp")
        kernel_folder = os.path.join(".", "kernel")
        region_reference_folder = "."
    os.makedirs(temp_folder, exist_ok=True)
    os.makedirs(kernel_folder, exist_ok=True)
    region_reference_file_path =  os.path.join(region_reference_folder, 'region_reference.json')
    image_name = Path(image_path).stem
    extension = Path(image_path).suffix
    region = image_name.split('_')[0]
    candidate_path = image_path
    reference_path = None
    try:
        if ref_image_path is not None:
            if os.path.exists(ref_image_path):
                reference_path = ref_image_path
            else:
                logging.info("Provided ref_image_path doesn't exists")
                logging.info("Switching to default region reference file")
        if reference_path is None:
            if os.path.exists(region_reference_file_path):
                reference_path = dict(json.load(open(region_reference_file_path))).get(region,{}).get(product, {}).get(source,{}).get('reference_image')
            else:
                raise Exception('Region reference file %s not present', (region_reference_file_path))
            if reference_path is None or not os.path.exists(reference_path): 
                raise Exception('Reference image %s does not exists' % (reference_path))
                
        logging.info('Selected %s as reference image', reference_path)
    except Exception as e:
        raise Exception("Reference image not set for the region %s and product %s" % (region, product))
    assert os.path.exists(image_path), "Image %s doesn't exists" % (image_path)
    
    reference_gimg = gimage.load(reference_path)
    candidate_gimg = gimage.load(candidate_path)
    if band_count is None:
        band_count = min(rasterio.open(reference_path).count, rasterio.open(candidate_path).count)

    alpha_c = True # alpha mask
    for band_num in range(band_count):
        alpha_band = np.logical_not(reference_gimg.bands[band_num]==0)
        if any(alpha_band):
            alpha_c = np.logical_and(alpha_c, alpha_band)
    for band_num in range(band_count):
        alpha_band = np.logical_not(candidate_gimg.bands[band_num]==0)
        if any(alpha_band):
            alpha_c = np.logical_and(alpha_c, alpha_band)

    temporary_gimg = gimage.GImage([candidate_gimg.bands[band_num] for band_num in range(band_count)], alpha_c, candidate_gimg.metadata)
    
    candidate_path = os.path.join(temp_folder, 'candidate_'+image_name+extension)
    reference_path = os.path.join(temp_folder, 'reference_'+image_name+extension)

    gimage.save(temporary_gimg, candidate_path)

    temporary_gimg = gimage.GImage([reference_gimg.bands[band_num] for band_num in range(band_count)], alpha_c, reference_gimg.metadata)

    gimage.save(temporary_gimg, reference_path)

    parameters = pif.pca_options(threshold=100)
    pif_mask = pif_wrapper.generate(candidate_path, reference_path, method='filter_PCA', last_band_alpha=True, method_options=parameters)

    transformations = transformation_wrapper.generate(candidate_path, reference_path, pif_mask, method='linear_relationship', last_band_alpha=True)

    normalised_gimg = normalize_wrapper.generate(candidate_path, transformations, last_band_alpha=True)

    bands = []

    for j in range(len(normalised_gimg.bands)):
        band = normalised_gimg.bands[j]
        band[np.logical_not(alpha_c)] = 0
        bands.append(band)

    norm_gimg = gimage.GImage([bands[band_num] for band_num in range(band_count)], alpha_c, reference_gimg.metadata)
    norm_path = outpath if out_path else os.path.join(out_directory or os.path.dirname(image_path), f"{image_name}_norm{extension}")
    gimage.save(norm_gimg, norm_path)
    
    result_path = outpath if out_path else os.path.join(out_directory or os.path.dirname(image_path), f"{image_name}_norm_deblur{extension}")

    kernel_path = os.path.join(kernel_folder, 'kernel_'+image_name+'.tif')
    estimate_kernel_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'planetscope_sharpness','estimate-kernel')
    bashCommand = '{} 35 {} {}'.format(os.path.realpath(estimate_kernel_path), norm_path, kernel_path)
    logging.info(bashCommand)
    process = subprocess.run(bashCommand.split(), stdout=1, stderr=2)
    out_image_path = norm_path
    if deblur:

        deconv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'planetscope_sharpness', 'deconv')
        bashCommand = '{} {} {} {}'.format(os.path.realpath(deconv_path), norm_path, kernel_path, result_path)
        logging.info(bashCommand)
        process = subprocess.run(bashCommand.split(), stdout=1, stderr=2)

        gimg_deblurred = gimage.load(result_path)
        temporary_gimg = gimage.GImage(gimg_deblurred.bands,alpha = norm_gimg.alpha, metadata = norm_gimg.metadata)
        gimage.save(temporary_gimg, result_path)
        os.remove(norm_path)
        out_image_path = result_path
        
    score = compute_score(kernel_path)
    os.remove(kernel_path)
    os.remove(candidate_path)
    os.remove(reference_path)
    return out_image_path, score
        
