import logging
import numpy
from collections import namedtuple
import cv2 as cv

def generate_pca_pifs(candidate_band, reference_band, combined_mask):

    valid_pixels = numpy.nonzero(combined_mask)

    sobelx_candidate = cv.Sobel(candidate_band,cv.CV_64F,1,0,ksize=3)
    sobely_candidate = cv.Sobel(candidate_band,cv.CV_64F,0,1,ksize=3)
    
    phase_candidate = cv.phase(sobelx_candidate, sobely_candidate)
    
    phase_candidate_blurred = cv.blur(phase_candidate, (3,3))
    
    sobelx_reference = cv.Sobel(reference_band,cv.CV_64F,1,0,ksize=3)
    sobely_reference = cv.Sobel(reference_band,cv.CV_64F,0,1,ksize=3)
    
    phase_reference = cv.phase(sobelx_reference, sobely_reference)
    
    phase_reference_blurred = cv.blur(phase_reference, (3,3))
    
    phase_difference = numpy.abs(phase_candidate_blurred-phase_reference_blurred)
    
    pif_mask = (phase_difference < 2.5*numpy.mean(phase_difference))
    
    pif_mask[numpy.logical_not(combined_mask)] = False

    return pif_mask