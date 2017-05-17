# Module overview

The full radiometric normalization workflow is contained within normalize.py. It calls on three analysis modules. 

1. Time stack analysis (time_stack.py) - DEPRECIATED
2. Pseudo-invariant feature generation (pif.py)
3. Radiometric transformation calculation (transformation.py)
4. Application of the radiometric transformation (normalize.py)

There is also an additional validation module (validation.py). This will be used in regression testing of the library.

## Time stack analysis - time_stack.py - DEPRECIATED

This is where a single reference image is create from a set of images of the same geographic location over time. 

### Input
* A list of paths to images
* Each image should cover exactly the same geographic location
* Each image should contain exactly the same number of pixels (rows and cols)
* Each image should have the same no data value

### Function

This module analyzes the set of images and creates a single output image that reflects the set of input images (e.g. the mean of each pixel over time) as well as a weighting alpha mask that gives an indication of how strong each pixel would be if used as a pseudo-invariant feature.

### Algorithm
* Mean with uniform weight: Currently only this method is implemented. It calculates the mean of each pixel over time and has a uniform weight (0 for pixels where it is a no data pixel for every image in the time stack) and 65535 for valid pixels. 

### Output
* A single image with an alpha mask
* The image contains data derived from the set of input images
* The datatype is always uint16

## Filtering - filtering.py

This module has some rough filters so that the fits calculated below are not influenced by outliers too much.

### 
* An array representing the image values for a single band in the candidate image
* An array representing the image values for a single band in the reference image
* A boolean mask representing the valid pixels in both images (0 indicates that the pixel is not valid)
* Both the reference image and the satellite image should contain exactly the same number of pixels (rows and cols) and be physically colocated

### Algorithm
* Filtering using a line: Uses the distance of a point from a line in candidate DN to reference DN space.
* Filtering using a histogram: Uses a 2D histogram of the population of candidate DN and reference DN values.

### Output
* An array to indicate if a pixel is a valid or not (0 indicates that the pixel is not valid)

## Pseudo-invariant feature generation - pif.py

The reference and candidate images are analyzed to find pixels that are invariant over the set. The motivation for this step is removing the effect of change (e.g. clouds or snow) from the normalization transformation calculation. `pif_wrapper.py` has example usage.

### Input
* An array representing the image values for a single band in the candidate image
* An array representing the image values for a single band in the reference image
* A boolean mask representing the valid pixels in both images (0 indicates that the pixel is not valid)
* Both the reference image and the satellite image should contain exactly the same number of pixels (rows and cols) and be physically colocated

### Algorithm
* Filtering out pixels with no data values: This method simply filters out all pixels that have no data (as indicated by a 0 in the alpha mask at that pixel location)
* Filtering using PCA fits: This method uses a PCA fit to filter out pixels that do not correspond closely to a linear relationship between the two bands.
* Filtering using robust fits: This method uses a robust linear fit on the data and a threshold around this fit to find PIF pixels.

### Output
* An array to indicate if a pixel is a pseudo invariant feature or not (0 indicates that the pixel is not a valid PIF)

## Calculating the radiometric transformation - transformation.py

Use the pseudo-invariant features to derive a transformation that will change the intensity distribution of the candidate image to one that is similar to the reference image. `transformation_wrapper.py` has example usage.

### Input
* An array representing the image values for a single band in the candidate image
* An array representing the image values for a single band in the reference image
* An array representing the pseudo invariant features of an image (0 indicates that the pixel is not a valid PIF)

### Algorithm
* Linear relationship: This method simply uses the mean and standard deviations of the data sets to the gain and offset to transform the candidate distribution to the reference distribution.
* Ordinary least squares regression: This method uses OLS regression to try and find the transformation.
* robust fit: This method uses various robust fitting methods to try and find the transformation.

### Output
* A linear transformation (gain and offset) to transform the candidate image to have similar intensities to the reference image

## Applying the radiometric transformation - normalize.py

Applies a linear transformation to an image. `normalize_wrapper.py` has example usage.

### Input
* An array representing the image values for a single band
* A linear transformation (gain and offset)

### Algorithm
* A look up table is calculated using the transformation and applied to the image bands.

### Output
* The input band transformed using the linear transformation.

## Validation - validation.py

Score how well two images match. This will be useful for regression testing. 

### Input
* Two GImages (in-memory representations of geographic rasters, see gimage.py)

### Algorithm
* RMSE: Calculate the root mean squared difference between the two images, this will be taken as the score.

### Output
* A score (float) of how similar the two images are
