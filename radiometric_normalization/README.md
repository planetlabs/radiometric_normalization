# Module overview #

The full radiometric normalization workflow is contained within normalize.py. It calls on three analysis modules. 

1. Time stack analysis (time_stack.py)
2. Pseudo-invariant feature generation (pif.py)
3. Radiometric transformation calculation (transformation.py)

There is also an additional validation module (validation.py). This will be used in regression testing of the library.

## Time stack analysis - time_stack.py ##

This is where the images of the same geographic location over time are prepared for further analysis. 

### Input ###
* A list of paths to images
* Each image should should cover exactly the same geographic location
* Each image should contain exactly the same number of pixels (rows and cols)
* Each image should have the same no data value

### Function ###

This module analyses the set of images and creates a single output image that reflects the set of input images (e.g. the mean of each pixel over time) as well as a weighting alpha mask that gives an indication of how strong each pixel would be if used as a pseudo-invariant feature.

### Algorithm ###
* Mean with uniform weight: Currently only this method is implemented. It calculates the mean of each pixel over time and has a uniform weight (0 for pixels where it is a no data pixel for every image in the time stack) and 65535 for valid pixels. 

### Output ####
* A single image with an alpha mask
* The image contains data derived from the set of input images
* The datatype is always uint16

## Pseudo-invariant feature generation - pif.py ##

The time stack image is analysed for pixels that are stable over time.

### Input ###
* A time stack image and a satellite image of the same geographic location as the time stack image.
* Both the time stack image and the satellite image should contain exactly the same number of pixels (rows and cols)

### Algorithm ###
* Filtering out pixels with no data values: This method simply filters out all pixels that have no data (as indicated by a 0 in the alpha mask at that pixel location)

### Output ###
* A list of data on pixel pairs:
* The data on each pixel pair consists of:
	* coordinates (int, int): Indicates the row and col entry this pixel pair refers to
	* weighting (float): Indicates how strong this pixel location is as an pseudo invariant feature
	* reference (list of numbers): A list of the intensity value of the time stack image at each band at this pixel location
	* candidate (list of numbers): A list of the intensity value of the satellite image at each band at this pixel location

## Radiometric transformation - transformation.py ##

Use the pseudo-invariant features to derive a transformation that will change the intensity distribution of the candidate image to one that is similar to the reference time stack.

### Input ###
* A list of data on each pixel pair

### Algorithm ###
* Linear relationship: This method simply uses the mean and standard deviations of the data sets to the gain and offset to transform the candidate distribution to the reference time stack one.

### Output ###
* A per-band look up table (as a numpy array)

## Validation - validation.py ##
This module is intended for functions that can score how well the algorithm is working. This will be useful for regression testing. 

### Input ###
* Two GImages

### Algorithm ###
* RMSE: Calculate the root mean squared difference between the two images, this will be taken as the score.

### Output ###
* A score of how similar the two images are
