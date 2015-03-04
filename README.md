# radiometric_normalization #
Implementation of radiometric normalization workflows.

## Module overview ##

The full radiometric normalisation workflow is contained within radiometric_normalization.py. It calls on three analysis modules. 

1. Time stack analysis (time_stack.py)
2. Pseudo-invariant feature generation (pif.py)
3. Radiometric transformation calculation (transformation.py)

There is also an additional validation module (validation.py).

### Time stack analysis ###

This is where the images of the same geographic location over time are prepared for further analysis. 

#### Input ####
* A list of paths to geotiffs
* Each image should should cover exactly the same geographic location
* Each image should contain exactly the same number of pixels (rows and cols)
* Each image should have the same no data value

#### Function ####

This module analyses the set of images and creates a single output image that reflects the set of input images (e.g. the mean of each pixel over time) as well as a weighting alpha mask that gives an indication of how strong each pixel would be if used as a pseudo-invariant feature.

#### Algorithm ####
* Mean with uniform weight: Currently only this method is implemented. It calculates the mean of each pixel over time and has a uniform weight (0 for pixels where it is a no data pixel for every image in the time stack) and 65535 for valid pixels. 

#### Output #####
* A single geotiff with an alpha mask
* The datatype is always uint16

