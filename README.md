# Radiometric Normalization

This library implements functionality for normalizing a candidate image to time-invariant features in a set of reference images covering a time series.  This includes generating a time-invariant reference image from a time stack, identifying features that are invariante between the reference time stack and a candidate image, calculating a linear transformation that normalizes the candidate image to the reference image, applying the linear transform, and validating the results.

The primary use case for this library is radiometrically normalizing a satellite image to a time series from a reference dataset.

## Development Environment

This library uses a Vagrant VM for the development environment and requires [Vagrant](https://www.vagrantup.com/) on the host computer.

To start the VM, in the root directory of the repo type:
```
vagrant up
```

Log into the VM by typing:
```
vagrant ssh
```

Navigate to the root directory:
```
cd /vagrant
```
This directory is shared between the VM and host.

Run the unit tests by typing:
```
nosetests ./tests
```


## Organization of the repo

The code in this repo is kept in two directories: 'radiometric_normalization' and 'tests'

'radiometric_normalization' contains the algorithm and functions of the library:
* 'time_stack.py' is the module that can calculate a time stack (average image) from a series of images (depreciated).
* 'pif.py' contains the functions that find pseudo invariant feature pixels on within an image.
* 'transformation.py' calculates the transformations to make the candidate data more consistent with the reference data.
* 'normalize.py' is the module that can apply transformations calculated in 'transformation.py' to an image.
* 'validate.py' is the module for validating radiometric normalization.

Each of the modules referenced above are intended to run on a single band at a time (represented as a numpy array). Examples of more complete functions that can handle image reading and multiple bands are within 'radiometric_normalization/utils'. These will be explained below.

'tests' contain unit tests for the functions in the library.


## Example Usage

The example below demonstrates the generation of per-band linear transformations that will normalize a candidate image to the mean values of a set of reference images. The candidate and reference images must be 16-bit. Additionally, all of the reference images in the set must have the same number and order bands and pixel dimensions as the candidate image.

`candidate_path` is a string specifying the location of the candidate image on disk. `reference_paths` is a list of strings, each specifying the location of a reference image on disk. `transformations` is a list of tuples, each specifying the gain (first entry) and offset (second entry) that will normalize the respective band of the candidate image.

Below is an example using two Landsat8 tiles: LC08_L1TP_044034_20170427_20170428_01_RT_B3 and LC08_L1TP_044034_20170105_20170218_01_T1_B3

```python

from radiometric_normalization.utils import pif_wrapper
from radiometric_normalization.utils import transformation_wrapper
from radiometric_normalization.utils import normalize_wrapper
from radiometric_normalization import gimage
from radiometric_normalization import pif

## OPTIONAL
import numpy
import subprocess
from osgeo import gdal
##

## OPTIONAL - Cut dataset to coincident sub scenes
full_candidate_path = 'LC08_L1TP_044034_20170427_20170428_01_RT_B3.TIF'
full_reference_path = 'LC08_L1TP_044034_20170105_20170218_01_T1_B3.TIF'  # Older scene
candidate_path = 'candidate.tif'
reference_path = 'reference.tif'
subprocess.check_call(['gdal_translate', '-projwin', '545000', '4136000', '601000', '4084000', full_candidate_path, candidate_path])
subprocess.check_call(['gdal_translate', '-projwin', '545000', '4136000', '601000', '4084000', full_reference_path, reference_path])
##

pif_mask = pif_wrapper.generate(candidate_path, reference_path, method='filter_alpha')

## OPTIONAL - Save out the PIF mask
candidate_ds = gdal.Open(candidate_path)
metadata = gimage.read_metadata(candidate_ds)
pif_gimg = gimage.GImage([pif_mask], numpy.ones(pif_mask.shape, dtype=numpy.bool), metadata)
gimage.save(pif_gimg, 'PIF_pixels.tif')
##

transformations = transformation_wrapper.generate(candidate_path, reference_path, pif_mask, method='linear_relationship')

## OPTIONAL - View the transformations
print transformations
##

normalised_gimg = normalize_wrapper.generate(candidate_path, transformations)
gimage.save(normalised_gimg, 'normalized.tif')
```
