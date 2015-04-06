# Radiometric Normalization #

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

'radiometric_normalization' contains the algorithm and functions of the library. 'normalize.py' is the top level module for the full workflow of running radiometric normalization and 'validate.py' is the module for validating radiometric normalization.

'tests' contain unit tests for the functions in the library.


## Example Usage

The example below demonstrates the generation of per-band linear transformations that will normalize a candidate image to the mean values of a set of reference images. The candidate and reference images must be 16-bit. Additionally, all of the reference images in the set must have the same number and order bands and pixel dimensions as the candidate image.

`candidate_path` is a string specifying the location of the candidate image on disk. `reference_paths` is a list of strings, each specifying the location of a reference image on disk. `transformations` is a list of tuples, each specifying the gain (first entry) and offset (second entry) that will normalize the respective band of the candidate image.

```python

 time_stack.generate(
    reference_paths, 'time_stack.tif',
    method='identity')

pif_weight, reference_img, candidate_img = pif.generate(
    candidate_path, reference_path='time_stack.tif',
    method='identity')

transformations = transformation.generate(
    pif_weight, reference_img, candidate_img,
    method='linear_relationship')
```
