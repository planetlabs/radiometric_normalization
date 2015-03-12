# Radiometric Normalization #

This library implements functionality for normalizing a candidate image to time-invariant features in a set of reference images covering a time series. This includes generating a time-invariant reference image from a time stack, identifying features that are invariante between the reference time stack and a candidate image, calculating a linear transformation that normalizes the candidate image to the reference image, applying the linear transform, and validating the results.

The primary use case for this library is radiometrically normalizing a satellite image to a time series from a reference dataset.

## Development Environment

This library uses a Vagrant VM for the development environment and requires [Vagrant](https://www.vagrantup.com/) on the host computer.

To start the VM, in the root directory of the repo type:
```
vagrant up
``

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
