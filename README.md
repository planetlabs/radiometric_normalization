# Radiometric Normalisation #

This library contains the functions to analyse a set of reference images for features that have a stable level of intensity over the whole reference set and use these features to normalise the intensity distribution of separate candidate image. 

It was originally created to radiometrically normalise a satellite image to a time series from a reference (atmospherically corrected) dataset. 

## How to use ##

This git repo contains a vagrant environment within which to run the library in. To start the vagrant environment, in the root directory of the repo type:

    vagrant up

Then log into the vagrant environment by typing:

    vagrant ssh

Once in, you can navigate to the root directory:

    cd /vagrant

And you can run the unit tests by typing:

    nosetests ./tests

If successful it should end with:

    vagrant@precise64:/vagrant$ nosetests ./tests 
    .............
    ----------------------------------------------------------------------
    Ran 13 tests in 0.112s

    OK

## Organisation of the repo ##

The code in this repo are kept in two directories: 'radiometric_normalization' and 'tests'

'radiometric_normalization' contains the algorithm and functions of the library. 'normalize.py' is the top level module for the full workflow of the radiometric normalisation implementation.

'tests' contain unit tests for the functions in the library.
