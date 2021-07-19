#! /bin/bash
# TV deconvolution example BASH shell script

# Echo shell commands
set -v

# Generate Gaussian noise with standard deviation 15 on "einstein.bmp"
# and save the result to "blurry.bmp".
./imblur K:disk:1 noise:gaussian:5 einstein.bmp blurry.bmp

# Perform TV regularized deconvolution with the split Bregman algorithm
# on "blurry.bmp" and save the result to "deconv.bmp".
./tvdeconv K:disk:1 lambda:50 blurry.bmp deconv.bmp

# Compare the original to the restored image
./imdiff einstein.bmp deconv.bmp

