#!/bin/bash

if [[ $EUID -eq 0 ]]; then
  echo "Do *not* run this as root, sudo will be used as needed."
  exit 1
fi

# change to the directory this script is running in, so work will
# be relative to a known path
TOP=$(cd $(dirname $0) && pwd)

# abort install if any errors occur and enable tracing
set -o errexit
set -o xtrace

sudo apt-get update
sudo apt-get install -y --force-yes python-software-properties
sudo add-apt-repository ppa:ubuntugis/ubuntugis-unstable

sudo apt-get update
sudo apt-get install -y --force-yes \
    build-essential \
    pylint \
    pep8 \
    python-dev \
    python-pip \
    python-numpy \
    python-nose \
    python-nosexcover \
    python-gdal \
    python-setuptools \
    python-scipy \
    python-matplotlib \

# install radiometric_normalization
echo $TOP | sudo tee /usr/local/lib/python2.7/dist-packages/radiometric_normalization.pth

sudo pip install pyflakes
sudo pip install scikit-learn

set +o xtrace
