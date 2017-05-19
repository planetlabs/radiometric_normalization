'''
Copyright 2015 Planet Labs, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''
import logging
import numpy

import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')

from matplotlib import pyplot as plt


def plot_pixels(file_name, candidate_data_single_band,
                reference_data_single_band, limits=None, fit_line=None):

    logging.info('Display: Creating pixel plot - {}'.format(file_name))
    fig = plt.figure()
    plt.hexbin(
        candidate_data_single_band, reference_data_single_band, mincnt=1)
    if not limits:
        min_value = 0
        _, ymax = plt.gca().get_ylim()
        _, xmax = plt.gca().get_xlim()
        max_value = max([ymax, xmax])
        limits = [min_value, max_value]
    plt.plot(limits, limits, 'k-')
    if fit_line:
        start = limits[0] * fit_line.gain + fit_line.offset
        end = limits[1] * fit_line.gain + fit_line.offset
        plt.plot(limits, [start, end], 'g-')
    plt.xlim(limits)
    plt.ylim(limits)
    plt.xlabel('Candidate DNs')
    plt.ylabel('Reference DNs')
    fig.savefig(file_name, bbox_inches='tight')
    plt.close(fig)


def plot_histograms(file_name, candidate_data_multiple_bands,
                    reference_data_multiple_bands=None,
                    # Default is for Blue-Green-Red-NIR:
                    colour_order=['b', 'g', 'r', 'y'],
                    x_limits=None, y_limits=None):
    logging.info('Display: Creating histogram plot - {}'.format(file_name))
    fig = plt.figure()
    plt.hold(True)
    for colour, c_band in zip(colour_order, candidate_data_multiple_bands):
        c_bh, c_bins = numpy.histogram(c_band, bins=256)
        plt.plot(c_bins[:-1], c_bh, color=colour, linestyle='-', linewidth=2)
    if reference_data_multiple_bands:
        for colour, r_band in zip(colour_order, reference_data_multiple_bands):
            r_bh, r_bins = numpy.histogram(r_band, bins=256)
            plt.plot(
                r_bins[:-1], r_bh, color=colour, linestyle='--', linewidth=2)
    plt.xlabel('DN')
    plt.ylabel('Number of pixels')
    if x_limits:
        plt.xlim(x_limits)
    if y_limits:
        plt.ylim(y_limits)
    fig.savefig(file_name, bbox_inches='tight')
    plt.close(fig)
