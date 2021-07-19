#pragma once

#include <functional>
#include <map>

#include "image.hpp"

namespace labeling {

    // compute the connected components of an image
    // using 8-connected neighbors
    template <typename T>
    int labels(img_t<int>& labels, const img_t<T>& img) {
        labels.resize(img);
        labels.set_value(0);

        std::vector<int> equiv;
        equiv.push_back(0);

        int nblabels = 0;
        for (int d = 0; d < img.d; d++) {
            for (int y = 0; y < img.h; y++) {
                for (int x = 0; x < img.w; x++) {
                    T val = img(x, y, d);
                    if (!val) {
                        continue;
                    }

                    int tl = y > 0 && x > 0 ? labels(x-1, y-1, d) : 0;
                    int t = y > 0 ? labels(x, y-1, d) : 0;
                    int tr = y > 0 && x < img.w-1 ? labels(x+1, y-1, d) : 0;
                    int l = x > 0 ? labels(x-1, y, d) : 0;

                    // no neighbor, add a connected component here
                    if (tl + t + tr + l == 0) {
                        nblabels++;
                        labels(x, y, d) = nblabels;
                        equiv.push_back(nblabels);
                        continue;
                    }

                    // otherwise, get the highest label and connect it
                    int max = std::max(tl, std::max(t, std::max(tr, l)));
                    labels(x, y, d) = max;
                    // and indicates that neighbors have equivalent label
                    if (tl && tl != max)
                        equiv[tl] = max;
                    if (t && t != max)
                        equiv[t] = max;
                    if (tr && tr != max)
                        equiv[tr] = max;
                    if (l && l != max)
                        equiv[l] = max;
                }
            }
        }

        // assign one label per connected component by taking the root
        std::function<int(int)> getroot = [&](int l) {
            if (l == equiv[l])
                return l;
            return getroot(equiv[l]);
        };
        for (int i = 0; i < labels.size; i++) {
            labels[i] = getroot(labels[i]);
        }

        return nblabels;
    }

    template <typename T>
    std::map<int, T> sum(const img_t<int>& labels, const img_t<T>& img) {
        std::map<int, T> sum;
        for (int i = 0; i < img.size; i++) {
            sum[labels[i]] += img[i];
        }
        return sum;
    }

};


