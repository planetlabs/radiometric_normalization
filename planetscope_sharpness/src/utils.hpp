#pragma once

#include "image.hpp"
#include "labeling.hpp"
#include "vec2.hpp"

namespace utils {
    extern "C" {
        // these are defined in imscript/upsa.c and imscript/downscale.c
        void zoom2(float *y, const float *x, int W, int H, int pd, int w, int h, float n, int zt);
        void downscale_image(float *y, float *x, int outw, int outh, int inw, int inh, float scale);
    }

    // upsample an image
    inline void upsample(img_t<float>& out, const img_t<float>& _in, float factor,
                         int targetw, int targeth, int interp=2/* bilinear */) {
        img_t<float> in = _in; // copy input
        out.resize(targetw, targeth, in.d);
        zoom2(&out[0], &in[0], out.w, out.h, out.d, in.w, in.h, factor, interp);
    }

    // downsample an image with Gaussian filtering
    void gaussian_downsample(img_t<float>& out, const img_t<float>& _in, float factor/* >= 1 */) {
        img_t<float> in = _in; // copy input
        if (factor == 1) {
            out = in;
            return;
        }

        out.resize(std::ceil(in.w/factor), std::ceil(in.h/factor), in.d);

        img_t<float> tmpout(out.w, out.h, 1);
        img_t<float> tmpin(in.w, in.h, 1);
        // process channel by channel since downscale_image accepts only grayscale images
        for (int d = 0; d < in.d; d++) {
            for (int i = 0; i < in.w * in.h; i++) {
                tmpin[i] = in[i*in.d+d];
            }
            downscale_image(&tmpout[0], &tmpin[0], tmpout.w, tmpout.h, tmpin.w, tmpin.h, factor);
            for (int i = 0; i < out.w * out.h; i++) {
                out[i*in.d+d] = tmpout[i];
            }
        }
    }

    // add symmetric padding of the size of the kernel
    template <typename T>
    img_t<T> add_padding(const img_t<T>& _f, const img_t<T>& K)
    {
        img_t<T> f(_f.w + K.w-1, _f.h + K.h-1, _f.d);
        f.set_value(T(0));
        for (int y = 0; y < _f.h; y++) {
            for (int x = 0; x < _f.w; x++) {
                for (int d = 0; d < _f.d; d++) {
                    f(x+K.w/2, y+K.h/2, d) = _f(x, y, d);
                }
            }
        }

        // replicate borders
        for (int y = 0; y < K.h/2; y++) {
            for (int x = 0; x < f.w; x++) {
                for (int l = 0; l < f.d; l++) {
                    f(x, y, l) = f(x, 2*(K.h/2) - y, l);
                    f(x, f.h-1-y, l) = f(x, f.h-1-2*(K.h/2)+y, l);
                }
            }
        }
        for (int y = 0; y < f.h; y++) {
            for (int x = 0; x < K.w/2; x++) {
                for (int l = 0; l < f.d; l++) {
                    f(x, y, l) = f(2*(K.w/2) - x, y, l);
                    f(f.w-1-x, y, l) = f(f.w-1-2*(K.w/2)+x, y, l);
                }
            }
        }
        return f;
    }

    // remove the padding added by add_padding
    template <typename T>
    img_t<T> remove_padding(const img_t<T>& f, const img_t<T>& K)
    {
        int w2 = K.w/2;
        int h2 = K.w/2;
        img_t<T> out(f.w - 2*w2, f.h - 2*h2, f.d);

        for (int y = 0; y < out.h; y++) {
            for (int x = 0; x < out.w; x++) {
                for (int l = 0; l < out.d; l++) {
                    out(x, y, l) = f(x+w2, y+h2, l);
                }
            }
        }
        return out;
    }

    template <typename T>
    void center_kernel(img_t<T>& kernel) {
        T dx = 0.f;
        T dy = 0.f;
        T sum = kernel.sum();
        for (int y = 0; y < kernel.h; y++) {
            for (int x = 0; x < kernel.w; x++) {
                dx += kernel(x, y) * x;
                dy += kernel(x, y) * y;
            }
        }
        dx = std::round(dx / sum);
        dy = std::round(dy / sum);

        img_t<T> copy(kernel);
        kernel.set_value(0);
        for (int y = 0; y < kernel.h; y++) {
            for (int x = 0; x < kernel.w; x++) {
                int nx = x + (int)dx - kernel.w/2;
                int ny = y + (int)dy - kernel.h/2;
                if (nx >= 0 && nx < kernel.w && ny >= 0 && ny < kernel.h) {
                    kernel(x, y) = copy(nx, ny);
                }
            }
        }
    }

    // compute connected component of the support of k
    // and set to zero pixels belonging to low intensity connected components
    template <typename T>
    void remove_isolated_cc(img_t<T>& k) {
        T sum = k.sum();
        for (int i = 0; i < k.size; i++)
            k[i] /= sum;
        img_t<int> lab;
        labeling::labels(lab, k);
        auto sums = labeling::sum(lab, k);
        for (int i = 0; i < k.size; i++) {
            if (sums[lab[i]] < T(0.1))
                k[i] = T(0);
        }
    }

    // compute the circular gradients by forward difference
    template <typename T>
    void circular_gradients(vec2<img_t<T>>& out, const img_t<T>& in) {
        out[0].resize(in);
        out[1].resize(in);

        int w = in.w;
        int h = in.h;
        int d = in.d;
        for (int l = 0; l < d; l++) {
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                    out[0](x, y, l) = in((x+1)%w, y, l) - in(x, y, l);

            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                    out[1](x, y, l) = in(x, (y+1)%h, l) - in(x, y, l);
        }
    }

    // compute the circular divergence by backward difference
    template <typename T>
    void circular_divergence(img_t<T>& out, const vec2<img_t<T>>& in) {
        out.resize(in[0]);

        int w = out.w;
        int h = out.h;
        int d = out.d;
        for (int l = 0; l < d; l++) {
            for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
                out(x, y, l) = in[0](x, y, l) - in[0]((x-1+w)%w, y, l)
                             + in[1](x, y, l) - in[1](x, (y-1+h)%h, l);
        }
    }

}

