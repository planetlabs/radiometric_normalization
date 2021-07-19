#pragma once

#include "image.hpp"
#include "fft.hpp"

/// smooth the borders of an image so that the result is more periodic
/// see matlab:'help edgetaper'
template <typename T>
void edgetaper(img_t<T>& out, const img_t<T>& in, const img_t<T>& kernel, int iterations=1)
{
    img_t<T> weights(in.w, in.h);
    // kind of tukey window
    for (int y = 0; y < in.h; y++) {
        T wy = 1.;
        if (y < kernel.h) {
            wy = std::pow(std::sin(y * M_PI / (kernel.h*2 - 1)), 2.);
        } else if (y > in.h - kernel.h) {
            wy = std::pow(std::sin((in.h-1 - y) * M_PI / (kernel.h*2 - 1)), 2.);
        }
        for (int x = 0; x < in.w; x++) {
            T wx = 1.;
            if (x < kernel.w) {
                wx = std::pow(std::sin(x * M_PI / (kernel.w*2 - 1)), 2.);
            } else if (x > in.w - kernel.w) {
                wx = std::pow(std::sin((in.w-1 - x) * M_PI / (kernel.w*2 - 1)), 2.);
            }
            weights(x, y) = wx * wy;
        }
    }

    img_t<std::complex<T>> kernel_ft;
    fft::psf2otf(kernel_ft, kernel, in.w, in.h, in.d);

    img_t<T> blurred(in.w, in.h, in.d);
    img_t<std::complex<T>> blurred_ft(in.w, in.h, in.d);

    out = in;
    for (int i = 0; i < iterations; i++) {
        // blur the image
        blurred_ft.map(fft::r2c(out));
        for (int y = 0; y < out.h; y++)
            for (int x = 0; x < out.w; x++)
                for (int l = 0; l < out.d; l++)
                    blurred_ft(x, y, l) *= kernel_ft(x, y);
        blurred.map(ifft::c2r(blurred_ft));

        // blend
        for (int y = 0; y < out.h; y++) {
            for (int x = 0; x < out.w; x++) {
                T w = weights(x, y);
                for (int l = 0; l < out.d; l++) {
                    out(x, y, l) = w * out(x, y, l) + (1. - w) * blurred(x, y, l);
                }
            }
        }
    }
}


