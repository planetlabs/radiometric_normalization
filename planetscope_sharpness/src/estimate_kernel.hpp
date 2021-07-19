#pragma once

#include <cassert>
#include "image.hpp"
#include "fft.hpp"
#include "utils.hpp"
#include "edgetaper.hpp"

struct options {
    bool verbose;
    std::string debug;
    std::string outputsharp;

    int ks;
    std::string input;
    std::string output;

    float lambda;
    float lambda_ratio;
    float lambda_min;
    float gamma;

    int iterations;
    bool multiscale;
    float scalefactor;

    float kernel_threshold_max;
    bool remove_isolated;
};

template <typename T>
class L0ImagePredictor {
    img_t<std::complex<T>> fv;
    img_t<T> DtD;
    img_t<T> v;

public:
    L0ImagePredictor(const img_t<T>& v) : fv(fft::r2c(v)), DtD(v.w, v.h), v(v) {
        // compute the fourier transform of \partial_x
        // with the size of the blurry image
        img_t<T> dx(3, 3);
        dx(0, 1) = 0; dx(1, 1) = -1; dx(2, 1) = 1;
        img_t<std::complex<T>> dx_otf;
        fft::psf2otf(dx_otf, dx, v.w, v.h);

        // same for \partial_y
        img_t<T> dy(3, 3);
        dy(1, 0) = 0; dy(1, 1) = -1; dy(1, 2) = 1;
        img_t<std::complex<T>> dy_otf;
        fft::psf2otf(dy_otf, dy, v.w, v.h);

        // compute |F(\partial_x)|^2 + |F(\partial_y)|^2
        for (int i = 0; i < DtD.size; i++) {
            // std::norm(c) returns the squared magnitude of c
            DtD[i] = std::norm(dx_otf[i]) + std::norm(dy_otf[i]);
        }
    }

    void operator()(img_t<T>& u, const img_t<T>& K,
                    T lambda, T beta_init, T beta_rate, T beta_max) {
        assert(K.w % 2);
        assert(K.h % 2);

        // vector field g
        vec2<img_t<T>> g;
        g[0].resize(v.w, v.h);
        g[1].resize(v.w, v.h);
        img_t<T> divergence;

        // compute F(k) at the size of the blurry image
        img_t<std::complex<T>> K_otf;
        fft::psf2otf(K_otf, K, v.w, v.h);

        // compute conj(F(k)).F(v)
        img_t<std::complex<T>> Ktf(v.w, v.h);
        for (int i = 0; i < Ktf.size; i++)
            Ktf[i] = std::conj(K_otf[i]) * fv[i];

        // compute |F(k)|^2
        img_t<T> KtK(v.w, v.h);
        for (int i = 0; i < KtK.size; i++)
            // std::norm(c) returns the squared magnitude of c
            KtK[i] = std::norm(K_otf[i]);

        img_t<std::complex<T>> div(v.w, v.h);

        T beta = beta_init;
        u = v;
        while (beta < beta_max) {
            // compute the gradient of u
            utils::circular_gradients(g, u);

            // hard-thresholding (solve the 'g' update)
            for (int i = 0; i < v.w * v.h; i++) {
                T n = std::pow(g[0][i], T(2)) + std::pow(g[1][i], T(2));
                if (n < lambda / beta) {
                    g[0][i] = 0;
                    g[1][i] = 0;
                }
            }

            // compute the divergence of the field
            utils::circular_divergence(divergence, g);
            img_t<std::complex<T>> adj = fft::r2c(divergence);

            // solve the 'u' update
            for (int i = 0; i < div.size; i++) {
                std::complex<T> num = Ktf[i] - beta * adj[i];
                T denom = KtK[i] + beta * DtD[i];
                div[i] = num / denom;
            }

            u = ifft::c2r(div);

            // increase beta
            beta *= beta_rate;
        }
    }
};

template <typename T>
class FourierKernelEstimator {
    int ks;
    vec2<img_t<std::complex<T>>> fgv;

public:
    FourierKernelEstimator(const img_t<T>& v, int ks) : ks(ks) {
        // precomputes F(\partial_x v) and F(\partial_y v)
        vec2<img_t<T>> gv;
        compute_gradients(gv, v);
        fgv[0] = fft::r2c(gv[0]);
        fgv[1] = fft::r2c(gv[1]);
    }

    // implements Algorithm 3
    void operator()(img_t<T>& k, const img_t<T>& u, const struct options& opts) {
        k.resize(ks, ks);

        // solves the Equation (29)
        img_t<std::complex<T>> div(u.w, u.h);
        {
            vec2<img_t<T>> gu;
            compute_gradients(gu, u);

            vec2<img_t<std::complex<T>>> fgu;
            fgu[0] = fft::r2c(gu[0]);
            fgu[1] = fft::r2c(gu[1]);

            // solve the linear system
            for (int i = 0; i < div.size; i++) {
                // std::norm(c) returns the squared magnitude of c
                std::complex<T> num = std::conj(fgu[0][i]) * fgv[0][i] + std::conj(fgu[1][i]) * fgv[1][i];
                T denum = std::norm(fgu[0][i]) + std::norm(fgu[1][i]) + opts.gamma;
                div[i] = num / denum;
            }
        }

        // compute the inverse discrete Fourier transform
        img_t<T> otf = fft::shift(ifft::c2r(div));

        // crop the center of the otf to get the kernel
        int left = otf.w / 2 - k.w / 2;
        int top = otf.h / 2 - k.h / 2;
        for (int y = 0; y < k.h; y++) {
            for (int x = 0; x < k.w; x++) {
                k(x, y) = otf(left + x, top + y);
            }
        }

        // enforce positivity of the kernel
        for (int i = 0; i < k.size; i++) {
            k[i] = std::max(T(0), k[i]);
        }

        // threshold the kernel at some percentage of the max value
        if (opts.kernel_threshold_max > 0.f) {
            T th = k.max() * opts.kernel_threshold_max;
            for (int i = 0; i < k.size; i++)
                k[i] = k[i] < th ? T(0) : k[i];
        }

        // remove isolated connected components
        if (opts.remove_isolated) {
            utils::remove_isolated_cc(k);
        }

        // center the kernel
        utils::center_kernel(k);

        // normalize
        T sum = k.sum();
        if (sum > 0) {
            for (int i = 0; i < k.size; i++) {
                k[i] /= sum;
            }
        }
    }

private:
    void compute_gradients(vec2<img_t<T>>& g, const img_t<T>& u) {
        g[0].resize(u);
        g[1].resize(u);
        g[0].gradientx(u);
        g[1].gradienty(u);
    }
};

// implements the inner loop of Algorithm 1
// estimates the sharp image and the kernel from a blurry image and an initialization of u
template <typename T>
void l0_kernel_estimation(img_t<T>& k, img_t<T>& u, const img_t<T>& v,
                          const img_t<T>& initu, struct options& opts) {
    L0ImagePredictor<T> sharp_predictor(v);
    FourierKernelEstimator<T> kernel_estimator(v, opts.ks);

    u = initu;

    // make sure lambda is not lower than lambda_min
    // in case the user changed lambda_min but not lambda
    opts.lambda = std::max(opts.lambda, opts.lambda_min);

    // alternate between estimating k and u, while decreasing lambda
    for (int i = 0; i < opts.iterations; i++) {
        if (opts.verbose) {
            printf("Iteration %d/%d: lambda=%f\n", i+1, opts.iterations, opts.lambda);
        }

        // estimate the kernel (Algorithm 3)
        kernel_estimator(k, u, opts);

        // estimate the sharp image (Algorithm 2)
        sharp_predictor(u, k, opts.lambda, 2*opts.lambda, T(2), T(1e5));

        // update the lambda (decay with a lower bound)
        opts.lambda = std::max(opts.lambda * opts.lambda_ratio, opts.lambda_min);

        if (!opts.debug.empty()) {
            static int it = 0;
            it++;
            u.save(string_format("%s/u_%03d.tiff", opts.debug.c_str(), it));
            v.save(string_format("%s/v_%03d.tiff", opts.debug.c_str(), it));
            k.save(string_format("%s/k_%03d.tiff", opts.debug.c_str(), it));
        }
    }
}

// implements Algorithm 1
// it assumes that the image was previously processed by preprocess_image
// the inner loop is implemented in l0_kernel_estimation
template <typename T>
void multiscale_l0_kernel_estimation(img_t<T>& k, img_t<T>& u, const img_t<T>& v, struct options& opts) {
    std::vector<img_t<T>> vs;
    std::vector<int> kernelSizes;

    // compute the subsampled versions of v and kernel sizes
    int ks = opts.ks;
    img_t<T> vv = v;
    do {
        // store the downsampled image and kernel size
        vs.push_back(vv);
        kernelSizes.push_back(ks);

        // downsample blurry image
        utils::gaussian_downsample(vv, vv, 1/opts.scalefactor);

        ks = ks * opts.scalefactor;
        // make the kernel odd-sized
        ks += (ks+1)%2;
    } while (vv.w > 1 && vv.h > 1 && ks >= 3);

    int Nscales = vs.size();
    for (int s = Nscales - 1; s >= 0; s--) {
        int ks = kernelSizes[s];
        const img_t<T>& v = vs[s];
        if (s == Nscales - 1) {
            u = vs[s];
        }

        if (opts.verbose) {
            printf("Estimation at scale %dx%d, kernel size=%d\n", v.w, v.h, ks);
        }

        opts.ks = ks;
        l0_kernel_estimation(k, u, v, u, opts);

        if (s > 0) {
            img_t<T>& nextv = vs[s-1];
            utils::upsample(u, u, 1/opts.scalefactor, nextv.w, nextv.h, 3);
        }
    }
}

// preprocess the input blurry image as describe in Section 2.1
template <typename T>
void preprocess_image(img_t<T>& out, const img_t<T>& _v, struct options& opts) {
    img_t<T> v(_v.w, _v.h);

    // convert to grayscale
    v.set_value(0);
    for (int i = 0; i < v.w * v.h; i++) {
        for (int d = 0; d < _v.d; d++) {
            v[i] += _v[i * _v.d + d];
        }
        v[i] /= _v.d;
    }

    // normalize the input between 0 and 1
    float min = v.min();
    for (int i = 0; i < v.size; i++)
        v[i] -= min;
    float max = v.max();
    for (int i = 0; i < v.size; i++)
        v[i] /= max;

    // crop the blurry image so that ffts are faster (at least for the finest scale)
    img_t<T> copy = v;
    int w = v.w;
    int h = v.h;
    int nw = fft::get_optimal_size_down(v.w);
    int nh = fft::get_optimal_size_down(v.h);
    int offx = (w - nw) / 2;
    int offy = (h - nh) / 2;
    v.resize(nw, nh);
    for (int y = 0; y < nh; y++) {
        for (int x = 0; x < nw; x++) {
            v(x, y) = copy(x + offx, y + offy);
        }
    }
    if (opts.verbose && (w != nw || h != nh)) {
        printf("Blurry image cropped from %dx%d to %dx%d.\n", w, h, nw, nh);
    }

    // apply an edgetaper to limit boundary condition errors
    img_t<T> k(opts.ks, opts.ks);
    k.set_value(1./(opts.ks*opts.ks));
    edgetaper(v, v, k);

    out = v;
}

