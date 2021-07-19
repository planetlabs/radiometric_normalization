#include "image.hpp"
#include "utils.hpp"
#include "edgetaper.hpp"

extern "C" {
#include "tvreg.h"
}

/// pad an image using constant boundaries
template <typename T>
static void padimage_replicate(img_t<T>& out, const img_t<T>& in, int padding)
{
    out.resize(in.w + padding*2, in.h + padding*2, in.d);

    for (int y = 0; y < in.h; y++) {
        for (int x = 0; x < in.w; x++) {
            for (int l = 0; l < in.d; l++) {
                out(x+padding, y+padding, l) = in(x, y, l);
            }
        }
    }

    // pad top and bottom
    for (int x = 0; x < out.w; x++) {
        int xx = std::min(std::max(0, x - padding), in.w-1);
        for (int l = 0; l < in.d; l++) {
            T val_top = in(xx, 0, l);
            T val_bottom = in(xx, in.h-1, l);
            for (int y = 0; y < padding; y++) {
                out(x, y, l) = val_top;
                out(x, out.h-1 - y, l) = val_bottom;
            }
        }
    }

    // pad left and right
    for (int y = 0; y < out.h; y++) {
        int yy = std::min(std::max(0, y - padding), in.h-1);
        for (int l = 0; l < in.d; l++) {
            T val_left = in(0, yy, l);
            T val_right = in(in.w-1, yy, l);
            for (int x = 0; x < padding; x++) {
                out(x, y, l) = val_left;
                out(out.w-1 - x, y, l) = val_right;
            }
        }
    }
}

/// deconvolve an image using Split bregman
/// deconvolve only the luminance
/// boundaries have to be handled elsewhere
template <typename T>
void deconvBregman(img_t<T>& u, const img_t<T>& f, const img_t<T>& K,
                  int numIter, T lambda, T beta)
{
    // reorder to planar
    img_t<T> f_planar(f.w, f.h, f.d);
    img_t<T> deconv_planar(f.w, f.h, f.d);
    if (f.d != 1) {
        for (int y = 0; y < f.h; y++) {
            for (int x = 0; x < f.w; x++) {
                for (int l = 0; l < f.d; l++) {
                    f_planar[x + f.w*(y + f.h*l)] = f(x, y, l);
                    deconv_planar[x + f.w*(y + f.h*l)] = f(x, y, l);
                }
            }
        }
    } else {
        f_planar.copy(f);
        deconv_planar.copy(f);
    }

    // deconvolve
    tvregopt* tv = TvRegNewOpt();
    TvRegSetKernel(tv, &K[0], K.w, K.h);
    TvRegSetLambda(tv, lambda);
    TvRegSetMaxIter(tv, numIter);
    TvRegSetGamma1(tv, beta);
    TvRegSetTol(tv, .000001);

    TvRegSetPlotFun(tv, 0, 0);
    TvRestore(&deconv_planar[0], &f_planar[0], f_planar.w, f_planar.h, f_planar.d, tv);

    TvRegFreeOpt(tv);

    // reorder to interleaved
    u.resize(deconv_planar.w, deconv_planar.h, deconv_planar.d);
    if (u.d != 1) {
        for (int y = 0; y < u.h; y++) {
            for (int x = 0; x < u.w; x++) {
                for (int l = 0; l < u.d; l++) {
                    u(x, y, l) = deconv_planar[x + u.w*(y + u.h*l)];
                }
            }
        }
    } else {
        u.copy(deconv_planar);
    }
}

#include "args.hxx"
#include <iostream>

struct options {
    std::string input;
    std::string input_kernel;
    std::string output;
    float alpha;
    float beta;
    int iterations;
};

static options parse_args(int argc, char** argv)
{
    args::ArgumentParser parser("");
    args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});
    args::Positional<std::string> input(parser, "input", "input blurry image file", args::Options::Required);
    args::Positional<std::string> input_kernel(parser, "input_kernel", "input kernel file", args::Options::Required);
    args::Positional<std::string> output(parser, "output", "deconvolution output file", args::Options::Required);
    args::ValueFlag<float> alpha(parser, "alpha", "total variation regularization weight", {"alpha"}, 3000.f);
    args::ValueFlag<float> beta(parser, "beta", "split bregman weight", {"beta"}, 30.f);
    args::ValueFlag<int> iterations(parser, "iterations", "number of iterations", {"iterations"}, 7);

    try {
        parser.ParseCLI(argc, argv);
    } catch (const args::Help&) {
        std::cout << parser;
        exit(0);
    } catch (const args::ParseError& e) {
        std::cerr << e.what() << std::endl;
        std::cerr << parser;
        exit(1);
    } catch (const args::ValidationError& e) {
        std::cerr << e.what() << std::endl;
        std::cerr << parser;
        exit(1);
    }

    options opts;
    opts.input = args::get(input);
    opts.input_kernel = args::get(input_kernel);
    opts.output = args::get(output);
    opts.alpha = args::get(alpha);
    opts.beta = args::get(beta);
    opts.iterations = args::get(iterations);
    return opts;
}

int main(int argc, char** argv)
{
    struct options opts = parse_args(argc, argv);

    // read the input image and kernel
    img_t<float> img = img_t<float>::load(opts.input);
    img_t<float> kernel = img_t<float>::load(opts.input_kernel);

    // normalize the image between 0 and 1
    float max = 0.;
    for (int i = 0; i < img.size; i++)
        max = std::max(max, img[i]);
    for (int i = 0; i < img.size; i++)
        img[i] /= max;

    // add padding and apply edge taper
    img_t<float> tapered;
    edgetaper(tapered, utils::add_padding(img, kernel), kernel, 3);

    // deconvolve the image
    img_t<float> deconv;
    deconvBregman(deconv, tapered, kernel, opts.iterations, opts.alpha, opts.beta);

    // remove the padding
    img_t<float> result = utils::remove_padding(deconv, kernel);

    // clamp the result and restore the original range
    for (int i = 0; i < result.size; i++)
        result[i] = std::max(std::min(float(1.), result[i]), 0.f);
    for (int i = 0; i < result.size; i++)
        result[i] *= max;

    // save the deblurred image
    result.save(opts.output);
}

