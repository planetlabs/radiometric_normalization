#pragma once

#include <memory>

#ifndef IMG_NO_FFTW
#include <fftw3.h>
#endif

#include "image.hpp"

#include <map>
#include <unordered_map>

struct dim_t {
    int h, w;
    int d;

    bool operator==(const dim_t &other) const {
        return h == other.h && w == other.w && d == other.d;
    }
};

namespace std {
    template <>
    struct hash<dim_t>
    {
        std::size_t operator()(const dim_t& k) const
        {
            return ((std::hash<int>()(k.h)
                  ^ (std::hash<int>()(k.w) << 1)) >> 1)
                  ^ (std::hash<int>()(k.d) << 1);
        }
    };
}

template <typename T>
struct plan_t {
};

template <>
struct plan_t<float> {
    typedef fftwf_plan plan_type;
    typedef fftwf_complex value_type;
    plan_type plan_forward = nullptr;
    plan_type plan_backward = nullptr;

    plan_t<float>(plan_t<float>&& p) {
        std::swap(plan_forward, p.plan_forward);
        std::swap(plan_backward, p.plan_backward);
    }

    plan_t<float>(dim_t dim, int flags) {
        img_t<std::complex<float>> img(dim.w, dim.h, dim.d);
        auto out = reinterpret_cast<value_type*>(&img[0]);
#ifdef _OPENMP
#pragma omp critical (fftw)
#endif
        {
            plan_forward = fftwf_plan_many_dft(2, &dim.h, dim.d, out, &dim.h, dim.d, 1, out,
                                               &dim.h, dim.d, 1, FFTW_FORWARD, flags);
            plan_backward = fftwf_plan_many_dft(2, &dim.h, dim.d, out, &dim.h, dim.d, 1, out,
                                                &dim.h, dim.d, 1, FFTW_BACKWARD, flags);
        }
        assert(plan_forward);
        assert(plan_backward);
    }

    ~plan_t<float>() {
        return; // don't free the plan...
#ifdef _OPENMP
#pragma omp critical (fftw)
#endif
        {
            if (plan_forward)
                fftwf_destroy_plan(plan_forward);
            if (plan_backward)
                fftwf_destroy_plan(plan_backward);
        }
    }

    void execute_forward(std::complex<float>* _out) const {
        auto out = reinterpret_cast<value_type*>(_out);
        fftwf_execute_dft(plan_forward, out, out);
    }

    void execute_backward(std::complex<float>* _out) const {
        auto out = reinterpret_cast<value_type*>(_out);
        fftwf_execute_dft(plan_backward, out, out);
    }

};

template <>
struct plan_t<double> {
    typedef fftw_plan plan_type;
    typedef fftw_complex value_type;
    plan_type plan_forward = nullptr;
    plan_type plan_backward = nullptr;

    plan_t<double>(plan_t<double>&& p) {
        std::swap(plan_forward, p.plan_forward);
        std::swap(plan_backward, p.plan_backward);
    }

    plan_t<double>(dim_t dim, int flags) {
        img_t<std::complex<double>> img(dim.w, dim.h, dim.d);
        auto out = reinterpret_cast<value_type*>(&img[0]);
#ifdef _OPENMP
#pragma omp critical (fftw)
#endif
        {
            plan_forward = fftw_plan_many_dft(2, &dim.h, dim.d, out, &dim.h, dim.d, 1, out,
                                              &dim.h, dim.d, 1, FFTW_FORWARD, flags);
            plan_backward = fftw_plan_many_dft(2, &dim.h, dim.d, out, &dim.h, dim.d, 1, out,
                                               &dim.h, dim.d, 1, FFTW_BACKWARD, flags);
        }
    }

    ~plan_t<double>() {
        return; // don't free the plan...
#ifdef _OPENMP
#pragma omp critical (fftw)
#endif
        {
            if (plan_forward)
                fftw_destroy_plan(plan_forward);
            if (plan_backward)
                fftw_destroy_plan(plan_backward);
        }
    }

    void execute_forward(std::complex<double>* _out) const {
        auto out = reinterpret_cast<value_type*>(_out);
        fftw_execute_dft(plan_forward, out, out);
    }

    void execute_backward(std::complex<double>* _out) const {
        auto out = reinterpret_cast<value_type*>(_out);
        fftw_execute_dft(plan_backward, out, out);
    }

};

template <typename T>
inline plan_t<T>* make_plan(dim_t dim, int flags)
{
    static std::unordered_map<dim_t, plan_t<T>> cache;
    if (cache.find(dim) == cache.end()) {
        cache.emplace(dim, plan_t<T>(dim, flags));
    }
    auto it = cache.find(dim);
    return &it->second;
}

namespace fft {

    template <typename T>
    img_t<T> c2c(const img_t<T>& o, bool fast=false) {
        using V = typename T::value_type;
        dim_t dim = {.h=o.h, .w=o.w, .d=o.d};
        const auto plan = make_plan<V>(dim, fast ? FFTW_ESTIMATE : FFTW_MEASURE);
        img_t<T> tmp = o;
        plan->execute_forward(&tmp[0]);
        return tmp;
    }

    template <typename T>
    img_t<std::complex<T>> r2c(const img_t<T>& o, bool fast=false) {
        img_t<std::complex<T>> tmp(o.w, o.h, o.d);
        for (int i = 0; i < tmp.size; i++) {
            tmp[i] = o[i];
        }
        return c2c(tmp, fast);
    }

    template <typename T>
    img_t<T> shift(const img_t<T>& in) {
        img_t<T> out;
        out.resize(in);

        int halfw = (in.w + 1) / 2.;
        int halfh = (in.h + 1) / 2.;
        int ohalfw = in.w - halfw;
        int ohalfh = in.h - halfh;
        for (int l = 0; l < in.d; l++) {
            for (int y = 0; y < halfh; y++) {
                for (int x = 0; x < ohalfw; x++) {
                    out(x, y + ohalfh, l) = in(x + halfw, y, l);
                }
            }
            for (int y = 0; y < halfh; y++) {
                for (int x = 0; x < halfw; x++) {
                    out(x + ohalfw, y + ohalfh, l) = in(x, y, l);
                }
            }
            for (int y = 0; y < ohalfh; y++) {
                for (int x = 0; x < ohalfw; x++) {
                    out(x, y, l) = in(x + halfw, y + halfh, l);
                }
            }
            for (int y = 0; y < ohalfh; y++) {
                for (int x = 0; x < halfw; x++) {
                    out(x + ohalfw, y, l) = in(x, y + halfh, l);
                }
            }
        }

        return out;
    }

    const int* get_optimal_table(int& _lut_size) {
        // based on the matlab code of Sunghyun Cho
        const int lut_size = 4096;
        static int is_optimal[lut_size] = {0};
        if (!is_optimal[1]) {
            for (int e2 = 1; e2 < lut_size; e2 *= 2)
            for (int e3 = e2; e3 < lut_size; e3 *= 3)
            for (int e5 = e3; e5 < lut_size; e5 *= 5)
            for (int e7 = e5; e7 < lut_size; e7 *= 7)
                is_optimal[e7] = true;
        }
        _lut_size = lut_size;
        return is_optimal;
    }

    int get_optimal_size_up(int size) {
        int lut_size;
        const int* is_optimal = get_optimal_table(lut_size);
        for (int i = size; i < lut_size; i++) {
            if (is_optimal[i]) {
                return i;
            }
        }
        return size;
    }

    int get_optimal_size_down(int size) {
        int lut_size;
        const int* is_optimal = get_optimal_table(lut_size);
        if (size >= lut_size)
            size = lut_size - 1;
        for (int i = size; i > 0; i--) {
            if (is_optimal[i]) {
                return i;
            }
        }
        return size;
    }

    template <typename T>
    void psf2otf(img_t<std::complex<T>>& out, const img_t<T>& k, int w, int h, int d=1)
    {
        out.resize(w, h, d);
        out.padcirc(k);
        out = c2c(out);
    }

}

namespace ifft {

    template <typename T>
    img_t<T> c2c(const img_t<T>& o, bool fast=false) {
        using V = typename T::value_type;
        dim_t dim = {.h=o.h, .w=o.w, .d=o.d};
        const auto plan = make_plan<V>(dim, fast ? FFTW_ESTIMATE : FFTW_MEASURE);
        img_t<T> tmp = o;
        plan->execute_backward(&tmp[0]);

        T norm = tmp.w * tmp.h;
        for (int i = 0; i < tmp.size; i++) {
            tmp[i] /= norm;
        }
        return tmp;
    }

    template <typename T>
    img_t<T> c2r(const img_t<std::complex<T>>& in, bool fast=false) {
        img_t<std::complex<T>> o = c2c(in, fast);
        img_t<T> tmp(o.w, o.h, o.d);
        for (int i = 0; i < tmp.size; i++) {
            tmp[i] = std::real(o[i]);
        }
        return tmp;
    }

    template <typename T>
    img_t<T> shift(const img_t<T>& in) {
        img_t<T> out;
        out.resize(in);

        int halfw = (in.w + 1) / 2.;
        int halfh = (in.h + 1) / 2.;
        int ohalfw = in.w - halfw;
        int ohalfh = in.h - halfh;
        for (int l = 0; l < in.d; l++) {
            for (int y = 0; y < ohalfh; y++) {
                for (int x = 0; x < halfw; x++) {
                    out(x, y + halfh, l) = in(x + ohalfw, y, l);
                }
            }
            for (int y = 0; y < ohalfh; y++) {
                for (int x = 0; x < ohalfw; x++) {
                    out(x + halfw, y + halfh, l) = in(x, y, l);
                }
            }
            for (int y = 0; y < halfh; y++) {
                for (int x = 0; x < halfw; x++) {
                    out(x, y, l) = in(x + ohalfw, y + ohalfh, l);
                }
            }
            for (int y = 0; y < halfh; y++) {
                for (int x = 0; x < ohalfw; x++) {
                    out(x + halfw, y, l) = in(x, y + ohalfh, l);
                }
            }
        }

        return out;
    }

}

