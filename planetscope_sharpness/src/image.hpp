#pragma once

#include <unordered_map>

#include <cassert>
#include <complex>
#include <limits>
#include <functional>
#include <vector>
#include <numeric>

#ifndef IMG_NO_FFTW
#include <fftw3.h>
#include "fftw_allocator.hpp"
#endif

#ifndef IMG_NO_OMP
#include <omp.h>
#endif

template <typename T>
class img_t {
public:
    typedef T value_type;
    int size, w, h, d;
#ifndef IMG_NO_FFTW
    std::vector<T, fftw_alloc<T>> data;
#else
    std::vector<T> data;
#endif

    img_t() : size(0), w(0), h(0), d(0) {
    }

    img_t(int w, int h, int d=1)
        : size(w*h*d), w(w), h(h), d(d), data(w*d*h) {
    }

    img_t(int w, int h, int d, T* data)
        : size(w*h*d), w(w), h(h), d(d) {
        this->data.assign(data, data+w*h*d);
    }

    img_t(const img_t<T>& o)
        : size(o.size), w(o.w), h(o.h), d(o.d), data(o.data) {
    }

    ~img_t() {
    }

    // indexing
    inline T& operator[](int i) {
        return data[i];
    }
    inline const T& operator[](int i) const {
        return data[i];
    }
    inline T& operator()(int x, int y, int dd=0) {
        return data[dd+d*(x+y*w)];
    }
    inline const T& operator()(int x, int y, int dd=0) const {
        return data[dd+d*(x+y*w)];
    }

    void set_value(const T& v) {
        std::fill(data.begin(), data.end(), v);
    }

    T sum() const {
        return fold<T>(std::plus<T>());
    }

    T max() const {
        return fold<T>([](const T& a, const T& b) { return a > b ? a : b; });
    }

    T min() const {
        return fold<T>([](const T& a, const T& b) { return a < b ? a : b; });
    }

    template <typename T2>
    T2 fold(const std::function<T2(const T&, const T2&)>& f) const {
        return std::accumulate(data.begin(), data.end(), T2(), f);
    }

    template <typename E>
    bool similar(const E& o) const {
        // should check if the cast makes sense in case T != E::value_type
        bool similar = w == o.w && h == o.h && d == o.d;
        if (!similar)
            fprintf(stderr, "%dx%dx%d (type %s) != %dx%dx%d (type %s)\n",
                    w, h, d, typeid(*this).name(), o.w, o.h, o.d, typeid(o).name());
        return similar;
    }

    void resize(int w, int h, int d=1) {
        if (this->w != w || this->h != h || this->d != d) {
            this->w = w;
            this->h = h;
            this->d = d;
            size = w * h * d;
            data.resize(w * h * d);
        }
    }

    template <typename T2>
    void resize(const img_t<T2>& o) {
        resize(o.w, o.h, o.d);
    }

    // map (no arg, xyd arg)
    //template <class E>
    //void map(const E& o) {
        //assert(o.similar(*this));
        //int n = w * h * d;
        //for (int i = 0; i < n; i++)
            //data[i] = o[i];
    //}

    template <typename T2>
    void map(const img_t<T2>& o) {
        assert(o.similar(*this));
        int n = w * h * d;
        for (int i = 0; i < n; i++)
            data[i] = o[i];
    }

    void mapf(const std::function<T(T)>& f) {
        for (int i = 0; i < size; i++) {
            (*this)[i] = f((*this)[i]);
        }
    }

    template <typename T2, class F>
    void map(const img_t<T2>& o, const F& f) {
        assert(o.similar(*this));
        for (int i = 0; i < size; i++) {
            (*this)[i] = f(o[i]);
        }
    }

    void copy(const img_t<T>& o) {
        assert(o.similar(*this));
        std::copy(o.data.begin(), o.data.end(), data.begin());
    }

    template <typename T2>
    void copy(const img_t<T2>& o) {
        assert(o.similar(*this));
        std::copy(o.data.begin(), o.data.end(), data.begin());
    }

    bool inside(int x, int y, int dd=0) const {
        return x >= 0 && x < w && y >= 0 && y < h && dd >= 0 && dd < d;
    }

    // forward differences
    template <typename T2>
    void gradients(const img_t<T2>& u_) {
        const img_t<typename T::value_type>& u = *static_cast<const img_t<typename T::value_type>*>(&u_);
        for (int l = 0; l < d; l++) {
            for (int y = 0; y < h; y++)
            for (int x = 0; x < w-1; x++)
                (*this)(x, y, l)[0] = u(x+1, y, l) - u(x, y, l);
            for (int y = 0; y < h; y++)
                (*this)(w-1, y, l)[0] = 0;

            for (int y = 0; y < h-1; y++)
            for (int x = 0; x < w; x++)
                (*this)(x, y, l)[1] = u(x, y+1, l) - u(x, y, l);
            for (int x = 0; x < w; x++)
                (*this)(x, h-1, l)[1] = 0;
        }
    }

    void gradientx(const img_t<T>& u) {
        for (int l = 0; l < d; l++) {
            for (int y = 0; y < h; y++)
            for (int x = 0; x < w-1; x++)
                (*this)(x, y, l) = u(x+1, y, l) - u(x, y, l);
            for (int y = 0; y < h; y++)
                (*this)(w-1, y, l) = 0;
        }
    }
    void gradienty(const img_t<T>& u) {
        for (int l = 0; l < d; l++) {
            for (int y = 0; y < h-1; y++)
            for (int x = 0; x < w; x++)
                (*this)(x, y, l) = u(x, y+1, l) - u(x, y, l);
            for (int x = 0; x < w; x++)
                (*this)(x, h-1, l) = 0;
        }
    }

    void fftshift() {
        img_t<T> out;
        out.resize(*this);

        int halfw = (this->w + 1) / 2.;
        int halfh = (this->h + 1) / 2.;
        int ohalfw = this->w - halfw;
        int ohalfh = this->h - halfh;
        for (int l = 0; l < this->d; l++) {
            for (int y = 0; y < halfh; y++) {
                for (int x = 0; x < ohalfw; x++) {
                    out(x, y + ohalfh, l) = (*this)(x + halfw, y, l);
                }
            }
            for (int y = 0; y < halfh; y++) {
                for (int x = 0; x < halfw; x++) {
                    out(x + ohalfw, y + ohalfh, l) = (*this)(x, y, l);
                }
            }
            for (int y = 0; y < ohalfh; y++) {
                for (int x = 0; x < ohalfw; x++) {
                    out(x, y, l) = (*this)(x + halfw, y + halfh, l);
                }
            }
            for (int y = 0; y < ohalfh; y++) {
                for (int x = 0; x < halfw; x++) {
                    out(x + ohalfw, y, l) = (*this)(x, y + halfh, l);
                }
            }
        }

        *this = out;
    }

    void ifftshift() {
        img_t<T> out;
        out.resize(*this);

        int halfw = (this->w + 1) / 2.;
        int halfh = (this->h + 1) / 2.;
        int ohalfw = this->w - halfw;
        int ohalfh = this->h - halfh;
        for (int l = 0; l < this->d; l++) {
            for (int y = 0; y < ohalfh; y++) {
                for (int x = 0; x < halfw; x++) {
                    out(x, y + halfh, l) = (*this)(x + ohalfw, y, l);
                }
            }
            for (int y = 0; y < ohalfh; y++) {
                for (int x = 0; x < ohalfw; x++) {
                    out(x + halfw, y + halfh, l) = (*this)(x, y, l);
                }
            }
            for (int y = 0; y < halfh; y++) {
                for (int x = 0; x < halfw; x++) {
                    out(x, y, l) = (*this)(x + ohalfw, y + ohalfh, l);
                }
            }
            for (int y = 0; y < halfh; y++) {
                for (int x = 0; x < ohalfw; x++) {
                    out(x + halfw, y, l) = (*this)(x, y + ohalfh, l);
                }
            }
        }

        *this = out;
    }

    template <typename T2>
    void padcirc(const img_t<T2>& o) {
        set_value(0);
        int ww = o.w / 2;
        int hh = o.h / 2;
        for (int dd = 0; dd < d; dd++) {
            int od;
            if (d == o.d)
                od = dd;
            else if (o.d == 1)
                od = 0;
            else
                assert(false);
            for (int y = 0; y < hh; y++) {
                for (int x = 0; x < ww; x++) {
                    (*this)(w  - ww + x, h  - hh + y, dd) = o(x, y, od);
                }
                for (int x = ww; x < o.w; x++) {
                    (*this)(- ww + x, h  - hh + y, dd) = o(x, y, od);
                }
            }
            for (int y = hh; y < o.h; y++) {
                for (int x = 0; x < ww; x++) {
                    (*this)(w  - ww + x, - hh + y, dd) = o(x, y, od);
                }
                for (int x = ww; x < o.w; x++) {
                    (*this)(- ww + x, - hh + y, dd) = o(x, y, od);
                }
            }
        }
    }

#ifndef IMG_NO_IIO
    static img_t<T> load(const std::string& filename);
    void save(const std::string& filename) const;
#endif
};

// from http://stackoverflow.com/a/26221725
#include <memory>
template<typename ... Args>
std::string string_format(const std::string& format, Args ... args )
{
    size_t size = std::snprintf(nullptr, 0, format.c_str(), args...) + 1; // Extra space for '\0'
    std::unique_ptr<char[]> buf(new char[size]);
    std::snprintf(buf.get(), size, format.c_str(), args...);
    return std::string(buf.get(), buf.get() + size - 1); // We don't want the '\0' inside
}

