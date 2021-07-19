#pragma once

#include <fftw3.h>

template <class T>
class fftw_alloc {
    public:
        typedef T        value_type;
        typedef T*       pointer;
        typedef const T* const_pointer;
        typedef T&       reference;
        typedef const T& const_reference;
        typedef std::size_t    size_type;
        typedef std::ptrdiff_t difference_type;

        template <class U>
        struct rebind {
            typedef fftw_alloc<U> other;
        };

        pointer address (reference value) const {
            return &value;
        }
        const_pointer address (const_reference value) const {
            return &value;
        }

        fftw_alloc() throw() {
        }
        fftw_alloc(const fftw_alloc&) throw() {
        }
        template <class U>
        fftw_alloc (const fftw_alloc<U>&) throw() {
        }
        ~fftw_alloc() throw() {
        }

        size_type max_size () const throw() {
            return std::numeric_limits<std::size_t>::max() / sizeof(T);
        }

        pointer allocate (size_type num, const void* = 0) {
            void* ptr;
#ifdef _OPENMP
#pragma omp critical (fftw)
#endif
            ptr = fftwf_malloc(num*sizeof(T));
            return (pointer) ptr;
        }

        void construct (pointer p, const T& value) {
            new((void*)p)T(value);
        }

        void destroy (pointer p) {
            p->~T();
        }

        void deallocate (pointer p, size_type num) {
            (void) num;
#ifdef _OPENMP
#pragma omp critical (fftw)
#endif
            fftwf_free(p);
        }
};

template <class T1, class T2>
bool operator== (const fftw_alloc<T1>&,
                 const fftw_alloc<T2>&) throw() {
    return true;
}

template <class T1, class T2>
bool operator!= (const fftw_alloc<T1>&,
                 const fftw_alloc<T2>&) throw() {
    return false;
}
