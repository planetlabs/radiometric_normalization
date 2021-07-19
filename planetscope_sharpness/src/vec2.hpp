#pragma once

#include <numeric>

template <typename T>
struct vec2 : public std::array<T, 2> {
    vec2() {}
    vec2(T v) : vec2(v, v) {}
    vec2(T x, T y) : std::array<T, 2>({x,y}) {}

    const vec2 operator+(const vec2& rhs) const { return vec2(*this) += rhs; }
    const vec2 operator-() const { return vec2(-(*this)[0], -(*this)[1]); };
    const vec2 operator-(const vec2& rhs) const { return vec2(*this) -= rhs; };
    const vec2 operator*(const vec2& rhs) const { return vec2(*this) *= rhs; };
    const vec2 operator/(const vec2& rhs) const { return vec2(*this) /= rhs; };

    vec2& operator+=(const vec2& rhs) { (*this)[0] += rhs[0]; (*this)[1] += rhs[1]; return *this; };
    vec2& operator-=(const vec2& rhs) { (*this)[0] -= rhs[0]; (*this)[1] -= rhs[1]; return *this; };
    vec2& operator*=(const vec2& rhs) { (*this)[0] *= rhs[0]; (*this)[1] *= rhs[1]; return *this; };
    vec2& operator/=(const vec2& rhs) { (*this)[0] /= rhs[0]; (*this)[1] /= rhs[1]; return *this; };
};

template <typename T>
vec2<T> operator+(const vec2<T>& v, T scalar) { return vec2<T>(v[0] + scalar, v[1] + scalar); }
template <typename T>
vec2<T> operator-(const vec2<T>& v, T scalar) { return vec2<T>(v[0] - scalar, v[1] - scalar); };
template <typename T>
vec2<T> operator*(const vec2<T>& v, T scalar) { return vec2<T>(v[0] * scalar, v[1] * scalar); };
template <typename T>
vec2<T> operator/(const vec2<T>& v, T scalar) { return vec2<T>(v[0] / scalar, v[1] / scalar); };

template <typename T>
vec2<T> operator+(T scalar, const vec2<T>& v) { return vec2<T>(v[0] + scalar, v[1] + scalar); }
template <typename T>
vec2<T> operator-(T scalar, const vec2<T>& v) { return vec2<T>(v[0] - scalar, v[1] - scalar); };
template <typename T>
vec2<T> operator*(T scalar, const vec2<T>& v) { return vec2<T>(v[0] * scalar, v[1] * scalar); };
template <typename T>
vec2<T> operator/(T scalar, const vec2<T>& v) { return vec2<T>(v[0] / scalar, v[1] / scalar); };

namespace std {
    template <typename T>
    T hypot(const vec2<T>& v) { return std::hypot(v[0], v[1]); }

    template <typename T>
    vec2<T> abs(const vec2<T>& v) { return vec2<T>(std::abs(v[0]), std::abs(v[1])); }

    template <typename T>
    vec2<T> max(const vec2<T>& v, T scalar) { return vec2<T>(std::max(v[0], scalar), std::max(v[1], scalar)); }
    template <typename T>
    vec2<T> max_noref(const vec2<T>& v, T scalar) { return vec2<T>(std::max(v[0], scalar), std::max(v[1], scalar)); }

    template <typename T>
    vec2<T> min(const vec2<T>& v, T scalar) { return vec2<T>(std::min(v[0], scalar), std::min(v[1], scalar)); }
    template <typename T>
    vec2<T> min_noref(const vec2<T>& v, T scalar) { return vec2<T>(std::min(v[0], scalar), std::min(v[1], scalar)); }

}

