#include "image.hpp"

#ifndef IMG_NO_IIO
extern "C" {
#include "iio.h"
}

template <>
img_t<uint8_t> img_t<uint8_t>::load(const std::string& filename)
{
    int w, h, d;
    uint8_t* data = iio_read_image_uint8_vec(filename.c_str(), &w, &h, &d);
    img_t<uint8_t> img(w, h, d, data);
    free(data);
    return img;
}

template <>
img_t<float> img_t<float>::load(const std::string& filename)
{
    int w, h, d;
    float* data = iio_read_image_float_vec(filename.c_str(), &w, &h, &d);
    img_t<float> img(w, h, d, data);
    free(data);
    return img;
}

template <>
img_t<double> img_t<double>::load(const std::string& filename)
{
    int w, h, d;
    double* data = iio_read_image_double_vec(filename.c_str(), &w, &h, &d);
    img_t<double> img(w, h, d, data);
    free(data);
    return img;
}

template <>
void img_t<uint8_t>::save(const std::string& filename) const
{
    iio_write_image_uint8_vec(const_cast<char*>(filename.c_str()), const_cast<uint8_t*>(&data[0]), w, h, d);
}

template <>
void img_t<int>::save(const std::string& filename) const
{
    iio_write_image_int_vec(const_cast<char*>(filename.c_str()), const_cast<int*>(&data[0]), w, h, d);
}

template <>
void img_t<float>::save(const std::string& filename) const
{
    iio_write_image_float_vec(const_cast<char*>(filename.c_str()), const_cast<float*>(&data[0]), w, h, d);
}

template <>
void img_t<double>::save(const std::string& filename) const
{
    iio_write_image_double_vec(const_cast<char*>(filename.c_str()), const_cast<double*>(&data[0]), w, h, d);
}

template <>
void img_t<std::complex<float>>::save(const std::string& filename) const
{
    iio_write_image_float_vec(const_cast<char*>(filename.c_str()), const_cast<float*>(reinterpret_cast<const float*>(&data[0])), w, h, 2 * d);
}

template <>
void img_t<std::complex<double>>::save(const std::string& filename) const
{
    iio_write_image_double_vec(const_cast<char*>(filename.c_str()), const_cast<double*>(reinterpret_cast<const double*>(&data[0])), w, h, 2 * d);
}
#endif

