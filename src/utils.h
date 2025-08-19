#pragma once

#include "openfhe.h"

#include <H5Cpp.h>
#include <stdexcept>
#include <vector>

#define LOCAL
#ifdef LOCAL
#include <cpglib/print.h>
#define debug(x...) _debug_print(0, #x, x);
#define Debug(x...) _debug_print(1, #x, x);
#define DEBUG(x...) _debug_print(2, #x, x);
#define PP cerr<<"\033[1;33mpause...\e[0m",terminal.ignore();
#else
#define debug(x...)
#define Debug(x...)
#define DEBUG(x...)
#define PP
#endif


struct Diagonal {
    int bs, gs;
    lbcrypto::Plaintext data;
};

typedef std::vector<Diagonal> Block;
typedef std::vector<Block> Row;
typedef std::vector<Row> Mat;
typedef std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> multi_cipher;
typedef std::vector<std::unordered_map<int, lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>> multi_cipher_bs;

lbcrypto::Ciphertext<lbcrypto::DCRTPoly>
blocks_dot_product(lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&,
                   Row&,
                   std::vector<std::unordered_map<int, lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>> &);

multi_cipher blocks_matrix_mult(lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&, Mat&, multi_cipher_bs&, int);

template<typename T> T read_attr(const H5::H5Object &obj, const char *name, const H5::PredType &t) {
    H5::Attribute a = obj.openAttribute(name);
    T v;
    a.read(t, &v);
    return v;
}

template<typename T> std::vector<T> read_array_1d(const H5::H5Object &obj, const char *name, const H5::PredType &t) {
    H5::DataSet ds = obj.openDataSet(name);
    H5::DataSpace sp = ds.getSpace();
    if (sp.getSimpleExtentNdims() != 1) { throw std::runtime_error("Expected 1D array"); }
    hsize_t len = 0;
    sp.getSimpleExtentDims(&len, nullptr);
    std::vector<T> vec(len);
    ds.read(vec.data(), t);
    return vec;
}

template<typename T> std::vector<std::vector<T>> read_array_2d(const H5::H5Object &obj, const char *name, const H5::PredType &t) {
    H5::DataSet ds = obj.openDataSet(name);
    H5::DataSpace sp = ds.getSpace();
    if (sp.getSimpleExtentNdims() != 2) {
        throw std::runtime_error("Expected 2D dataset");
    }
    hsize_t dims[2]; sp.getSimpleExtentDims(dims, nullptr);
    const hsize_t R = dims[0], C = dims[1];
    std::vector<float> flat(static_cast<size_t>(R * C));
    ds.read(flat.data(), H5::PredType::NATIVE_FLOAT);
    std::vector<std::vector<T>> data(R, std::vector<T>(C));
    for (int r = 0; r < R; ++r) {
        auto *src = flat.data() + r * C;
        std::copy(src, src + C, data[r].begin());
    }
    return data;
}


