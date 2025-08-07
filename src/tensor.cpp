// clang-format off
// comp := cmake --build .
// run  := ./openfhe_demo
// dir  := /home/project/openfhe-demo/build
// kid  :=
// #include <iostream>
// #include "tensor.h"
// #include <iomanip>
// #include <cassert>
// #include <chrono>
// #include <random>

// Tensor::Tensor(vector<int> _shape): shape(_shape) {
//     int n = size();
//     std::mt19937 gen(std::chrono::steady_clock::now().time_since_epoch().count());
//     int total = 1;
//     for (int dim: shape) total *= dim;
//     assert(total == n);
//     uniform_real_distribution<> dist(0.0, 1.0);
//     data.resize(n);
//     for (int i = 0; i < n; ++i) { data[i] = dist(gen); }
// }

// int Tensor::size() {
//     int n = 1;
//     for (int dim : shape) n *= dim;
//     return n;
// }

// std::ostream& operator<<(std::ostream& os, Tensor& obj) {
//     int n = obj.size();
//     os << "size = " << n << endl;
//     os << "shape = [";
//     for (size_t i = 0; i < obj.shape.size(); ++i) {
//         os << obj.shape[i];
//         if (i < obj.shape.size() - 1) os << ", ";
//     }
//     os << "]" << endl;
//     for (int i = 0; i < n; ++i) {
//         os << std::fixed << std::setprecision(2) << obj.data[i] << " ";
//         if ((i+1) % 10 == 0) os << endl;
//     }
//     return os;
// }

// void Tensor::reshape(vector<int> shape) {
//     int n = 1;
//     for (int dim : shape) n *= dim;
//     assert(n == size());
//     this->shape = shape;
// }

// double Tensor::get(vector<int> idx) {
//     assert(idx.size() == shape.size());
//     int n = size();
//     int i = 0;
//     for (size_t j = 0; j < idx.size(); ++j) {
//         assert(idx[j] >= 0 && idx[j] < shape[j]);
//         n /= shape[j];
//         i += idx[j] * n;
//     }
//     return data[i];
// }

// void Tensor::set(vector<int> idx, double x) {
//     assert(idx.size() == shape.size());
//     int n = size();
//     int i = 0;
//     for (size_t j = 0; j < idx.size(); ++j) {
//         assert(idx[j] >= 0);
//         assert(idx[j] < shape[j]);
//         n /= shape[j];
//         i += idx[j] * n;
//     }
//     cout << i << endl;
//     data[i] = x;
// }

// Tensor mul(Tensor& lhs, Tensor& rhs) {
//     assert(lhs.shape.size() == 2);
//     assert(rhs.shape.size() == 1);
//     assert(lhs.shape[1] == rhs.shape[0]);
//     Tensor result({lhs.shape[0]});
//     for (int i = 0; i < lhs.shape[0]; ++i) {
//         double sum = 0.0;
//         for (int k = 0; k < lhs.shape[1]; ++k) {
//             sum += lhs.get({i, k}) * rhs.get({k});
//         }
//         result.set({i}, sum);
//     }
//     return result;
// }


// Tensor conv(Tensor& img, Tensor& kernel) {
//     assert(img.shape.size() == 2);
//     assert(kernel.shape.size() == 2);
//     assert(img.shape[0] >= kernel.shape[0]);
//     assert(img.shape[1] >= kernel.shape[1]);
//     int h_out = img.shape[0] - kernel.shape[0] + 1;
//     int w_out = img.shape[1] - kernel.shape[1] + 1;
//     Tensor result({h_out, w_out});
//     for (int i = 0; i < h_out; ++i) {
//         for (int j = 0; j < w_out; ++j) {
//             double sum = 0.0;
//             for (int ki = 0; ki < kernel.shape[0]; ++ki) {
//                 for (int kj = 0; kj < kernel.shape[1]; ++kj) {
//                     sum += img.get({i + ki, j + kj}) * kernel.get({ki, kj});
//                 }
//             }
//             result.set({i, j}, sum);
//         }
//     }
//     return result;
// }
