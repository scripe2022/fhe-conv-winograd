#include <H5Cpp.h>
#include <vector>
#include "utils.h"

using namespace std;
using namespace H5;
using namespace lbcrypto;

int NUM_ADDS = 0;
int NUM_MULTS = 0;
int NUM_ROTS = 0;

Ciphertext<DCRTPoly> blocks_dot_product(lbcrypto::CryptoContext<lbcrypto::DCRTPoly> &cc, Row &row, vector<unordered_map<int, Ciphertext<DCRTPoly>>> &inputs) {
    unordered_map<int, vector<Ciphertext<DCRTPoly>>> groups;
    for (int y = 0; y < (int)row.size(); ++y) {
        Block &block = row[y];
        for (Diagonal &diag: block) {
            groups[diag.gs].push_back(cc->EvalMult(diag.data, inputs[y][diag.bs]));
            ++NUM_MULTS;
        }
    }
    vector<Ciphertext<DCRTPoly>> sums;
    for (auto &[gs, vs]: groups) {
        sums.push_back(gs != 0 ? cc->EvalRotate(cc->EvalAddMany(vs), gs) : cc->EvalAddMany(vs));
        if (gs != 0) ++NUM_ROTS;
        NUM_ADDS += vs.size() - 1;
    }
    return cc->EvalAddMany(sums);
    NUM_ADDS += sums.size() - 1;
}

multi_cipher blocks_matrix_mult(CryptoContext<DCRTPoly> &cc, Mat &mat, multi_cipher_bsgs &inputs, int T_out) {
    multi_cipher outputs(T_out);
    for (int x = 0; x < T_out; ++x) outputs[x] = blocks_dot_product(cc, mat[x], inputs);
    return outputs;
}
