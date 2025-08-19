// comp := cmake --build .
// run  := OMP_NUM_THREADS=1 ./openfhe_conv winograd ../pack/winograd.h5
// run  := OMP_NUM_THREADS=1 ./openfhe_conv orion ../pack/orion.h5
// dir  := /home/jyh/project/openfhe-conv/build
// kid  :=

#include <H5Cpp.h>
#include "openfhe.h"
#include "utils.h"
#include "config.h"
#include <cassert>

using namespace lbcrypto;
using namespace std;
using namespace H5;

Block read_block_winograd(CryptoContext<DCRTPoly> &cc, const H5Object &obj, int k, int x, int y) {
    vector<int> bs = read_array_1d<int>(obj, ("/blocks/bs_" + to_string(x) + "_" + to_string(y)).c_str(), PredType::NATIVE_INT);
    vector<int> gs = read_array_1d<int>(obj, ("/blocks/gs_" + to_string(x) + "_" + to_string(y)).c_str(), PredType::NATIVE_INT);
    DataSet ds = obj.openDataSet(("/blocks/diags_" + to_string(k) + "_" + to_string(x) + "_" + to_string(y)).c_str());
    DataSpace sp = ds.getSpace();
    if (sp.getSimpleExtentNdims() != 2) {
        throw std::runtime_error("Expected 2D dataset");
    }
    hsize_t dims[2]; sp.getSimpleExtentDims(dims, nullptr);
    const hsize_t R = dims[0], C = dims[1];
    vector<float> flat(static_cast<size_t>(R * C));
    ds.read(flat.data(), PredType::NATIVE_FLOAT);
    vector<Diagonal> diags(R);
    for (int r = 0; r < R; ++r) {
        auto *src = flat.data() + r * C;
        diags[r].bs = bs[r];
        diags[r].gs = gs[r];
        vector<double> data(C);
        std::copy(src, src + C, data.begin());
        diags[r].data = cc->MakeCKKSPackedPlaintext(data);
    }
    return diags;
}

Row read_row_blocks_winograd(CryptoContext<DCRTPoly> &cc, const H5::H5Object &obj, int k, int x, int T_in) {
    Row row(T_in);
    for (int t = 0; t < T_in; ++t) {
        row[t] = read_block_winograd(cc, obj, k, x, t);
    }
    return row;
}

multi_cipher read_input_winograd(CryptoContext<DCRTPoly> &cc, KeyPair<DCRTPoly> &keys, const H5Object &obj, int k) {
    DataSet ds = obj.openDataSet("/global/input_" + to_string(k));
    DataSpace sp = ds.getSpace();
    if (sp.getSimpleExtentNdims() != 2) {
        throw std::runtime_error("Expected 2D dataset");
    }
    hsize_t dims[2]; sp.getSimpleExtentDims(dims, nullptr);
    const hsize_t R = dims[0], C = dims[1];
    vector<float> flat(static_cast<size_t>(R * C));
    ds.read(flat.data(), PredType::NATIVE_FLOAT);
    vector<Ciphertext<DCRTPoly>> inputs(R);
    for (int r = 0; r < R; ++r) {
        auto *src = flat.data() + r * C;
        vector<double> data(C);
        std::copy(src, src + C, data.begin());
        inputs[r] = cc->Encrypt(keys.publicKey, cc->MakeCKKSPackedPlaintext(data));
    }
    return inputs;
}

vector<vector<vector<double>>> read_reference_winograd(const H5Object &obj, int ts_out) {
    vector<vector<vector<double>>> ref(ts_out);
    for (int i = 0; i < ts_out; ++i) {
        ref[i] = read_array_2d<double>(obj, ("/global/ref_" + to_string(i)).c_str(), PredType::NATIVE_FLOAT);
    }
    return ref;
}

vector<vector<pair<int, int>>> read_input2c_tarnsform_winograd(const H5Object &obj, int ts_out) {
    vector<vector<int>> input2c_target = read_array_2d<int>(obj, "/global/input2c_target", PredType::NATIVE_INT32);
    vector<vector<int>> input2c_offset = read_array_2d<int>(obj, "/global/input2c_offset", PredType::NATIVE_INT32);
    vector<vector<pair<int, int>>> input2c(ts_out, vector<pair<int, int>>(ts_out));
    for (int i = 0; i < ts_out; ++i) {
        for (int j = 0; j < ts_out; ++j) {
            input2c[i][j] = {input2c_target[i][j], input2c_offset[i][j]};
        }
    }
    return input2c;
}

Ciphertext<DCRTPoly> linear_sum_tree_01(CryptoContext<DCRTPoly> &cc, vector<Ciphertext<DCRTPoly>> &inputs, vector<int> &weights) {
    int wsum = 0;
    for (int i: weights) wsum += i;
    int negate = wsum >= 0 ? 0 : 1;

    vector<int> pool[2];
    for (int i = 0; i < (int)weights.size(); ++i) {
        if (weights[i] == 1) pool[0 ^ negate].push_back(i);
        else if (weights[i] == -1) pool[1 ^ negate].push_back(i);
    }
    vector<Ciphertext<DCRTPoly>> c;
    uint32_t i = 0, j = 0;
    for (; i < pool[0].size() && j < pool[1].size(); ++i, ++j) {
        c.push_back(cc->EvalSub(inputs[pool[0][i]], inputs[pool[1][j]]));
    }
    for (; i + 1 < pool[0].size(); i += 2) {
        c.push_back(cc->EvalAdd(inputs[pool[0][i]], inputs[pool[0][i + 1]]));
    }
    if (i < pool[0].size()) {
        c.push_back(inputs[pool[0][i]]);
    }
    Ciphertext<DCRTPoly> result = cc->EvalAddMany(c);
    if (negate == 1) cc->EvalNegateInPlace(result);
    return result;
}

void winograd(char *filename) {
    H5File file(filename, H5F_ACC_RDONLY);

    // NOTE: global
    int n_slots = read_attr<int>(file, "n_slots", PredType::NATIVE_INT);
    int T_in = read_attr<int>(file, "T_in", PredType::NATIVE_INT);
    int T_out = read_attr<int>(file, "T_out", PredType::NATIVE_INT);
    int H = read_attr<int>(file, "H", PredType::NATIVE_INT);
    int W = read_attr<int>(file, "W", PredType::NATIVE_INT);
    int R = read_attr<int>(file, "R", PredType::NATIVE_INT);
    int S = read_attr<int>(file, "S", PredType::NATIVE_INT);
    int Ht = read_attr<int>(file, "Ht", PredType::NATIVE_INT);
    int Wt = read_attr<int>(file, "Wt", PredType::NATIVE_INT);

    assert(H % Ht == 0 && W % Wt == 0);
    int NHt = H / Ht, NWt = W / Wt;
    int Ht_in = Ht + R - 1, Wt_in = Wt + S - 1;
    int ts_in = Ht_in * Wt_in;
    int ts_out = Ht * Wt;

    vector<int> global_rots = read_array_1d<int>(file, "/global/rotations", PredType::NATIVE_INT);
    vector<vector<int>> v_bs;
    for (int i = 0; i < T_in; ++i) v_bs.push_back(read_array_1d<int>(file, ("/global/bs" + to_string(i)).c_str(), PredType::NATIVE_INT));

    // NOTE: fhe
    CCParams<CryptoContextCKKSRNS> params;
    params.SetSecurityLevel(HEStd_NotSet);
    params.SetMultiplicativeDepth(MULT_DEPTH);
    params.SetScalingModSize(SCALING_FACTOR);
    params.SetBatchSize(n_slots);
    params.SetRingDim(n_slots * 2);
    CryptoContext<DCRTPoly> cc = GenCryptoContext(params);
    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);
    KeyPair<DCRTPoly> keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);
    cc->EvalRotateKeyGen(keys.secretKey, global_rots);

    auto cipherprint = [&](Ciphertext<DCRTPoly> &c, int len = -1) {
        Plaintext ptxt;
        cc->Decrypt(keys.secretKey, c, &ptxt);
        auto prec = ptxt->GetLogPrecision();
        vector<double> values = ptxt->GetRealPackedValue();
        if (len == -1) len = values.size();
        cerr << "Precision: " << prec << ", Values: ";
        for (int i = 0; i < len; ++i) cerr << values[i] << " \n"[i == len-1];
    };

    auto plainprint = [&](Plaintext &p, int len = -1) {
        vector<double> values = p->GetRealPackedValue();
        if (len == -1) len = values.size();
        for (int i = 0; i < len; ++i) cerr << values[i] << " \n"[i == len-1];
    };

    const int CyclotomicOrder = cc->GetCyclotomicOrder();

    // NOTE: read diagonals
    vector<Mat> mats(ts_in, Mat(T_out));
    for (int k = 0; k < ts_in; ++k) {
        for (int i = 0; i < T_out; ++i) {
            mats[k][i] = read_row_blocks_winograd(cc, file, k, i, T_in);
        }
    }

    // NOTE: read input && transforms
    vector<multi_cipher> inputs_raw(ts_out), c(T_in, multi_cipher(ts_in));
    for (int k = 0; k < ts_out; ++k) {
        inputs_raw[k] = read_input_winograd(cc, keys, file, k);
        assert(inputs_raw[k].size() == T_in);
    }
    vector<vector<pair<int, int>>> input2c = read_input2c_tarnsform_winograd(file, ts_out);
    vector<vector<int>> BT = read_array_2d<int>(file, "/global/BT", PredType::NATIVE_INT32);
    vector<vector<int>> AT = read_array_2d<int>(file, "/global/AT", PredType::NATIVE_INT32);

    auto TEST_TIME_START = std::chrono::high_resolution_clock::now();
    // NOTE: input transform
    for (int k = 0; k < ts_out; ++k) {
        for (int i = 0; i < T_in; ++i) {
            auto ct_precomp = cc->EvalFastRotationPrecompute(inputs_raw[k][i]);
            for (int j = 0; j < ts_out; ++j) {
                auto [tar, off] = input2c[k][j];
                c[i][tar] = cc->EvalFastRotation(inputs_raw[k][i], off, CyclotomicOrder, ct_precomp);
            }
        }
    }
    // NOTE: precompute bs
    vector<multi_cipher_bs> D_tilde_bs(ts_in, multi_cipher_bs(T_in));
    for (int i = 0; i < ts_in; ++i) {
        for (int j = 0; j < T_in; ++j) {
            const Ciphertext<DCRTPoly> t = linear_sum_tree_01(cc, c[j], BT[i]);
            auto ct_precomp = cc->EvalFastRotationPrecompute(t);
            for (int offset: v_bs[j]) {
                D_tilde_bs[i][j].insert({offset, offset != 0 ? cc->EvalFastRotation(t, offset, CyclotomicOrder, ct_precomp) : t});
            }
        }
    }
    // NOTE: matrix multiplication
    vector<multi_cipher> E_tilde(T_out, multi_cipher(ts_in));
    for (int i = 0; i < ts_in; ++i) {
        auto outputs = blocks_matrix_mult(cc, mats[i], D_tilde_bs[i], T_out);
        for (int j = 0; j < T_out; ++j) {
            E_tilde[j][i] = outputs[j];
        }
    }

    // NOTE: interpolation
    vector<multi_cipher> y(ts_out, multi_cipher(T_out));
    for (int i = 0; i < T_out; ++i) {
        for (int j = 0; j < ts_out; ++j) {
            y[j][i] = linear_sum_tree_01(cc, E_tilde[i], AT[j]);
        }
    }

    auto TEST_TIME_END = std::chrono::high_resolution_clock::now();
    auto TEST_TIME_DURATION = std::chrono::duration_cast<std::chrono::microseconds>(TEST_TIME_END - TEST_TIME_START);
    std::cerr << "Running time: " << TEST_TIME_DURATION.count() / 1000 << "ms" << std::endl;

    // NOTE: check
    vector<vector<vector<double>>> ref = read_reference_winograd(file, ts_out);
    double mae = 0, max_err = 0;
    vector<double> precisions;
    for (int idx = 0; idx < ts_out; ++idx) {
        for (int i = 0; i < T_out; ++i) {
            Plaintext ptxt;
            cc->Decrypt(keys.secretKey, y[idx][i], &ptxt);
            precisions.push_back(ptxt->GetLogPrecision());
            vector<double> values = ptxt->GetRealPackedValue();
            for (int k = 0; k < n_slots; ++k) {
                double diff = abs(values[k] - ref[idx][i][k]);
                mae += diff;
                max_err = max(max_err, diff);
            }
        }
    }
    double avg_precision = accumulate(precisions.begin(), precisions.end(), 0.0) / precisions.size();
    cerr << "MAE: " << mae / (n_slots * ts_out * T_out) << ", " << "MAX_ERR: " << max_err << ", avg precisions: " << avg_precision << endl;
}
