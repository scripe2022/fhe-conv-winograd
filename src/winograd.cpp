// comp := cmake --build .
// run  := ./openfhe_conv winograd ../pack/winograd.h5
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
    vector<double> flat(static_cast<size_t>(R * C));
    ds.read(flat.data(), PredType::NATIVE_DOUBLE);
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
    vector<double> flat(static_cast<size_t>(R * C));
    ds.read(flat.data(), PredType::NATIVE_DOUBLE);
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
        ref[i] = read_array_2d<double>(obj, ("/global/ref_" + to_string(i)).c_str(), PredType::NATIVE_DOUBLE);
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
        ++NUM_ADDS;
    }
    for (; i + 1 < pool[0].size(); i += 2) {
        c.push_back(cc->EvalAdd(inputs[pool[0][i]], inputs[pool[0][i + 1]]));
        ++NUM_ADDS;
    }
    if (i < pool[0].size()) {
        c.push_back(inputs[pool[0][i]]);
    }
    Ciphertext<DCRTPoly> result = cc->EvalAddMany(c);
    NUM_ADDS += c.size() - 1;
    if (negate == 1) cc->EvalNegateInPlace(result), ++NUM_ADDS;
    return result;
}

unordered_map<int, Ciphertext<DCRTPoly>> blocks_dot_product_gs(CryptoContext<DCRTPoly> &cc, Row &row, vector<unordered_map<int, Ciphertext<DCRTPoly>>> &inputs) {
    unordered_map<int, vector<Ciphertext<DCRTPoly>>> groups;
    for (int y = 0; y < (int)row.size(); ++y) {
        Block &block = row[y];
        for (Diagonal &diag: block) {
            groups[diag.gs].push_back(cc->EvalMult(diag.data, inputs[y][diag.bs]));
            ++NUM_MULTS;
        }
    }
    unordered_map<int, Ciphertext<DCRTPoly>> sums;
    for (auto &[gs, vs]: groups) {
        if (sums.find(gs) == sums.end()) sums[gs] = cc->EvalAddMany(vs), NUM_ADDS += vs.size() - 1;
        else cc->EvalAddInPlace(sums[gs], cc->EvalAddMany(vs)), NUM_ADDS += vs.size();
    }
    return sums;
}

void winograd(char *filename) {
    H5File file_input("../pack/winograd_input.h5", H5F_ACC_RDONLY);

    // NOTE: global
    int n_slots = read_attr<int>(file_input, "n_slots", PredType::NATIVE_INT);
    int T_in = read_attr<int>(file_input, "T_in", PredType::NATIVE_INT);
    int T_out = read_attr<int>(file_input, "T_out", PredType::NATIVE_INT);
    int H = read_attr<int>(file_input, "H", PredType::NATIVE_INT);
    int W = read_attr<int>(file_input, "W", PredType::NATIVE_INT);
    int R = read_attr<int>(file_input, "R", PredType::NATIVE_INT);
    int S = read_attr<int>(file_input, "S", PredType::NATIVE_INT);
    int Ht = read_attr<int>(file_input, "Ht", PredType::NATIVE_INT);
    int Wt = read_attr<int>(file_input, "Wt", PredType::NATIVE_INT);

    assert(H % Ht == 0 && W % Wt == 0);
    int NHt = H / Ht, NWt = W / Wt;
    int Ht_in = Ht + R - 1, Wt_in = Wt + S - 1;
    int TS_in = Ht_in * Wt_in;
    int TS_out = Ht * Wt;

    vector<int> global_rots = read_array_1d<int>(file_input, "/global/rotations", PredType::NATIVE_INT);
    vector<vector<int>> v_bs;
    for (int i = 0; i < T_in; ++i) v_bs.push_back(read_array_1d<int>(file_input, ("/global/bs" + to_string(i)).c_str(), PredType::NATIVE_INT));

    CCParams<CryptoContextCKKSRNS> params;
    SecretKeyDist secretKeyDist = UNIFORM_TERNARY;
    params.SetSecretKeyDist(secretKeyDist);
    params.SetSecurityLevel(HEStd_NotSet);

    uint32_t levelsAvailableAfterBootstrap = 8;

    #ifdef WINOGRAD_BOOTSTRAP
    vector<uint32_t> levelBudget = {3, 3};
    vector<uint32_t> bsgsDim = {0, 0};
    usint depth = levelsAvailableAfterBootstrap + FHECKKSRNS::GetBootstrapDepth(levelBudget, secretKeyDist);
    params.SetMultiplicativeDepth(depth);
    cout << depth << endl;
    #else
    params.SetMultiplicativeDepth(levelsAvailableAfterBootstrap);
    #endif

    // {29, 26, 26, 26, 26, 26}
    params.SetFirstModSize(60);
    params.SetScalingModSize(40);

    params.SetScalingTechnique(FIXEDMANUAL);
    // params.SetScalingTechnique(FIXEDAUTO);

    params.SetBatchSize(n_slots);
    params.SetRingDim(n_slots * 2);

    CryptoContext<DCRTPoly> cc = GenCryptoContext(params);

    cc->Enable(PKE);
    cc->Enable(FHE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);

    KeyPair<DCRTPoly> keys = cc->KeyGen();
    #ifdef WINOGRAD_BOOTSTRAP
    cc->EvalBootstrapSetup(levelBudget);
    cc->EvalBootstrapKeyGen(keys.secretKey, n_slots);
    #endif
    cc->EvalMultKeyGen(keys.secretKey);
    cc->EvalRotateKeyGen(keys.secretKey, global_rots);

    auto cipherprint = [&](Ciphertext<DCRTPoly> &c, int len = -1) {
        Plaintext ptxt;
        cc->Decrypt(keys.secretKey, c, &ptxt);
        auto prec = ptxt->GetLogPrecision();
        vector<double> values = ptxt->GetRealPackedValue();
        if (len == -1) len = values.size();
        cerr << "Precision: " << prec << ", Values: \n";
        for (int i = 0; i < len; ++i) cerr << i << ": " << (fabs(values[i]) < 1e-10 ? 0 : values[i]) << "\n";
    };

    auto plainprint = [&](Plaintext &ptxt, int len = -1) {
        auto prec = ptxt->GetLogPrecision();
        vector<double> values = ptxt->GetRealPackedValue();
        if (len == -1) len = values.size();
        cerr << "Precision: " << prec << ", Values: \n";
        for (int i = 0; i < len; ++i) cerr << i << ": " << (fabs(values[i]) < 1e-10 ? 0 : values[i]) << "\n";

    };

    auto stat = [&](Ciphertext<DCRTPoly> &c) {
        cerr << "Level: " << c->GetLevel() << ", NoiseScaleDeg: " << c->GetNoiseScaleDeg() << ", Scale: " << c->GetScalingFactor() << ", # primes: " << c->GetElements()[0].GetNumOfElements() << endl;
    };

    const int CyclotomicOrder = cc->GetCyclotomicOrder();

    vector<multi_cipher> inputs(TS_out);
    for (int k = 0; k < TS_out; ++k) {
        inputs[k] = read_input_winograd(cc, keys, file_input, k);
        assert(inputs_raw[k].size() == T_in);
    }
    vector<vector<pair<int, int>>> input2c = read_input2c_tarnsform_winograd(file_input, TS_out);
    vector<vector<int>> BT = read_array_2d<int>(file_input, "/global/BT", PredType::NATIVE_INT32);
    vector<vector<int>> AT = read_array_2d<int>(file_input, "/global/AT", PredType::NATIVE_INT32);

    auto run_conv = [&](vector<multi_cipher> &inputs_raw, string fn) {
        H5File file(fn, H5F_ACC_RDONLY);

        // NOTE: read diagonals
        vector<Mat> mats(TS_in, Mat(T_out));
        for (int k = 0; k < TS_in; ++k) {
            for (int i = 0; i < T_out; ++i) {
                mats[k][i] = read_row_blocks_winograd(cc, file, k, i, T_in);
            }
        }

        // NOTE: read input && transforms
        vector<multi_cipher> c(T_in, multi_cipher(TS_in));

        // NOTE: read batchnorm adds
        vector<Plaintext> bn_adds(T_out);
        {
            vector<vector<double>> bn_adds_clear = read_array_2d<double>(file, "/global/bn_add", PredType::NATIVE_DOUBLE);
            for (int i = 0; i < T_out; ++i) {
                bn_adds[i] = cc->MakeCKKSPackedPlaintext(bn_adds_clear[i]);
            }
        }

        auto TEST_TIME_START = std::chrono::high_resolution_clock::now();
        // NOTE: input transform
        for (int k = 0; k < TS_out; ++k) {
            for (int i = 0; i < T_in; ++i) {
                auto ct_precomp = cc->EvalFastRotationPrecompute(inputs_raw[k][i]);
                for (int j = 0; j < TS_out; ++j) {
                    auto [tar, off] = input2c[k][j];
                    c[i][tar] = cc->EvalFastRotation(inputs_raw[k][i], off, CyclotomicOrder, ct_precomp);
                    ++NUM_ROTS;
                }
            }
        }
        // NOTE: precompute bs
        vector<multi_cipher_bsgs> D_tilde_bs(TS_in, multi_cipher_bsgs(T_in));
        for (int i = 0; i < TS_in; ++i) {
            for (int j = 0; j < T_in; ++j) {
                const Ciphertext<DCRTPoly> t = linear_sum_tree_01(cc, c[j], BT[i]);
                auto ct_precomp = cc->EvalFastRotationPrecompute(t);
                for (int offset: v_bs[j]) {
                    D_tilde_bs[i][j].insert({offset, offset != 0 ? cc->EvalFastRotation(t, offset, CyclotomicOrder, ct_precomp) : t});
                    if (offset != 0) ++NUM_ROTS;
                }
            }
        }

        #define ADVANCED_BSGS 1

        #ifdef ADVANCED_BSGS
        for (int t_out = 0; t_out < T_out; ++t_out) {
            multi_cipher_bsgs y_bsgs(TS_out);
            auto add = [&y_bsgs, &cc](int ts_out, int gs, Ciphertext<DCRTPoly> &c, int weight) {
                if (weight == 0) return;
                else if (weight == 1) {
                    if (y_bsgs[ts_out].find(gs) == y_bsgs[ts_out].end()) y_bsgs[ts_out][gs] = c;
                    else cc->EvalAddInPlace(y_bsgs[ts_out][gs], c), ++NUM_ADDS;
                }
                else if (weight == -1) {
                    if (y_bsgs[ts_out].find(gs) == y_bsgs[ts_out].end()) y_bsgs[ts_out][gs] = cc->EvalNegate(c);
                    else cc->EvalSubInPlace(y_bsgs[ts_out][gs], c), ++NUM_ADDS;
                }
                else throw runtime_error("unreachable");
            };
            for (int i = 0; i < TS_in; ++i) {
                unordered_map<int, Ciphertext<DCRTPoly>> E_tilde_bsgs_i = blocks_dot_product_gs(cc, mats[i][t_out], D_tilde_bs[i]);
                for (auto &[gs, c]: E_tilde_bsgs_i) {
                    for (int j = 0; j < TS_out; ++j) {
                        add(j, gs, c, AT[j][i]);
                    }
                }
            }
            for (int ts_out = 0; ts_out < TS_out; ++ts_out) {
                vector<Ciphertext<DCRTPoly>> sums;
                for (auto &[gs, v]: y_bsgs[ts_out]) {
                    sums.push_back(gs != 0 ? cc->EvalRotate(v, gs) : v);
                    if (gs != 0) ++NUM_ROTS;
                }
                inputs[ts_out][t_out] = cc->EvalAddMany(sums);
                NUM_ADDS += sums.size() - 1;
            }
        }
        #else
        // NOTE: matrix multiplication
        vector<multi_cipher> E_tilde(T_out, multi_cipher(TS_in));
        for (int i = 0; i < TS_in; ++i) {
            auto outputs = blocks_matrix_mult(cc, mats[i], D_tilde_bs[i], T_out);
            for (int j = 0; j < T_out; ++j) {
                E_tilde[j][i] = outputs[j];
            }
        }

        // NOTE: interpolation
        vector<multi_cipher> y(TS_out, multi_cipher(T_out));
        for (int i = 0; i < T_out; ++i) {
            for (int j = 0; j < TS_out; ++j) {
                y[j][i] = linear_sum_tree_01(cc, E_tilde[i], AT[j]);
            }
        }
        #endif
        for (int i = 0; i < TS_out; ++i) {
            for (int j = 0; j < T_out; ++j) {
                cc->EvalAddInPlace(inputs[i][j], bn_adds[j]);
                cc->RescaleInPlace(inputs[i][j]);
            }
        }
        auto TEST_TIME_END = std::chrono::high_resolution_clock::now();
        auto TEST_TIME_DURATION = std::chrono::duration_cast<std::chrono::microseconds>(TEST_TIME_END - TEST_TIME_START);
        std::cerr << "(" << fn << ") " "Running time: " << TEST_TIME_DURATION.count() / 1000 << "ms" << std::endl;
    };


    auto run_act = [&](vector<multi_cipher> &inputs) {
        auto TEST_TIME_START = std::chrono::high_resolution_clock::now();
        for (auto &cs: inputs) {
            for (auto &c: cs) {
                auto x2 = cc->EvalSquare(c);
                cc->RescaleInPlace(x2);
                c = cc->EvalSub(x2, c);
            }
        }
        auto TEST_TIME_END = std::chrono::high_resolution_clock::now();
        auto TEST_TIME_DURATION = std::chrono::duration_cast<std::chrono::microseconds>(TEST_TIME_END - TEST_TIME_START);
        std::cerr << "(act) Running time: " << TEST_TIME_DURATION.count() / 1000 << "ms" << std::endl;
    };

    stat(inputs[0][0]);
    cout << endl;

    run_conv(inputs, "../pack/winograd_conv_0.h5");
    stat(inputs[0][0]);
    cout << endl;

    run_act(inputs);
    stat(inputs[0][0]);
    cout << endl;

    run_conv(inputs, "../pack/winograd_conv_1.h5");
    stat(inputs[0][0]);
    cout << endl;

    run_act(inputs);
    stat(inputs[0][0]);
    cout << endl;

    run_conv(inputs, "../pack/winograd_conv_2.h5");
    stat(inputs[0][0]);
    cout << endl;

    run_act(inputs);
    stat(inputs[0][0]);
    cout << endl;

    // run_conv(inputs, "../pack/winograd_conv_2.h5");
    // stat(inputs[0][0]);
    // cout << endl;

    // run_act(inputs);
    // stat(inputs[0][0]);
    // cout << endl;

    // cipherprint(inputs[0][0], 10);
    cout << endl;
    // NOTE: check
    H5File file_ref("../pack/winograd_ref.h5", H5F_ACC_RDONLY);
    vector<vector<vector<double>>> ref = read_reference_winograd(file_ref, TS_out);
    double mae = 0, max_err = 0;
    vector<double> precisions;
    vector<int> levels;
    for (int idx = 0; idx < TS_out; ++idx) {
        for (int i = 0; i < T_out; ++i) {
            Plaintext ptxt;
            cc->Decrypt(keys.secretKey, inputs[idx][i], &ptxt);
            precisions.push_back(ptxt->GetLogPrecision());
            levels.push_back(ptxt->GetLevel());
            vector<double> values = ptxt->GetRealPackedValue();
            for (int k = 0; k < n_slots; ++k) {
                double diff = abs(values[k] - ref[idx][i][k]);
                mae += diff;
                max_err = max(max_err, diff);
            }
        }
    }
    double avg_precision = accumulate(precisions.begin(), precisions.end(), 0.0) / precisions.size();
    cerr << "MAE: " << mae / (n_slots * TS_out * T_out) << ", " << "MAX_ERR: " << max_err << ", avg precisions: " << avg_precision << endl;
    cerr << "NUM_ADDS: " << NUM_ADDS << ", NUM_MULTS: " << NUM_MULTS << ", NUM_ROTS: " << NUM_ROTS << endl;
}
