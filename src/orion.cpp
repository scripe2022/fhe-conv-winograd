// comp := cmake --build .
// run  := OMP_NUM_THREADS=1 ./openfhe_conv orion ../pack/orion.h5
// run  := OMP_NUM_THREADS=1 ./openfhe_conv winograd ../pack/winograd.h5
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

Block read_block_orion(CryptoContext<DCRTPoly> &cc, const H5Object &obj, int x, int y) {
    string idx = to_string(x) + "_" + to_string(y);
    vector<int> bs = read_array_1d<int>(obj, ("/blocks/bs_" + idx).c_str(), PredType::NATIVE_INT);
    vector<int> gs = read_array_1d<int>(obj, ("/blocks/gs_" + idx).c_str(), PredType::NATIVE_INT);
    DataSet ds = obj.openDataSet(("/blocks/diags_" + idx).c_str());
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

Row read_row_blocks_orion(CryptoContext<DCRTPoly> &cc, const H5::H5Object &obj, int x, int T_in) {
    Row row(T_in);
    for (int t = 0; t < T_in; ++t) {
        row[t] = read_block_orion(cc, obj, x, t);
    }
    return row;
}

vector<Ciphertext<DCRTPoly>> read_input_orion(CryptoContext<DCRTPoly> &cc, KeyPair<DCRTPoly> &keys, const H5Object &obj) {
    DataSet ds = obj.openDataSet("/global/input");
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

vector<vector<double>> read_reference_orion(const H5Object &obj) {
    DataSet ds = obj.openDataSet("/global/reference");
    DataSpace sp = ds.getSpace();
    if (sp.getSimpleExtentNdims() != 2) {
        throw std::runtime_error("Expected 2D dataset");
    }
    hsize_t dims[2]; sp.getSimpleExtentDims(dims, nullptr);
    const hsize_t R = dims[0], C = dims[1];
    vector<float> flat(static_cast<size_t>(R * C));
    ds.read(flat.data(), PredType::NATIVE_FLOAT);
    vector<vector<double>> ref(R, vector<double>(C));
    for (int r = 0; r < R; ++r) {
        auto *src = flat.data() + r * C;
        std::copy(src, src + C, ref[r].begin());
    }
    return ref;
}

void orion(char *filename) {
    H5File file(filename, H5F_ACC_RDONLY);

    // NOTE: global
    int n_slots = read_attr<int>(file, "n_slots", PredType::NATIVE_INT);
    int T_in = read_attr<int>(file, "T_in", PredType::NATIVE_INT);
    int T_out = read_attr<int>(file, "T_out", PredType::NATIVE_INT);

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
        vector<double> values = ptxt->GetRealPackedValue();
        if (len == -1) len = values.size();
        for (int i = 0; i < len; ++i) cerr << values[i] << " \n"[i == len-1];
    };

    auto plainprint = [&](Plaintext &p, int len = -1) {
        vector<double> values = p->GetRealPackedValue();
        if (len == -1) len = values.size();
        for (int i = 0; i < len; ++i) cerr << values[i] << " \n"[i == len-1];
    };

    const int CyclotomicOrder = cc->GetCyclotomicOrder();

    // NOTE: read diagonals
    Mat mat(T_out);
    for (int i = 0; i < T_out; ++i) mat[i] = read_row_blocks_orion(cc, file, i, T_in);

    // NOTE: read input
    multi_cipher_bs inputs(T_in);
    multi_cipher inputs_raw = read_input_orion(cc, keys, file);
    assert(inputs_raw.size() == T_in);

    // NOTE: orion
    auto TEST_TIME_START = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < T_in; ++i) {
        auto ct_precomp = cc->EvalFastRotationPrecompute(inputs_raw[i]);
        for (int bs: v_bs[i]) {
            inputs[i].insert({bs, bs != 0 ? cc->EvalFastRotation(inputs_raw[i], bs, CyclotomicOrder, ct_precomp) : inputs_raw[i]});
        }
    }

    multi_cipher outputs = blocks_matrix_mult(cc, mat, inputs, T_out);

    auto TEST_TIME_END = std::chrono::high_resolution_clock::now();
    auto TEST_TIME_DURATION = std::chrono::duration_cast<std::chrono::microseconds>(TEST_TIME_END - TEST_TIME_START);
    std::cerr << "Running time: " << TEST_TIME_DURATION.count() / 1000 << "ms" << std::endl;

    // NOTE: check
    vector<vector<double>> ref = read_reference_orion(file);
    double mae = 0, max_err = 0;
    vector<double> precisions(outputs.size());
    for (int i = 0; i < (int)outputs.size(); ++i) {
        Plaintext ptxt;
        cc->Decrypt(keys.secretKey, outputs[i], &ptxt);
        precisions[i] = ptxt->GetLogPrecision();
        vector<double> values = ptxt->GetRealPackedValue();
        for (int k = 0; k < n_slots; ++k) {
            double diff = abs(values[k] - ref[i][k]);
            mae += diff;
            max_err = max(max_err, diff);
        }
    }
    double avg_precision = accumulate(precisions.begin(), precisions.end(), 0.0) / precisions.size();
    cerr << "MAE: " << mae / (n_slots * (int)outputs.size()) << ", " << "MAX_ERR: " << max_err << ", avg precisions: " << avg_precision << endl;
}
