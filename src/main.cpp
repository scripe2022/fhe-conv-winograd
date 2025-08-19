// clang-format off
// comp := cmake --build .
// run  := OMP_NUM_THREADS=1 ./openfhe_conv test 1
// run  := OMP_NUM_THREADS=1 ./openfhe_conv winograd ../pack/winograd.h5
// run  := OMP_NUM_THREADS=1 ./openfhe_conv orion ../pack/orion.h5
// dir  := /home/jyh/project/openfhe-conv/build
// kid  :=
#include <bits/stdc++.h>
#include <H5Cpp.h>
#include "config.h"

using namespace std;

#include "openfhe.h"
#include "config.h"
#include <cassert>
using namespace lbcrypto;
using namespace H5;

void test() {
    mt19937 rng(42);
    std::uniform_real_distribution<double> uniform(0.0, 1.0);

    int n_slots = 8192;

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
    cc->EvalRotateKeyGen(keys.secretKey, {});

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

    auto gen_plain = [&]() {
        vector<double> data(n_slots);
        for (double &i: data) i = uniform(rng);
        return cc->MakeCKKSPackedPlaintext(data);
    };

    auto gen_cipher = [&]() {
        Ciphertext<DCRTPoly> c = cc->Encrypt(keys.publicKey, gen_plain());
        return c;
    };

    const int CyclotomicOrder = cc->GetCyclotomicOrder();

    int N = 100;
    vector<Ciphertext<DCRTPoly>> c(N);
    for (int i = 0; i < N; ++i) c[i] = gen_cipher();

    vector<double> weights(N, 0);
    int steps = 25;
    for (int i = 0; i < 100; i += 4) {
        weights[i] = (i/4) % 2 == 0 ? 1.0 : -1.0;
    }

    // D_BEGIN: debug time start
    #include <chrono>
    auto TEST_TIME_START = std::chrono::high_resolution_clock::now();
    // D_END:

    for (int it = 0; it < N; ++it) {
        auto sum = cc->EvalLinearWSumMutable(c, weights);
    }

    // D_BEGIN: time end
    auto TEST_TIME_END = std::chrono::high_resolution_clock::now();
    auto TEST_TIME_DURATION = std::chrono::duration_cast<std::chrono::microseconds>(TEST_TIME_END - TEST_TIME_START);
    std::cerr << "Running time: " << TEST_TIME_DURATION.count() / 1000 << "ms" << std::endl;
    // D_END:
}

int main(int argc, char *argv[]) {
    if (argc != 3 || (string(argv[1]) != "orion" && string(argv[1]) != "winograd" && string(argv[1]) != "test")) {
        cerr << "Usage: " << argv[0] << " orion|winograd <path_to_h5_file>" << endl;
        return 1;
    }
    if (string(argv[1]) == "orion") {
        orion(argv[2]);
    }
    else if (string(argv[1]) == "winograd") {
        winograd(argv[2]);
    }
    else if (string(argv[1]) == "test") {
        test();
    }
}
