// clang-format off
// comp := cmake --build .
// run  := ./openfhe_conv test 1
// run  := ./openfhe_conv winograd ../pack/winograd.h5
// run  := OMP_NUM_THREADS=1 ./openfhe_conv test 1
// run  := OMP_NUM_THREADS=1 ./openfhe_conv winograd ../pack/winograd.h5
// dir  := /home/jyh/project/openfhe-conv/build
// kid  :=
#include <bits/stdc++.h>
#include <H5Cpp.h>
#include "ciphertext-fwd.h"
#include "config.h"
#include <chrono>

using namespace std;

#include "openfhe.h"
#include "config.h"
#include <cassert>
using namespace lbcrypto;
using namespace H5;

void test() {
    mt19937 rng(41);
    std::uniform_real_distribution<double> uniform(0.0, 1.0);

    int n_slots = 1 << 14;

    CCParams<CryptoContextCKKSRNS> params;
    SecretKeyDist secretKeyDist = UNIFORM_TERNARY;
    params.SetSecretKeyDist(secretKeyDist);
    // params.SetSecurityLevel(HEStd_128_classic);
    params.SetSecurityLevel(HEStd_NotSet);

    // vector<uint32_t> levelBudget = {3, 3};
    // vector<uint32_t> bsgsDim = {0, 0};
    // uint32_t levelsAvailableAfterBootstrap = 10;
    // usint depth = levelsAvailableAfterBootstrap + FHECKKSRNS::GetBootstrapDepth(levelBudget, secretKeyDist);
    // params.SetMultiplicativeDepth(depth);
    params.SetMultiplicativeDepth(1);

    // {29, 26, 26, 26, 26, 26}
    params.SetFirstModSize(60);
    params.SetScalingModSize(59);

    // params.SetScalingTechnique(FIXEDMANUAL);
    params.SetScalingTechnique(FIXEDMANUAL);

    params.SetBatchSize(n_slots);
    params.SetRingDim(n_slots * 2);

    CryptoContext<DCRTPoly> cc = GenCryptoContext(params);

    cc->Enable(PKE);
    cc->Enable(FHE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);

    // cc->EvalBootstrapSetup(levelBudget);

    KeyPair<DCRTPoly> keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);
    // cc->EvalBootstrapKeyGen(keys.secretKey, n_slots);

    auto cipherprint = [&](Ciphertext<DCRTPoly> c, int len = -1) {
        Plaintext ptxt;
        cc->Decrypt(keys.secretKey, c, &ptxt);
        cout << "Level: " << c->GetLevel() << ", Precision: " << ptxt->GetLogPrecision() << endl;
        vector<double> values = ptxt->GetRealPackedValue();
        if (len == -1) len = values.size();
        for (int i = 0; i < len; ++i) cerr << values[i] << " \n"[i == len-1];
    };

    auto plainprint = [&](Plaintext &p, int len = -1) {
        vector<double> values = p->GetRealPackedValue();
        if (len == -1) len = values.size();
        for (int i = 0; i < len; ++i) cerr << values[i] << " \n"[i == len-1];
    };

    auto gen_cipher = [&]() {
        vector<double> v(n_slots);
        for (double &i: v) i = uniform(rng);
        Ciphertext<DCRTPoly> c = cc->Encrypt(keys.publicKey, cc->MakeCKKSPackedPlaintext(v));
        return c;
    };

    auto gen_plain = [&](int scale) {
        vector<double> v(n_slots);
        // for (double &i: v) i = uniform(rng);
        for (double &i: v) i = 1;
        Plaintext p = cc->MakeCKKSPackedPlaintext(v, scale);
        return p;
    };


    auto c = gen_cipher();
    cipherprint(c, 10);
    auto p = gen_plain(1);
    auto t = cc->EvalMult(c, p);
    auto tt = cc->EvalMult(t, p);
    cc->RescaleInPlace(tt);
    cipherprint(tt, 10);
    // cipherprint(c, 10);
    // for (int i = 0; i < 9; ++i) {
    //     c = cc->EvalMult(c, p);
    // }
    // cipherprint(c, 10);
    // c = cc->EvalBootstrap(c);
    // cipherprint(c, 10);
    // for (int i = 0; i < 9; ++i) {
    //     c = cc->EvalMult(c, p);
    // }
    // cipherprint(c, 10);
    // c = cc->EvalBootstrap(c);
    // cipherprint(c, 10);
    // for (int i = 0; i < 9; ++i) {
    //     c = cc->EvalMult(c, p);
    // }
    // cipherprint(c, 10);
    // c = cc->EvalBootstrap(c);
    // cipherprint(c, 10);
    // for (int i = 0; i < 9; ++i) {
    //     c = cc->EvalMult(c, p);
    // }
    // cipherprint(c, 10);
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
