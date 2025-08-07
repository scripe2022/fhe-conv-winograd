// clang-format off
// comp := cmake --build .
// run  := OMP_NUM_THREADS=1 ./openfhe_conv
// dir  := /home/jyh/project/openfhe-conv/build
// kid  :=
#include <bits/stdc++.h>
#include <ctime>
#define LOCAL
#ifdef LOCAL
// #include <cpglib/print.h>
// #define debug(x...) _debug_print(0, #x, x);
// #define Debug(x...) _debug_print(1, #x, x);
// #define DEBUG(x...) _debug_print(2, #x, x);
// std::ifstream terminal("/dev/tty");
// #define PP cerr<<"\033[1;33mpause...\e[0m",terminal.ignore();
#else
#define debug(x...)
#define Debug(x...)
#define DEBUG(x...)
#define PP
#endif

#include "openfhe.h"
#include "tensor.h"

#define MULT_DEPTH 3
#define PROFILE
using namespace lbcrypto;
using namespace std;

// static std::mt19937 gen(std::random_device{}());
static std::mt19937 gen(42);
static std::uniform_real_distribution<double> dist(0.0, 1.0);

struct Diagonal {
    int bs, gs;
    Plaintext poly;
    Diagonal(int _bs, int _gs, Plaintext _poly) : bs(_bs), gs(_gs), poly(std::move(_poly)) {}
};

void orion() {
    auto time_start_read = std::chrono::high_resolution_clock::now();
    ifstream fin("../pack/orion-pack.txt");
    int n_slots, H, W, C, M, R, S, n1, n2;
    fin >> n_slots >> H >> W >> C >> M >> R >> S >> n1 >> n2;

    CCParams<CryptoContextCKKSRNS> params;
    params.SetSecurityLevel(HEStd_NotSet);
    params.SetMultiplicativeDepth(MULT_DEPTH);
    params.SetScalingModSize(59);
    params.SetBatchSize(n_slots);
    params.SetRingDim(n_slots * 2);
    CryptoContext<DCRTPoly> cc = GenCryptoContext(params);
    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);

    int n_bs, n_gs;
    fin >> n_bs >> n_gs;
    vector<int> rots, bs(n_bs), gs(n_gs);
    for (int i = 0; i < n_bs; ++i) {
        int x; fin >> x;
        bs[i] = x;
        if (x > 0) rots.push_back(x);
    }
    for (int i = 0; i < n_gs; ++i) {
        int x; fin >> x;
        gs[i] = x;
        if (x > 0) rots.push_back(x);
    }

    KeyPair<DCRTPoly> keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);
    cc->EvalRotateKeyGen(keys.secretKey, rots);

    int n_diags, x_max, y_max; fin >> n_diags >> x_max >> y_max;
    vector<vector<vector<Diagonal>>> diagonals1(x_max, vector<vector<Diagonal>>(y_max));
    for (int i = 0; i < n_diags; ++i) {
        int x, y, bs, gs; fin >> x >> y >> bs >> gs;
        vector<double> coefs(n_slots);
        for (double &c: coefs) fin >> c;
        diagonals1[x][y].push_back(Diagonal(bs, gs, cc->MakeCKKSPackedPlaintext(coefs)));
    }
    vector<vector<vector<Diagonal>>> diagonals2(x_max, vector<vector<Diagonal>>(y_max));
    for (int i = 0; i < n_diags; ++i) {
        int x, y, bs, gs; fin >> x >> y >> bs >> gs;
        vector<double> coefs(n_slots);
        for (double &c: coefs) fin >> c;
        diagonals2[x][y].push_back(Diagonal(bs, gs, cc->MakeCKKSPackedPlaintext(coefs)));
    }

    int nt; fin >> nt;
    vector<unordered_map<int, Ciphertext<DCRTPoly>>> ciphers1(nt);
    for (int i = 0; i < nt; ++i) {
        vector<double> data(n_slots);
        for (double &num: data) fin >> num;
        ciphers1[i][0] = cc->Encrypt(keys.publicKey, cc->MakeCKKSPackedPlaintext(data));
    }

    auto time_end_read = std::chrono::high_resolution_clock::now();
    auto time_duration_read = std::chrono::duration_cast<std::chrono::microseconds>(time_end_read - time_start_read);
    std::cerr << "Reading time: " << time_duration_read.count() / 1000 << "ms" << std::endl;

    // NOTE: begin
    auto time_start_fhe = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < nt; ++i) {
        auto ct_precomp = cc->EvalFastRotationPrecompute(ciphers1[i][0]);
        for (int rot: bs) {
            if (rot > 0) ciphers1[i][rot] = cc->EvalFastRotation(ciphers1[i][0], rot, cc->GetCyclotomicOrder(), ct_precomp);
        }
    }
    vector<unordered_map<int, Ciphertext<DCRTPoly>>> ciphers2(nt);
    for (int x = 0; x < x_max; ++x) {
        vector<Ciphertext<DCRTPoly>> out_block;
        for (int y = 0; y < y_max; ++y) {
            for (int g = 0; g < n2; ++g) {
                vector<Ciphertext<DCRTPoly>> out_group;
                for (int b = 0; b < R*S*n1; ++b) {
                    int bs = diagonals1[x][y][g*R*S*n1 + b].bs;
                    out_group.push_back(cc->EvalMult(diagonals1[x][y][g*R*S*n1 + b].poly, ciphers1[y][bs]));
                }
                int gs = g * n1 * H * W;
                Ciphertext<DCRTPoly> result = gs > 0 ? cc->EvalRotate(cc->EvalAddMany(out_group), gs) : cc->EvalAddMany(out_group);
                out_block.push_back(result);
            }
        }
        ciphers2[x][0] = cc->EvalAddMany(out_block);
    }

    for (int i = 0; i < nt; ++i) {
        auto ct_precomp = cc->EvalFastRotationPrecompute(ciphers2[i][0]);
        for (int rot: bs) {
            if (rot > 0) ciphers2[i][rot] = cc->EvalFastRotation(ciphers2[i][0], rot, cc->GetCyclotomicOrder(), ct_precomp);
        }
    }
    vector<Ciphertext<DCRTPoly>> final;
    for (int x = 0; x < x_max; ++x) {
        vector<Ciphertext<DCRTPoly>> out_block;
        for (int y = 0; y < y_max; ++y) {
            for (int g = 0; g < n2; ++g) {
                vector<Ciphertext<DCRTPoly>> out_group;
                for (int b = 0; b < R*S*n1; ++b) {
                    int bs = diagonals2[x][y][g*R*S*n1 + b].bs;
                    out_group.push_back(cc->EvalMult(diagonals2[x][y][g*R*S*n1 + b].poly, ciphers2[y][bs]));
                }
                int gs = g * n1 * H * W;
                Ciphertext<DCRTPoly> result = gs > 0 ? cc->EvalRotate(cc->EvalAddMany(out_group), gs) : cc->EvalAddMany(out_group);
                out_block.push_back(result);
            }
        }
        final.push_back(cc->EvalAddMany(out_block));
    }


    auto time_end_fhe = std::chrono::high_resolution_clock::now();
    auto time_duration_fhe = std::chrono::duration_cast<std::chrono::microseconds>(time_end_fhe - time_start_fhe);
    std::cerr << "Running time: " << time_duration_fhe.count() / 1000 << "ms" << std::endl;

    auto check = [&](string filename) {
        ifstream ref(filename);
        double mae = 0, max_err = 0;
        vector<double> precisions(final.size());
        for (int x = 0; x < (int)final.size(); ++x) {
            Plaintext ptxt;
            cc->Decrypt(keys.secretKey, final[x], &ptxt);
            precisions[x] = ptxt->GetLogPrecision();
            vector<double> values = ptxt->GetRealPackedValue();
            for (int y = 0; y < n_slots; ++y) {
                double sol; ref >> sol;
                double out = values[y];
                mae += abs(out - sol);
                max_err = max(max_err, abs(out - sol));
            }
        }
        double avg_precision = accumulate(precisions.begin(), precisions.end(), 0.0) / precisions.size();
        cerr << "MAE: " << mae / (n_slots * (int)final.size()) << ", " << "MAX_ERR: " << max_err << ", avg precisions: " << avg_precision << endl;
    };
    check("../pack/orion-ref.txt");
}

void winograd() {
    auto time_start_read = std::chrono::high_resolution_clock::now();
    ifstream fin("../pack/winograd-pack.txt");
    int n_slots, H, W, C, M, R, S;
    fin >> n_slots >> H >> W >> C >> M >> R >> S;

    CCParams<CryptoContextCKKSRNS> params;
    params.SetSecurityLevel(HEStd_NotSet);
    params.SetMultiplicativeDepth(MULT_DEPTH);
    params.SetScalingModSize(59);
    params.SetBatchSize(n_slots);
    params.SetRingDim(n_slots * 2);
    CryptoContext<DCRTPoly> cc = GenCryptoContext(params);
    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);

    KeyPair<DCRTPoly> keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);

    int n_rots; fin >> n_rots;
    vector<int> rots;
    for (int i = 0; i < n_rots; ++i) {
        int x; fin >> x;
        if (x != 0) rots.push_back(x);
    }
    cc->EvalRotateKeyGen(keys.secretKey, rots);

    int nt; fin >> nt;
    vector<vector<Ciphertext<DCRTPoly>>> inputs(nt);
    for (int k = 0; k < nt; ++k) {
        for (int i = 0; i < 4; ++i) {
            vector<double> data(n_slots);
            for (double &num: data) fin >> num;
            inputs[k].push_back(cc->Encrypt(keys.publicKey, cc->MakeCKKSPackedPlaintext(data)));
        }
    }

    vector<vector<vector<Plaintext>>> gs1(nt, vector<vector<Plaintext>>(M));
    vector<vector<vector<Plaintext>>> gs2(M, vector<vector<Plaintext>>(nt));
    for (int k = 0; k < nt; ++k) {
        for (int m = 0; m < M; ++m) {
            for (int i = 0; i < 16; ++i) {
                vector<double> data(n_slots);
                for (double &num: data) fin >> num;
                gs1[k][m].push_back(cc->MakeCKKSPackedPlaintext(data));
            }
        }
    }
    for (int m = 0; m < M; ++m) {
        for (int k = 0; k < nt; ++k) {
            for (int i = 0; i < 16; ++i) {
                vector<double> data(n_slots);
                for (double &num: data) fin >> num;
                gs2[m][k].push_back(cc->MakeCKKSPackedPlaintext(data));
            }
        }
    }

    int tiles_per_row = W / 2, tiles_per_col = H / 2;
    int channels_per_cipher = tiles_per_row * tiles_per_col;

    auto cipherprint = [&](Ciphertext<DCRTPoly> &c) {
        Plaintext ptxt;
        cc->Decrypt(keys.secretKey, c, &ptxt);
        vector<double> values = ptxt->GetRealPackedValue();
    };

    auto plainprint = [&](Plaintext &p) {
        vector<double> values = p->GetRealPackedValue();
    };

    auto time_end_read = std::chrono::high_resolution_clock::now();
    auto time_duration_read = std::chrono::duration_cast<std::chrono::microseconds>(time_end_read - time_start_read);
    std::cerr << "Reading time: " << time_duration_read.count() / 1000 << "ms" << std::endl;
    auto sum_rows_keys = cc->EvalSumRowsKeyGen(keys.secretKey, nullptr, channels_per_cipher);

    // NOTE: begin
    auto time_start_fhe = std::chrono::high_resolution_clock::now();
    // NOTE: for nc
    vector<vector<Ciphertext<DCRTPoly>>> c(nt, vector<Ciphertext<DCRTPoly>>(16));
    for (int k = 0; k < nt; ++k) {
        auto ct_precomp_0 = cc->EvalFastRotationPrecompute(inputs[k][0]);
        c[k][5] = inputs[k][0];
        c[k][7] = cc->EvalFastRotation(inputs[k][0], 1, cc->GetCyclotomicOrder(), ct_precomp_0);
        c[k][13] = cc->EvalFastRotation(inputs[k][0], tiles_per_row, cc->GetCyclotomicOrder(), ct_precomp_0);
        c[k][15] = cc->EvalFastRotation(inputs[k][0], tiles_per_row + 1, cc->GetCyclotomicOrder(), ct_precomp_0);

        auto ct_precomp_1 = cc->EvalFastRotationPrecompute(inputs[k][1]);
        c[k][4] = cc->EvalFastRotation(inputs[k][1], -1, cc->GetCyclotomicOrder(), ct_precomp_1);
        c[k][6] = inputs[k][1];
        c[k][12] = cc->EvalFastRotation(inputs[k][1], tiles_per_row - 1, cc->GetCyclotomicOrder(), ct_precomp_1);
        c[k][14] = cc->EvalFastRotation(inputs[k][1], tiles_per_row, cc->GetCyclotomicOrder(), ct_precomp_1);

        auto ct_precomp_2 = cc->EvalFastRotationPrecompute(inputs[k][2]);
        c[k][1] = cc->EvalFastRotation(inputs[k][2], -tiles_per_row, cc->GetCyclotomicOrder(), ct_precomp_2);
        c[k][3] = cc->EvalFastRotation(inputs[k][2], -tiles_per_row + 1, cc->GetCyclotomicOrder(), ct_precomp_2);
        c[k][9] = inputs[k][2];
        c[k][11] = cc->EvalFastRotation(inputs[k][2], 1, cc->GetCyclotomicOrder(), ct_precomp_2);

        auto ct_precomp_3 = cc->EvalFastRotationPrecompute(inputs[k][3]);
        c[k][0] = cc->EvalFastRotation(inputs[k][3], -tiles_per_row - 1, cc->GetCyclotomicOrder(), ct_precomp_3);
        c[k][2] = cc->EvalFastRotation(inputs[k][3], -tiles_per_row, cc->GetCyclotomicOrder(), ct_precomp_3);
        c[k][8] = cc->EvalFastRotation(inputs[k][3], -1, cc->GetCyclotomicOrder(), ct_precomp_3);
        c[k][10] = inputs[k][3];
    }

    // faster than EvalLinearWSum
    vector<vector<Ciphertext<DCRTPoly>>> sums(M, vector<Ciphertext<DCRTPoly>>(4));
    for (int m = 0; m < M; ++m) {
        vector<vector<Ciphertext<DCRTPoly>>> per(4);
        for (int k = 0; k < nt; ++k) {
            vector<Ciphertext<DCRTPoly>> D(16);
            D[0] = cc->EvalMult(cc->EvalAddMany({c[k][0], cc->EvalNegate(c[k][2]), cc->EvalNegate(c[k][8]), c[k][10]}), gs1[k][m][0]);
            D[1] = cc->EvalMult(cc->EvalAddMany({c[k][1], c[k][2], cc->EvalNegate(c[k][9]), cc->EvalNegate(c[k][10])}), gs1[k][m][1]);
            D[2] = cc->EvalMult(cc->EvalAddMany({cc->EvalNegate(c[k][1]), c[k][2], c[k][9], cc->EvalNegate(c[k][10])}), gs1[k][m][2]);
            D[3] = cc->EvalMult(cc->EvalAddMany({c[k][1], cc->EvalNegate(c[k][3]), cc->EvalNegate(c[k][9]), c[k][11]}), gs1[k][m][3]);

            D[4] = cc->EvalMult(cc->EvalAddMany({c[k][4], cc->EvalNegate(c[k][6]), c[k][8], cc->EvalNegate(c[k][10])}), gs1[k][m][4]);
            D[5] = cc->EvalMult(cc->EvalAddMany({c[k][5], c[k][6], c[k][9], c[k][10]}), gs1[k][m][5]);
            D[6] = cc->EvalMult(cc->EvalAddMany({cc->EvalNegate(c[k][5]), c[k][6], cc->EvalNegate(c[k][9]), c[k][10]}), gs1[k][m][6]);
            D[7] = cc->EvalMult(cc->EvalAddMany({c[k][5], cc->EvalNegate(c[k][7]), c[k][9], cc->EvalNegate(c[k][11])}), gs1[k][m][7]);

            D[8] = cc->EvalMult(cc->EvalAddMany({cc->EvalNegate(c[k][4]), c[k][6], c[k][8], cc->EvalNegate(c[k][10])}), gs1[k][m][8]);
            D[9] = cc->EvalMult(cc->EvalAddMany({cc->EvalNegate(c[k][5]), cc->EvalNegate(c[k][6]), c[k][9], c[k][10]}), gs1[k][m][9]);
            D[10] = cc->EvalMult(cc->EvalAddMany({c[k][5], cc->EvalNegate(c[k][6]), cc->EvalNegate(c[k][9]), c[k][10]}), gs1[k][m][10]);
            D[11] = cc->EvalMult(cc->EvalAddMany({cc->EvalNegate(c[k][5]), c[k][7], c[k][9], cc->EvalNegate(c[k][11])}), gs1[k][m][11]);

            D[12] = cc->EvalMult(cc->EvalAddMany({c[k][4], cc->EvalNegate(c[k][6]), cc->EvalNegate(c[k][12]), c[k][14]}), gs1[k][m][12]);
            D[13] = cc->EvalMult(cc->EvalAddMany({c[k][5], c[k][6], cc->EvalNegate(c[k][13]), cc->EvalNegate(c[k][14])}), gs1[k][m][13]);
            D[14] = cc->EvalMult(cc->EvalAddMany({cc->EvalNegate(c[k][5]), c[k][6], c[k][13], cc->EvalNegate(c[k][14])}), gs1[k][m][14]);
            D[15] = cc->EvalMult(cc->EvalAddMany({c[k][5], cc->EvalNegate(c[k][7]), cc->EvalNegate(c[k][13]), c[k][15]}), gs1[k][m][15]);

            per[0].push_back(cc->EvalAddMany({D[0], D[1], D[2], D[4], D[5], D[6], D[8], D[9], D[10]}));
            per[1].push_back(cc->EvalAddMany({D[1], cc->EvalNegate(D[2]), cc->EvalNegate(D[3]), D[5], cc->EvalNegate(D[6]), cc->EvalNegate(D[7]), D[9], cc->EvalNegate(D[10]), cc->EvalNegate(D[11])}));
            per[2].push_back(cc->EvalAddMany({D[4], D[5], D[6], cc->EvalNegate(D[8]), cc->EvalNegate(D[9]), cc->EvalNegate(D[10]), cc->EvalNegate(D[12]), cc->EvalNegate(D[13]), cc->EvalNegate(D[14])}));
            per[3].push_back(cc->EvalAddMany({D[5], cc->EvalNegate(D[6]), cc->EvalNegate(D[7]), cc->EvalNegate(D[9]), D[10], D[11], cc->EvalNegate(D[13]), D[14], D[15]}));
        }
        for (int i = 0; i < 4; ++i) {
            Ciphertext<DCRTPoly> sum = cc->EvalAddMany(per[i]);
            sums[m][i] = cc->EvalSumRows(sum, channels_per_cipher, *sum_rows_keys);
        }
    }

    // NOTE: for nm
    vector<vector<vector<Ciphertext<DCRTPoly>>>> outputs(nt, vector<vector<Ciphertext<DCRTPoly>>>(4));
    for (int m = 0; m < M; ++m) {
        vector<Ciphertext<DCRTPoly>> c(16);
        auto ct_precomp_0 = cc->EvalFastRotationPrecompute(sums[m][0]);
        c[5] = sums[m][0];
        c[7] = cc->EvalFastRotation(sums[m][0], 1, cc->GetCyclotomicOrder(), ct_precomp_0);
        c[13] = cc->EvalFastRotation(sums[m][0], tiles_per_row, cc->GetCyclotomicOrder(), ct_precomp_0);
        c[15] = cc->EvalFastRotation(sums[m][0], tiles_per_row + 1, cc->GetCyclotomicOrder(), ct_precomp_0);

        auto ct_precomp_1 = cc->EvalFastRotationPrecompute(sums[m][1]);
        c[4] = cc->EvalFastRotation(sums[m][1], -1, cc->GetCyclotomicOrder(), ct_precomp_1);
        c[6] = sums[m][1];
        c[12] = cc->EvalFastRotation(sums[m][1], tiles_per_row - 1, cc->GetCyclotomicOrder(), ct_precomp_1);
        c[14] = cc->EvalFastRotation(sums[m][1], tiles_per_row, cc->GetCyclotomicOrder(), ct_precomp_1);

        auto ct_precomp_2 = cc->EvalFastRotationPrecompute(sums[m][2]);
        c[1] = cc->EvalFastRotation(sums[m][2], -tiles_per_row, cc->GetCyclotomicOrder(), ct_precomp_2);
        c[3] = cc->EvalFastRotation(sums[m][2], -tiles_per_row + 1, cc->GetCyclotomicOrder(), ct_precomp_2);
        c[9] = sums[m][2];
        c[11] = cc->EvalFastRotation(sums[m][2], 1, cc->GetCyclotomicOrder(), ct_precomp_2);

        auto ct_precomp_3 = cc->EvalFastRotationPrecompute(sums[m][3]);
        c[0] = cc->EvalFastRotation(sums[m][3], -tiles_per_row - 1, cc->GetCyclotomicOrder(), ct_precomp_3);
        c[2] = cc->EvalFastRotation(sums[m][3], -tiles_per_row, cc->GetCyclotomicOrder(), ct_precomp_3);
        c[8] = cc->EvalFastRotation(sums[m][3], -1, cc->GetCyclotomicOrder(), ct_precomp_3);
        c[10] = sums[m][3];

        for (int k = 0; k < nt; ++k) {
            vector<Ciphertext<DCRTPoly>> D(16);
            D[0] = cc->EvalMult(cc->EvalAddMany({c[0], cc->EvalNegate(c[2]), cc->EvalNegate(c[8]), c[10]}), gs2[m][k][0]);
            D[1] = cc->EvalMult(cc->EvalAddMany({c[1], c[2], cc->EvalNegate(c[9]), cc->EvalNegate(c[10])}), gs2[m][k][1]);
            D[2] = cc->EvalMult(cc->EvalAddMany({cc->EvalNegate(c[1]), c[2], c[9], cc->EvalNegate(c[10])}), gs2[m][k][2]);
            D[3] = cc->EvalMult(cc->EvalAddMany({c[1], cc->EvalNegate(c[3]), cc->EvalNegate(c[9]), c[11]}), gs2[m][k][3]);

            D[4] = cc->EvalMult(cc->EvalAddMany({c[4], cc->EvalNegate(c[6]), c[8], cc->EvalNegate(c[10])}), gs2[m][k][4]);
            D[5] = cc->EvalMult(cc->EvalAddMany({c[5], c[6], c[9], c[10]}), gs2[m][k][5]);
            D[6] = cc->EvalMult(cc->EvalAddMany({cc->EvalNegate(c[5]), c[6], cc->EvalNegate(c[9]), c[10]}), gs2[m][k][6]);
            D[7] = cc->EvalMult(cc->EvalAddMany({c[5], cc->EvalNegate(c[7]), c[9], cc->EvalNegate(c[11])}), gs2[m][k][7]);

            D[8] = cc->EvalMult(cc->EvalAddMany({cc->EvalNegate(c[4]), c[6], c[8], cc->EvalNegate(c[10])}), gs2[m][k][8]);
            D[9] = cc->EvalMult(cc->EvalAddMany({cc->EvalNegate(c[5]), cc->EvalNegate(c[6]), c[9], c[10]}), gs2[m][k][9]);
            D[10] = cc->EvalMult(cc->EvalAddMany({c[5], cc->EvalNegate(c[6]), cc->EvalNegate(c[9]), c[10]}), gs2[m][k][10]);
            D[11] = cc->EvalMult(cc->EvalAddMany({cc->EvalNegate(c[5]), c[7], c[9], cc->EvalNegate(c[11])}), gs2[m][k][11]);

            D[12] = cc->EvalMult(cc->EvalAddMany({c[4], cc->EvalNegate(c[6]), cc->EvalNegate(c[12]), c[14]}), gs2[m][k][12]);
            D[13] = cc->EvalMult(cc->EvalAddMany({c[5], c[6], cc->EvalNegate(c[13]), cc->EvalNegate(c[14])}), gs2[m][k][13]);
            D[14] = cc->EvalMult(cc->EvalAddMany({cc->EvalNegate(c[5]), c[6], c[13], cc->EvalNegate(c[14])}), gs2[m][k][14]);
            D[15] = cc->EvalMult(cc->EvalAddMany({c[5], cc->EvalNegate(c[7]), cc->EvalNegate(c[13]), c[15]}), gs2[m][k][15]);

            outputs[k][0].push_back(cc->EvalAddMany({D[0], D[1], D[2], D[4], D[5], D[6], D[8], D[9], D[10]}));
            outputs[k][1].push_back(cc->EvalAddMany({D[1], cc->EvalNegate(D[2]), cc->EvalNegate(D[3]), D[5], cc->EvalNegate(D[6]), cc->EvalNegate(D[7]), D[9], cc->EvalNegate(D[10]), cc->EvalNegate(D[11])}));
            outputs[k][2].push_back(cc->EvalAddMany({D[4], D[5], D[6], cc->EvalNegate(D[8]), cc->EvalNegate(D[9]), cc->EvalNegate(D[10]), cc->EvalNegate(D[12]), cc->EvalNegate(D[13]), cc->EvalNegate(D[14])}));
            outputs[k][3].push_back(cc->EvalAddMany({D[5], cc->EvalNegate(D[6]), cc->EvalNegate(D[7]), cc->EvalNegate(D[9]), D[10], D[11], cc->EvalNegate(D[13]), D[14], D[15]}));
        }
    }
    vector<vector<Ciphertext<DCRTPoly>>> final(nt, vector<Ciphertext<DCRTPoly>>(4));
    for (int k = 0; k < nt; ++k) {
        for (int i = 0; i < 4; ++i) {
            final[k][i] = cc->EvalAddMany(outputs[k][i]);
        }
    }

    auto time_end_fhe = std::chrono::high_resolution_clock::now();
    auto time_duration_fhe = std::chrono::duration_cast<std::chrono::microseconds>(time_end_fhe - time_start_fhe);
    std::cerr << "Running time: " << time_duration_fhe.count() / 1000 << "ms" << std::endl;

    auto check = [&](string filename) {
        ifstream ref(filename);
        double mae = 0, max_err = 0;
        vector<double> precisions;
        for (int k = 0; k < nt; ++k) {
            for (int i = 0; i < 4; ++i) {
                Plaintext ptxt;
                cc->Decrypt(keys.secretKey, final[k][i], &ptxt);
                precisions.push_back(ptxt->GetLogPrecision());
                vector<double> values = ptxt->GetRealPackedValue();
                for (int j = 0; j < n_slots; ++j) {
                    double sol; ref >> sol;
                    double out = values[j];
                    mae += abs(out - sol);
                    max_err = max(max_err, abs(out - sol));
                }
            }
        }
        double avg_precision = accumulate(precisions.begin(), precisions.end(), 0.0) / precisions.size();
        cerr << "MAE: " << mae / (n_slots * (int)final.size()) << ", " << "MAX_ERR: " << max_err << ", avg precisions: " << avg_precision << endl;
    };
    check("../pack/winograd-ref.txt");
}

int main() {
    // orion();
    winograd();
}
