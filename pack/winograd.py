#!/home/jyh/project/conv-chain/test-venv/bin/python3
# run  := python3 winograd.py
# dir  := .
# kid  :=
# alternate nested loop order

from __future__ import annotations
import numpy as np
import torch
from typing import List
import math

class Poly:
    N: int = 0
    recording: bool = False
    num_add: int = 0 # vector addition
    num_rot: int = 0 # vector rotation
    num_vxv: int = 0 # vector * vector
    num_vxs: int = 0 # vector * scalar

    sum_num_add: int = 0 # vector addition
    sum_num_rot: int = 0 # vector rotation
    sum_num_vxv: int = 0 # vector * vector
    sum_num_vxs: int = 0 # vector * scalar

    def __init__(self, coef: np.ndarray=np.array([])):
        assert len(coef.shape) == 1, "Coefficient must be a 1D array."
        assert coef.shape[0] <= Poly.N
        self.coef = np.pad(coef, (0, Poly.N - len(coef)), mode="constant")

    def __add__(self, rhs: Poly) -> Poly:
        assert isinstance(rhs, Poly)
        if Poly.recording:
            Poly.num_add += 1
        return Poly(self.coef + rhs.coef)

    def __sub__(self, rhs: Poly) -> Poly:
        assert isinstance(rhs, Poly)
        if Poly.recording:
            Poly.num_add += 1
        return Poly(self.coef - rhs.coef)

    def __mul__(self, rhs: float | Poly) -> Poly:
        if isinstance(rhs, float):
            if Poly.recording:
                Poly.num_vxs += 1
            return Poly(self.coef * rhs)
        elif isinstance(rhs, Poly):
            if Poly.recording:
                Poly.num_vxv += 1
            return Poly(self.coef * rhs.coef)
        else:
            assert False

    def rotate(self, offsets: List[int]) -> List[Poly]:
        for off in offsets:
            assert off > -Poly.N and off < Poly.N
        if Poly.recording:
            Poly.num_rot += 1
        return [
            Poly(np.roll(self.coef, off))
            for off in offsets
        ]

    def __neg__(self):
        return Poly(-self.coef)

    def __str__(self) -> str:
        return f"{self.coef}"

    @staticmethod
    def flush():
        Poly.sum_num_add += Poly.num_add
        Poly.sum_num_rot += Poly.num_rot
        Poly.sum_num_vxv += Poly.num_vxv
        Poly.sum_num_vxs += Poly.num_vxs
        Poly.num_add = 0
        Poly.num_rot = 0
        Poly.num_vxv = 0
        Poly.num_vxs = 0

    @staticmethod
    def statistics(sum: bool = False) -> str:
        if sum:
            Poly.flush()
            return f"N={Poly.N}, num_add={Poly.sum_num_add}, num_rot={Poly.sum_num_rot}, num_vxv={Poly.sum_num_vxv}, num_vxs={Poly.sum_num_vxs}"
        else:
            return f"N={Poly.N}, num_add={Poly.num_add}, num_rot={Poly.num_rot}, num_vxv={Poly.num_vxv}, num_vxs={Poly.num_vxs}"

class Ref:
    def __init__(self, input: torch.Tensor):
        self.input = input
        self.kernels = []

    def conv2d(self, kernel: torch.Tensor):
        self.kernels.append(kernel)

    def run(self) -> torch.Tensor:
        tensor = self.input
        for kernel in self.kernels:
            tensor = torch.nn.functional.conv2d(tensor, kernel, padding=1)
        return tensor

class Winograd:
    B = np.array([[1, 0, -1, 0], [0, 1, 1, 0], [0, -1, 1, 0], [0, 1, 0, -1]], dtype=np.float32)
    G = np.array([[1, 0, 0], [0.5, 0.5, 0.5], [0.5, -0.5, 0.5], [0, 0, 1]], dtype=np.float32)
    A = np.array([[1, 1, 1, 0], [0, 1, -1, -1]], dtype=np.float32)
    B_kron = np.kron(B, B)
    G_kron = np.kron(G, G)
    A_kron = np.kron(A, A)

    # weights is a sparse vector of {-1, 0, 1}
    # returns the inner product of weights and operands
    @staticmethod
    def inner01(weights, operands):
        weights = np.array(weights, dtype=np.int8)
        assert np.isin(weights, (-1, 0, 1)).all()
        idx = np.flatnonzero(weights)
        pairs = list(zip(idx, weights[idx]))
        i, w = pairs.pop(0)
        sum = operands[i] if w == 1 else -operands[i]
        while pairs:
            i, w = pairs.pop(0)
            sum = sum + operands[i] if w == 1 else sum - operands[i]
        return sum

    def __init__(self, img: torch.Tensor, Poly_N: int):
        Poly.N = Poly_N
        img = torch.nn.functional.pad(img, (0, 2, 0, 2), mode="constant", value=0)
        _, channels, self.h, self.w = img.shape
        self.channels, self.C, self.M = channels, channels, channels
        self.tiles_per_row = self.w // 2
        self.tiles_per_col = self.h // 2
        self.channels_per_cipher = math.ceil(Poly.N / (self.tiles_per_row * self.tiles_per_col))
        self.Nt = math.ceil(self.channels / self.channels_per_cipher)
        inputs = []

        for c in range(self.Nt):
            cipher = [[], [], [], []]
            for channel in range(c*self.channels_per_cipher, (c+1)*self.channels_per_cipher):
                for i in range(0, self.h, 2):
                    for j in range(0, self.w, 2):
                        for k in range(4):
                            x = i + (k // 2)
                            y = j + (k % 2)
                            cipher[k].append(float(img[0][channel][x][y]))
            inputs.append([Poly(np.array(i)) for i in cipher])
        self.G_tilde_in = []
        self.G_tilde_out = []
        self.n_layers = 0
        self.inputs = inputs

        mask_padding = np.zeros((self.tiles_per_col, self.tiles_per_row))
        mask_padding[:-1, :-1] = 1
        mask_padding = mask_padding.ravel()
        self.mask_padding = np.tile(mask_padding, self.channels_per_cipher)


    def conv2d(self, kernel: torch.Tensor):
        self.n_layers += 1
        out_channels, in_channels, _, _ = kernel.shape
        kf_in = kernel.numpy().reshape(out_channels * in_channels, 3*3, 1)
        self.G_tilde_in.append(
            (Winograd.G_kron @ kf_in).reshape(self.M, self.Nt, self.channels_per_cipher, 16)
        )
        kf_out = np.transpose(kernel.numpy(), (1, 0, 2, 3)).reshape(in_channels * out_channels, 3*3, 1)
        self.G_tilde_out.append(
            (Winograd.G_kron @ kf_out).reshape(self.C, self.Nt, self.channels_per_cipher, 16)
        )

    def run(self) -> torch.Tensor:
        Poly.recording = True

        def reduction(v):
            off = self.channels_per_cipher // 2
            while off > 0:
                t = v.rotate([-off * N])[0]
                v = v + t
                off //= 2
            return v

        rots = list(set([0, -1, -self.tiles_per_row, -self.tiles_per_row - 1, 1, 0, -self.tiles_per_row + 1, -self.tiles_per_row, self.tiles_per_row, self.tiles_per_row - 1, 0, -1, self.tiles_per_row + 1, self.tiles_per_row, 1, 0]))
        rots = [-i for i in rots]
        with open("winograd-pack.txt", "a") as f:
            print(len(rots), file=f)
            for i in rots:
                print(i, end=" ", file=f)
            print(file=f)

        with open("winograd-pack.txt", "a") as f:
            print(len(self.inputs), file=f)
            for input in self.inputs:
                for p in input:
                    for num in p.coef:
                        print(num, file=f, end=" ")
                    print(file=f)

        for layer in range(self.n_layers):
            N = self.tiles_per_row * self.tiles_per_col

            def for_nc_nht_nwt():
                g = self.G_tilde_in[layer]
                mask = []
                for i in range(self.channels_per_cipher):
                    t = np.concatenate([np.zeros(i*N, dtype=int), np.ones(N, dtype=int)])
                    mask.append(Poly(t))

                all = [[], [], [], []]
                for it, input in enumerate(self.inputs):
                    c5, c7, c13, c15 = input[0].rotate([0, -1, -self.tiles_per_row, -self.tiles_per_row - 1])
                    c4, c6, c12, c14 = input[1].rotate([1, 0, -self.tiles_per_row + 1, -self.tiles_per_row])
                    c1, c3, c9, c11 = input[2].rotate([self.tiles_per_row, self.tiles_per_row - 1, 0, -1])
                    c0, c2, c8, c10 = input[3].rotate([self.tiles_per_row + 1, self.tiles_per_row, 1, 0])
                    c = [c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15]

                    for m in range(self.channels):
                        D_tilde = []
                        for i in range(16):
                            gs = np.repeat(g[m, it, :, i].flatten(), N) * self.mask_padding
                            b = Winograd.inner01(Winograd.B_kron[i], c)
                            with open("winograd-pack.txt", "a") as f:
                                for num in gs:
                                    print(num, end=" ", file=f)
                                print(file=f)
                            D_tilde.append(Poly(gs) * b)

                        out = [
                            Winograd.inner01(Winograd.A_kron[i], D_tilde)
                            for i in range(4)
                        ]
                        for i in range(4):
                            all[i].append(out[i])

                sums = []
                for i in range(self.channels):
                    sums.append([
                        reduction(Winograd.inner01(
                            np.ones(self.channels // self.channels_per_cipher),
                            all[j][i::self.channels][:(self.channels // self.channels_per_cipher)]
                        ))
                        for j in range(4)
                    ])
                self.inputs = sums

            def for_nm_nht_nwt():
                g = self.G_tilde_out[layer]
                sums = []
                for it, input in enumerate(self.inputs):
                    c5, c7, c13, c15 = input[0].rotate([0, -1, -self.tiles_per_row, -self.tiles_per_row - 1])
                    c4, c6, c12, c14 = input[1].rotate([1, 0, -self.tiles_per_row + 1, -self.tiles_per_row])
                    c1, c3, c9, c11 = input[2].rotate([self.tiles_per_row, self.tiles_per_row - 1, 0, -1])
                    c0, c2, c8, c10 = input[3].rotate([self.tiles_per_row + 1, self.tiles_per_row, 1, 0])
                    c = [c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15]
                    for nt in range(0, self.channels // self.channels_per_cipher):
                        D_tilde = []
                        for i in range(16):
                            gs = np.repeat(g[it, nt, :, i].flatten(), N) * self.mask_padding
                            with open("winograd-pack.txt", "a") as f:
                                for num in gs:
                                    print(num, end=" ", file=f)
                                print(file=f)
                            b = Winograd.inner01(Winograd.B_kron[i], c)
                            D_tilde.append(Poly(gs) * b)
                        out = [
                            Winograd.inner01(Winograd.A_kron[i], D_tilde)
                            for i in range(4)
                        ]
                        if it == 0:
                            sums.append(out)
                        else:
                            for j in range(4):
                                sums[nt][j] += out[j]
                self.inputs = sums

            if layer % 2 == 0:
                for_nc_nht_nwt()
            else:
                for_nm_nht_nwt()

            print(f"layer {layer}: {Poly.statistics()}")
            Poly.flush()

            with open("winograd-ref.txt", "w") as f:
                for k in range(self.channels // self.channels_per_cipher):
                    for i in range(4):
                        for num in self.inputs[k][i].coef:
                            print(num, file=f)

            if layer == len(self.G_tilde_in) - 1:
                Poly.recording = False
                h_out, w_out = self.h - 2, self.w - 2
                mat = np.zeros((self.channels, h_out, w_out))
                for c in range(self.channels):
                    nc = c // self.channels_per_cipher
                    nidx = c % self.channels_per_cipher
                    for k in range(4):
                        for i in range(h_out // 2):
                            for j in range(w_out // 2):
                                x = 2*i + (k // 2)
                                y = 2*j + (k % 2)
                                mat[c][x][y] = self.inputs[nc][k].coef[nidx*self.tiles_per_row*self.tiles_per_col + i*self.tiles_per_row + j]
                return torch.tensor(mat)
        assert False, "unreachable"

if __name__ == "__main__":
    np.random.seed(42)

    n_channels = 32
    input_size = 32
    Poly_N = 2**13

    Poly.N = Poly_N
    H, W = input_size, input_size
    C, M = n_channels, n_channels
    R, S = 3, 3

    with open("winograd-pack.txt", "w") as f:
        print(Poly_N, H, W, C, M, R, S, file=f)

    num_layers = 2
    img = np.random.rand(1, M, H-2, W-2)
    kernels = [np.random.rand(M, C, 3, 3) for _ in range(num_layers)]

    ref = Ref(torch.tensor(img))
    for kernel in kernels:
        ref.conv2d(torch.tensor(kernel))
    sol = ref.run()
    # print(sol[0])

    winograd = Winograd(torch.tensor(img), Poly_N)
    for kernel in kernels:
        winograd.conv2d(torch.tensor(kernel))
    my = winograd.run()
    # print(my)

    print(np.allclose(my, sol[0], atol=1e-9))
    print(Poly.statistics(True))

    def estimate():
        def winograd():
            N_T = (H//2 * W//2 * C) // Poly.N
            num_adds = (
                M * N_T * (16*(4-1) + 4*(9-1)) +
                M * 4 * (int(math.log2(C // N_T)) + (N_T - 1)) +
                4 * N_T * (M//N_T - 1)
            )
            # num_rots = 4*N_T + 4*M*int(math.log2(C//N_T))
            num_rots = ((4*N_T + 4*M*int(math.log2(C//N_T))) + (4*M)) // 2
            # num_vxvs = N_T * M * 16 + 4 * M
            num_vxvs = N_T * M * 16
            # print(f"N={Poly.N}, num_add={num_layers * num_adds}, num_rot={num_layers * num_rots}, num_vxv={num_layers * num_vxvs}")
            print(f"WINO: N={Poly.N}, num_add={num_adds}, num_rot={num_rots}, num_vxv={num_vxvs}")
        def bsgs():
            num_adds = (Poly.N // (H*W) * R * S - 1) * (H*W*C//Poly.N) * (H*W*M//Poly.N)
            num_rots = (H*W*M//Poly.N) * 2*int(math.sqrt(R*S*(Poly.N//(H*W))))
            num_vxvs = H*W*C*M*R*S//Poly.N
            print(f"BSGS: N={Poly.N}, num_add={num_adds}, num_rot={num_rots}, num_vxv={num_vxvs}")

        winograd()
        bsgs()

    estimate()
