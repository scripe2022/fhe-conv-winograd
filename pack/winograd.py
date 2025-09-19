#!/home/jyh/project/openfhe-conv/openfhe-venv/bin/python3
# run  := python3 winograd.py
# dir  := .
# kid  :=

import math
import time

import numpy as np
import scipy.sparse as sp
import torch
import tqdm
import h5py

def construct_conv2d_winograd(G_tilde, NHt, NWt):
    M, C = G_tilde.shape[:2]
    N = NHt * NWt
    data = []
    rows = []
    cols = []
    for m in range(M):
        for r in range(N):
            rows.extend([m*N + r] * C)
            cols.extend(np.arange(r, N*C, N).tolist())
            if ((r+1) % NWt == 0 or r >= (NHt-1)*NWt):
                data.extend((np.zeros(C, dtype=np.int32)-1).tolist())
            else:
                data.extend(np.arange(m*C, (m+1)*C).tolist())
    mat = sp.csr_matrix((data, (rows, cols)), shape=(M*N, N*C))
    return mat

# credit: orion
def diagonalize(
    matrix: sp.csr_matrix,
    num_slots: int,
    embed_method: str,
    is_last_layer: bool,
):
    assert matrix.shape is not None
    matrix_height, matrix_width = matrix.shape[0], matrix.shape[1]
    num_block_rows = math.ceil(matrix_height / num_slots)
    num_block_cols = math.ceil(matrix_width / num_slots)
    print(f"├── embed method: {embed_method}")
    print(f"├── original matrix shape: {matrix.shape}")
    print(f"├── # blocks (rows, cols) = {(num_block_rows, num_block_cols)}")

    if num_block_rows == 1 and embed_method == "hybrid" and not is_last_layer:
        block_height = 2 ** math.ceil(math.log2(matrix_height))
        output_rotations = int(math.log2(num_slots // block_height))
    else:
        block_height = num_slots
        output_rotations = 0

    # Inflate dimensions of the sparse matrix
    matrix.resize(num_block_rows * block_height, num_block_cols * num_slots)

    print(f"├── resized matrix shape: {matrix.shape}")
    print(f"├── # output rotations: {output_rotations}")

    # Prepare indices for diagonal extraction 
    row_idx = torch.arange(block_height).repeat(num_slots // block_height)
    col_idx = torch.arange(block_height)[:, None] + torch.arange(num_slots)[None, :]
    col_idx = torch.where(col_idx >= num_slots, col_idx - num_slots, col_idx)

    diagonals_by_block = {}
    total_diagonals = 0

    # Process each block 
    progress_bar = tqdm.tqdm(
        total=num_block_rows * num_block_cols,
        desc="|    Processing blocks",
        leave=False,
    )
    start_time = time.time()
    for block_row in range(num_block_rows):
        for block_col in range(num_block_cols):
            row_start = num_slots * block_row
            col_start = num_slots * block_col
            block_sparse = matrix[
                row_start: row_start + block_height,
                col_start: col_start + num_slots,
            ]
            block_dense = torch.tensor(block_sparse.todense(), dtype=torch.float32)
            block_diagonals = block_dense[row_idx, col_idx]

            # Collect non-zero diagonals
            nonzero_diagonals = {}
            for i in range(block_height):
                if torch.any(block_diagonals[i]):
                    nonzero_diagonals[i] = block_diagonals[i].tolist()

            total_diagonals += len(nonzero_diagonals)
            diagonals_by_block[(block_row, block_col)] = (
                nonzero_diagonals or {0: [0.0] * num_slots}
            )

            progress_bar.set_postfix({
                "Current Block": f"({block_row},{block_col})",
                "Total Diagonals": total_diagonals,
            })
            progress_bar.update(1)

    progress_bar.close()
    elapsed_time = time.time() - start_time
    print(f"├── time to pack (s): {elapsed_time:.2f}")
    print(f"├── # diagonals = {total_diagonals}")

    return diagonals_by_block, num_block_rows, num_block_cols, output_rotations

def binary_search(diags, n_slots):
    s = set()
    for _, block in diags.items():
        for i in block.keys():
            s.add(i)
    left = 1
    right = n_slots
    mid = -1
    bs = set()
    gs = set()
    while left < right:
        mid = (left + right) // 2
        bs.clear()
        gs.clear()
        for i in s:
            bs.add(i % mid)
            gs.add((i // mid) * mid)
        if len(bs) == len(gs):
            break
        elif len(bs) > len(gs):
            right = mid - 1
        else:
            left = mid + 1
    return mid

def orion_search(offsets, n_slots):
    n1 = 1
    while n1 < n_slots:
        bs = set()
        gs = set()
        for d in offsets:
            bs.add(d % n1)
            gs.add((d // n1) * n1)
        bs.discard(0)
        gs.discard(0)
        if len(bs) == len(gs):
            return n1
        if len(bs) > len(gs):
            return n1 // 2
        n1 *= 2

def pad_to_multiple_cipher(x, n_slots, fill=0):
    x = np.asarray(x.flatten())
    if x.ndim != 1:
        raise ValueError("must be 1D array")
    if n_slots <= 0:
        raise ValueError("n_slots must be positive")
    n = x.shape[0]
    pad = (-n) % n_slots
    if pad == 0:
        return x
    return np.pad(x, (0, pad), mode="constant", constant_values=fill)

def rotate(arr, *args):
    return [
        np.roll(arr, -offset)
        for offset in args
    ]

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

if __name__ == "__main__":
    H, W = 32, 32
    C, M = 32, 32
    n_slots = 2**13

    H, W = 58, 58
    C, M = 32, 32
    n_slots = 2**13

    R, S = 3, 3
    Ht, Wt = 2, 2
    NHt, NWt = H // Ht, W // Wt

    B = np.array([[1, 0, -1, 0], [0, 1, 1, 0], [0, -1, 1, 0], [0, 1, 0, -1]], dtype=np.float32)
    G = np.array([[1, 0, 0], [0.5, 0.5, 0.5], [0.5, -0.5, 0.5], [0, 0, 1]], dtype=np.float32)
    A = np.array([[1, 1, 1, 0], [0, 1, -1, -1]], dtype=np.float32)
    B_kron = np.kron(B, B)
    G_kron = np.kron(G, G)
    A_kron = np.kron(A, A)

    np.random.seed(42)
    img = np.random.rand(1, C, H, W)[:, :, :H-Ht, :W-Wt]
    kernel = np.random.rand(M, C, 3, 3)
    ref = torch.nn.functional.conv2d(torch.tensor(img), torch.tensor(kernel), padding=1)
    img = torch.nn.functional.pad(torch.tensor(img), (0, 2, 0, 2), mode="constant", value=0).numpy()

    inputs = [
        img[:, :, ::2, ::2].reshape(-1),
        img[:, :, ::2, 1::2].reshape(-1),
        img[:, :, 1::2, ::2].reshape(-1),
        img[:, :, 1::2, 1::2].reshape(-1)
    ]
    ref = torch.nn.functional.pad(ref, (0, 2, 0, 2), mode="constant", value=0).numpy()
    ref = [
        ref[:, :, ::2, ::2].reshape(-1),
        ref[:, :, ::2, 1::2].reshape(-1),
        ref[:, :, 1::2, ::2].reshape(-1),
        ref[:, :, 1::2, 1::2].reshape(-1)
    ]

    input2c_target = np.array([
        [5, 7, 13, 15],
        [4, 6, 12, 14],
        [1, 3, 9, 11],
        [0, 2, 8, 10]
    ])
    input2c_offset = np.array([
        [0, 1, NWt, NWt+1],
        [-1, 0, NWt-1, NWt],
        [-NWt, -NWt+1, 0, 1],
        [-NWt-1, -NWt, -1, 0]
    ])

    G_tilde = np.matmul(G_kron[:, :], kernel.reshape(M, C, -1)[..., None]).squeeze(-1)

    mat_base = construct_conv2d_winograd(G_tilde, NHt, NWt)
    mat_base_data = [int(i) for i in mat_base.data]
    mats = []

    diags, T_in, T_out, __ = diagonalize(mat_base, n_slots, "", False)
    offsets = set()
    for k, v in diags.items():
        offsets.update(v.keys())
    n1 = binary_search(diags, n_slots)

    global_rots = set(list(input2c_offset.flatten()))
    block_rots = [[(set(), set()) for _ in range(T_in)] for _ in range(T_out)]
    for idx, block in diags.items():
        tx, ty = idx
        for d in block.keys():
            bs = d % n1
            gs = (d // n1) * n1
            global_rots.add(bs)
            global_rots.add(gs)
            block_rots[tx][ty][0].add(bs)
            block_rots[tx][ty][1].add(gs)

    v_bs = [set() for _ in range(T_in)]
    row_gs = [set() for _ in range(T_out)]
    for i in range(T_out):
        for j in range(T_in):
            v_bs[j].update(block_rots[i][j][0])
            row_gs[i].update(block_rots[i][j][1])
    for j in range(T_in):
        print(len(v_bs[j]))
    print()
    for i in range(T_out):
        print(len(row_gs[i]))

    with h5py.File("winograd.h5", "w") as f:
        f.attrs["H"] = np.int32(H)
        f.attrs["W"] = np.int32(W)
        f.attrs["C"] = np.int32(C)
        f.attrs["M"] = np.int32(M)
        f.attrs["R"] = np.int32(R)
        f.attrs["S"] = np.int32(S)
        f.attrs["n_slots"] = np.int32(n_slots)
        f.attrs["T_in"] = np.int32(T_in)
        f.attrs["T_out"] = np.int32(T_out)
        f.attrs["Ht"] = np.int32(Ht)
        f.attrs["Wt"] = np.int32(Wt)

        global_group = f.create_group("global")
        global_group.create_dataset("rotations", data=np.array(sorted(global_rots), dtype=np.int32))
        for i in range(T_in):
            global_group.create_dataset(f"bs{i}", data=np.array(sorted(v_bs[i]), dtype=np.int32))
        for i in range(len(inputs)):
            global_group.create_dataset(f"input_{i}", data=pad_to_multiple_cipher(inputs[i], n_slots).reshape(-1, n_slots), dtype=np.float32)

        global_group.create_dataset("input2c_target", data=input2c_target, dtype=np.int32)
        global_group.create_dataset("input2c_offset", data=input2c_offset, dtype=np.int32)
        global_group.create_dataset("BT", data=B_kron, dtype=np.int32)
        global_group.create_dataset("AT", data=A_kron, dtype=np.int32)
        for i in range(len(ref)):
            global_group.create_dataset(f"ref_{i}", data=pad_to_multiple_cipher(ref[i], n_slots).reshape(-1, n_slots), dtype=np.float32)

        blocks_group = f.create_group("blocks")
        for i in range(T_out):
            for j in range(T_in):
                block = diags.get((i, j), {})
                bs = [offset % n1 for offset in block.keys()]
                gs = [(offset // n1) * n1 for offset in block.keys()]

                vecs_base = np.array(list(block.values()))
                mask = vecs_base != -1
                for k in range(16):
                    vecs = np.zeros_like(vecs_base)
                    vecs[mask] = G_tilde.reshape(M*C, -1)[vecs_base[mask].astype(int), k]
                    vecs_rots = [np.roll(vecs[k], gs[k]) for k in range(len(vecs))]
                    blocks_group.create_dataset(f"diags_{k}_{i}_{j}", data=np.array(vecs_rots, dtype=np.float32))

                blocks_group.create_dataset(f"bs_{i}_{j}", data=np.array(bs, dtype=np.int32))
                blocks_group.create_dataset(f"gs_{i}_{j}", data=np.array(gs, dtype=np.int32))

