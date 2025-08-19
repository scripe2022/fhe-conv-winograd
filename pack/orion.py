#!/home/jyh/project/openfhe-conv/openfhe-venv/bin/python3
# run  := python3 orion.py
# dir  := .
# kid  :=

import math
import time

import numpy as np
import scipy.sparse as sp
import torch
import tqdm
import h5py

# credit: orion
def construct_conv2d_toeplitz(weight, H, W):
    N = 1
    on_Co, on_Ci = weight.shape[:2]
    on_Hi, on_Wi = H, W
    on_Ho, on_Wo = H, W
    Ho, Wo = H, W
   
    P = 1
    D = 1
    iG = 1
    oG = 1
    kW, kH = weight.shape[2:]

    def compute_first_kernel_position():
        mpx_anchors = valid_image_indices[:, :iG, :iG].reshape(-1, 1)

        row_idxs = torch.arange(0, kH*D*iG, D*iG).reshape(-1, 1)
        col_idxs = torch.arange(0, kW*D*iG, D*iG)
        kernel_offsets = valid_image_indices[0, row_idxs, col_idxs].flatten()
        
        img_pixels_touched = mpx_anchors + kernel_offsets
        return img_pixels_touched.flatten()
    
    def compute_row_interchange_map():
        output_indices = torch.arange(on_Ho * on_Wo).reshape(on_Ho, on_Wo)
        
        start_indices = output_indices[:oG, :oG].flatten()
        corner_indices = output_indices[0:(Ho*oG):oG, 0:(Wo*oG):oG].reshape(-1, 1)
        return corner_indices + start_indices
    
    # Padded input dimensions with multiplexing
    Hi_pad = on_Hi + 2*P*iG 
    Wi_pad = on_Wi + 2*P*iG

    # Initialize our sparse Toeplitz matrix
    n_rows = on_Co * on_Ho * on_Wo
    n_cols = on_Ci * Hi_pad * Wi_pad
    toeplitz = sp.lil_matrix((n_rows, n_cols), dtype="f")

    # Create an index grid for the padded input image.
    valid_image_indices = torch.arange(n_cols).reshape(on_Ci, Hi_pad, Wi_pad)

    # Pad the kernel's input and output channels to the nearest multiple
    # of gap^2 to ensure that multiplexing works.
    kernel = torch.zeros(on_Co * oG**2, on_Ci * iG**2, kW, kH) 
    kernel[:weight.shape[0], :weight.shape[1], ...] = weight

    # All the indices the kernel initially touches
    initial_kernel_position = compute_first_kernel_position()

    # Create our row-interchange map that dictates how we permute rows for 
    # optimal packing. Also return all indices that the first top-left filter 
    # value touches throughout the convolution.
    row_map = compute_row_interchange_map()
    corner_indices = valid_image_indices[0, 0:(Ho*oG):oG, 0:(Wo*oG):oG].flatten() 

    # Create offsets for the multiplexed output channels.
    out_channels = (torch.arange(on_Co) * (on_Ho * on_Wo)).reshape(on_Co, 1)

    # Flattened kernel populates rows of our Toeplitz matrix
    kernel_flat = kernel.reshape(kernel.shape[0], -1)

    # Iterate over all positions that the top-left kernel element can touch 
    # populating the correct (permuted) rows of our Toeplitz matrix.
    for i, start_idx in enumerate(corner_indices):
        rows = (row_map[i] + out_channels).reshape(-1, 1)
        cols = initial_kernel_position + start_idx
        toeplitz[rows.numpy(), cols.numpy()] = kernel_flat

    # Keep only the columns corresponding to the non-padded input image.
    row_idxs = torch.arange(P*iG, P*iG + on_Hi).reshape(-1, 1)
    col_idxs = torch.arange(P*iG, P*iG + on_Wi)
    image_indices = valid_image_indices[:, row_idxs, col_idxs].flatten()
    toeplitz = toeplitz.tocsc()[:, image_indices.numpy()]

    # Support batching
    toeplitz = sp.kron(sp.eye(N, dtype="f"), toeplitz, format="csr")
    return toeplitz

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

if __name__ == "__main__":
    H, W = 32, 32
    C, M = 32, 32
    R, S = 3, 3
    n_slots = 2**13

    np.random.seed(42)
    input = np.random.rand(1, C, H, W)
    kernel = np.random.rand(M, C, 3, 3)
    ref = torch.nn.functional.conv2d(torch.tensor(input), torch.tensor(kernel), padding=1)

    toeplitz = construct_conv2d_toeplitz(torch.tensor(kernel), H, W)
    diags, T_in, T_out, __ = diagonalize(toeplitz, n_slots, "", False)

    global_rots = set()
    block_rots = [[(set(), set()) for _ in range(T_in)] for _ in range(T_out)]
    block_n1 = [[0 for _ in range(T_in)] for _ in range(T_out)]
    for idx, block in diags.items():
        tx, ty = idx
        n1 = orion_search(list(block.keys()), n_slots)
        for d in block.keys():
            bs = d % n1
            gs = (d // n1) * n1
            global_rots.add(bs)
            global_rots.add(gs)
            block_rots[tx][ty][0].add(bs)
            block_rots[tx][ty][1].add(gs)
            block_n1[tx][ty] = n1

    v_bs = [set() for _ in range(T_in)]
    for i in range(T_out):
        for j in range(T_in):
            v_bs[j].update(block_rots[i][j][0])

    with h5py.File("/home/jyh/project/openfhe-conv/pack/orion.h5", "w") as f:
        f.attrs["n_slots"] = np.int32(n_slots)
        f.attrs["T_in"] = np.int32(T_in)
        f.attrs["T_out"] = np.int32(T_out)

        global_group = f.create_group("global")
        global_group.create_dataset("rotations", data=np.array(sorted(global_rots), dtype=np.int32))
        for i in range(T_in):
            global_group.create_dataset(f"bs{i}", data=np.array(sorted(v_bs[i]), dtype=np.int32))
        global_group.create_dataset("input", data=pad_to_multiple_cipher(input, n_slots).reshape(-1, n_slots), dtype=np.float32)
        global_group.create_dataset("reference", data=pad_to_multiple_cipher(ref.numpy(), n_slots).reshape(-1, n_slots), dtype=np.float32)

        blocks_group = f.create_group("blocks")
        for i in range(T_out):
            for j in range(T_in):
                block = diags.get((i, j), {})
                n1 = block_n1[i][j]
                bs = [offset % n1 for offset in block.keys()]
                gs = [(offset // n1) * n1 for offset in block.keys()]

                vecs = list(block.values())
                vecs_rots = [np.roll(vecs[k], gs[k]) for k in range(len(vecs))]

                blocks_group.create_dataset(f"diags_{i}_{j}", data=np.array(vecs_rots, dtype=np.float32))

                blocks_group.create_dataset(f"bs_{i}_{j}", data=np.array(bs, dtype=np.int32))
                blocks_group.create_dataset(f"gs_{i}_{j}", data=np.array(gs, dtype=np.int32))

