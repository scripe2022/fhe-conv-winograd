#!/home/jyh/project/conv-chain/conv-venv/bin/python3
# run  := python3 orion.py
# dir  := .
# kid  :=

import math
import time

import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import math
import time
import torch
import scipy.sparse as sp
import tqdm
import numpy as np

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

def diagonalize(
    matrix: sp.csr_matrix,
    num_slots: int,
    embed_method: str,
    is_last_layer: bool,
):
    """
    For each (slots, slots) block of the input matrix, this function 
    extracts the generalized diagonals and stores them in a dictionary. 
    Each key ((i,j)) in the dictionary block_{i,j}, and the value is 
    another dictionary mapping diagonal indices to their values.

    Args:
        matrix (scipy.sparse.csr_matrix): A 4D tensor representing a weight matrix 
            for a fully-connected or convolutional layer. The shape must 
            conform to (num_blocks_y, num_blocks_x, slots, slots).
        slots (int): The number of SIMD plaintext slots, dictating the 
            block size.

    Returns:
        dict: A dictionary where each key is a tuple (i, j) corresponding 
              to the (i, j)th (slots, slots) block of `matrix`. The value 
              for each key is another dictionary that maps diagonal indices 
              within the block to the diagonal's tensor values.

    Examples:
        >>> matrix = torch.tensor([[[[ 0,  1,  2,  3],
                                     [ 4,  5,  6,  7],
                                     [ 8,  9, 10, 11],
                                     [12, 13, 14, 15]]]])
        >>> # Example with slots=4, showing processing of a single block
        >>> print(diagonalize(matrix, slots=4)) 
        {(0, 0): {0: [0., 5., 10., 15.], 
                  1: [1., 6., 11., 12.], 
                  2: [2., 7., 8., 13.], 
                  3: [3., 4., 9., 14.]}}

        >>> # Example with slots=2, showing processing of four blocks or 
              sub-matrices
        >>> print(diagonalize(matrix, slots=2)) 
        {(0, 0): {0: [0., 5.], 
                  1: [1., 4.]}, 
         (0, 1): {0: [2., 7.], 
                  1: [3., 6.]}, 
         (1, 0): {0: [8., 13.], 
                  1: [9., 12.]}, 
         (1, 1): {0: [10., 15.], 
                  1: [11., 14.]}}
    """

    matrix_height, matrix_width = matrix.shape
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

    return diagonals_by_block, output_rotations

def split_near_9(N: int):
    if N % 9:
        raise ValueError()
    M = N // 9
    best = (1, M)
    best_err = abs(best[1] - 9*best[0])
    for d in range(1, int(math.isqrt(M)) + 1):
        if M % d == 0:
            err = abs(M//d - 9*d)
            if err < best_err:
                best = (d, M//d)
                best_err = err
                if err == 0:
                    break
    return best

np.random.seed(42)
torch.manual_seed(42)

n_channels = 64
input_size = 32
n_slots = 2**13

H, W = input_size, input_size
C, M = n_channels, n_channels
R, S = 3, 3
input = torch.rand(1, M, H, W)
kernel1 = torch.rand(M, C, R, S)
kernel2 = torch.rand(M, C, R, S)
matrix1 = construct_conv2d_toeplitz(kernel1, H, W)
matrix2 = construct_conv2d_toeplitz(kernel2, H, W)
blocks1, _ = diagonalize(matrix1, num_slots=n_slots, embed_method="", is_last_layer=False)
blocks2, _ = diagonalize(matrix2, num_slots=n_slots, embed_method="", is_last_layer=False)

channel_per_cipher = n_slots // (H * W)
n1, n2 = split_near_9(R*S*channel_per_cipher)
base_gs = n1 * H * W

def get_bs(k):
    return k % (n1 * H * W)

def get_gs(k):
    return (k // (n1 * H * W)) * H * W


o1 = torch.nn.functional.conv2d(input, kernel1, stride=1, padding=1)
o2 = torch.nn.functional.conv2d(o1, kernel2, stride=1, padding=1).flatten()
with open("orion-ref.txt", "w") as f:
    for i in range(o2.shape[0]):
        print(float(o2[i]), file=f)

with open("orion-pack.txt", "w") as f:
    print(n_slots, H, W, C, M, R, S, n1, n2, file=f)
    bs = []
    gs = []
    for k, v in blocks1.items():
        for i, (rot, vec) in enumerate(v.items()):
            if i < n1 * 9:
                bs.append(rot)
            if i % (n1 * 9) == 0:
                gs.append(rot)
        break

    print(len(bs), len(gs), file=f)
    for i in bs:
        print(i, end=' ', file=f)
    print(file=f)
    for i in gs:
        print(i, end=' ', file=f)
    print(file=f)

    n_diags, x_max, y_max = 0, 0, 0
    for k, v in blocks1.items():
        n_diags += len(v)
        x_max = max(x_max, k[0] + 1)
        y_max = max(y_max, k[1] + 1)
    print(n_diags, x_max, y_max, file=f)
    for k, v in blocks1.items():
        for i, (rot, vec) in enumerate(v.items()):
            print(k[0], k[1], get_bs(rot), get_gs(rot), file=f)
            for num in np.roll(vec, get_gs(rot)):
                print(num, end=' ', file=f)
            print(file=f)
    for k, v in blocks2.items():
        for i, (rot, vec) in enumerate(v.items()):
            print(k[0], k[1], get_bs(rot), get_gs(rot), file=f)
            for num in np.roll(vec, get_gs(rot)):
                print(num, end=' ', file=f)
            print(file=f)

    print(C*H*W // n_slots, file=f)
    cipher_full = input.numpy().flatten()
    for i in range(0, C*H*W, n_slots):
        c = cipher_full[i:i+n_slots]
        for num in c:
            print(num, end=' ', file=f)
        print(file=f)
