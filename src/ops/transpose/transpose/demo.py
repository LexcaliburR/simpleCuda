import torch
import numpy as np

src_3d = torch.tensor([[[1, 2, 3, 4], [5, 6, 7, 8]], [[9, 10, 11, 12], [13, 14, 15, 16]]])
transposed_3d = src_3d.permute(0, 2, 1)

print(src_3d)
print(src_3d.size())

print(transposed_3d)
print(transposed_3d.size())

# 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16

# 1, 5, 2, 6, 3, 7, 4, 8, 9, 13, 10, 14, 11, 15, 12, 16

src_dim0 = 2
src_dim1 = 2
src_dim2 = 4

dist_dim0 = 2
dist_dim1 = 4
dist_dim2 = 2


# dim1 和 dim2 交换
# for i in range(src_dim0):
#     for j in range(src_dim1):
#         for k in range(src_dim2):
#             print(f"src_3d[{i}][{j}][{k}] = {src_3d[i][j][k]}")
            # dst_idx = k * dist_dim1 + j * dist_dim2 + i * dist_dim1 * dist_dim2


