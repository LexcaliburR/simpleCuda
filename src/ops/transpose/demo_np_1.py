import numpy as np

# src_3d = np.arry([[[1, 2, 3, 4], [5, 6, 7, 8]], [[9, 10, 11, 12], [13, 14, 15, 16]]])

src_3d = np.array([[[1, 2, 3, 4], [5, 6, 7, 8]], [[9, 10, 11, 12], [13, 14, 15, 16]]])
transposed_3d = src_3d.transpose(0, 2, 1)
src_3d_platte = src_3d.flatten()

print(src_3d)
print(src_3d.shape)

print(transposed_3d)
print(transposed_3d.shape)

print(src_3d_platte)
print(transposed_3d.flatten())


src_dim0 = 2
src_dim1 = 2
src_dim2 = 4

dist_dim0 = 2
dist_dim1 = 4
dist_dim2 = 2

dist_3d = np.zeros((dist_dim0, dist_dim1, dist_dim2), dtype=np.int32)

new_array = np.zeros((dist_dim0 * dist_dim1 * dist_dim2), dtype=np.int32)
# src_stride = [src_dim1 * src_dim2, src_dim2, 1]
a = []
for i in range(src_dim0):
    for j in range(src_dim1):
        for k in range(src_dim2):
            idx_src = k + j * src_dim2 + i * src_dim1 * src_dim2
            dist_3d[i][k][j] = src_3d[i][j][k]

            a.append(i * src_dim1 * src_dim2 + k * src_dim1 + j)
            dist_idx = j + k * dist_dim2 + i * dist_dim1 * dist_dim2
            new_array[dist_idx] = src_3d_platte[idx_src]

print(dist_3d.flatten())
print(new_array)

print("--------------------------- new demo ---------------------------")
print("----------------------------------------------------------------")
src_stride = [src_dim1 * src_dim2, src_dim2, 1]
dist_stride = [src_dim1 * src_dim2, src_dim1, 1]
new_array_m2 = np.zeros((dist_dim0 * dist_dim1 * dist_dim2), dtype=np.int32)
for i in range(src_dim0):
    for j in range(src_dim1):
        for k in range(src_dim2):
            idx_src = i * src_stride[0] + j * src_stride[1] + k * src_stride[2]
            dist_idx = i * dist_stride[0] + k * dist_stride[1] + j * dist_stride[2]
            new_array_m2[dist_idx] = src_3d_platte[idx_src]

print(new_array_m2)