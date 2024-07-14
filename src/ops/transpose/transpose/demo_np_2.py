import numpy as np

# src_3d = np.arry([[[1, 2, 3, 4], [5, 6, 7, 8]], [[9, 10, 11, 12], [13, 14, 15, 16]]])
# shape [2, 3, 2, 2]  -> 2 * 3 * 2 * 2 = 24

src_4d = np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]], [[[13, 14], [15, 16]], [[17, 18], [19, 20]], [[21, 22], [23, 24]]]])
transposed_4d = src_4d.transpose(1, 0, 2, 3)
src_4d_platte = src_4d.flatten()

print(src_4d)
print(src_4d.shape)

print(transposed_4d)
print(transposed_4d.shape)

print(src_4d_platte)
print(transposed_4d.flatten())

src_dims = [2, 3, 2, 2]
dist_dims = [3, 2, 2, 2]
cnt = 1
for i in range(len(src_dims)):
    cnt *= src_dims[i]


print("--------------------------- new demo ---------------------------")
print("----------------------------------------------------------------")
# src_stride = [src_dims[0] * src_dim2, src_dim2, 1]
src_stride = [src_dims[1] * src_dims[2] * src_dims[3], src_dims[2] * src_dims[3], src_dims[3], 1]
dist_stride = [dist_dims[1] * dist_dims[2] * dist_dims[3], dist_dims[2] * dist_dims[3], dist_dims[3], 1]
dist_3d = np.zeros(dist_dims, dtype=np.int32)
new_array = np.zeros(cnt, dtype=np.int32)


for i in range(src_dims[0]):
    for j in range(src_dims[1]):
        for k in range(src_dims[2]):
            for l in range(src_dims[3]):
                idx_src = i * src_stride[0] + j * src_stride[1] + k * src_stride[2] + l * src_stride[3]
                dist_idx = j * dist_stride[0] + i * dist_stride[1] + k * dist_stride[2] + l * dist_stride[3]
                new_array[dist_idx] = src_4d_platte[idx_src]
print(new_array)

# print(new_array_m2)


            