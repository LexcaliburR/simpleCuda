import numpy as np

# src shape [4, 2, 4, 32]
src_4d = np.arange(1024).reshape(4, 2, 4, 32)
print(src_4d)

transposed_4d = src_4d.transpose(1, 0, 2, 3)
src_4d_platte = src_4d.flatten()

print(src_4d)
print(src_4d.shape)

print(transposed_4d)
print(transposed_4d.shape)

print(src_4d_platte)
print(transposed_4d.flatten())


# 最多一次只能处理256个数字
# 需要满足：
# 分tile后，每个tile的总数据量 < 256
# 从某个维度开始分段，计算tile的数据量，从该维度开始，一直到最后一个维度

# 例1， [0, 1]维度变换，[2, 3]维度不变换
src_dims = [4, 2, 4, 32]
dist_dims = [32, 2, 4, 4]
k = 4
cnt = 1
for i in range(len(src_dims)):
    cnt *= src_dims[i]

src_stride = [src_dims[1] * src_dims[2] * src_dims[3], src_dims[2] * src_dims[3], src_dims[3], 1]
dist_stride = [dist_dims[1] * dist_dims[2] * dist_dims[3], dist_dims[2] * dist_dims[3], dist_dims[3], 1]

tile_size = cnt / k
tile_shape = [src_dims[0]/k, src_dims[1], src_dims[2], src_dims[3]]
tile_dist_shape = [dist_dims[0], dist_dims[1], dist_dims[2], dist_dims[3]/k]
# for tile in range(k):
#     print(f"processing input data [{}, {}]")

dist_index = []
for i in range(1):
    for j in range(src_dims[1]):
        for k in range(src_dims[2]):
            for l in range(src_dims[3]):
                idx_src = i * src_stride[0] + j * src_stride[1] + k * src_stride[2] + l * src_stride[3]
                dist_idx = l * dist_stride[0] + j * dist_stride[1] + k * dist_stride[2] + i * dist_stride[3]
                dist_index.append(dist_idx)
                # new_array[dist_idx] = src_4d_platte[idx_src]

print(f'min = {min(dist_index)}, max = {max(dist_index)}, size = {len(dist_index)}')