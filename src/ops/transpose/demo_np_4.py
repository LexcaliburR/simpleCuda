import numpy as np

# src shape [4, 2, 4, 32]
src_4d = np.arange(128).reshape(4, 2, 4, 4)
print(src_4d.flatten())

tras_4d_01 = src_4d.transpose(1, 0, 2, 3)
tras_4d_12 = src_4d.transpose(0, 2, 1, 3)
tras_4d_23 = src_4d.transpose(0, 1, 3, 2)

print(tras_4d_01.flatten())
print(tras_4d_12.flatten())
print(tras_4d_23.flatten())
