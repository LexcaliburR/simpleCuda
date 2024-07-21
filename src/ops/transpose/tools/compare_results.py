import numpy as np

# src shape [6, 256, 32, 128]
src_4d = np.arange(6*256*32*128).reshape(6, 256, 32, 128)
print(src_4d)

# to [32, 6, 256, 128]
src_4d.transpose(2, 0, 1, 3)
src_4d_platten = src_4d.flatten()



