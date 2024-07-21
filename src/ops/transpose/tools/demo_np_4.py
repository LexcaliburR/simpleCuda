import numpy as np

# src shape [4, 2, 4, 32]
src_4d = np.arange(128).reshape(4, 2, 4, 4)
print(src_4d.flatten())
perm = [1, 0, 2, 3]
transposed_4d = src_4d.transpose(perm)
print(transposed_4d.flatten())



inputStride = [32, 16,4, 1]
transstride = []

for i in range(4):
    transstride.append(inputStride[perm[i]])

outputShape = [2, 4, 4, 4]
print(src_4d.strides)
print(transstride)


ret = []
for i in range(2):
    for j in range(4):
        for k in range(4):
            for l in range(4):
                idx = i * transstride[0] + j * transstride[1] + k * transstride[2] + l * transstride[3]
                ret.append(src_4d.flatten()[idx])

print(ret)
