import numpy as np

mean=np.load('spectrum_imagenet_mean.npy')
std=np.load('spectrum_imagenet_std.npy')

print(mean.mean(axis=-1).mean(axis=-1))
print(std.mean(axis=-1).mean(axis=-1))
print('/n')
