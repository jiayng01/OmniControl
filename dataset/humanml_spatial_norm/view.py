import numpy as np

raw_mean = np.load("./dataset/humanml_spatial_norm/Mean_raw.npy")
raw_std = np.load("./dataset/humanml_spatial_norm/Std_raw.npy")
print("Raw Mean:", raw_mean)
print("Raw Std:", raw_std)
