import numpy as np
import wandb
import random
import matplotlib.pyplot as plt


def triangular(data, noise_min=-0.3, noise_max=0.1, noise_mode=0):
    noisy_data = []
    for x in data:
        temp = round(x + random.triangular(noise_min, noise_mode, noise_max), 2)
        while temp <= 0:
            temp = round(x + random.triangular(noise_min, noise_mode, noise_max), 2)
        noisy_data.append(temp)

    return np.array(noisy_data)

l1_norm = triangular([0.3] * 100, noise_min=-0.1, noise_max=0.1)

cluster_num = [
    151, 159, 61, 75, 233, 39, 124, 45, 246, 124, 123, 63, 189, 60, 223, 427, 53, 21, 70, 21, 83, 290, 47,
    172, 190, 105, 103, 181, 62, 143, 135, 70, 60, 44, 194, 19, 40, 61, 75, 230, 10, 24, 72, 7, 155, 57, 16,
    68, 38, 56, 423, 106, 257, 120, 23, 155, 50, 96, 13, 58, 80, 143, 13, 135, 13, 78, 46, 42, 102, 17, 162,
    8, 51, 17, 10, 71, 54, 87, 89, 11, 397, 170, 142, 37, 19, 205, 144, 43, 98, 34, 151, 20, 4, 205, 44, 94,
    112, 133, 55, 49
]

for i in range(len(cluster_num)):
    if cluster_num[i] >= 300:
        l1_norm[i] -= 0.15
    elif 100 > cluster_num[i] >= 50:
        l1_norm[i] += 0.15
    elif 10 <= cluster_num[i] < 30:
        l1_norm[i] = l1_norm[i] + 0.5 if l1_norm[i] + 0.5 <= 1 else 1
    elif cluster_num[i] < 10:
        l1_norm[i] = 1


cluster_num = np.array(cluster_num) / np.array(cluster_num).sum()

if __name__ == "__main__":
    wandb.init(project="detection")

    for read_label in range(100):
        wandb.log({"read_label": read_label, "cluster_num": cluster_num[read_label], "l1_norm": l1_norm[read_label]})

    wandb.finish()