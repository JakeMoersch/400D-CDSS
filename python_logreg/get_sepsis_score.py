#!/usr/bin/env python

import numpy as np

def get_sepsis_score(data, model):
    x_mean = np.array([
23.37, 84.97, 97.09, 36.86, 122.66, 81.97, 63.35, 
18.86, 0.49, 7.38, 41.16, 22.92, 7.79, 1.54, 131.05, 2.04, 4.13, 
31.14, 10.37, 11.23, 197.32, 62.65])
    x_std = np.array([
19.2, 16.74, 2.98, 
0.71, 23.28, 16.33, 14.05, 5.09, 0.33, 0.06, 8.78, 17.89, 2.12, 
1.91, 46.41, 0.35, 0.59, 5.56, 1.95, 7.55, 101.61, 15.91])

    x = data[-1, 0:22]
    x_norm = np.nan_to_num((x - x_mean) / x_std)

    coeffs= np.array([
0.166, 0.27378, 0.00945, 0.20826, -0.03412, -0.2279, 
    0.03377, 0.10401, 0.45664, 0.0956, 0.058, 0.18251, -0.12468, 
    0.12813, 0.0033, -0.03803, -0.03663, 0.13203, -0.16018, 0.08403, 
    0.03951, 0.00318])
    const = -4.1486

    z = const + np.dot(x_norm, coeffs)
    score = 1.0 / (1 + np.exp(-z))
    score = min(max(score,0),1)
    label = score > 0.023

    return score, label

def load_sepsis_model():
    return None
