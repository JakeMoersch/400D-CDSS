#!/usr/bin/env python

import numpy as np
import lightgbm as lgb

def get_sepsis_score(data, model):
    x_mean = np.array([
23.37, 84.97, 97.09, 36.86, 122.66, 81.97, 63.35, 
18.86, 0.49, 7.38, 41.16, 22.92, 7.79, 1.54, 131.05, 2.04, 4.13, 
31.14, 10.37, 11.23, 197.32, 62.65,0])
    x_std = np.array([
19.2, 16.74, 2.98, 
0.71, 23.28, 16.33, 14.05, 5.09, 0.33, 0.06, 8.78, 17.89, 2.12, 
1.91, 46.41, 0.35, 0.59, 5.56, 1.95, 7.55, 101.61, 15.91,1])

    x = data[-1, 0:23]
    x_norm = np.nan_to_num((x - x_mean) / x_std)
    x_norm = np.array(x_norm)
    x_norm = x_norm.reshape(-1,23)
#    x_norm = x_norm.astype(np.float64)
    score=model.predict(x_norm)
    score=min(max(score,0),1)
    label = score > 0.026

    return score, label

def load_sepsis_model():
    return lgb.Booster(model_file='lightgbm.model')
