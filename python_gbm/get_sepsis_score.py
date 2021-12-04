#!/usr/bin/env python

import numpy as np
import lightgbm as lgb
import pandas as pd
from pandas.core.frame import DataFrame
import time

def get_sepsis_score(data, model):
    #start = time.time()
    df = pd.DataFrame(data, columns = ['patient','ICULOS','HR','O2Sat','Temp','SBP','MAP','DBP','Resp','FiO2','pH','PaCO2','BUN','Calcium','Creatinine','Magnesium','Potassium','Hct','Hgb','WBC','Platelets','Age','Sex'])
    #end = time.time()
    #print(end-start)

    x_mean = np.array([
23.37, 85, 97.1, 36.85, 122.73, 82.01, 62.56, 
18.84, 0.53, 7.38, 41.16, 22.92, 7.79, 1.54, 131.05, 2.04, 4.13, 
31.14, 10.37, 11.23, 197.32, 62.65,0])
    x_std = np.array([
19.2, 16.74, 2.98, 
0.71, 23.28, 16.33, 14.05, 5.09, 0.33, 0.06, 8.78, 17.89, 2.12, 
1.91, 46.41, 0.35, 0.59, 5.56, 1.95, 7.55, 101.61, 15.91,1])
    #start = time.time()
    df.interpolate(method = 'linear')
    #end = time.time()
    #print(end-start)

    #start = time.time()
    data2 = df.to_numpy()
    #end = time.time()
    #print(end-start)
    x = data2[-1, 0:23]
    x_norm = np.nan_to_num((x - x_mean) / x_std)
    x_norm = np.array(x_norm)
    x_norm = x_norm.reshape(-1,23)
  #  x_norm = x_norm.astype(np.float64)
    #start = time.time()
    score=model.predict(x_norm)
    #end = time.time()
    #print(end-start)
    score=min(max(score,0),1)
    label = score > 0.0234

    return score, label

def load_sepsis_model():
    return lgb.Booster(model_file='lightgbm.model')
