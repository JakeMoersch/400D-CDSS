#!/usr/bin/env python

import numpy as np
import lightgbm as lgb
import pandas as pd
from pandas.core.frame import DataFrame
import time

def get_sepsis_score(data, model):
    df = pd.DataFrame(data, columns = ['ICULOS','HR','O2Sat','Temp','SBP','MAP','DBP','Resp','FiO2','pH','PaCO2','BUN','Calcium','Creatinine','Glucose', 'Magnesium','Potassium','Hct','Hgb','WBC','Platelets','Age','Sex'])




    x_mean = np.array([2.337000e+01, 8.499000e+01, 9.709000e+01, 3.686000e+01, 1.226800e+02,
 8.200000e+01, 6.259000e+01, 1.886000e+01, 5.100000e-01, 7.380000e+00,
 4.132000e+01, 2.278000e+01, 7.990000e+00, 1.550000e+00, 1.309400e+02,
 2.030000e+00, 4.120000e+00, 3.107000e+01, 1.033000e+01, 1.108000e+01,
 1.951900e+02, 6.265000e+01, 5.700000e-01, 1.041568e+04, 5.341490e+03,
 1.191176e+04, 1.035012e+04, 1.470170e+03, 7.827760e+03, 8.247920e+03,
 6.080560e+03, 3.553000e+01, 1.619560e+03, 2.076700e+02, 1.619560e+03,
 7.827760e+03])

#11911.76,10350.12,
#1470.170,7827.760,8247.920,6080.560,35.53000,1619.560,207.6700,1619.560,7827.760]

    x_std = np.array([1.92000e+01, 1.66900e+01, 2.95000e+00, 6.80000e-01, 2.31600e+01, 1.62400e+01,
 1.35100e+01, 5.03000e+00, 5.30000e-01, 6.00000e-02, 8.30000e+00, 1.66800e+01,
 1.59000e+00, 1.80000e+00, 4.38300e+01, 3.00000e-01, 5.40000e-01, 5.29000e+00,
 1.83000e+00, 6.98000e+00, 9.70700e+01, 1.59100e+01, 5.00000e-01, 2.83455e+03,
 1.66058e+03, 2.27942e+03, 3.93220e+03, 1.31199e+03, 2.87781e+03, 1.61872e+03,
 1.55145e+03, 3.29100e+01, 6.07090e+02, 2.81800e+01, 6.07090e+02, 2.87781e+03])

#2279.42,3932.20,1311.99,2877.81,1618.72,1551.45,32.9100,607.090,28.1800,607.090,2877.81





    df.interpolate(method = 'linear')

    df['multiplyHRSBP'] = df['SBP'] * df['HR']
    df['multiplyHRDBP'] = df['DBP'] * df['HR']
    df['multiplyO2SatSBP'] = df['SBP'] * df['O2Sat']
    df['multiplyMAPSBP'] = df['SBP'] * df['MAP']
    df['multiplyHRSBP'] = df['SBP'] * df['HR']
    df['multiplyAgeICULOS'] = df['Age'] * df['ICULOS']
    df['multiplySBPDBP'] = df['SBP'] * df['DBP']
    df['multiplyHRO2Sat'] = df['HR'] * df['O2Sat']
    df['multiplyAgeO2'] = df['Age'] * df['O2Sat']
    df['multiplyAgeSex'] = df['Age'] * df['Sex']
    df['multiplyRespHR'] = df['Resp'] * df['HR']
    df['addHRSBP'] = df['HR'] + df['SBP']
    df['addHRResp'] = df['HR'] * df['Resp']
    df['addSBPDBP'] = df['SBP'] * df['DBP']

    df = df.fillna(0)
    #df.drop(['patient'],axis=1,inplace=True)

    #print(df.iloc[:,0:35])
    
    data2 = df.to_numpy()
    #print(data2)
    x = data2[-1, 0:36]
    #print(x)
    #print(x_mean)
    
    x_norm = np.nan_to_num((x - x_mean) / x_std)
    x_norm = np.array(x_norm)
    x_norm = x_norm.reshape(-1,36)
    score=model.predict(x_norm)
    score=min(max(score,0),1)
    label = score > 0.0243

    return score, label

def load_sepsis_model():
    return lgb.Booster(model_file='lightgbm.model')
