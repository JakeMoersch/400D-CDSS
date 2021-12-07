#!/usr/bin/env python

import numpy as np
import lightgbm as lgb
import pandas as pd


def get_sepsis_score(data, model):
    df = pd.DataFrame(data, columns = ['ICULOS','HR','O2Sat','Temp','SBP','MAP','DBP','Resp','FiO2','pH','PaCO2','BUN','Calcium','Creatinine','Glucose', 'Magnesium','Potassium','Hct','Hgb','WBC','Platelets','Age','Sex'])




    x_mean = np.array([23.37,  84.99,  97.09,  36.86, 122.68,  82.,    62.59,  18.86,   0.51,   7.38,
  41.32,  22.78,   7.99,   1.55, 130.94,   2.03,   4.12,  31.07,  10.33,  11.08,
 195.19,  62.65,   0.57, 207.67, 147.57, 219.77, 204.68, 163.99, 145.46, 182.08,
 159.74, 133.76, 103.85, 207.67, 103.85, 185.26])



    x_std = np.array([1.920e+01, 1.669e+01, 2.950e+00, 6.800e-01, 2.316e+01, 1.624e+01, 1.351e+01,
 5.030e+00, 5.300e-01, 6.000e-02, 8.300e+00, 1.668e+01, 1.590e+00, 1.800e+00,
 4.383e+01, 3.000e-01, 5.400e-01, 5.290e+00, 1.830e+00, 6.980e+00, 9.707e+01,
 1.591e+01, 5.000e-01, 2.818e+01, 2.250e+01, 2.340e+01, 3.717e+01, 2.473e+01,
 2.912e+01, 1.673e+01, 1.605e+01, 2.387e+01, 1.838e+01, 2.818e+01, 1.838e+01,
 3.192e+01])






    df.interpolate(method = 'linear')

    df['multiplyHRSBP'] = df['SBP'] + df['HR']
    df['multiplyHRDBP'] = df['DBP'] + df['HR']
    df['multiplyO2SatSBP'] = df['SBP'] + df['O2Sat']
    df['multiplyMAPSBP'] = df['SBP'] + df['MAP']
    df['multiplyHRSBP'] = df['SBP'] + df['HR']
    df['comboSBPPaCO2'] = df['SBP'] + df['PaCO2']
    df['comboSBPBUN'] = df['SBP'] + df['BUN']
    df['multiplyHRO2Sat'] = df['HR'] + df['O2Sat']
    df['multiplyAgeO2'] = df['Age'] + df['O2Sat']
    df['ComboSBPWBC'] = df['SBP'] + df['WBC']
    df['multiplyRespHR'] = df['Resp'] + df['HR']
    df['addHRSBP'] = df['HR'] + df['SBP']
    df['addHRResp'] = df['HR'] + df['Resp']
    df['addSBPDBP'] = df['SBP'] + df['DBP']

    df = df.fillna(1)
    #df.drop(['patient'],axis=1,inplace=True)

    #print(df.iloc[:,0:35])
    
    data2 = df.to_numpy()
    #print(data2)
    x = data2[-1, 0:36]
    #x = data2[-1, 0:23]
    #print(x)
    #print(x_mean)
    
    x_norm = np.nan_to_num((x - x_mean) / x_std)
    x_norm = np.array(x_norm)
    x_norm = x_norm.reshape(-1,36)
    #x_norm = x_norm.reshape(-1,)
    score=model.predict(x_norm)
    score=min(max(score,0),1)
    label = score > 0.04

    return score, label

def load_sepsis_model():
    return lgb.Booster(model_file='lightgbm.model')
