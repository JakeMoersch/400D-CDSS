#!/usr/bin/env python

import numpy as np
import lightgbm as lgb
import pandas as pd
from pandas.core.frame import DataFrame
import time

def get_sepsis_score(data, model):
    df = pd.DataFrame(data, columns = ['ICULOS','HR','O2Sat','Temp','SBP','MAP','DBP','Resp','FiO2','pH','PaCO2','BUN','Calcium','Creatinine','Glucose', 'Magnesium','Potassium','Hct','Hgb','WBC','Platelets','Age','Sex'])




    x_mean = np.array([1,
23.37, 84.99, 97.09, 36.86, 122.68, 82.00, 62.59, 
18.86, 0.51, 7.38, 41.132, 22.72, 7.99, 1.55, 130.94, 2.03, 4.12, 
31.07, 10.33, 11.08, 195.19, 62.65,10415.68,5341.490,11911.76,10350.12,
1470.170,7827.760,8247.920,6080.560,35.53000,1619.560,207.6700,1619.560,7827.76])

#11911.76,10350.12,
#1470.170,7827.760,8247.920,6080.560,35.53000,1619.560,207.6700,1619.560,7827.760]

    x_std = np.array([1,
19.2, 16.74, 2.98, 0.71, 23.28, 16.33, 14.05, 5.09, 0.33, 0.06, 8.78, 17.89, 2.12, 
1.91, 46.41, 0.35, 0.59, 5.56, 1.95, 7.55, 101.61, 15.91,
2834.55,1660.58,2279.42,3932.20,1311.99,2877.81,1618.72,1551.45,32.9100,607.090,28.1800,607.090,2877.81])

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
