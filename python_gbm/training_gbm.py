#!/usr/bin/env python
import os, glob
import numpy as np
import pandas as pd
from pandas import Series

## Load all the data so we can quickly combine it and explore it. 
pfile = 'CinC.pickle'
if os.path.isfile(pfile):
  CINCdat = pd.read_pickle(pfile)
else:
  os.chdir("../training_2021-11-15")
  extension = 'csv'
  all_filenames = [i for i in glob.glob('*.{}'.format(extension))]
  CINCdat = pd.concat([pd.read_csv(f).assign(patient=os.path.basename(f).split('.')[0]) for f in all_filenames ])
  os.chdir(os.path.dirname(__file__))
  CINCdat.to_pickle(pfile)

print(len(CINCdat)) # should be n=198774

# want to add a column of # of measurements of another column
#HgbUnique = CINCdat['Hgb'].nunique()
#CINCdat['HgbUnique'] = CINCdat.groupby('patient').apply(HgbUnique)
#ICUTotalStay = CINCdat['ICULOS'].groupby('patient').count()
#CINCdat['ICUTotalStay'] = CINCdat.groupby('patient').add(insert(0,'ICUTotalStay', ICUTotalStay))
#CINCdat['Measurements'] = CINCdat.groupby('patient').apply(lambda group: group.apply(lambda column: column.nunique(CINCdat['Hgb'], dropna=False)))

## Forward-fill missing values up to 5 sections




##CINCdat['BP+HR'] = CINCdat['BP'] + CINCdat['HR']

CINCdat = CINCdat.groupby('patient').apply(lambda column: column.interpolate(method = 'linear'))
CINCdat = CINCdat.interpolate(method = 'linear')
#clipvals = np.linspace(0,0,38)
#clipvalsSer = pd.Series(clipvals)

#CINCdat = CINCdat.clip(clipvalsSer,axis=1) #replace negative data with 0
##CINCdat.update(CINCdat.groupby('patient').ffill())
# general statement; want to forward fill; then when we encounter a different value, we want to backfill such that i -> i+j elements each have a value equal to (v(j)-v(i) / j-i) + i
CINCdat['multiplyHRSBP'] = CINCdat['SBP'] * CINCdat['HR']
CINCdat['multiplyHRDBP'] = CINCdat['DBP'] * CINCdat['HR']
CINCdat['multiplyO2SatSBP'] = CINCdat['SBP'] * CINCdat['O2Sat']
CINCdat['multiplyMAPSBP'] = CINCdat['SBP'] * CINCdat['MAP']
CINCdat['multiplyHRSBP'] = CINCdat['SBP'] * CINCdat['HR']
CINCdat['multiplyAgeICULOS'] = CINCdat['Age'] * CINCdat['ICULOS']
CINCdat['multiplySBPDBP'] = CINCdat['SBP'] * CINCdat['DBP']
CINCdat['multiplyHRO2Sat'] = CINCdat['HR'] * CINCdat['O2Sat']
CINCdat['multiplyAgeO2'] = CINCdat['Age'] * CINCdat['O2Sat']
CINCdat['multiplyAgeSex'] = CINCdat['Age'] * CINCdat['Sex']
CINCdat['multiplyRespHR'] = CINCdat['Resp'] * CINCdat['HR']
CINCdat['addHRSBP'] = CINCdat['HR'] + CINCdat['SBP']
CINCdat['addHRResp'] = CINCdat['HR'] * CINCdat['Resp']
CINCdat['addSBPDBP'] = CINCdat['SBP'] * CINCdat['DBP']


## add a filter for multiorgan failure; 

## Get reference ranges for variables using only non-sepsis patients as 'normal'
CINCdat_NOsepsis = CINCdat[~CINCdat.patient.isin(np.unique(CINCdat.patient[CINCdat.SepsisLabel==1]))]
CINCdat_NOsepsis = CINCdat_NOsepsis[CINCdat_NOsepsis.ICULOS>1]
CINCdat_NOsepsis.drop(['patient','SepsisLabel','Sex'],axis=1,inplace=True)
meanCINCdat = round(CINCdat_NOsepsis.mean(axis=0),2)
sdCINCdat = round(CINCdat_NOsepsis.std(axis=0),2)
print('meanCINCdat',np.array(meanCINCdat))
print('sdCINCdat',np.array(sdCINCdat))

## Obtain the z-scores for all the variables
CINCdat_zScores = CINCdat
cols = CINCdat_zScores.columns.drop(['patient','SepsisLabel','Sex'])
for c in cols:
  CINCdat_zScores[c] = (CINCdat_zScores[c]-meanCINCdat[c])/sdCINCdat[c]

## Replace values still missing with the mean
CINCdat_zScores = CINCdat_zScores.fillna(0)

## Build a linear regression model using all the training data
## Try a LightGBM approach
import lightgbm as lgb
#train_data = lgb.Dataset(data=CINCdat_zScores.iloc[:,0:23], label=CINCdat_zScores.SepsisLabel)
train_data = lgb.Dataset(data=CINCdat_zScores.iloc[:,0:(23+14)], label=CINCdat_zScores.SepsisLabel)
param = {'objective': 'binary'}
bst = lgb.train(param, train_data, 10)

## Quick but not necessarily great way to find a threshold
CINCdat_zScores = CINCdat_zScores.assign(probSepsisGBM=bst.predict(data=CINCdat_zScores.iloc[:,0:23]))

# Plot the AUC
from sklearn.metrics import roc_curve, roc_auc_score
fpr, tpr, thresholds = roc_curve(CINCdat_zScores.SepsisLabel,CINCdat_zScores.probSepsisGBM)
print('AUC:',round(roc_auc_score(CINCdat_zScores.SepsisLabel,CINCdat_zScores.probSepsisGBM),2))

# Save the model and get the threshold for use as a model
bst.save_model('lightgbm.model')
print('Threshold:',round(thresholds[np.argmax(tpr - fpr)],4))
#print('tpr',tpr)
#print('fpr', fpr) 