#!/usr/bin/env python
import os, glob
import numpy as np
import pandas as pd

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

## Forward-fill missing values up to 5 sections


CINCdat.update(CINCdat.groupby('patient').ffill(limit = 7))

## Backward-fill missing values up to 5 sections
CINCdat.update(CINCdat.groupby('patient').bfill(limit=3))

## delete any rows with any remaining missing values
##CINCdat.dropna(subset = ['FiO2','pH','PaCO2','BUN','Calcium','Creatinine','Glucose','Magnesium','Potassium','Hct','Hgb','WBC','Platelets'])
CINCdat.fillna(-1)

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
train_data = lgb.Dataset(data=CINCdat_zScores.iloc[:,0:23], label=CINCdat_zScores.SepsisLabel)
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