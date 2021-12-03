## Load some libraries I need to manipulate the data more easily
library('dplyr')
library('reshape2') 
library('zoo')
library('pROC')
library('lightgbm')

## Clear workspace
rm(list=ls())

## Load all the data so we can quickly combine it and explore it. 
source('load_cinc_data.R')
CINCdat <- load_cinc_data(fromfile=T)
nrow(CINCdat) # should be n=198774

## Forward-fill missing values
CINCdat <- CINCdat %>% group_by(patient) %>% mutate_all(funs(na.locf(., na.rm = FALSE)))

## Get reference ranges for variables using 
## only non-sepsis patients as 'normal'
CINCdat_NOsepsis <- CINCdat[!CINCdat$patient %in% unique(CINCdat$patient[CINCdat$SepsisLabel==1]),]
CINCdat_NOsepsis <- CINCdat_NOsepsis[CINCdat_NOsepsis$ICULOS>1,seq(2,ncol(CINCdat_NOsepsis)-2)]
meanCINCdat <- round(sapply(CINCdat_NOsepsis,mean,na.rm=T),2)
sdCINCdat <- round(sapply(CINCdat_NOsepsis,sd,na.rm=T),2)

## Obtain the z-scores for all the variables
cols <- colnames(CINCdat)
cols <- cols[!(cols %in% c("patient","SepsisLabel","Sex"))]
CINCdat_zScores <- CINCdat[,2:ncol(CINCdat)]
for (c in cols){
  CINCdat_zScores[[c]] <- (CINCdat[[c]]-meanCINCdat[[c]])/sdCINCdat[[c]]
}

## Replace values still missing with the mean
for (c in cols){
  CINCdat_zScores[[c]][is.na(CINCdat_zScores[[c]])]<-0
  CINCdat_zScores[[c]][is.infinite(CINCdat_zScores[[c]])]<-0
}

## Build a linear regression model using all the training data
## Try a LightGBM approach
library('lightgbm')
bst <- lightgbm(
  data = as.matrix(CINCdat_zScores[,1:23])
  , label = CINCdat_zScores$SepsisLabel
  , objective = "binary"
)

## Quick but not necessarily great way to find a threshold
CINCdat_zScores$probSepsisGBM <- predict(bst,data=as.matrix(CINCdat_zScores[,1:23]))

# Plot the AUC
roc_GBM <- roc(SepsisLabel ~ probSepsisGBM,data=CINCdat_zScores)
plot(roc_GBM,main=paste0('AUC=',round(roc_GBM$auc,3)))
thresh<-coords(roc_GBM, "b", best.method="youden", input = "threshold", transpose = T,
               ret = c("threshold", "sensitivity","specificity","ppv","npv","fp","tp","fn","tn"))

# Save the model and get the threshold for use as a model
lgb.save(bst, "lightgbm.model")
round(thresh[1],3)
