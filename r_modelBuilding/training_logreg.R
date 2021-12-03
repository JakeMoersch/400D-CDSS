## Load some libraries I need to manipulate the data more easily
library('dplyr')
library('reshape2') 
library('zoo')
library('pROC')

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
cnames<-colnames(CINCdat_zScores)
form <- as.formula(paste0("SepsisLabel ~ ",paste0(cols,sep="",collapse="+")))
logReg <- glm(form,data=CINCdat_zScores,family=binomial(link='logit'))
summary(logReg$coefficients)
logReg_const <- logReg$coefficients[1]
logReg_coeffs <- logReg$coefficients[2:length(logReg$coefficients)]

## Quick but not necessarily great way to find a threshold
CINCdat_zScores$probSepsis <- predict(logReg,newdata=CINCdat_zScores,type=c("response"))
roc_logReg <- roc(SepsisLabel ~ probSepsis,data=CINCdat_zScores)
thresh<-coords(roc_logReg, "b", best.method="youden", input = "threshold", transpose = T,
               ret = c("threshold", "sensitivity","specificity","ppv","npv","fp","tp","fn","tn"))
thresh
# Plot the AUC
plot(roc_logReg,main=paste0('AUC=',round(roc_logReg$auc,3)))

## Report the values to put into my get_sepsis_score's load_sepsis_model function
myModel<- NULL
myModel$x_mean <- as.vector(meanCINCdat)
myModel$x_std <- as.vector(sdCINCdat)
myModel$const <- round(logReg_const,5)
myModel$coeffs <- round(as.vector(logReg_coeffs),5)
myModel$thresh <- round(thresh[1],3)
dput(myModel)

## Quick validation of score
x_norm <- CINCdat_zScores[200,1:22]
correct <- predict(logReg,x_norm,type="response")
score <- plogis(myModel$const + sum(x_norm * myModel$coeffs))
c(predict=correct,calculated=score,difference=abs(correct-score)<1e-5)
