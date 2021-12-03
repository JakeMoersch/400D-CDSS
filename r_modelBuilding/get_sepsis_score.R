#!/usr/bin/Rscript

## And now continue as before.
get_sepsis_score = function(data, myModel){
    # Get the current row we are predicting
    x <- data[nrow(data),1:22]
    # Make the z-score
    x_norm <- (x - myModel$x_mean) / myModel$x_std
    # Missing is normal
    x_norm[is.na(x_norm)] <- 0

    score <- plogis(myModel$const + sum(x_norm * myModel$coeffs))
    score <- min(max(score,0),1)
    #score <- round(scores,digits=5)
    label <- score > myModel$thresh
  return(c(score,label))
}

load_sepsis_model <- function(){
  myModel<-list(x_mean = c(23.37, 84.97, 97.09, 36.86, 122.66, 81.97, 63.35, 
                           18.86, 0.49, 7.38, 41.16, 22.92, 7.79, 1.54, 131.05, 2.04, 4.13, 
                           31.14, 10.37, 11.23, 197.32, 62.65),
                x_std = c(19.2, 16.74, 2.98, 0.71, 23.28, 16.33, 14.05, 5.09, 
                          0.33, 0.06, 8.78, 17.89, 2.12, 1.91, 46.41, 0.35,
                          0.59, 5.56, 1.95, 7.55, 101.61, 15.91), 
                const = c(`(Intercept)` = -4.14866), 
                coeffs = c(0.166, 0.27378, 0.00945, 0.20826, -0.03412, -0.2279, 
                           0.03377, 0.10401, 0.45664, 0.0956, 0.058, 0.18251, 
                           -0.12468, 0.12813, 0.0033, -0.03803, -0.03663, 0.13203, 
                           -0.16018, 0.08403, 0.03951, 0.00318), 
                thresh = c(threshold = 0.023))
  return(myModel)
}