load_cinc_data <- function(fromfile=T){
  if (fromfile==T){
    load("CinC.RData")
  }else{
    files <- list.files('../training_2021-11-15', full.names=TRUE)
    cinc_dat <- NULL
    for (f in files){
      fname <- substr(basename(f), 1, nchar(basename(f))-4)
      #print(fname)
      pdat <- read.delim(f, sep=",",na = "NA")
      pdat <- cbind(patient=fname,pdat)
      cinc_dat<-rbind(cinc_dat,pdat)
    }
    # Save the data
    save(cinc_dat, file = "CinC.RData")
  }
  return(cinc_dat)
}
