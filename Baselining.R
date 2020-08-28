#setwd("~/Documents/DataSets/TrafficLabelling ")

######################## Include/Install packages ########################
include_packages <- c("dplyr","tidyr","lubridate","ggplot2","tibble","readxl","caret","MASS","car","stringr","BBmisc","gdata","DAAG","car","ggthemes")

for (pckgs in include_packages) {
  tryCatch({
    print(paste("Loading library", pckgs))
    library(pckgs,character.only = TRUE)
  }, error = function(err) {
    print(paste("Package not present. Installing package", pckgs))
    install.packages(pckgs,character.only = TRUE)
    library(pckgs,character.only = TRUE)
  })
}

install.packages("ggthemes")
library(e1071)
library(caret)
library(kernlab)
library(car)
library(ggthemes)
library(caret)
library(caTools)
library(graphics)
library(forecast)
new_dos <- read.csv("new_dos.csv",stringsAsFactors = FALSE) 

######################################################################################################################################################
###### BASELINING
######################################################################################################################################################
#hist(pinger_parameters$Max_RTT,breaks = 2000)
hist(dos_data$Open,breaks=2000)
baseline<-function(data, sigma, nu,xlab__, ylab__){
  ksvm_<-ksvm(data, kpar = list(sigma =sigma), type = "one-svc", nu=nu, scaled = T )
  
  x<-1:length(data)
  x1<- x[-ksvm_@alphaindex] 
  y1<- data[-ksvm_@alphaindex]
  
  x2<- ksvm_@alphaindex
  y2<- data[ksvm_@alphaindex]
  xrange<-range(c(x1, x2))
  yrange<-range(c(y1, y2))
  
  plot(x2, y2, col = 'red', xlim = c(xrange), ylim = c(yrange), xlab = xlab__, ylab = ylab__
  )
  
  points(x1, y1, type = 'p', col = '#01a982' )
  legend("topright",
         c("Inlier","Outlier"),
         fill=c("#01a982","red"),
         cex = 0.70
  )
}

prediction<-function(model, data){
  data<-sort(data, decreasing = F)
  for(i in 1:length(data)){
    if(as.character(predict(model, data[i])))
      return(data[i])
  }
}

dos_15days <- new_dos[1370:1384,3]
median(dos_15days)
m24 <- baseline(dos_15days,0.25,0.05,"data_points","Number of DoS")