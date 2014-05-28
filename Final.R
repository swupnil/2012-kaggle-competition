#Stat 151B Final
setwd("/Users/Swupnil/Documents/School/Berkeley/12 Spring/Stat 151B/Final /151b_final_project_data")

library(MASS)
library(lars)
library(class)
library(e1071)
library(gbm)
library(glmnet)
library(kknn)

train.predictors = scan("train.predictors.csv", what=numeric(), sep=",")
train.responses = scan("train.responses.csv", what=numeric(), sep=",")
test.predictors = scan("test.predictors.csv", what=numeric(), sep=",")
train.predictors = matrix(train.predictors, nrow=1500, byrow=T)
test.predictors = matrix(test.predictors, nrow=370, byrow=T)
column.descriptions = as.matrix(read.csv("column_descriptions.csv"))

zerocolumns = which(apply(train.predictors, 2, function(x) all(x) == 0) == TRUE)
cleanpred = train.predictors[,-zerocolumns]
cleantest = test.predictors[,-zerocolumns]
cleandes = column.descriptions[-(zerocolumns-1),]

#Partition the training set into training, validation, and test sets.
#Random sample to partition the 1500 rows into 3 sets of 500 rows.
partition <- sample(1:1500,1500,replace=FALSE)
train <- c(1:1000)
validation <-c(1001:1250)
test <- c(1251:1500)
cleanpred<-cleanpred[partition,]
resp = train.responses[partition]

#EXPLORATORY ANALYSIS
#loop to generate correlation only on training partition
cors1<-c(1:9893)
for(i in 1:9893){
  cors1[i]<-cor(cleanpred[1:1000,i],resp[1:1000])
}
cors<-cors1[-1]
cor.abs1<-(cors1^2)^0.5
cor.abs<-cor.abs1[-1]

#Prelimenary Plots
plot(cor.abs,cleandes[,3],main="Correlation v. Size",xlab="Corr(Xi,Y)",ylab="Size")
plot(cor.abs,sqrt(cleandes[,4]^2+cleandes[,5]^2),main="Correlation v. Distance",xlab="Corr(Xi,Y)",ylab="Euclidean Distance")
#separate features by size
size1 = which(cleandes[,3]==1)
size2 = which(cleandes[,3]==2)
size3 = which(cleandes[,3]==4)
size4 = which(cleandes[,3]==8)
size5 = which(cleandes[,3]==16)
#plot Corr v. Orientation per size
plot(cor.abs[size1],cleandes[size1,2],main="Size 1", xlab="Corr(Xi,Y)",ylab="Orientation")
plot(cor.abs[size2],cleandes[size2,2],main="Size 2", xlab="Corr(Xi,Y)",ylab="Orientation")
plot(cor.abs[size3],cleandes[size3,2],main="Size 4", xlab="Corr(Xi,Y)",ylab="Orientation")
plot(cor.abs[size4],cleandes[size4,2],main="Size 8", xlab="Corr(Xi,Y)",ylab="Orientation")
plot(cor.abs[size5],cleandes[size5,2],main="Size 16", xlab="Corr(Xi,Y)",ylab="Orientation")
#plot Corr v. Euclidean Distance per size
plot(cor.abs[size1],cleandes[size1,4]^2+cleandes[size1,5]^2, main="Size 1", xlab="Corr(Xi,Y)", ylab="Euclidean Distance")
plot(cor.abs[size2],cleandes[size2,4]^2+cleandes[size2,5]^2, main="Size 2", xlab="Corr(Xi,Y)", ylab="Euclidean Distance")
plot(cor.abs[size3],cleandes[size3,4]^2+cleandes[size3,5]^2, main="Size 4", xlab="Corr(Xi,Y)", ylab="Euclidean Distance")
plot(cor.abs[size4],cleandes[size4,4]^2+cleandes[size4,5]^2, main="Size 8", xlab="Corr(Xi,Y)", ylab="Euclidean Distance")
plot(cor.abs[size5],cleandes[size5,4]^2+cleandes[size5,5]^2, main="Size 16", xlab="Corr(Xi,Y)", ylab="Euclidean Distance")
-
#VARIABLE SELECTION
#Optimize Correlation Cutoff for Removal of Weaker Variables
cutoff=seq(1:40)*.005
sve_gbm<-c(1:40)
sve_gbm_best=1
for(i in 1:40){
  #Cutoff Variables
  weakcor <- c(which(cor.abs1<=0.15+cutoff[i]))
  strongpred = cleanpred[,-weakcor]
  strongtest = cleantest[,-weakcor]
  strongcor = cor.abs1[-weakcor]
  strongdes = cleandes[-(weakcor[-1]-1),]
  #Run GBM on New Variables
  gbm_model = gbm.fit(
    x=strongpred[1:1000,],
    y=resp[1:1000],
    distribution="gaussian",
    n.trees=100,
    shrinkage=0.11,
    train.fraction=0.8,
    interaction.depth=2)
  #determine n.tress with lowest validation standard error
  best.iter = gbm.perf(gbm_model, method="test")
  gbm.guess = predict(gbm_model, newdata=strongpred[1001:1250,], n.trees=best.iter)
  sve_gbm[i] = (mean((gbm.guess-resp[1001:1250])^2))^0.5
  if(sve_gbm[i]<=min(sve_gbm_best)){
      gbm_best=gbm_model;
      sve_gbm_best=sve_gbm[i];
  }
}
plot(cutoff+0.15,sve_gbm,main="Tuning Gbm.Fit Cutoff",xlab="Corr Cutoff",ylab="Validation Error")
sve_gbm_best-min(sve_gbm)

#After running loop several times, found best validation error on average
#was for correlation cutoff of 0.225
weakcor <- c(which(cor.abs1<=0.225))
strongpred = cleanpred[,-weakcor]
strongtest = cleantest[,-weakcor]
strongcor = cor.abs1[-weakcor]
strongdes = cleandes[-(weakcor[-1]-1),]
-
#Post-Cleaning Plots
size1 = which(strongdes[,3]==1)
size2 = which(strongdes[,3]==2)
size3 = which(strongdes[,3]==4)
size4 = which(strongdes[,3]==8)
size5 = which(strongdes[,3]==16)
#plot Corr v. Orientation per size
plot(strongcor[size1],strongdes[size1,2],main="Size 1", xlab="Corr(Xi,Y)",ylab="Orientation")
plot(strongcor[size2],strongdes[size2,2],main="Size 2", xlab="Corr(Xi,Y)",ylab="Orientation")
plot(strongcor[size3],strongdes[size3,2],main="Size 4", xlab="Corr(Xi,Y)",ylab="Orientation")
plot(strongcor[size4],strongdes[size4,2],main="Size 8", xlab="Corr(Xi,Y)",ylab="Orientation")
plot(strongcor[size5],strongdes[size5,2],main="Size 16", xlab="Corr(Xi,Y)",ylab="Orientation")
#plot Corr v. Euclidean Distance per size
plot(strongcor[size1],strongdes[size1,4]^2+strongdes[size1,5]^2, main="Size 1", xlab="Corr(Xi,Y)", ylab="Euclidean Distance")
plot(strongcor[size2],strongdes[size2,4]^2+strongdes[size2,5]^2, main="Size 2", xlab="Corr(Xi,Y)", ylab="Euclidean Distance")
plot(strongcor[size3],strongdes[size3,4]^2+strongdes[size3,5]^2, main="Size 4", xlab="Corr(Xi,Y)", ylab="Euclidean Distance")
plot(strongcor[size4],strongdes[size4,4]^2+strongdes[size4,5]^2, main="Size 8", xlab="Corr(Xi,Y)", ylab="Euclidean Distance")
plot(strongcor[size5],strongdes[size5,4]^2+strongdes[size5,5]^2, main="Size 16", xlab="Corr(Xi,Y)", ylab="Euclidean Distance")



#REGRESSION ANALYSIS

#Model 1A: GLMNet Using Training Partition and Clean Variables
fit1=cv.glmnet(cleanpred[1:1000,],resp[1:1000])
plot(fit1)
print(fit1)

#Validation Partition Error
guess_glm = predict(fit1,newx=cleanpred[1001:1250,],s="lambda.min")
sve_glm=mean((guess_glm-resp[1001:1250])^2)^0.5
sve_glm

#Test Partition Error
guess_glm = predict(fit1,newx=cleanpred[1251:1500,],s="lambda.min")
ste_glm=mean((guess_glm-resp[1251:1500])^2)^0.5
ste_glm

-

#Model 1B: GLMNet Using Training Partition and Strong Variables
fit2=cv.glmnet(strongpred[1:1000,],resp[1:1000])
plot(fit2)
print(fit2)

#Validation Partition Error
guess_glm2 = predict(fit2,newx=strongpred[1001:1250,],s="lambda.min")
sve_glm2=mean((guess_glm2-resp[1001:1250])^2)^0.5
sve_glm2

#Test Partition Error
guess_glm2 = predict(fit2,newx=strongpred[1251:1500,],s="lambda.min")
ste_glm2=mean((guess_glm2-resp[1251:1500])^2)^0.5
ste_glm2

-
glm.ideal<-fit2
--

#Model 2: KNN
dat=as.data.frame(cbind(strongpred, resp))
attach(dat)
sve1_knn<-c(1:30)
k<-c(1:30)*10
#tune for best possible K
for(i in 1:30){
  knn_model = kknn(resp~., dat[1:1000,], dat[1001:1250,], k=k[i])
  knn_fits=knn_model$fitted.values
  sve1_knn[i]=sqrt(mean((knn_fits-resp[1001:1250])^2))
  if(sve1_knn[i]<=(sve_knn_best)){
      k_best=k[i];
      sve_knn_best=sve1_knn[i];
  }
}
plot(k,sve1_knn,xlab="K",ylab="Validation Error",main="Tuning of Number of Neighbors")
sve_knn_best - min(sve1_knn)
#Found Ideal to be K=50

#Test Partition Error
fitknn = kknn(resp~., dat[1:1000,], dat[1251:1500,], k=50, kernel="biweight")
knn.guess = fitknn$fitted.values
ste_knn=sqrt(mean((knn.guess-resp[1251:1500])^2))
ste_knn

--

#Model 3: Lasso
beta.fracs = seq(from = 0, to = 1, length = 100)
lambda.err = rep(0, times = 100)
beta.ideal<-0
lasso_model = lars(
  x= strongpred[1:1000,],
  y=resp[1:1000],
  type="lasso",
  trace=T,
  normalize=F,
  intercept=F,
  max.steps=300,
  use.Gram=F)

# validate over all lambdas
j<-1
mse<-0
val.guess = (predict.lars(lasso_model, newx=strongpred[1001:1250,],mode="fraction"))
lars_guess = (val.guess[4])
for(j in 1:100) {
  mse = sqrt(mean((val.guess-resp[1001:1250])^2))
  lambda.err[j] = lambda.err[j] + mse
}
j.ideal<-min(lambda.err)

#Validation Partition Error
test.guess <- predict(lass_model, cleanpred[1251:1500,])
ste_lasso <- (mean(test.guess-resp[1251:1500])^2)^0.5

#Test Partition Error
test.guess <- predict(lass_model, cleanpred[1251:1500,])
ste_lasso <- (mean(test.guess-resp[1251:1500])^2)^0.5

--

#Model 4: Boosted Decision Trees
#Optimize Model for Number of Trees
trees=seq(1:20)*10
sve_gbm<-c(1:20)
gbm_model<-c(1:20)
sve_gbm_best = 10
for(i in 1:20){
  gbm_model = gbm.fit(
    x=strongpred[1:1000,],
    y=resp[1:1000],
    distribution="gaussian",
    n.trees=trees[i],
    shrinkage=0.1,
    train.fraction=0.8,
    interaction.depth=2)
  #determine n.tress with lowest validation standard error
  best.iter = gbm.perf(gbm_model, method="test")
  gbm.guess = predict(gbm_model, newdata=strongpred[1001:1250,], n.trees=best.iter)
  sve_gbm[i] = (mean((gbm.guess-resp[1001:1250])^2))^0.5
  if(sve_gbm[i]<=(sve_gbm_best)){
      gbm_best=gbm_model; 
      sve_gbm_best = sve_gbm[i];
  }
}
plot(trees,sve_gbm,main="Tuning Gbm.Fit Trees",xlab="n.trees",ylab="Validation Error")
min(sve_gbm)
#After trying this loop several times, we found that the best validation
#error was achieved for n.trees=90 in the initial model creation
#and then using gbm.perf on that model
-
#Optimize Model for Shrinkage
shrink=seq(1:10)*.01
sve_gbm<-c(1:10)
for(i in 1:10){
  gbm_model = gbm.fit(
    x=strongpred[1:1000,],
    y=resp[1:1000],
    distribution="gaussian",
    n.trees=90,
    shrinkage=shrink[i]+0.05,
    train.fraction=0.8,
    interaction.depth=2)
  #determine n.tress with lowest validation standard error
  best.iter = gbm.perf(gbm_model, method="test")
  gbm.guess = predict(gbm_model, newdata=strongpred[1001:1250,], n.trees=best.iter)
  sve_gbm[i] = (mean((gbm.guess-resp[1001:1250])^2))^0.5
  if(sve_gbm[i]<=(sve_gbm_best)){
    gbm_best=gbm_model; 
    sve_gbm_best = sve_gbm[i];
  }
}
plot(shrink+0.05,sve_gbm,main="Tuning Gbm.Fit Shrinkage",xlab="Shrinkage",ylab="Validation Error")
sve_gbm_best - min(sve_gbm)
sve_gbm_best
#After trying this loop several times, we found that the best validation
#error was achieved for shrinkage = 0.11
-
#Optimize Model for Interaction Depth
depth=seq(1:10)
sve_gbm<-c(1:10)
for(i in 1:10){
  gbm_model = gbm.fit(
    x=strongpred[1:1000,],
    y=resp[1:1000],
    distribution="gaussian",
    n.trees=90,
    shrinkage=0.11,
    train.fraction=0.8,
    interaction.depth=depth[i])
  #determine n.tress with lowest validation standard error
  best.iter = gbm.perf(gbm_model, method="test")
  gbm.guess = predict(gbm_model, newdata=strongpred[1001:1250,], n.trees=best.iter)
  sve_gbm[i] = (mean((gbm.guess-resp[1001:1250])^2))^0.5
  if(sve_gbm[i]<=(sve_gbm_best)){
    gbm_best=gbm_model; 
    sve_gbm_best = sve_gbm[i];
  }
}
plot(depth,sve_gbm,main="Tuning Gbm.Fit Interaction Depth",xlab="Interaction Depth",ylab="Validation Error")
sve_gbm_best-min(sve_gbm)
#After trying this loop several times, we found that the best validation
#error was achieved for interaction depth=2
-
#Save Best GBM Model After Running all Tuning Loops Several Times
gbm_ideal<-gbm_best
#sve_gbm_best was 0.7452262
ideal.iter=gbm.perf(gbm_ideal,method="test")
-
#Test Partition Error
gbm.guess = predict(gbm_ideal, newdata=strongpred[1251:1500,], n.trees=ideal.iter)
ste_gbm = (mean((gbm.guess-resp[1251:1500])^2))^0.5
ste_gbm

--

#Best Blend of Models
finalguess<-as.data.frame(cbind(resp[1251:1500],gbm.guess,knn.guess,glm.ideal))
names(finalguess)=c("a","b","c","d")
ideal.blend<-lm(a~.,data=finalguess)
summary(ideal.blend)
#best.ste was .8036

--

#Use Models to Predict on Kaggle Test Set
ideal.gbm.guess<-predict(gbm_ideal, newdata=strongtest, n.trees=ideal.iter)
ideal.knn = kknn(resp~., dat[1:1000,], as.data.frame(strongtest), k=50, kernel="biweight")
ideal.knn.guess=ideal.knn$fitted.values
ideal.glm.guess<-predict(glm.ideal,newx=cleantest,s="lambda.min")
#Blend Kaggle Predictions
newdata=as.data.frame(cbind(ideal.gbm.guess,ideal.knn.guess,ideal.glm.guess))
names(newdata)=c("b","c","d")
guesses<-predict(ideal.blend,newdata)
#Write Kaggle Predictions
write.table(guesses, "test.predictions.csv", sep=",", quote=F, row.names=F, col.names=F)