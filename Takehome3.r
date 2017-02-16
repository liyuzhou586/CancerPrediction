###IMPORT LIBRARIES
library(e1071)
library(pROC)
library(randomForest)
library(ada)
library(nnet)
library(quadprog)
library(ggplot2)
##READ IN DATA AND DATA CLEANING
importData<-read.csv("~/AngleClosure.csv",header=TRUE,na.string=c("."," ","NA"))
deletelist <- c(which(names(importData)=="GENDER"),which(names(importData)=="EYE"),which(names(importData)=="ETHNIC"))
importData<-importData[,-deletelist]
dim(importData)
datatemp<-data.matrix(importData)
Deletena<-which(apply(datatemp,1,function(xx){sum(is.na(xx))})>=1)
Deletena
importData<-importData[-Deletena,]
dim(importData)
###(b)
#Remove the following data column
Partbrmls=c(which(names(importData)=="HGT"),
  which(names(importData)=="WT"),
  which(names(importData)=="ASPH"),
  which(names(importData)=="ACYL"),
  which(names(importData)=="SE"),
  which(names(importData)=="AXL"),
  which(names(importData)=="CACD"),
  which(names(importData)=="AGE"),
  which(names(importData)=="CCT.OD"),
  which(names(importData)=="PCCURV_mm"))
Partbrmls
Datab <- importData[,-Partbrmls]
myLabel=Datab[,12]
mySource<- Datab[,-12]
myY=as.numeric(myLabel=="YES")
##Normalize data
coldim = dim(mySource)[2]
meansource= matrix(0,coldim,1)
for(ii in 1:coldim){
	meansource[ii,1]=mean(mySource[,ii])
}
sdsource = matrix(0,coldim,1)
for(ii in 1:coldim){
	sdsource[ii,1]=sd(mySource[,ii])
}

for (ii in 1:dim(mySource)[1]){
	for (jj in 1:dim(mySource)[2]){
		mySource[ii,jj]=(mySource[ii,jj]-meansource[jj,1])/sdsource[jj,1]
	}
}
myData<-mySource

###b)&c)
#Define iteration number is 10 and use 10 fold and calculate AUC, and find the best tunning parameter.
#################First begin with logistic regression model###
niter = 10
nfolds =10
numberoftest = round(dim(myData)[1]/nfolds)
##Build test set
AUC=c()
kk = c(2,3,4,5,6,7,8,9)
mAUCL=c()
for(kkk in 1:length(kk)){
for (iter in 1:niter){
	##Build Training data
	spnum=sample(dim(myData)[1])[1:numberoftest]
	myTestX=myData[spnum,]
	myTestY=myY[spnum]
	myTrainX=myData[-spnum,]
	myTrainY=myY[-spnum]
	method=glm(myTrainY~AOD750+TISA750+IT750+IT2000+ITCM+
        IAREA+ICURV+ACW_mm+ACA+ACV+LENSVAULT,data=cbind(myTrainX,myTrainY))
	LRstep = step(method,k=kk[kkk],direction = "both")
	LRProb = predict(LRstep, newdata = myTestX, type = "response")
	rocv = roc(myTestY,predict(LRstep,newdata=myTestX))
	AUC=append(AUC,rocv$auc)
}
MeanAUC=mean(AUC)
mAUCL=append(mAUCL,MeanAUC)
}
infomatrixLG=rbind(kk,mAUCL)
Bestk<-which.max(infomatrixLG[2,])
###So, the best k in this model is: k=3 AUC = 0.9408409 
kk[Bestk]
###PLOT PART
dev.new(width=4,height=2.5)
par(mai=c(0.5,0.5,0.3,0.05),cex=0.8)
plot(infomatrixLG[1,],infomatrixLG[2,],type="l",
     axes=FALSE,xlab="",ylab="",main="LOGISTIC")
box()
axis(1,padj=-0.5)
title(xlab="Parameter k",line=1.65)
axis(2,padj=0.5)
title(ylab="AUC",line=2)



############SVM without kernel (linear kernal)
AUCSVM0=c()
cs=c(2,3,4,5,6,7,8,9,10)
mAUCSVM0=c()
for (kk in 1:length(cs)){
for (iter in 1:niter){
	##Build Training data
	spnum=sample(dim(myData)[1])[1:numberoftest]
	myTestX=myData[spnum,]
	myTestY=myY[spnum]
	myTrainX=myData[-spnum,]
	myTrainY=myY[-spnum]
	method=svm(myTrainX,myTrainY,type="C",kernel="linear",cost=cs[kk],probability=TRUE)
	svmpredict=predict(method, myTestX)
	svmproba = predict(method, myTestX, probability = TRUE)
	rocv = roc(myTestY,attributes(svmproba)$probabilities[,1])
	AUCSVM0=append(AUCSVM0,rocv$auc)
} 
MeanAUCSVM0=mean(AUCSVM0)
mAUCSVM0=append(mAUCSVM0,MeanAUCSVM0)
}
infomatrixSV0=rbind(cs,mAUCSVM0)
Bestcost<-which.max(infomatrixSV0[2,])
##So, the best cost parameter is cost= 3, AUC=0.9693185
cs[Bestcost]
###PLOT 
dev.new(width=4,height=2.5)
par(mai=c(0.5,0.5,0.3,0.05),cex=0.8)
plot(infomatrixSV0[1,],infomatrixSV0[2,],type="l",
     axes=FALSE,xlab="",ylab="",main="SVM-Linear")
box()
axis(1,padj=-0.5)
title(xlab="Parameter cost",line=1.65)
axis(2,padj=0.5)
title(ylab="AUC",line=2)


########randomForest Method####
AUCRF=c()
mm=c(2,3,4,5,6,7,8,9,10)
mAUCRF=c()
for(mmm in 1:length(mm)){
for (iter in 1:niter){
	spnum=sample(dim(myData)[1])[1:numberoftest]
	myTestX=myData[spnum,]
	myTestY=myY[spnum]
	myTrainX=myData[-spnum,]
	myTrainY=myY[-spnum]
	method=randomForest(y=myTrainY,x=myTrainX,ntree=1000,mtry=mm[mmm])
	randomfpred=predict(method,myTestX)
	rdfproba = predict(method,newdata=myTestX,type="response",proximity=TRUE)
	rocv=roc(myTestY,rdfproba$predicted)
	AUCRF=append(AUCRF,rocv$auc)
}
MeanAUCRF=mean(AUCRF)
mAUCRF<-append(mAUCRF,MeanAUCRF)
}
infomatrixRF<-rbind(mm,mAUCRF)
Bestmtry<-which.max(infomatrixRF[2,])
##So,the best mtry factor is mtry=2, and the AUC = 0.9552338
mm[Bestmtry]
###PLOT 
dev.new(width=4,height=2.5)
par(mai=c(0.5,0.5,0.3,0.05),cex=0.8)
plot(infomatrixRF[1,],infomatrixRF[2,],type="l",
     axes=FALSE,xlab="",ylab="",main="RandomForest")
box()
axis(1,padj=-0.5)
title(xlab="Parameter mtry",line=1.65)
axis(2,padj=0.5)
title(ylab="AUC",line=2)





#####Boost Algorithm#####
AUCB=c()
nulist=c(0.001,0.01,0.02,0.05,0.1,0.15,0.2,0.5)
mAUCB=c()
for(nn in 1:length(nulist)){
for (iter in 1:niter){
	spnum=sample(dim(myData)[1])[1:numberoftest]
	myTestX=myData[spnum,]
	myTestY=myY[spnum]
	myTrainX=myData[-spnum,]
	myTrainY=myY[-spnum]
	#traindata =data.frame(myTrainY,myTrainX)
	method=ada(x=myTrainX,y=myTrainY,nu=nulist[nn])
	Adapred=predict(method,myTestX)
	Adaprob=predict(method,myTestX,type=c("probs"))
	rocv=roc(myTestY,Adaprob[,2])

	AUCB=append(AUCB,rocv$auc)
}
MeanAUCB=mean(AUCB)
mAUCB=append(mAUCB,MeanAUCB)
}
informatrixADB=rbind(nulist,mAUCB)
Bestnu<-which.max(informatrixADB[2,])
##So,the best nu factor is nu=0.1, AUC = 0.950323
nulist[Bestnu]
##PLOT
dev.new(width=4,height=2.5)
par(mai=c(0.5,0.5,0.3,0.05),cex=0.8)
plot(informatrixADB[1,],informatrixADB[2,],type="l",
     axes=FALSE,xlab="",ylab="",main="Adaboost")
box()
axis(1,padj=-0.5)
title(xlab="Parameter nu",line=1.65)
axis(2,padj=0.5)
title(ylab="AUC",line=2)


######Neural Network Method#####
decaylist=c(0,0.001,0.01,0.1,0.2,0.5,1,2,5)
sizelist=c(1,2,5,10,15)

mAUCN=matrix(0,length(decaylist),length(sizelist))

for (dd in 1:length(decaylist)){
for(ss in 1:length(sizelist)){
	AUCN=c()
for(iter in 1:niter){
	spnum=sample(dim(myData)[1])[1:numberoftest]
	myTestX=myData[spnum,]
	myTestY=myY[spnum]
	myTrainX=myData[-spnum,]
	myTrainY=myY[-spnum]
	method=nnet(myTrainY~.,data=myTrainX,size=sizelist[ss],decay=decaylist[dd],MaxNWts=10000,maxit=250)
	Nnetprob=predict(method,newdata=myTestX,type="raw")
	rocv=roc(myTestY,Nnetprob[,1])
	AUCN=append(AUCN,rocv$auc)
}
MeanAUCN=mean(AUCN)
mAUCN[dd,ss]=MeanAUCN
}
}
which.max(mAUCN)
###The result is when decay = 0.2, size =1, the AUC is  0.9740386
tt=matrix(0,45,3)
tt[,1]=rep(c(0,0.001,0.01,0.1,0.2,0.5,1,2,5),5)
tt[,2]=rep(c("1","2","5","10","15"),9)
for(ii in 1:45){
	tt[ii,3]=mAUCN[ii]
}
ttt<-data.frame(tt)
attributes(ttt)$names=c("decay","size","MeanAUC")
qplot(decay,MeanAUC, data=ttt,shape =size)



###STACKING MODEL#####
###5 INITIAL MODEL WITH OPTIMIZED PARAMETER
miu = NULL
y=c()
for(iter in 1:niter){
##Redefine the data part
	spnum=sample(dim(myData)[1])[1:numberoftest]
	myTestX=myData[spnum,]
	myTestY=myY[spnum]
	myTrainX=myData[-spnum,]
	myTrainY=myY[-spnum]
	resultcombine=NULL

##1.LOGISTIC MODEL with k=3
	methodLG=glm(myTrainY~AOD750+TISA750+IT750+IT2000+ITCM+
        IAREA+ICURV+ACW_mm+ACA+ACV+LENSVAULT,data=cbind(myTrainX,myTrainY))
	LRstep = step(methodLG,k=3,direction = "both")
	LRProb = predict(LRstep, newdata = myTestX, type = "response")
##2.SVM without kernel(linear kernel) cost =2
	methodSVM=svm(myTrainX,myTrainY,type="C",kernel="linear",cost=2,probability=TRUE)
	svmpredict=predict(methodSVM, myTestX)
	svmproba = predict(methodSVM, myTestX, probability = TRUE)
##3.Random Forest Method mtry=3
	methodRF=randomForest(y=myTrainY,x=myTrainX,ntree=1000,mtry=2)
	randomfpred=predict(methodRF,myTestX)
	rdfproba = predict(methodRF,newdata=myTestX,type="response",proximity=TRUE)
##4. Adaboost Method with nu=0.2
	methodAD=ada(x=myTrainX,y=myTrainY,nu=0.1)
	Adapred=predict(methodAD,myTestX)
	Adaprob=predict(methodAD,myTestX,type=c("probs"))
##5. Neural Network Method with decay =0.01, size =1
	method=nnet(myTrainY~.,data=myTrainX,size=1,decay=0.2,MaxNWts=10000,maxit=250)
	Nnetprob=predict(method,newdata=myTestX,type="raw")
#####COMBINE THE RESULT
	resultcombine = cbind(as.vector(LRProb),as.vector(attributes(svmproba)$probabilities[,1]),as.vector(rdfproba$predicted),as.vector(Adaprob[,2]),as.vector(Nnetprob[,1]))
	miu =rbind(miu,resultcombine)
	y=append(y,myTestY)
}
QPstep<-solve.QP(t(miu)%*%miu,t(y)%*%miu,cbind(matrix(1,5,1),diag(1,5,5)),c(1,0,0,0,0,0),meq=1)
##So the constrained solution is: 2.290224e-18  5.054845e-01  1.843784e-01  3.101371e-01 -2.775558e-17
weightcon<-QPstep$solution
##Unconstrained solution is: -0.03071441  0.65996695  0.23104732  0.28151009 -0.14808687
weightun<-QPstep$unconstrained.solution

conpred<-miu%*%weightcon
unconpred<-miu%*%weightun
AUCCON<-roc(y,as.vector(conpred))$auc  ##0.9623
AUCUN<-roc(y,as.vector(unconpred))$auc  ##0.9621




#######Validation Part#####
valicase=read.csv("~/AngleClosure_ValidationCases.csv",header=TRUE,na.string=c("."," ","NA"))
##attributes(valicase)$names
##[1] "Age"       "Gender"    "Ethnic"    "Diagnosis" "lAOD250"   "lAOD500"   "lAOD750"   "lTISA500"  "lTISA750"  "lARA"      "lIT750"   
##[12] "lIT2000"   "lITCM"     "lIAREA"    "lICURV"    "lTILT"     "rAOD250"   "rAOD500"   "rAOD750"   "rTISA500"  "rTISA750"  "rARA"     
##[23] "rIT750"    "IT2000"    "rITCM"     "rIAREA"    "rICURV"    "rTILT"     "ACDmm"     "ACWmm"     "ACA"       "ACV"       "PCCURVmm" 
##[34] "ACCURVmm"  "CTHICK"    "LENSVAULT" "CLENGTHmm"
valicont=read.csv("~/AngleClosure_ValidationControls.csv",header=TRUE,na.string=c("."," ","NA"))
##attributes(valicont)$names
# [1] "File"        "Age"         "gender"      "lAOD250"     "lAOD500"     "lAOD750"     "lTISA500"    "lTISA750"    "lARA"       
#[10] "lIT750"      "lIT2000"     "lITCM"       "lIAREA"      "lICURV."     "lTILT"       "rAOD250"     "rAOD500"     "rAOD750"    
#[19] "rTISA500"    "rTISA750"    "rARA"        "rIT750"      "rIT2000"     "rITCM"       "rIAREA"      "rICURV"      "rTILT"      
#[28] "ACD.mm."     "ACW.mm."     "ACA"         "ACV"         "PCCURV.mm."  "ACCURV.mm."  "CTHICK"      "LENSVAULT"   "CLENGTH.mm."

###Because right eye data are preferentially used here, I would use left eye data to substitute the right eye data.
rightlist = c("rAOD750","rTISA750","rIT750","IT2000","rITCM","rIAREA","rICURV", "ACWmm", "ACA", "ACV", "LENSVAULT")
leftlist = c("rAOD750","lTISA750","lIT750","IT2000","lITCM","lIAREA","lICURV", "ACWmm", "ACA", "ACV", "LENSVAULT")
myDatacase_r = valicase[,attributes(valicase)$names %in% rightlist]
myDatacase_l = valicase[,attributes(valicase)$names %in% leftlist]
for(ii in 1:dim(myDatacase_r)[1]){
	for(jj in 1:dim(myDatacase_r)[2]){
		if(is.na(myDatacase_r[ii,jj])==1){
			myDatacase_r[ii,jj]=myDatacase_l[ii,jj]
		}
	}
}
###DELETE THE MISSING VALUES
casedatatemp<-data.matrix(myDatacase_r)
casedelete<-which(apply(casedatatemp,1,function(xx){sum(is.na(xx))})>=1)
casedelete
myDatacase_r<-myDatacase_r[-casedelete,]
###Validation control file.
rtlist=c("rAOD750","rTISA750","rIT750","rIT2000","rITCM","rIAREA","rICURV", "ACW.mm.", "ACA", "ACV", "LENSVAULT")
lflist=c("lAOD750","lTISA750","lIT750","lIT2000","lITCM","lIAREA","lICURV.", "ACW.mm.", "ACA", "ACV", "LENSVAULT")
myDatacont_r=valicont[,attributes(valicont)$names %in% rtlist]
myDatacont_l=valicont[,attributes(valicont)$names %in% lflist]
##Similarly, use left to fill missing value
for(ii in 1:dim(myDatacont_r)[1]){
	for(jj in 1:dim(myDatacont_r)[2]){
		if(is.na(myDatacont_r[ii,jj])==1){
			myDatacont_r[ii,jj]=myDatacont_l[ii,jj]
		}
	}
}
##Delete the missing rows (Actually there is no row has missing values)
#contdatatemp<-data.matrix(myDatacont_r)
#contdelete<-which(apply(contdatatemp,1,function(xx){sum(is.na(xx))})>=1)
#contdelete
#myDatacont_r<-myDatacont_r[-contdelete,]
##Normalize the data column names
attributes(myDatacase_r)$names<-c("AOD750","TISA750","IT750","IT2000","ITCM","IAREA","ICURV","ACW_mm","ACA","ACV","LENSVAULT")
attributes(myDatacont_r)$names<-c("AOD750","TISA750","IT750","IT2000","ITCM","IAREA","ICURV","ACW_mm","ACA","ACV","LENSVAULT")

testX<-rbind(myDatacase_r,myDatacont_r)
testY<-rbind(matrix(1,dim(myDatacase_r)[1],1),matrix(0,dim(myDatacont_r)[1],1))
coltestx=dim(testX)[2]
meantestx=matrix(0,coltestx,1)
sdtestx=matrix(0,coltestx,1)
for(ii in 1:coltestx){
	meantestx[ii,1]=mean(testX[,ii])
	sdtestx[ii,1]=sd(testX[,ii])
}
for(ii in 1:dim(testX)[1]){
	for(jj in 1:dim(testX)[2]){
		testX[ii,jj]=(testX[ii,jj]-meantestx[jj,1])/sdtestx[jj,1]
	}
}


###Model I will use:

miuEstif=NULL
yy=c()
for(iter in 1:1){
##Redefine the data part
	miuEsti = NULL
	resultcombine=NULL

##1.LOGISTIC MODEL with k=3
	methodLG=glm(myY~AOD750+TISA750+IT750+IT2000+ITCM+
        IAREA+ICURV+ACW_mm+ACA+ACV+LENSVAULT,data=cbind(myData,myY))
	LRstep = step(methodLG,k=3,direction = "both")
	LRProb = predict(LRstep, newdata = testX, type = "response")
##2.SVM without kernel(linear kernel) cost =2
	methodSVM=svm(myData,myY,type="C",kernel="linear",cost=2,probability=TRUE)
	#svmpredict=predict(methodSVM, myTestX)
	svmproba = predict(methodSVM, testX, probability = TRUE)
##3.Random Forest Method mtry=3
	methodRF=randomForest(y=myY,x=myData,ntree=1000,mtry=3)
	#randomfpred=predict(methodRF,myTestX)
	rdfproba = predict(methodRF,newdata=testX,type="response",proximity=TRUE)
##4. Adaboost Method with nu=0.2
	methodAD=ada(x=myData,y=myY,nu=0.2)
	#Adapred=predict(methodAD,myTestX)
	Adaprob=predict(methodAD,testX,type=c("probs"))
##5. Neural Network Method with decay =0.2, size =10
	methodNN=nnet(myY~.,data=myData,size=10,decay=0.2,MaxNWts=10000,maxit=250)
	Nnetprob=predict(methodNN,newdata=testX,type="raw")

#####COMBINE THE RESULT
	resultcombine=c()
	resultcombine = cbind(as.vector(LRProb),as.vector(attributes(svmproba)$probabilities[,1]),as.vector(rdfproba$predicted),as.vector(Adaprob[,2]),as.vector(Nnetprob[,1]))
	miuEsti =rbind(miuEsti,resultcombine)
	yy=append(yy,testY)
##6. Constrained Model
	ConsEstiProb<-resultcombine%*%weightcon
	
##7. Unconstrained Model
	UnconsEstiProb<-resultcombine%*%weightun

	miuEsti=cbind(miuEsti,ConsEstiProb)
	miuEsti=cbind(miuEsti,UnconsEstiProb)
	miuEstif = rbind(miuEstif,miuEsti)
	
}
###ROC FUNCTION
for(ii in 1:7){
print(roc(yy,miuEstif[,ii]))
}
####PLOT ROC CURVE
dev.new(width=5,height=5)
par(mai=c(0.5,0.5,0.3,0.1),cex=0.8)
plot(roc(testY,miuEstif[,1]),main="Logistic",print.auc=TRUE)

dev.new(width=5,height=5)
par(mai=c(0.5,0.5,0.3,0.1),cex=0.8)
plot(roc(testY,miuEstif[,2]),main="SVM-Linear",print.auc=TRUE)

dev.new(width=5,height=5)
par(mai=c(0.5,0.5,0.3,0.1),cex=0.8)
plot(roc(testY,miuEstif[,3]),main="RandomForest",print.auc=TRUE)

dev.new(width=5,height=5)
par(mai=c(0.5,0.5,0.3,0.1),cex=0.8)
plot(roc(testY,miuEstif[,4]),main="Adaboost",print.auc=TRUE)

dev.new(width=5,height=5)
par(mai=c(0.5,0.5,0.3,0.1),cex=0.8)
plot(roc(testY,miuEstif[,5]),main="Nueral Network",print.auc=TRUE)

dev.new(width=5,height=5)
par(mai=c(0.5,0.5,0.3,0.1),cex=0.8)
plot(roc(testY,miuEstif[,6]),main="Constrained Model",print.auc=TRUE)

dev.new(width=5,height=5)
par(mai=c(0.5,0.5,0.3,0.1),cex=0.8)
plot(roc(testY,miuEstif[,7]),main="Unconstrained Model",print.auc=TRUE)












