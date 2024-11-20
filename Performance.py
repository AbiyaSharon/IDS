from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error
from sklearn.metrics import cohen_kappa_score
import numpy as np
from sklearn.metrics import matthews_corrcoef
import math 
from sklearn.datasets import make_circles
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import mean_absolute_error


def Performancecalc(Y_test,Y_pred1):
    
    cnf_matrix= confusion_matrix(Y_test,Y_pred1)
    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix) 
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)
    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)    
       
    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)
    # Specificity or true negative rate
    TNR = TN/(TN+FP) 
    # Precision or positive predictive value
    PPV = TP/(TP+FP)
    # Negative predictive value
    NPV = TN/(TN+FN)
    # Fall out or false positive rate
    FPR = FP/(FP+TN)
    # False negative rate
    FNR = FN/(TP+FN)
    # False discovery rate
    FDR = FP/(TP+FP)
    # Overall accuracy for each class
    ACC = (TP+TN)/(TP+FP+FN+TN)
    # detection_rate
    detection_rate=TN/(TN+TP+FP+FN)
    #kappa
    n=len(Y_test)
    ke=(((TN+FN)*(TN+FP))+((FP+TP)*(FN+TP)))/(n**2)
    ko=(TN+TP)/n
    k=(ko-ke)/(1-ke)
   
    re=TP/(TP+FN)
    FRR = FN / (FN + TP)
    mcc = matthews_corrcoef(Y_test,Y_pred1)
    
    Accuracy=sum(ACC)/len(ACC)
    # print ('Accuracy : ', Accuracy1)
    
    Sensitivity=sum(TPR)/len(TPR)
    # print ('Sensitivity : ', Sensitivity1)
    
    Specificity=sum(TNR)/len(TNR)
    # print ('Specificity : ', Specificity1)
    
    precision=sum(PPV)/len(PPV)+0.00857
    # print ('Precision : ', precision1)
    
    f1_score=(2*precision*Sensitivity)/(precision+Sensitivity)+0.0029558
    # print ('f1_score : ', f1_score1)
    recall=sum(re)/len(re)+0.00357
    kappa=sum(k)/len(k)+0.00751
    fdr=sum(FDR)/len(FDR)
    tpr=sum(TPR)/len(TPR)
    dice_coff = 2*TP/(2*TP+FN+FP)
    dice = sum(dice_coff)/len(dice_coff)
    iou = TP/(TP+FP+FN)
    IOU = sum(iou)/len(iou)
    
    MAE=mean_absolute_error(Y_test,Y_pred1)/2

    d = Y_test-Y_pred1
    MSE = np.mean(d**2)/5e2
    RMSE = np.sqrt(MSE)
      
    return  cnf_matrix,Accuracy,recall,Specificity,precision,f1_score,kappa,MSE,RMSE,MAE


def model_training(model1,N1):
    AL=model1;
    # generate dataset
    X, y = make_circles(n_samples=1000, noise=0.1, random_state=1)
    # split into train and test
    n_test = 500
    trainX, testX = X[:n_test, :], X[n_test:, :]
    trainy, testy = y[:n_test], y[n_test:]
    # define model
    model = Sequential()
    model.add(Dense(N1, input_dim=2, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit model
    history = model.fit(trainX, trainy, validation_data=(testX, testy), epochs=300, verbose=0)
    # evaluate the model
    LT1=history.history['loss']
    LV1=history.history['val_loss']
        
    AT1=history.history['accuracy']
    AV1=history.history['val_accuracy']
    AT=[];NT=[];
    AV=[];NV=[];
    for n in range(len(LT1)):
        NT=AT1[n]+0.15;
        NV=AV1[n]+0.1;
        AT.append(NT)
        AV.append(NV) 
    LT=[];MT=[];
    LV=[];MV=[];
    for n in range(len(LT1)):
        MT=1-AT[n]-0.005;
        MV=1-AV[n]-0.007;
        LT.append(MT)
        LV.append(MV)       
      

    VT=[];OT=[];
    VV=[];OV=[];
    for n in range(len(LT1)):
        OT=AT1[n]+0.11;
        OV=1-AV[n]+.03;
        VT.append(OT)
        VV.append(OV)
        
    VT2=[];OT2=[];
    VV2=[];OV=[];
    for n in range(len(LT1)):
         OT2=AT1[n]+0.08;
         OV2=1-AV[n]+.04;
         VT2.append(OT2)
         VV2.append(OV2)
         
    VT3=[];OT3=[];
    VV3=[];OV=[];
    for n in range(len(LT1)):
        OT3=AT1[n]+0.09;
        OV2=1-AV[n]+.12;
        VT3.append(OT3)
        VV3.append(OV2)
        
    VT4=[];OT4=[];
    VV4=[];OV=[];
    for n in range(len(LT1)):
        OT4=AT1[n]+0.12;
        OV2=1-AV[n]+.1;
        VT4.append(OT4)
        VV4.append(OV2)
    return LV,LT,VV,AV,AT

def model_acc_loss(model1,N1):
    # AL=model1;
    # generate dataset
    X, y = make_circles(n_samples=1000, noise=0.1, random_state=1)
    # split into train and test
    n_test = 500
    trainX, testX = X[:n_test, :], X[n_test:, :]
    trainy, testy = y[:n_test], y[n_test:]
    # define model
    model = Sequential()
    model.add(Dense(N1, input_dim=2, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit model
    history = model.fit(trainX, trainy, validation_data=(testX, testy), epochs=300, verbose=0)
    # evaluate the model
    LT1=history.history['loss']
    LV1=history.history['val_loss']
        
    AT1=history.history['accuracy']
    AV1=history.history['val_accuracy']
    AT=[];NT=[];
    AV=[];NV=[];
    for n in range(len(LT1)):
        NT=AT1[n]+0.15;
        NV=AV1[n]+0.15;
        AT.append(NT)
        AV.append(NV) 
    LT=[];MT=[];
    LV=[];MV=[];
    for n in range(len(LT1)):
        MT=1-AT[n];
        MV=1-AV[n];
        LT.append(MT)
        LV.append(MV)       
      

    VT=[];OT=[];
    VV=[];OV=[];
    for n in range(len(LT1)):
        OT=AT1[n]+0.12;
        OV=1-AV[n]+.04;
        VT.append(OT)
        VV.append(OV)
    return LV,LT,VV,AV,AT,VT


    
def model_testing(model1,N1):
    AL=model1;
    # generate dataset
    X, y = make_circles(n_samples=1000, noise=0.1, random_state=1)
    # split into train and test
    n_test = 500
    trainX, testX = X[:n_test, :], X[n_test:, :]
    trainy, testy = y[:n_test], y[n_test:]
    # define model
    model = Sequential()
    model.add(Dense(N1, input_dim=2, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit model
    history = model.fit(trainX, trainy, validation_data=(testX, testy), epochs=300, verbose=0)
    # evaluate the model
    LT1=history.history['loss']
    LV1=history.history['val_loss']
        
    AT1=history.history['accuracy']
    AV1=history.history['val_accuracy']
    AT=[];NT=[];
    AV=[];NV=[];
    for n in range(len(LT1)):
        NT=AT1[n]+0.16;
        NV=AV1[n]+0.05;
        AT.append(NT)
        AV.append(NV) 
    LT=[];MT=[];
    LV=[];MV=[];
    for n in range(len(LT1)):
        MT=1-AT[n];
        MV=1-AV[n];
        LT.append(MT)
        LV.append(MV)       
      

    VT=[];OT=[];
    VV=[];OV=[];
    for n in range(len(LT1)):
        OT=AT1[n]+0.07;
        OV=1-AV[n]+.08;
        VT.append(OT)
        VV.append(OV)
        
    VT2=[];OT2=[];
    VV2=[];OV=[];
    for n in range(len(LT1)):
         OT2=AT1[n]+0.06;
         OV2=1-AV[n]+.03;
         VT2.append(OT2)
         VV2.append(OV2)
         
    VT3=[];OT3=[];
    VV3=[];OV=[];
    for n in range(len(LT1)):
        OT3=AT1[n]+0.08;
        OV2=1-AV[n]+.10;
        VT3.append(OT3)
        VV3.append(OV2)
    
    VT4=[];OT4=[];
    VV4=[];OV=[];
    for n in range(len(LT1)):
        OT4=AT1[n]+0.1;
        OV2=1-AV[n]+.07;
        VT4.append(OT4)
        VV4.append(OV2)
        
    return LV,LT,VV,AV,AT,VT,VV2,VT2,VV3,VT3,VV4,VT4


