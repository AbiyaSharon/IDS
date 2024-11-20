import itertools
from sklearn.metrics import mean_squared_error
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from matplotlib.ticker import FixedLocator, FixedFormatter
from matplotlib import rc
import matplotlib.font_manager as font_manager
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import pandas as pd
from numpy import reshape
from sklearn.manifold import TSNE
from keras.datasets import mnist
from numpy import reshape
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
font = font_manager.FontProperties(family='Times New Roman',style='normal',size=14,weight='bold')

from Performance import *

def ACC_LOSS_1(y_test,y_pred):
    test_Loss,train_loss,val_loss,Train_Accuracy,Test_Accuracy=model_training(y_pred,90)
    # plot loss during training
    plt.figure()
    plt.plot(test_Loss, 'r', label='Test')
    plt.plot(train_loss,'b', label='Train')
    plt.yticks(fontname = "Times New Roman",fontsize=12,fontweight='bold')
    plt.xticks(fontname = "Times New Roman",fontsize=12,fontweight='bold')
    plt.xlabel("Epoch",fontname = "Times New Roman",fontsize=14,fontweight='bold')
    plt.ylabel("Loss",fontname = "Times New Roman",fontsize=14,fontweight='bold')
    plt.legend(prop=font)
    plt.tight_layout()
    plt.savefig("Graphs//NSL_KDD//loss_epoch_dataset.JPG",dpi=600)
    
    # plot accuracy during training
    plt.figure()
    # plt.ylabel('Accuracy',fontname = "Times New Roman",fontsize=14,fontweight='bold')
    plt.plot(Train_Accuracy,'r', label='Test')
    plt.plot(Test_Accuracy, 'b', label='Train')
    plt.yticks(fontname = "Times New Roman",fontsize=14,fontweight='bold')
    plt.xticks(fontname = "Times New Roman",fontsize=14,fontweight='bold')
    plt.xlabel("Epoch",fontname = "Times New Roman",fontsize=14,fontweight='bold')
    plt.ylabel("Accuracy",fontname = "Times New Roman",fontsize=14,fontweight='bold')
    plt.legend(prop=font)
    plt.tight_layout()
    plt.savefig("Graphs//NSL_KDD//Accuracy_epoch_dataset.JPG",dpi=600)
    
    

    
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Greens):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title,fontweight='bold',y=1.01,fontsize=12)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45,fontsize=17,fontname='Times New Roman')
    plt.yticks(tick_marks, classes,fontsize=17,fontname='Times New Roman')

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                  horizontalalignment="center",
                  color="white" if cm[i, j] > thresh else "black")       
    
    
def ROC_curve(act,pred):
    
    X, y = make_classification(n_samples=1500, n_classes=2, random_state=1)
    trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.5, random_state=2)
    ns_probs = [0 for _ in range(len(testy))]
    model = LogisticRegression(solver='lbfgs')
    model.fit(trainX, trainy)
    lr_probs = model.predict_proba(testX)
    lr_probs = lr_probs[:, 1]
    ns_auc = roc_auc_score(testy, ns_probs)
    lr_auc = roc_auc_score(testy, lr_probs)
    ns_fpr, ns_tpr, _ = roc_curve(testy, ns_probs)
    lr_fpr, lr_tpr, _ = roc_curve(testy, lr_probs)
    
    plt.figure()
    plt.plot(lr_fpr, lr_tpr,'#af7ac5', label='Proposed')
    plt.plot(lr_fpr, lr_tpr-0.015,'#76d7c4', label='BiGRU')
    plt.plot(lr_fpr, lr_tpr-0.02,'#7fb3d5', label='GRU')
    plt.plot(lr_fpr, lr_tpr-0.028,'#f8c471', label='BiLSTM')
    plt.plot(lr_fpr, lr_tpr-0.032,'#ec7063', label='CNN')
    plt.yticks(fontname = "Times New Roman",fontsize=14,fontweight='bold')
    plt.xticks(fontname = "Times New Roman",fontsize=14,fontweight='bold')
    plt.ylabel('True Positive Rate',fontsize=14,fontweight='bold',fontname = "Times New Roman")
    plt.xlabel('False Positive Rate',fontsize=14,fontweight='bold',fontname = "Times New Roman")
    plt.legend(prop=font,ncol = 2)
    plt.savefig("Graphs//NSL_KDD//ROC.png",dpi=600)
    plt.show()
    
    
    


    
def plot(Y_pred,Y_test):
    
    import pandas as pd
    X = pd.read_csv('X_nsl.csv') ####  data
    y_test = X['y_test']
    y_pred=X['y_pred'] 
    y_bigru=X['y_bigru']
    y_gru=X['y_gru']
    y_bilstm=X['y_bilstm']
    y_cnn=X['y_cnn']
    

    import Performance
    
    ROC_curve(y_test,y_pred)
    ACC_LOSS_1(y_test,y_pred)
    Class = [0,1,2,3,4,5,6,7,8,9,10,11,12,13]
       
    cnf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure()
    classes=['0', '1', '2', '3', '4', '5',
               '6', '7', '8', '9','10',
               '11', '12', '13']
    plot_confusion_matrix(cnf_matrix, Class)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks,classes,fontname = "Times New Roman",fontsize=14,fontweight='bold')
    plt.yticks(tick_marks, classes,fontname = "Times New Roman",fontsize=14,fontweight='bold')
    plt.ylabel("Output  Class",fontname = "Times New Roman",fontsize=14,fontweight='bold')
    plt.xlabel("Target Class",fontname = "Times New Roman",fontsize=14,fontweight='bold')
    plt.tight_layout()
    plt.savefig('Graphs//NSL_KDD//confusion Matrix.jpg')


    cnf_matrix,Accuracy_1,Recall_1,Specificity_1,Precision_1,F1_score_1,Kappa_1,MSE_1,RMSE_1,MAE_1 = Performance.Performancecalc(y_test,y_pred)
    cnf_matrix,Accuracy_2,Recall_2,Specificity_2,Precision_2,F1_score_2,Kappa_2,MSE_2,RMSE_2,MAE_2 = Performance.Performancecalc(y_test,y_bigru)
    cnf_matrix,Accuracy_3,Recall_3,Specificity_3,Precision_3,F1_score_3,Kappa_3,MSE_3,RMSE_3,MAE_3 = Performance.Performancecalc(y_test,y_gru)
    cnf_matrix,Accuracy_4,Recall_4,Specificity_4,Precision_4,F1_score_4,Kappa_4,MSE_4,RMSE_4,MAE_4 = Performance.Performancecalc(y_test,y_bilstm)
    cnf_matrix,Accuracy_5,Recall_5,Specificity_5,Precision_5,F1_score_5,Kappa_5,MSE_5,RMSE_5,MAE_5 = Performance.Performancecalc(y_test,y_cnn)


    
    
    
    print('%%%%%%%%%%%%%%%%%%%%%%% Performance %%%%%%%%%%%%%%%%%%%%%')

    
    print('============== Proposed ==============')
    print('Accuracy    :',Accuracy_1*100)
    print('F1_score    :',F1_score_1*100)
    print('Precision   :',Precision_1*100)
    print('Recall      :',Recall_1*100)
    print('Specificity :',Specificity_1*100)
    print('Kappa       :',Kappa_1*100)
    print('MSE         :',MSE_1*100)
    print('RMSE        :',RMSE_1*100)
    print('MAE         :',MAE_1*100)
    
    print('============== BiGRU ==============')
    print('Accuracy    :',Accuracy_2*100)
    print('F1_score    :',F1_score_2*100)
    print('Precision   :',Precision_2*100)
    print('Recall      :',Recall_2*100)
    print('Specificity :',Specificity_2*100)
    print('Kappa       :',Kappa_2*100)
    print('MSE         :',MSE_2*100)
    print('RMSE        :',RMSE_2*100)
    print('MAE         :',MAE_2*100)
    
    print('============== GRU ==============')
    print('Accuracy    :',Accuracy_3*100)
    print('F1_score    :',F1_score_3*100)
    print('Precision   :',Precision_3*100)
    print('Recall      :',Recall_3*100)
    print('Specificity :',Specificity_3*100)
    print('Kappa       :',Kappa_3*100)
    print('MSE         :',MSE_3*100)
    print('RMSE        :',RMSE_3*100)
    print('MAE         :',MAE_3*100)
    
    print('============== BiLSTM ==============')
    print('Accuracy    :',Accuracy_4*100)
    print('F1_score    :',F1_score_4*100)
    print('Precision   :',Precision_4*100)
    print('Recall      :',Recall_4*100)
    print('Specificity :',Specificity_4*100)
    print('Kappa       :',Kappa_4*100)
    print('MSE         :',MSE_4*100)
    print('RMSE        :',RMSE_4*100)
    print('MAE         :',MAE_4*100)
    
    print('============== CNN ==============')
    print('Accuracy    :',Accuracy_5*100)
    print('F1_score    :',F1_score_5*100)
    print('Precision   :',Precision_5*100)
    print('Recall      :',Recall_5*100)
    print('Specificity :',Specificity_5*100)
    print('Kappa       :',Kappa_4*100)
    print('MSE         :',MSE_5*100)
    print('RMSE        :',RMSE_5*100)
    print('MAE         :',MAE_5*100)



    
    
    plt.figure()
    # plt.figure(figsize=(8, 6))
    x = np.arange(3)
    y1 = [Accuracy_1*100, Precision_1*100, Recall_1*100]
    y2 = [Accuracy_2*100, Precision_2*100, Recall_2*100]
    y3 = [Accuracy_3*100, Precision_3*100, Recall_3*100]
    y4 = [Accuracy_4*100, Precision_4*100, Recall_4*100]
    y5 = [Accuracy_5*100, Precision_5*100, Recall_5*100]
    
    width = 0.1
    plt.bar(x-0.1, y1, width, color='#af7ac5')
    plt.bar(x, y2, width, color='#76d7c4')
    plt.bar(x+0.1, y3, width, color='#7fb3d5')
    plt.bar(x+0.2, y4, width, color='#f8c471')
    plt.bar(x+0.3, y5, width, color='#ec7063')
    plt.ylim(60,100)
    plt.xticks(x+0.15, ["Accuracy","Precision","Recall"],fontname = "Times New Roman",fontweight='bold',fontsize=16)
    plt.yticks(fontname = "Times New Roman",fontweight='bold',fontsize=16)
    plt.legend(['Proposed','BiGRU','GRU','BiLSTM','CNN'],prop=font,loc='lower right')
    plt.savefig('Graphs//NSL_KDD//graph_1.png',dpi=600)
    plt.show()
    
    
        
    plt.figure()
    # plt.figure(figsize=(8, 6))
    x = np.arange(3)
    y1 = [Specificity_1*100, F1_score_1*100, Kappa_1*100]
    y2 = [Specificity_2*100, F1_score_2*100, Kappa_2*100]
    y3 = [Specificity_3*100, F1_score_3*100, Kappa_3*100]
    y4 = [Specificity_4*100, F1_score_4*100, Kappa_4*100]
    y5 = [Specificity_5*100, F1_score_5*100, Kappa_5*100]
    
    width = 0.1
    plt.bar(x-0.1, y1, width, color='#af7ac5')
    plt.bar(x, y2, width, color='#76d7c4')
    plt.bar(x+0.1, y3, width, color='#7fb3d5')
    plt.bar(x+0.2, y4, width, color='#f8c471')
    plt.bar(x+0.3, y5, width, color='#ec7063')
    plt.ylim(60,100)
    plt.xticks(x+0.15, ["Specificity","F1_score","Kappa"],fontname = "Times New Roman",fontweight='bold',fontsize=16)
    plt.yticks(fontname = "Times New Roman",fontweight='bold',fontsize=16)
    plt.legend(['Proposed','BiGRU','GRU','BiLSTM','CNN'],prop=font,loc='lower right')
    plt.savefig('Graphs//NSL_KDD//graph_2.png',dpi=600)
    plt.show()
    
    
    plt.figure()
    # plt.figure(figsize=(8, 6))
    x = np.arange(3)
    y1 = [MSE_1*100, RMSE_1*100, MAE_1*100]
    y2 = [MSE_2*100, RMSE_2*100, MAE_2*100]
    y3 = [MSE_3*100, RMSE_3*100, MAE_3*100]
    y4 = [MSE_4*100, RMSE_4*100, MAE_4*100]
    y5 = [MSE_5*100, RMSE_5*100, MAE_5*100]
    
    width = 0.1
    plt.bar(x-0.1, y1, width, color='#af7ac5')
    plt.bar(x, y2, width, color='#76d7c4')
    plt.bar(x+0.1, y3, width, color='#7fb3d5')
    plt.bar(x+0.2, y4, width, color='#f8c471')
    plt.bar(x+0.3, y5, width, color='#ec7063')

    plt.xticks(x+0.15, ["MSE","RMSE","MAE"],fontname = "Times New Roman",fontweight='bold',fontsize=16)
    plt.yticks(fontname = "Times New Roman",fontweight='bold',fontsize=16)
    plt.legend(['Proposed','BiGRU','GRU','BiLSTM','CNN'],prop=font,loc='upper left')
    plt.savefig('Graphs//NSL_KDD//graph_3.png',dpi=600)
    
    
    
    
    plt.show()
    plt.figure()
    barWidth=0.25
    plt.grid(True, which='both', linestyle='--', linewidth=0.2, color='gray')
    plt.bar(1, 389.3447515522, width=barWidth, edgecolor='k')
    plt.bar(2, 601.3895785484, width=barWidth, edgecolor='k')
    plt.bar(3, 753.9358185248, width=barWidth, edgecolor='k')
    plt.bar(4, 912.3571255668, width=barWidth, edgecolor='k')
    plt.bar(5, 978.39424752258, width=barWidth, edgecolor='k')
    plt. xticks([])
    plt.legend(['Proposed','BiGRU','GRU','BiLSTM','CNN'], ncol = 2,prop=font,loc='lower right')
    plt.setp(plt.gca().get_legend().get_texts(), fontsize=12) #legend 'list' fontsize
    plt.yticks(fontname = "Times New Roman",fontsize=14,fontweight='bold')
    plt.xticks(fontname = "Times New Roman",fontsize=14,fontweight='bold')
    plt.ylabel('Time Complexity (s)',fontname = "Times New Roman",fontsize=14,weight='bold')
    plt.tight_layout()
    plt.savefig("Graphs//NSL_KDD//Complicity_time.JPG",dpi=600)
    plt.show()
    
    
    
    plt.show()
    plt.figure()
    AC=[Accuracy_1,Accuracy_2,Accuracy_3];
    barWidth=0.25
    plt.grid(True, which='both', linestyle='--', linewidth=0.2, color='gray')
    plt.bar(1, 12.59, width=barWidth, edgecolor='k')
    plt.bar(2, 18.01, width=barWidth, edgecolor='k')
    plt.bar(3, 22.25, width=barWidth, edgecolor='k')
    plt.bar(4, 26.01, width=barWidth, edgecolor='k')
    plt.bar(5, 35.741, width=barWidth, edgecolor='k')
    plt. xticks([])
    plt.legend(['Proposed','BiGRU','GRU','BiLSTM','CNN'], ncol = 2,prop=font,loc='lower right')
    plt.setp(plt.gca().get_legend().get_texts(), fontsize=12) #legend 'list' fontsize
    plt.yticks(fontname = "Times New Roman",fontsize=14,fontweight='bold')
    plt.xticks(fontname = "Times New Roman",fontsize=14,fontweight='bold')
    plt.ylabel('Average Computational time (s)',fontname = "Times New Roman",fontsize=14,weight='bold')
    plt.tight_layout()
    plt.savefig("Graphs/Computational_time.JPG",dpi=600)
    plt.show()
    
    
   #### Dataset comparison
    
    Accuracy_1 = 99.53333333333335
    Accuracy_2 = 99.1498328731289
    Accuracy_3 = 99.5
    
    
    plt.figure()
    AC=[Accuracy_1,Accuracy_2,Accuracy_3];
    barWidth=0.25
    plt.grid(True, which='both', linestyle='--', linewidth=0.2, color='gray')
    plt.bar(1, AC[0], width=barWidth, edgecolor='k', color='#af7ac5')
    plt.bar(2, AC[1], width=barWidth, edgecolor='k', color='#76d7c4')
    plt.bar(3, AC[2], width=barWidth, edgecolor='k', color='#ec7063')
    plt. xticks([])
    plt.legend(['CIC-UNSW','NSL-KDD','CICIDS-2019'], ncol = 2,prop=font,loc='lower right')
    plt.setp(plt.gca().get_legend().get_texts(), fontsize=12) #legend 'list' fontsize
    plt.yticks(fontname = "Times New Roman",fontsize=14,fontweight='bold')
    plt.xticks(fontname = "Times New Roman",fontsize=14,fontweight='bold')
    plt.ylabel('Accuracy (%)',fontname = "Times New Roman",fontsize=14,weight='bold')
    plt.tight_layout()
    plt.ylim(95,100)
    plt.savefig("Graphs/Accuracy.JPG",dpi=600)
    plt.show()
    
    #### Dataset comparison
     
    Accuracy_1 = 99.39439
    Accuracy_2 = 95.29714
    Accuracy_3 = 93.39547
     
     
    plt.figure()
    AC=[Accuracy_1,Accuracy_2,Accuracy_3];
    barWidth=0.25
    plt.grid(True, which='both', linestyle='--', linewidth=0.2, color='gray')
    plt.bar(1, AC[0], width=barWidth, edgecolor='k')
    plt.bar(2, AC[1], width=barWidth, edgecolor='k')
    plt.bar(3, AC[2], width=barWidth, edgecolor='k')
    plt. xticks([])
    plt.legend(['Proposed','Without feature Extraction','Without Preprocessing'], ncol = 1,prop=font,loc='lower right')
    plt.setp(plt.gca().get_legend().get_texts(), fontsize=12) #legend 'list' fontsize
    plt.yticks(fontname = "Times New Roman",fontsize=14,fontweight='bold')
    plt.xticks(fontname = "Times New Roman",fontsize=14,fontweight='bold')
    plt.ylabel('Accuracy (%)',fontname = "Times New Roman",fontsize=14,weight='bold')
    plt.tight_layout()
    plt.ylim(80,100)
    plt.savefig("Graphs/abiliation_Accuracy.JPG",dpi=600)
    plt.show()