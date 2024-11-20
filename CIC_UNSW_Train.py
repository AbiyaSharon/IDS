import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils import to_categorical
import time
import tensorflow as tf
import ila_smote
import Modified_ViT
import Model

def train():

    ########### reading data and labels ###########
    df = pd.read_csv('datasets/CIC-UNSW/Data.csv')
    df = df.replace(np.nan,0)
    label = pd.read_csv('datasets/CIC-UNSW/Label.csv')
    
    ########### Conerting to array format ###########
    data = np.array(df)
    label = np.array(label)
    
    data = data[:10000,:]
    label = label[:10000,:]
    
    x_train,x_test,Y_train,Y_test=train_test_split(
            data,label,test_size=0.2,random_state=0,stratify=label)
    
    np.save('Y_test_cic_unsw', Y_test)            
    np.save('x_test_cic_unsw', x_test)   
    
    ########### Data normalization ###########
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(x_train)
    
    ########### Data standardization  ###########
    mean = np.mean(normalized_data)  # Calculate the mean of the normalized data
    std_dev = np.std(normalized_data)# Calculate the standard deviation of the normalized data
    standardized_data = (normalized_data - mean) / std_dev# Standardize the data using the formula
    
    ########### Data balancing uiang ILA_SMOTE ###########
    X_resampled, y_resampled = ila_smote.ILA_SMOTE(standardized_data, Y_train)
    
    
    ########### Feature Extraction using modified vision transformer model  ###########
    X_data = np.expand_dims(X_resampled,axis=1)
    Y_data = y_resampled
    
    # Convert NumPy array to TensorFlow tensor
    input_data_tf = tf.convert_to_tensor(X_data, dtype=tf.float32)
    # Pass the tensor through the ViT model to extract features
    
    vit_model = Modified_ViT.mod_ViT()
    extracted_features = vit_model(input_data_tf)
    feature = np.array(extracted_features)
    features = np.expand_dims(feature,axis=1)
    
    # Example usage
    input_shape = features[0].shape  # Updated input shape as per your requirements
    num_classes = 10  # Change this to your number of classes
    y_train = to_categorical(Y_data, num_classes=10)
    
    # stime = time.time()
    
    model =Model.Datt_GBiGRU(input_shape, num_classes)
    
    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # # Summary of the model
    # model.summary()
    
    model.fit(features,y_train,epochs=300,batch_size=64)
    
    # etime = time.time()
    # comp = etime - stime
    

    
    # model.save('model_cic_unsw.h5')
    
