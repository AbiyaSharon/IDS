import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import time
import ila_smote
import Modified_ViT
import Model


def train():
    ########### reading data and labels ###########
    df = pd.read_csv('datasets/nsl-kdd/kdd_train.csv')
    
    df = df.replace(np.nan,0)
    
    
    df['protocol_type'].replace(to_replace=['tcp', 'udp', 'icmp'],
                             value=[0,1,2],inplace=True)
    
    df=df.drop(['labels_name'], axis=1)
    
    df['service'].replace(to_replace=['ftp_data', 'other', 'private', 'http', 'remote_job', 'name',
           'netbios_ns', 'eco_i', 'mtp', 'telnet', 'finger', 'domain_u',
           'supdup', 'uucp_path', 'Z39_50', 'smtp', 'csnet_ns', 'uucp',
           'netbios_dgm', 'urp_i', 'auth', 'domain', 'ftp', 'bgp', 'ldap',
           'ecr_i', 'gopher', 'vmnet', 'systat', 'http_443', 'efs', 'whois',
           'imap4', 'iso_tsap', 'echo', 'klogin', 'link', 'sunrpc', 'login',
           'kshell', 'sql_net', 'time', 'hostnames', 'exec', 'ntp_u',
           'discard', 'nntp', 'courier', 'ctf', 'ssh', 'daytime', 'shell',
           'netstat', 'pop_3', 'nnsp', 'IRC', 'pop_2', 'printer', 'tim_i',
           'pm_dump', 'red_i', 'netbios_ssn', 'rje', 'X11', 'urh_i',
           'http_8001', 'aol', 'http_2784', 'tftp_u', 'harvest'],
                             value=list(range(70)),inplace=True)
    
    df['flag'].replace(to_replace=['SF', 'S0', 'REJ', 'RSTR', 'SH', 
                                   'RSTO', 'S1', 'RSTOS0', 'S3','S2', 'OTH'],
                             value=list(range(11)),inplace=True)
    
    
    
    data=df.iloc[:,:-1]
    label=df.iloc[:,-1]
    
    
    ########### Conerting to array format ###########
    data = np.array(df)
    label = np.array(label)
    
    x_train,x_test,Y_train,Y_test=train_test_split(
            data,label,test_size=0.2,random_state=0,stratify=label)
    
    np.save('Y_test_nsl_kdd', Y_test)            
    np.save('x_test_nsl_kdd', x_test)   
    
    
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
    num_classes = 21  # Change this to your number of classes
    y_train = to_categorical(Y_data, num_classes=21)
    
    # stime = time.time()
    
    model =Model.Datt_GBiGRU(input_shape, num_classes)
    
    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # # Summary of the model
    # model.summary()
    
    model.fit(features,y_train,epochs=300,batch_size=64)
    
    # etime = time.time()
    # comp = etime - stime
    
    # model.save('model_nsl_kdd.h5')