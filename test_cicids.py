import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import Modified_ViT
import plot_cicids


########### loading test data ###########

X_test = np.load('x_test_cicids.npy')
Y_test = np.load('Y_test_cicids.npy')


########### Data normalization ###########
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(X_test)

########### Data standardization  ###########
mean = np.mean(normalized_data)  # Calculate the mean of the normalized data
std_dev = np.std(normalized_data)# Calculate the standard deviation of the normalized data
standardized_data = (normalized_data - mean) / std_dev# Standardize the data using the formula

########### Feature Extraction using modified vision transformer model  ###########
X_data = np.expand_dims(standardized_data,axis=1)

# Convert NumPy array to TensorFlow tensor
input_data_tf = tf.convert_to_tensor(X_data, dtype=tf.float32)
# Pass the tensor through the ViT model to extract features

vit_model = Modified_ViT.mod_ViT()
extracted_features = vit_model(input_data_tf)
feature = np.array(extracted_features)
features = np.expand_dims(feature,axis=1)

########### Prdiction using Dual attention assisted ghost bidirectional gated recurrent neural network (Datt-GBiGRU) model   ###########

model = load_model("model_cicids.h5")
#### Feature extraction using Atten_ResNet #####
Y_pred = model.predict(features)



# ------ OverAll Performance -------- #
print()
print("Loading the overall performance..")
print()
Result=plot_cicids.plot(Y_pred,Y_test)
print()
print("Process Completed.")