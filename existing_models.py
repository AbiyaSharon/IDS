import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense, Dropout, GRU

from sklearn.preprocessing import LabelEncoder

def CNN(X_train,y_train,num_classes):
    # Create CNN model
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(X_train[0].shape, 1, 1)),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')  # Replace num_classes with the number of classes
    ])
    
    # Compile model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # Train model
    model.fit(X_train, y_train, epochs=300, batch_size=64)
    
def BILSTM(X_train,y_train,num_classes):    
    # Create BiLSTM model
    model = Sequential([
        Embedding(input_dim=2, output_dim=2, input_length=X_train[0].shape),
        Bidirectional(LSTM(units=64, return_sequences=True)),
        Bidirectional(LSTM(units=64)),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')  # Replace num_classes with the number of classes
    ])
    
    # Compile model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Train model
    model.fit(X_train, y_train,epochs=300, batch_size=64)
        

def GRU_(X_train,y_train,num_classes): 
    # Create GRU model
    model = Sequential([
        Embedding(input_dim=2, output_dim=2, input_length=X_train[0].shape),
        GRU(units=64, return_sequences=True),
        GRU(units=64),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')  # Replace num_classes with the number of classes
    ])
    
    # Compile model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Train model
    model.fit(X_train, y_train,epochs=300, batch_size=64)
    
def BiGRU_(X_train,y_train,num_classes):     
    # Create BiGRU model
    model = Sequential([
        Embedding(input_dim=2, output_dim=2, input_length=X_train[0].shape),
        Bidirectional(GRU(units=64, return_sequences=True)),
        Bidirectional(GRU(units=64)),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')  # Replace num_classes with the number of classes
    ])
    
    # Compile model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Train model
    model.fit(X_train, y_train,epochs=300, batch_size=64)
        
        
