import tensorflow as tf
from tensorflow.keras.layers import Layer, Input, Dense, GlobalAveragePooling1D, Multiply, Reshape
from tensorflow.keras.models import Model

# Channel Attention Block
class ChannelAttention(Layer):
    def __init__(self, filters, ratio=8, **kwargs):
        super(ChannelAttention, self).__init__(**kwargs)
        self.global_avg_pool = GlobalAveragePooling1D()
        self.fc1 = Dense(filters // ratio, activation='relu')
        self.fc2 = Dense(filters, activation='sigmoid')

    def call(self, inputs):
        # Global Average Pooling along the time dimension
        avg_out = self.global_avg_pool(inputs)  # Shape (batch_size, channels)
        
        # Reduce channel dimension
        avg_out = self.fc1(avg_out)  # Dense layer for dimensionality reduction
        avg_out = self.fc2(avg_out)  # Final Dense layer with sigmoid activation
        
        # Expand avg_out back to the original dimensions of inputs for element-wise multiplication
        avg_out = tf.expand_dims(avg_out, axis=1)  # Shape (batch_size, 1, channels)
        
        # Ensure avg_out has the same time dimension as inputs by tiling
        avg_out = tf.tile(avg_out, [1, inputs.shape[1], 1])  # Shape (batch_size, timesteps, channels)
        
        # Element-wise multiplication of inputs and avg_out
        return Multiply()([inputs, avg_out])

# Dual Attention Block (combines channel and other types of attention)
class DualAttention(Layer):
    def __init__(self, filters, **kwargs):
        super(DualAttention, self).__init__(**kwargs)
        self.channel_attention = ChannelAttention(filters)

    def call(self, inputs):
        # Apply Channel Attention
        x = self.channel_attention(inputs)
        return x  # Add more attention mechanisms here if needed

# Define the Model with Dual Attention and Bidirectional GRU
def Datt_GBiGRU(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    
    # BiGRU layer
    x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(64, return_sequences=True))(inputs)
    
    # Apply Dual Attention
    x = DualAttention(filters=128)(x)  # Updated to match the input channel size (128)

    # Flatten the output before passing it to the Dense layer
    x = tf.keras.layers.Flatten()(x)

    # Dense layer for classification
    x = Dense(num_classes, activation='softmax')(x)

    # Create Model
    model = Model(inputs=inputs, outputs=x)
    
    return model

