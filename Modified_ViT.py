import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
class PatchEmbedding(layers.Layer):
    def __init__(self, embedding_dim):
        super(PatchEmbedding, self).__init__()
        self.projection = layers.Dense(embedding_dim)  # Project 12 features to embedding_dim
        self.position_embedding = layers.Embedding(input_dim=100, output_dim=embedding_dim)
    def call(self, x):
        positions = tf.range(start=0, limit=100, delta=1)
        x = self.projection(x)
        x += self.position_embedding(positions)
        return x
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)
    def call(self, inputs, training):
        attention_output = self.attention(inputs, inputs)
        attention_output = self.dropout1(attention_output, training=training)
        out1 = self.layernorm1(inputs + attention_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
class VisionTransformer(tf.keras.Model):
    def __init__(self, embedding_dim, num_heads, ff_dim, num_transformer_blocks, num_features):
        super(VisionTransformer, self).__init__()
        self.patch_embed = PatchEmbedding(embedding_dim=embedding_dim)
        self.transformer_blocks = [TransformerBlock(embed_dim=embedding_dim, num_heads=num_heads, ff_dim=ff_dim)
                                   for _ in range(num_transformer_blocks)]
        self.global_avg_pool = layers.GlobalAveragePooling1D()
        self.feature_output = layers.Dense(num_features, activation=None)  # Extracted features
    def call(self, inputs):
        x = self.patch_embed(inputs)
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)
        x = self.global_avg_pool(x)
        return self.feature_output(x)
    
def mod_ViT():  
    # Define the parameters for the ViT model
    embedding_dim = 64  # You can increase this if needed
    num_heads = 8  # Number of attention heads
    ff_dim = 128  # Feed-forward layer dimension
    num_transformer_blocks = 4  # Number of transformer layers/blocks
    num_features = 128  # Number of output features for extraction
    # Create the Vision Transformer model for feature extraction
    vit_model = VisionTransformer(
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        ff_dim=ff_dim,
        num_transformer_blocks=num_transformer_blocks,
        num_features=num_features)
    return vit_model

# X_ = np.expand_dims(X_resampled,axis=1)

# # Convert NumPy array to TensorFlow tensor
# input_data_tf = tf.convert_to_tensor(X_[:1000,:,:], dtype=tf.float32)
# # Pass the tensor through the ViT model to extract features
# extracted_features = vit_model(input_data_tf)
# print(extracted_features.shape)


# feature = np.array(extracted_features)
