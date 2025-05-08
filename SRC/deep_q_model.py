import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from collections import deque

class DQNetwork(tf.keras.Model):
    def __init__(self, state_size, action_size, learning_rate, name='DQNetwork'):
        super(DQNetwork, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        
        self.conv1 = layers.Conv2D(32, kernel_size=(8,8), strides=(4,4), padding="valid", 
                                   kernel_initializer=tf.keras.initializers.GlorotUniform(), name="conv1")
        self.bn1 = layers.BatchNormalization(epsilon=1e-5, name="batch_norm1")
        
        self.conv2 = layers.Conv2D(64, kernel_size=(4,4), strides=(2,2), padding="valid",
                                   kernel_initializer=tf.keras.initializers.GlorotUniform(), name="conv2")
        self.bn2 = layers.BatchNormalization(epsilon=1e-5, name="batch_norm2")
        
        self.conv3 = layers.Conv2D(128, kernel_size=(4,4), strides=(2,2), padding="valid",
                                   kernel_initializer=tf.keras.initializers.GlorotUniform(), name="conv3")
        self.bn3 = layers.BatchNormalization(epsilon=1e-5, name="batch_norm3")
        
        self.flatten = layers.Flatten()
        self.fc = layers.Dense(512, activation=tf.nn.elu, kernel_initializer=tf.keras.initializers.GlorotUniform(), name="fc1")
        self.output_layer = layers.Dense(action_size, activation=None, kernel_initializer=tf.keras.initializers.GlorotUniform())

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.elu(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = tf.nn.elu(x)
        x = self.conv3(x)
        x = self.bn3(x, training=training)
        x = tf.nn.elu(x)
        x = self.flatten(x)
        x = self.fc(x)
        output = self.output_layer(x)
        return output


class Memory:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        buffer_size = len(self.buffer)
        index = np.random.choice(np.arange(buffer_size), size=batch_size, replace=False)
        return [self.buffer[i] for i in index]