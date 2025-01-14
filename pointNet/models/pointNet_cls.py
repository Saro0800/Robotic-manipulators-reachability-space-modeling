import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers

def input_transform_net(input, batch_size, num_points, K=3):

    input = tf.expand_dims(input, -1)

    x = layers.Conv2D(filters=64,
                      kernel_size=(1,3),
                      padding="valid",
                      strides=(1,1),
                      activation="leaky_relu")(input)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(filters=128,
                      kernel_size=(1,1),
                      padding="valid",
                      strides=(1,1),
                      activation="leaky_relu")(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(filters=1024,
                      kernel_size=(1,1),
                      padding="valid",
                      strides=(1,1),
                      activation="leaky_relu")(x)
    x = layers.BatchNormalization()(x)

    x = layers.MaxPool2D(pool_size=(num_points, 1), strides=(2,2), padding="valid")(x)

    x = layers.Flatten()(x)

    x = layers.Dense(512,
                     activation="leaky_relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256,
                     activation="leaky_relu")(x)
    x = layers.BatchNormalization()(x)

    assert(K==3)
    weights = tf.zeros((256,3*K))

    biases = tf.zeros(3*K)
    
    biases += tf.constant([1,0,0,0,1,0,0,0,1], dtype=tf.float32)
    transform = tf.matmul(x, weights)
    transform = tf.nn.bias_add(transform, biases)

    transform = layers.Reshape([3, K], name="rshape_input_tnet")(transform)

    return transform

def feature_transform_net(input, batch_size, num_points, K=64):

    x = layers.Conv2D(filters=64,
                      kernel_size=(1,1),
                      padding="valid",
                      strides=(1,1),
                      activation="leaky_relu")(input)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(filters=128,
                      kernel_size=(1,1),
                      padding="valid",
                      strides=(1,1),
                      activation="leaky_relu")(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(filters=1024,
                      kernel_size=(1,1),
                      padding="valid",
                      strides=(1,1),
                      activation="leaky_relu")(x)
    x = layers.BatchNormalization()(x)

    x = layers.MaxPool2D(pool_size=(num_points, 1), strides=(2,2), padding="valid")(x)

    x = layers.Flatten()(x)

    x = layers.Dense(512,
                     activation="leaky_relu")(x)
    x = layers.BatchNormalization()(x)

    x = layers.Dense(256,
                     activation="leaky_relu")(x)
    x = layers.BatchNormalization()(x)

    weights = tf.zeros((256,K*K))

    biases = tf.zeros(K*K)
    
    biases += tf.constant(tf.reshape(tf.eye(K),[-1], name="rshape1_feature_tnet"), dtype=tf.float32)
    transform = tf.matmul(x, weights)
    transform = tf.nn.bias_add(transform, biases)

    transform = layers.Reshape([K,K], name="rshape2_feature_tnet")(transform)
    return transform


def get_model(batch_size=8, num_points=2048):
    input = layers.Input((num_points, 3), batch_size=batch_size)

    # input transform net
    input_transform = input_transform_net(input, batch_size=batch_size, num_points=num_points)
    x = tf.matmul(input, input_transform)

    # 2 MLP as Conv2D
    x = tf.expand_dims(x,-1)
    x = layers.Conv2D(filters=64,
                      kernel_size=(1,3),
                      padding="valid",
                      strides=(1,1),
                      activation="leaky_relu")(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(filters=64,
                      kernel_size=(1,1),
                      padding="valid",
                      strides=(1,1),
                      activation="leaky_relu")(x)
    x = layers.BatchNormalization()(x)

    # feature transform net
    feature_transform = feature_transform_net(x, batch_size=batch_size, num_points=num_points)
    net_transformed = tf.matmul(tf.squeeze(x, axis=[2]), feature_transform)

    # 3 MLP as Conv2d
    x = tf.expand_dims(net_transformed, [2])
    x = layers.Conv2D(filters=64,
                      kernel_size=(1,1),
                      padding="valid",
                      strides=(1,1),
                      activation="leaky_relu")(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(filters=128,
                      kernel_size=(1,1),
                      padding="valid",
                      strides=(1,1),
                      activation="leaky_relu")(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(filters=1024,
                      kernel_size=(1,1),
                      padding="valid",
                      strides=(1,1),
                      activation="leaky_relu")(x)
    x = layers.BatchNormalization()(x)

    # max pooling -> GLOBAL FEATURES
    global_features = layers.MaxPool2D(pool_size=(num_points, 1), strides=(2,2), padding="valid")(x)
    global_features = layers.Flatten()(global_features)

    # MLP for parameter estimation
    x = layers.Dense(512,
                     activation="leaky_relu")(global_features)
    # x = layers.Dropout(0.3)(x)
    x = layers.Dense(256,
                     activation="leaky_relu")(x)
    # x = layers.Dropout(0.3)(x)
    out = layers.Dense(3,
                     activation="leaky_relu")(x)   



    model = keras.Model(inputs=input, outputs=out, name=str.split(str.split(__file__,'/')[-1],'.')[0])
    return model

if __name__=="__main__":
    model = get_model(100, 2048)
    model.summary()






