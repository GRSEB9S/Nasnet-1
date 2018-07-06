from keras.applications.nasnet import NASNetLarge, NASNetMobile
from keras.layers import Dense,Conv2D,Input,Flatten,Conv3D
from keras.models import Model
from keras import losses

import keras.backend as K
import tensorflow as tf
import sys

def loss_fn(y_true, y_pred):
    #y_true = tf.squeeze(y_true)
    #y_true = tf.cast(y_true, dtype=tf.int32)
    #return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
    #                                        labels=y_true, logits=y_pred))
    y_pred = K.reshape(y_pred, (-1, K.int_shape(y_pred)[-1]))
    log_softmax = tf.nn.log_softmax(y_pred)

    y_true = K.one_hot(tf.to_int32(K.flatten(y_true)), K.int_shape(y_pred)[-1])
    #y_true = tf.stack(unpacked[:-1], axis=-1)

    cross_entropy = -K.sum(y_true * log_softmax, axis=1)
    cross_entropy_mean = K.mean(cross_entropy)

    return cross_entropy_mean

def ensemble_loss_fn(y_true, y_pred):
    
    """
    default image size = 224, default last_dim = 6
    """
    batch_size = 10
    total_loss = tf.Variable(0, dtype=tf.float32)
    for idx in range(0, 3):
        tmp = tf.slice(y_pred, [idx, 0, 0, 0], [1, 224, 224, 6])
        tmp = tf.squeeze(tmp)
        if idx == 0:
            total_loss = loss_fn(y_true, tmp)
        else:
            total_loss = tf.add(total_loss, loss_fn(y_true, tmp))
    return total_loss

def build_model_flat(input_image):

    input_tensor=Input(shape=(input_image))
    base_model = NASNetLarge(input_shape=input_image,input_tensor=input_tensor,include_top = False,weights=None)
    flat=Flatten()(base_model.output)
    fc_1=Dense(4096,activation='relu')(flat)
    output=Dense(10*10,activation='sigmoid')(fc_1)
    model=Model(input=input_tensor,output=output)
    model.compile(optimizer='adam',loss=losses.binary_crossentropy)
    return model

def build_model_conv(input_image):

    input_tensor=Input(shape=(input_image))
    base_model = NASNetLarge(input_shape=input_image,input_tensor=input_tensor,include_top = False,weights=None)
    conv_1=Conv2D(input_shape=(160,160,409),filters=1,kernel_size=(1,1),activation='sigmoid',padding='valid')(base_model.output)
    flat=Flatten()(conv_1)
    model=Model(input=input_tensor,output=conv_1)
    model.compile(optimizer='adam',loss=losses.binary_crossentropy)
    return model

model=build_model_conv([5120,5120,3])
model.summary()
