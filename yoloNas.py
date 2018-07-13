from keras.applications.nasnet import NASNetLarge, NASNetMobile
from keras.layers import Dense,Conv2D,Input,Flatten,MaxPooling2D
from keras.models import Model
from keras import losses

import keras.backend as K
import tensorflow as tf
import sys
import cv2



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
    model=Model(inputs=input_tensor,outputs=output)
    model.compile(optimizer='adam',loss=losses.binary_crossentropy)
    return model

def build_model_conv(input_image):

    input_tensor=Input(shape=(input_image))
    base_model = NASNetLarge(input_shape=input_image,input_tensor=input_tensor,include_top = False,weights=None)
    
        
    conv_1=Conv2D(input_shape=(160,160,4033),filters=1,kernel_size=(1,1),activation='sigmoid',padding='valid')(base_model.output)
    maxpool=MaxPooling2D(pool_size=(4,4))(conv_1)
    model=Model(inputs=input_tensor,outputs=maxpool)
    model.compile(optimizer='adam',loss=losses.binary_crossentropy)
    return model

def easy_model(input_image):
    input_tensor=Input(shape=(input_image))
    conv_1=Conv2D(input_shape=input_image,filters=256,kernel_size=(3,3))(input_tensor)
    conv_2=Conv2D(filters=256,kernel_size=(3,3))(conv_1)
    conv_3=Conv2D(filters=256,kernel_size=(3,3))(conv_2) 
    flat=Flatten()(conv_3)
    output=Dense(1024)(flat)
    model=Model(inputs=input_tensor,outputs =output)
    model.compile(optimizer='adam',loss=losses.binary_crossentropy)
    return model    

#model=build_model_conv([5120,5120,3])
#model.summary()

label_row=[]
input_data=cv2.imread('image.jpg')
input_data/=255
label_txt=open('label.txt','r')
for i in range(40):
    label_row.append(label_txt.read())


if __name__ == "__main__":
    model=build_model_conv([5120,5120,3])
    model.summary()
    model.fit(x=input_data,y=y_label,batch_size=1,epochs=10)(
    model.predict(x,batch_size=1)
