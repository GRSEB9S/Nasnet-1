from keras.applications.nasnet import NASNetLarge, NASNetMobile
from keras.layers import Dense,Conv2D
from keras.models import Model

def build_model():
      
    base_model = NASNetLarge(include_top = False, pooling = 'avg')
    fc_1=Dense(4096,activation='relu')(base_model.output)
    output=Dense(4096,activation='softmax')(fc_1)
    model=Model(input=base_model.input,output=output)
    return model

model=build_model()
model.summary()