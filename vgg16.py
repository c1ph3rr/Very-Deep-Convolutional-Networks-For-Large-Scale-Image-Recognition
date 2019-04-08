from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from tensorflow.keras import Input

def conv_block(filters, name):
    x = Conv2D(filters=filters, kernel_size=3, padding='same', activation='relu', name=name)
    return x

def pool_block(pool_size, strides, name):
    pool = MaxPool2D(pool_size=pool_size, strides=strides, name=name)
    return pool

def dense(units, activation, name):
    fc = Dense(units=units, activation=activation, name=name)
    return fc

inp = Input(shape=(224,224,3), name='Input')

block1 = conv_block(64, 'layer1') (inp)
block1 = conv_block(64, 'layer2') (block1)
pool1 = pool_block(2, 2, 'pool1') (block1)

block2 = conv_block(128, 'layer3') (pool1)
block2 = conv_block(128, 'layer4') (block2)
pool2 = pool_block(2, 2, 'pool2') (block2)

block3 = conv_block(256, 'layer5') (pool2)
block3 = conv_block(256, 'layer6') (block3)
block3 = conv_block(256, 'layer7') (block3)
block3 = conv_block(256, 'layer8') (block3)
pool3 = pool_block(2, 2, 'pool3') (block3)

block4 = conv_block(512, 'layer9') (pool3)
block4 = conv_block(512, 'layer10') (block4)
block4 = conv_block(512, 'layer11') (block4)
block4 = conv_block(512, 'layer12') (block4)
pool4 = pool_block(2, 2, 'pool4') (block4)

block5 = conv_block(512, 'layer13') (pool4)
block5 = conv_block(512, 'layer14') (block5)
block5 = conv_block(512, 'layer15') (block5)
block5 = conv_block(512, 'layer16') (block5)
pool5 = pool_block(2, 2, 'pool5') (block5)

flatten = Flatten(name='flatten') (pool5)

fc1 = dense(4096, 'relu', 'fc1') (flatten)
drop = Dropout(0.5)
fc2 = dense(4096, 'relu', 'fc2') (fc1)
fc3 = dense(1000, 'relu', 'fc3') (fc2)

output = dense(10, 'softmax', 'output') (fc3)

model = Model(inputs=inp, outputs=output)
model.summary()