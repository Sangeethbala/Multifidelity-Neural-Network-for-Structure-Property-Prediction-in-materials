"""
Model mapping I to I
"""
import numpy as np
from sklearn.model_selection import train_test_split
import keras
import cdata
from keras.layers import Conv2D, MaxPooling2D, Dropout, Activation, BatchNormalization, Input, Conv2DTranspose
from keras import optimizers
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Model
from keras.callbacks import CSVLogger
#
# Image configuration
#
final_image_shape = [128, 128, 1]
raw_image_shape = [101, 101]
#
# Training parameters
#
epochs = 200
batch_size = 32
#
# Load input data (images and label)
#
all_data, all_labels, size_morph, _ = cdata.main()
all_data = all_data.reshape(all_data.shape[0], 128, 128, 1)

x_train1, x_test1 = train_test_split(all_data, test_size=0.3, random_state=0)

def conv2d_block(final_image_shape, filt3, kernel_size = 2, batch_norm = True):
    """Function to add 2 convolutional layers with the parameters passed to it"""
    # first layer
    x = Conv2D(filters=filt3, kernel_size = (kernel_size, kernel_size), kernel_initializer='he_normal',
               padding='same')(final_image_shape)
    if batch_norm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)

    return x

dropout = 0.1
batch_norm = True
shape1 = (2, 2)
shape2 = (2, 2)
shape3 = (2, 2)
shape4 = (2, 2)

filt1 = 128
filt2 = 64
filt3 = 1
#
# Contracting Path
#
inp_img = Input(shape=final_image_shape, name='input')
c1 = conv2d_block(inp_img, filt1, kernel_size=3, batch_norm = batch_norm)
p1 = MaxPooling2D(shape1)(c1)
p1 = Dropout(dropout)(p1)

c2 = conv2d_block(p1, filt2, kernel_size=3, batch_norm = batch_norm)
p2 = MaxPooling2D(shape2)(c2)
p2 = Dropout(dropout)(p2)

c33 = conv2d_block(p2, filt3, kernel_size=3, batch_norm = batch_norm)
p33 = MaxPooling2D(shape3)(c33)
p33 = Dropout(dropout)(p33)
#
# Expansive Path
#
c3 = conv2d_block(p33, filt3, kernel_size=3, batch_norm = batch_norm)
c44 = UpSampling2D(shape3)(c3)
c44 = Dropout(dropout)(c44)

u4 = conv2d_block(c44, filt2, kernel_size=3, batch_norm = batch_norm)
c4 = UpSampling2D(shape2)(u4)
c4 = Dropout(dropout)(c4)

u5 = conv2d_block(c4, filt1, kernel_size=3, batch_norm = batch_norm)
c5 = UpSampling2D(shape1)(u5)
c5 = Dropout(dropout)(c5)

output1 = Conv2D(1, (1, 1), activation='sigmoid')(c5)
model1 = keras.Model(inputs=inp_img, outputs=output1)
sub_net_1 = keras.models.Model(inputs=inp_img, outputs=output1)
sgd = optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
sub_net_1.compile(loss=['mse'], optimizer=sgd)

es = EarlyStopping(monitor='val_loss', mode='auto', verbose=1, patience=100)
mc = ModelCheckpoint('best_state_i_x_i.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
csv_logger = CSVLogger('training_i_x_i.log', separator=',', append=False)
history = sub_net_1.fit(x_train1, x_train1, batch_size=batch_size, epochs=epochs,
                            verbose=1, validation_split=0.3, callbacks=[es, mc, csv_logger])
#
# Save model
#
sub_net_1.save("model_i_x_i.h5")
print("Network saved to disk".format("model_i_x_i.h5"))

