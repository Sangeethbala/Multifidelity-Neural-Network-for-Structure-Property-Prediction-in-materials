"""
Model mapping I to I and simultaneously getting G from the latent representation(X)
"""
import numpy as np
from keras.layers import Dense, Flatten
from sklearn.model_selection import train_test_split
import keras
import cdata
from keras.layers import Conv2D, MaxPooling2D, Dropout, Activation, BatchNormalization, Input, Conv2DTranspose,\
    concatenate
from keras import optimizers
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Model
from sklearn.externals import joblib
from keras.callbacks import CSVLogger
import h5py
from keras.models import load_model

def load_data_labels(filename):
    with h5py.File(filename, 'r') as h5f:
        h5_data = np.asarray(list(h5f['graspi_results']))
    print('Data and labels are loaded!')
    return h5_data

def conv2d_block(final_image_shape, filt3, kernel_size = 2, batch_norm = True):
    """Function to add 2 convolutional layers with the parameters passed to it"""
    # first layer
    x = Conv2D(filters=filt3, kernel_size=(kernel_size, kernel_size), kernel_initializer='he_normal',
               padding='same')(final_image_shape)
    if batch_norm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)

    return x

H5FILENAME_MORPH = 'all_labels.h5'
morph_desc = load_data_labels(H5FILENAME_MORPH)
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
dropout = 0.1
#
# Load input data (images and label)
#
all_data, all_labels, size_morph, _ = cdata.main()
all_data = all_data.reshape(all_data.shape[0], 128, 128, 1)

# Rescale all_labels
#
scaler = MinMaxScaler(feature_range=(0, 1))
scaler = scaler.fit(morph_desc)
morph_desc = scaler.transform(morph_desc)
print("labels normalized !")
scaler_filename = "scaler_x_g.save"
joblib.dump(scaler, scaler_filename)
neuralnetwork = load_model('best_state_i_x_i.h5')

x_train1, x_test1, morph_train1, morph_test1 = train_test_split(all_data, morph_desc, test_size=0.3, random_state=100)

encoder = keras.models.Model(inputs=neuralnetwork.get_layer('input').input,
                             outputs=neuralnetwork.get_layer('dropout_3').output)
enc_out = encoder.predict(x_train1)

inp_enc = Input(shape=(enc_out.shape[1], enc_out.shape[2], enc_out.shape[3]))
c1 = conv2d_block(inp_enc, 2, kernel_size=3, batch_norm=True)
p1 = MaxPooling2D((2, 2))(c1)
p1 = Dropout(dropout)(p1)

c2 = conv2d_block(p1, 64, kernel_size=3, batch_norm=True)
p2 = MaxPooling2D((2, 2))(c2)
p2 = Dropout(dropout)(p2)

image_vector_size = 64*4*4
input_vec = Flatten()(p1)
a1 = (Dense(64, activation='relu', input_shape=(image_vector_size, ),
                 kernel_initializer='normal'))(input_vec)

a2 = (Dense(32, activation='relu', kernel_initializer='normal'))(a1)
a3 = (Dense(16, activation='relu', kernel_initializer='normal'))(a2)
hidden_yl5 = (Dense(6, activation='relu', kernel_initializer='normal', name='low_fid'))(a3)

sub_net_1 = keras.models.Model(inputs=inp_enc, outputs=hidden_yl5)
sgd = optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
sub_net_1.compile(loss='mse', optimizer=sgd)

es = EarlyStopping(monitor='val_loss', mode='auto', verbose=1, patience=150)
mc = ModelCheckpoint('best_state_x_gg.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
csv_logger = CSVLogger('training_x_gg.log', separator=',', append=False)
history = sub_net_1.fit(enc_out, morph_train1, batch_size=batch_size, epochs=epochs,
                            verbose=1, validation_split=0.3, callbacks=[es, mc, csv_logger])
#
# Save model
#
sub_net_1.save("model_x_gg.h5")
print("Network saved to disk".format("model_x_gg.h5"))

