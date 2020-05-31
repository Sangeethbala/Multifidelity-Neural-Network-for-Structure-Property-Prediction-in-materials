"""
Model mapping I to I and simultaneously getting G, J from the latent representation(X)
"""
import numpy as np
from keras.layers import Dense, Flatten
from sklearn.model_selection import train_test_split
import keras
import cdata
from keras.layers import Conv2D, Dropout, Activation, BatchNormalization, Input,\
    concatenate, MaxPooling2D
from keras import optimizers
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Model
from keras.callbacks import CSVLogger
from keras.models import load_model
from sklearn.neighbors import KernelDensity
import tensorflow as tf
from scipy import spatial
from sklearn.neighbors import KDTree
import h5py

def load_data_labels(filename):
    with h5py.File(filename, 'r') as h5f:
        lowfid_data = np.asarray(list(h5f['graspi_results']))
        highfid_data = np.asarray(list(h5f['xdd_results']))
    print('Labels are loaded!')
    return lowfid_data, highfid_data

def conv2d_block(final_image_shape, filt3, kernel_size = 2, batch_norm = True):
    """Function to add 2 convolutional layers with the parameters passed to it"""
    # first layer
    x = Conv2D(filters=filt3, kernel_size = (kernel_size, kernel_size), kernel_initializer='he_normal',
               padding='same')(final_image_shape)
    if batch_norm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)

    return x

H5FILENAME_MORPH = 'all_labels.h5'
morph_desc, high_fid = load_data_labels(H5FILENAME_MORPH)

neuralnetwork_x1 = load_model('best_state_i_x_i.h5')
neuralnetwork_x1.save_weights("weights_i_x_i.h5")
#
# Image configuration
#
final_image_shape = [128, 128, 1]
raw_image_shape = [101, 101]
#
# Training parameters
#
epochs = 150
batch_size = 32
#
# Load input data (images and label)
#
all_data, all_labels, size_morph, _ = cdata.main()
all_data = all_data.reshape(all_data.shape[0], 128, 128, 1)
all_labels = high_fid
#
# Rescale all_labels
#
scaler = MinMaxScaler(feature_range=(0, 1))
scaler = scaler.fit(all_labels)
all_labels = scaler.transform(all_labels)
print("labels normalized !")

neuralnetwork_x2 = load_model('best_state_x_gg.h5')

dropout = 0.1
batch_norm = True
shape1 = (2, 2)
shape2 = (2, 2)
shape3 = (2, 2)
shape4 = (2, 2)
filt1 = 128
filt2 = 64
filt3 = 1

encoder = keras.models.Model(inputs=neuralnetwork_x1.get_layer('input').input,
                             outputs=neuralnetwork_x1.get_layer('dropout_3').output)
encoder.summary()
encoded_img = encoder.predict(all_data)
encoded_img1 = encoded_img
morph_desc = neuralnetwork_x2.predict(encoded_img1)
#
# Reshape encoded image(X)
#
print(" Encoded Image Shape: ", encoded_img.shape)
image_vector_size = np.prod(encoder.output_shape[1 : ])
encoded_img = encoded_img.reshape(encoded_img.shape[0], image_vector_size)
np.random.seed(0)

kde = KernelDensity(bandwidth=.1).fit(encoded_img.reshape(encoded_img.shape[0], 256))
num_samples = int(.1 * len(all_data))
morph_dim = morph_desc.shape[1]
for num_iter in range(10, 11):
    if num_iter > 10:
         sub_net_1.load_weights('model.h5')

    new_data = kde.sample(num_samples, random_state=num_iter)
    distance, enc_idxs = spatial.KDTree(encoded_img).query(new_data)

    all_data_1 = all_data[enc_idxs]
    all_labels_1 = all_labels[enc_idxs]
    morph_desc_1 = morph_desc[enc_idxs]
    encoded_img_1 = encoded_img1[enc_idxs]

    enc_train1, enc_test1, y_train1, y_test1 = train_test_split(encoded_img_1, all_labels_1, test_size=0.3, random_state=100)
    enc_train1, enc_test1, morph_train1, morph_test1 = train_test_split(encoded_img_1, morph_desc_1, test_size=0.3, random_state=100)

    inp_enc = Input(shape=(encoded_img1.shape[1],encoded_img1.shape[2], encoded_img1.shape[3]))
    c1 = conv2d_block(inp_enc, 2, kernel_size=3, batch_norm=batch_norm)
    p1 = MaxPooling2D(shape1)(c1)
    p1 = Dropout(dropout)(p1)
    input_vec = Flatten()(p1)

    hidden_yh2 = Dense(64, activation='relu', name='dense_5')(input_vec)
    hidden_yh3 = Dense(32, activation='relu', name='dense_3')(hidden_yh2)
    hidden_yh33 = Dense(16, activation='relu', name='dense_4')(hidden_yh3)

    inp_morph = Input(shape=(morph_dim, ))
    hidden_yh22 = Dense(64, activation='relu', name='dense_26')(inp_morph)
    hidden_yh23 = Dense(32, activation='relu', name='dense_27')(hidden_yh22)
    hidden_yh43 = Dense(16, activation='relu', name='dense_28')(hidden_yh23)

    added = keras.layers.Concatenate(axis=-1)([hidden_yh33, hidden_yh43])
    hidden_yh5 = Dense(5, activation='relu', name='dense_77')(added)

    sub_net_1 = keras.models.Model(inputs=[inp_enc, inp_morph], outputs=[hidden_yh5])

    sgd = optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    sub_net_1.compile(loss=['mse'], optimizer=sgd)
    if num_iter == 10:
        sub_net_1.save_weights('model.h5')

    es = EarlyStopping(monitor='val_loss', mode='auto', verbose=1, patience=100)
    mc = ModelCheckpoint(np.str(num_iter) + '_best_state_xg_j2.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
    csv_logger = CSVLogger(np.str(num_iter) + '_training_xg_j2.log', separator=',', append=False)
    history = sub_net_1.fit([enc_train1, morph_train1], [y_train1], batch_size=batch_size, epochs=epochs,
                                      verbose=1, validation_split=0.3, callbacks=[es, mc, csv_logger])
    #
    # Save model
    #
    sub_net_1.save(np.str(num_iter) + "_model_xg_j2.h5")
    print("Network saved to disk".format(np.str(num_iter) + "_model_xg_j2.h5"))

    sub_net_1.save_weights(np.str(num_iter) + "_weights_xg_j.h5")

