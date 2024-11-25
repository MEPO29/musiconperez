from keras.layers import Dense, Dropout, Activation, Flatten, Permute, Lambda, Input, BatchNormalization, Embedding, LSTM, Bidirectional, Reshape, GRU, Concatenate, Conv1D, GlobalMaxPooling1D, Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, MaxPooling1D
from keras.regularizers import l2, l1
from keras.models import Sequential, Model
import logging
from keras import backend as K
import keras
import numpy as np
import pickle
import common

params_1 = {
    # dataset params
    'dataset': {
        'fact': '',
        'dim': 200,
        'dataset': '',
        'window': 15,
        'nsamples': 'all',
        'npatches': 3
    },

    # training params
    'training': {
        'decay': 1e-6,
        'learning_rate': 0.1,
        'momentum': 0.95,
        'n_epochs': 100,
        'n_minibatch': 32,
        'nesterov': True,
        'validation': 0.1,
        'test': 0.1,
        'loss_func': 'cosine',
        'optimizer': 'adam'
    },
    # cnn params

    'cnn' : {
        'dropout_factor' : 0.5,
        'sequence_length' : 500,
        'embedding_dim' : 300,
        'filter_sizes' : (2, 3),
        'num_filters' : 150,
        'dropout_prob' : (0.5, 0.8),
        'hidden_dims' : 2048,
        'batch_size' : 32,
        'num_epochs' : 100,
        'val_split' : 0.1,
        'model_variation' : 'CNN-rnd',
        'n_out' : 200,
        'n_frames' : '',
        'n_mel' : 96,
        'architecture' : 82,
        'n_metafeatures' : 7927,
        'final_activation' : 'linear'
    },
    'predicting' : {
        'trim_coeff' : 0.15
    },
    'evaluating' : {
        'get_map' : False,
        'get_p' : True,
        'get_knn' : False
    }
}

# AUDIO ARCH CNNs
def get_model_1(params):
    # Define the model architecture here
    pass

# Audio ARCH with graph api
def get_model_11(params):
    # Define the model architecture here
    pass

# AUDIO multiple filters
def get_model_12(params):
    # Define the model architecture here
    pass

# Multimodal ARCH text + audio
def get_model_2(params):
    inputs = Input(shape=(params["input_shape"],))
    x = Dense(params["n_dense"], activation='relu')(inputs)
    x = Dropout(params["dropout_factor"])(x)
    x = Flatten()(x)
    logging.debug("Output Flatten: %s" % str(x.shape))

    inputs2 = Input(shape=(params["n_metafeatures"],))
    x2 = Dense(params["n_dense"], activation='relu')(inputs2)
    x2 = Dropout(params["dropout_factor"])(x2)
    logging.debug("Output Dense: %s" % str(x2.shape))

    xout = Concatenate()([x, x2])
    xout = Dense(params["n_out"], activation='linear')(xout)
    logging.debug("Output Dense: %s" % str(xout.shape))

    xout = Lambda(lambda x: K.l2_normalize(x, axis=1))(xout)

    model = Model(inputs=[inputs, inputs2], outputs=xout)
    return model

# METADATA ARCH
def get_model_3(params):
    inputs2 = Input(shape=(params["n_metafeatures"],))
    x2 = Dropout(params["dropout_factor"])(inputs2)

    if params["n_dense"] > 0:
        x2 = Dense(params["n_dense"], activation='relu')(x2)

    if params["n_dense_2"] > 0:
        x2 = Dense(params["n_dense_2"], activation='relu')(x2)

    xout = Dense(params["n_out"], activation=params['final_activation'])(x2)
    logging.debug("Output Dense: %s" % str(xout.shape))

    model = Model(inputs=inputs2, outputs=xout)
    return model

# Metadata 2 inputs, post-merge with dense layers
def get_model_31(params):
    inputs = Input(shape=(params["n_metafeatures"],))
    x = BatchNormalization()(inputs)
    x = Dropout(params["dropout_factor"])(x)
    x = Dense(params["n_dense"], activation='relu')(x)
    logging.debug("Output Dense: %s" % str(x.shape))
    x = Dropout(params["dropout_factor"])(x)

    inputs2 = Input(shape=(params["n_metafeatures2"],))
    x2 = BatchNormalization()(inputs2)
    x2 = Dropout(params["dropout_factor"])(x2)
    x2 = Dense(params["n_dense"], activation='relu')(x2)
    logging.debug("Output Dense: %s" % str(x2.shape))
    x2 = Dropout(params["dropout_factor"])(x2)

    xout = Concatenate()([x, x2])
    xout = Dense(params["n_out"], activation=params['final_activation'])(xout)
    logging.debug("Output Dense: %s" % str(xout.shape))

    model = Model(inputs=[inputs, inputs2], outputs=xout)
    return model

# Metadata 2 inputs, pre-merge and l2
def get_model_32(params):
    inputs = Input(shape=(params["n_metafeatures"],))
    x1 = Lambda(lambda x: K.l2_normalize(x, axis=1))(inputs)

    inputs2 = Input(shape=(params["n_metafeatures2"],))
    x2 = Lambda(lambda x: K.l2_normalize(x, axis=1))(inputs2)

    x = Concatenate()([x1, x2])
    x = Dropout(params["dropout_factor"])(x)

    if params['n_dense'] > 0:
        x = Dense(params["n_dense"], activation='relu')(x)

    xout = Dense(params["n_out"], activation=params['final_activation'])(x)
    logging.debug("Output Dense: %s" % str(xout.shape))

    model = Model(inputs=[inputs, inputs2], outputs=xout)
    return model

# Metadata 3 inputs, pre-merge and l2
def get_model_33(params):
    inputs = Input(shape=(params["n_metafeatures"],))
    x1 = Lambda(lambda x: K.l2_normalize(x, axis=1))(inputs)

    inputs2 = Input(shape=(params["n_metafeatures2"],))
    x2 = Lambda(lambda x: K.l2_normalize(x, axis=1))(inputs2)

    inputs3 = Input(shape=(params["n_metafeatures3"],))
    x3 = Lambda(lambda x: K.l2_normalize(x, axis=1))(inputs3)

    x = Concatenate()([x1, x2, x3])
    x = Dropout(params["dropout_factor"])(x)

    if params['n_dense'] > 0:
        x = Dense(params["n_dense"], activation='relu')(x)

    xout = Dense(params["n_out"], activation=params['final_activation'])(x)
    logging.debug("Output Dense: %s" % str(xout.shape))

    model = Model(inputs=[inputs, inputs2, inputs3], outputs=xout)
    return model

# Metadata 4 inputs, pre-merge and l2
def get_model_34(params):
    inputs = Input(shape=(params["n_metafeatures"],))
    x1 = Lambda(lambda x: K.l2_normalize(x, axis=1))(inputs)

    inputs2 = Input(shape=(params["n_metafeatures2"],))
    x2 = Lambda(lambda x: K.l2_normalize(x, axis=1))(inputs2)

    inputs3 = Input(shape=(params["n_metafeatures3"],))
    x3 = Lambda(lambda x: K.l2_normalize(x, axis=1))(inputs3)

    inputs4 = Input(shape=(params["n_metafeatures4"],))
    x4 = Lambda(lambda x: K.l2_normalize(x, axis=1))(inputs4)

    x = Concatenate()([x1, x2, x3, x4])
    x = Dropout(params["dropout_factor"])(x)

    if params['n_dense'] > 0:
        x = Dense(params["n_dense"], activation='relu')(x)

    xout = Dense(params["n_out"], activation=params['final_activation'])(x)
    logging.debug("Output Dense: %s" % str(xout.shape))

    model = Model(inputs=[inputs, inputs2, inputs3, inputs4], outputs=xout)
    return model

params_w2v = {
    # dataset params
    'dataset' : {
        'fact' : 'als',
        'dim' : 200,
        'dataset' : 'W2',
        'window' : 15,
        'nsamples' : 'all',
        'npatches' : 1,
        'meta-suffix' : ''
    },

    # training params
    'training' : {
        'decay' : 1e-6,
        'learning_rate' : 0.1,
        'momentum' : 0.95,
        'n_epochs' : 100,
        'n_minibatch' : 32,
        'nesterov' : True,
        'validation' : 0.1,
        'test' : 0.1,
        'loss_func' : 'cosine',
        'optimizer' : 'sgd'
    },
    # cnn params
    'cnn' : {
        'dropout_factor' : 0.5,
        'sequence_length' : 500,
        'embedding_dim' : 300,
        'filter_sizes' : (2, 3),
        'num_filters' : 150,
        'dropout_prob' : (0.5, 0.8),
        'hidden_dims' : 2048,
        'batch_size' : 32,
        'num_epochs' : 100,
        'val_split' : 0.1,
        'model_variation' : 'CNN-rnd',
        'n_out' : 200,
        'n_frames' : '',
        'n_mel' : 96,
        'architecture' : 82,
        'n_metafeatures' : 7927,#5393
        'final_activation' : 'linear'
    },
    'predicting' : {
        'trim_coeff' : 0.15
    },
    'evaluating' : {
        'get_map' : False,
        'get_p' : True,
        'get_knn' : False
    }
}

# word2vec ARCH with CNNs
def get_model_4(params):
    embedding_weights = pickle.load(open(common.TRAINDATA_DIR + "/embedding_weights_w2v_%s.pk" % params['embeddings_suffix'], "rb"))
    graph_in = Input(shape=(params['sequence_length'], params['embedding_dim']))
    convs = []
    for fsz in params['filter_sizes']:
        conv = Conv1D(filters=params['num_filters'], kernel_size=fsz, activation='relu')(graph_in)
        pool = GlobalMaxPooling1D()(conv)
        convs.append(pool)

    if len(params['filter_sizes']) > 1:
        out = Concatenate()(convs)
    else:
        out = convs[0]

    graph = Model(inputs=graph_in, outputs=out)

    model = Sequential()
    if params['model_variation'] != 'CNN-static':
        model.add(Embedding(input_dim=len(embedding_weights[0]), output_dim=params['embedding_dim'], input_length=params['sequence_length'], weights=[embedding_weights], trainable=True))
    model.add(Dropout(params['dropout_prob'][0]))
    model.add(graph)
    model.add(Dense(params['n_dense'], activation='relu'))
    model.add(Dropout(params['dropout_prob'][1]))
    model.add(Dense(params["n_out"], activation=params['final_activation']))
    logging.debug("Output Dense: %s" % str(model.output_shape))

    return model

# word2vec ARCH with LSTM
def get_model_41(params):
    embedding_weights = pickle.load(open(common.TRAINDATA_DIR + "/embedding_weights_w2v_%s.pk" % params['embeddings_suffix'], "rb"))
    model = Sequential()
    model.add(Embedding(input_dim=len(embedding_weights[0]), output_dim=params['embedding_dim'], input_length=params['sequence_length'], weights=[embedding_weights], trainable=True))
    model.add(LSTM(2048))
    model.add(Dense(params["n_out"], activation=params['final_activation']))
    logging.debug("Output Dense: %s" % str(model.output_shape))

    return model

# CRNN Arch for audio
def get_model_5(params):
    input_tensor = None
    include_top = True

    if K.image_data_format() == 'channels_first':
        input_shape = (1, params['input_shape'][0], params['input_shape'][1])
        channel_axis = 1
        freq_axis = 2
    else:
        input_shape = (params['input_shape'][0], params['input_shape'][1], 1)
        channel_axis = -1
        freq_axis = 1

    if input_tensor is None:
        melgram_input = Input(shape=input_shape)
    else:
        melgram_input = Input(tensor=input_tensor, shape=input_shape)

    x = ZeroPadding2D(padding=(0, 37))(melgram_input)
    x = BatchNormalization(axis=freq_axis, name='bn_0_freq')(x)
    x = Permute((1, 3, 2))(x)

    x = Conv2D(64, (3, 3), padding='same', name='conv1')(x)
    x = BatchNormalization(axis=channel_axis, name='bn1')(x)
    x = Activation('elu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool1')(x)
    x = Dropout(0.1, name='dropout1')(x)

    x = Conv2D(128, (3, 3), padding='same', name='conv2')(x)
    x = BatchNormalization(axis=channel_axis, name='bn2')(x)
    x = Activation('elu')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(3, 3), name='pool2')(x)
    x = Dropout(0.1, name='dropout2')(x)

    x = Conv2D(128, (3, 3), padding='same', name='conv3')(x)
    x = BatchNormalization(axis=channel_axis, name='bn3')(x)
    x = Activation('elu')(x)
    x = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), name='pool3')(x)
    x = Dropout(0.1, name='dropout3')(x)

    x = Conv2D(128, (3, 3), padding='same', name='conv4')(x)
    x = BatchNormalization(axis=channel_axis, name='bn4')(x)
    x = Activation('elu')(x)
    x = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), name='pool4')(x)
    x = Dropout(0.1, name='dropout4')(x)

    if K.image_data_format() == 'channels_first':
        x = Reshape((128, -1))(x)
    else:
        x = Reshape((-1, 128))(x)

    x = GRU(32, return_sequences=True, name='gru1')(x)
    x = GRU(32, return_sequences=False, name='gru2')(x)
    x = Dropout(0.3)(x)
    if include_top:
        x = Dense(params["n_out"], activation=params['final_activation'])(x)

    model = Model(inputs=melgram_input, outputs=x)
    return model

# AUDIO Features
def get_model_6(params):
    inputs2 = Input(shape=(params["n_metafeatures"],))

    if params["n_dense"] > 0:
        x2 = Dense(params["n_dense"], activation='relu')(inputs2)

    if params["n_dense_2"] > 0:
        x2 = Dense(params["n_dense_2"], activation='relu')(x2)

    xout = Dense(params["n_out"], activation=params['final_activation'])(x2)
    logging.debug("Output Dense: %s" % str(xout.shape))

    model = Model(inputs=inputs2, outputs=xout)
    return model