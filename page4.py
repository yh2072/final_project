import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import tensorflow.compat.v1 as tf
from tensorflow import keras
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Conv1D, MaxPooling1D, Bidirectional, LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler


def show():

    def calculate_median(vector):
        # Sort the vector
        sorted_vector = sorted(vector)

        # Find the number of elements in the vector
        n = len(sorted_vector)

        # Calculating the median
        if n % 2 == 0:  # If even number of elements
            median1 = sorted_vector[n//2]
            median2 = sorted_vector[n//2 - 1]
            median = (median1 + median2) / 2
        else:  # If odd number of elements
            median = sorted_vector[n//2]

        return median

    def transform_vector(vector):
        median = calculate_median(vector)
        return [0 if x < median else 1 for x in vector]

    data = pd.read_csv('data.csv')
    bands = ["Alpha", "Theta", "Gamma", "Beta", "Delta"]

    data["segment"] = ((data["time"] - 1) // 30) + 1

    y = data.drop(columns=["Delta", "Theta", "Alpha", "Beta", "Gamma"])
    y["Sum"] = np.array(transform_vector(y["Sum"]))
    x = data.drop(columns=['Sum'])

    unique_ids = data['id'].unique()
    train_ids, test_ids = train_test_split(unique_ids, test_size=0.5, random_state=42)
    x_train = x[x['id'].isin(train_ids)]
    x_test = x[x['id'].isin(test_ids)]
    y_train = y[data['id'].isin(train_ids)]
    y_test = y[data['id'].isin(test_ids)]

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train[bands])
    x_train[bands] = x_train_scaled
    x_test_scaled = scaler.transform(x_test[bands])
    x_test[bands] = x_test_scaled

    y_train_grouped = y_train.groupby(['id', 'condition', 'segment']).apply(
        lambda x: x.drop(columns=['id', 'condition', 'segment', 'time']).values.tolist())
    y_train_grouped = y_train_grouped.reset_index()
    y_train_grouped = y_train_grouped.drop(columns=["id", "condition", "segment"])
    y_train_new = np.empty((len(y_train_grouped), 1))
    for i in range(len(y_train_grouped)):
        y_train_new[i] = np.array(y_train_grouped.iloc[i, 0][0])
    y_train = np.array(y_train_new)

    y_test_grouped = y_test.groupby(['id', 'condition', 'segment']).apply(
        lambda x: x.drop(columns=['id', 'condition', 'segment', 'time']).values.tolist())
    y_test_grouped = y_test_grouped.reset_index()
    y_test_grouped = y_test_grouped.drop(columns=["id", "condition", "segment"])
    y_test_new = np.empty((len(y_test_grouped), 1))
    for i in range(len(y_test_grouped)):
        y_test_new[i] = np.array(y_test_grouped.iloc[i, 0][0])
    y_test = np.array(y_test_new)

    x_train_grouped = x_train.groupby(['id', 'condition', 'segment']).apply(
        lambda x: x.drop(columns=['id', 'condition', 'segment', 'time']).values.tolist())
    x_train_grouped = x_train_grouped.reset_index()
    x_train_grouped = x_train_grouped.drop(columns=["id", "condition", "segment"])
    x_train_new = np.empty((len(x_train_grouped), 5, 30))
    for i in range(len(x_train_grouped)):
        x_train_new[i] = np.array(x_train_grouped.iloc[i, 0]).T

    x_test_grouped = x_test.groupby(['id', 'condition', 'segment']).apply(
        lambda x: x.drop(columns=['id', 'condition', 'segment', 'time']).values.tolist())
    x_test_grouped = x_test_grouped.reset_index()
    x_test_grouped = x_test_grouped.drop(columns=["id", "condition", "segment"])
    x_test_new = np.empty((len(x_test_grouped), 5, 30))
    for i in range(len(x_test_grouped)):
        x_test_new[i] = np.array(x_test_grouped.iloc[i, 0]).T

    x_test = np.array(x_test_new).reshape(-1, 5, 30)
    x_train = np.array(x_train_new).reshape(-1, 5, 30)
    st.write(x_train.shape, x_test.shape, y_train.shape, y_test.shape)


    options = ['Transformer', 'CF-Bi-LSTM','Bi-LSTM']
    selected_model = st.sidebar.radio("Select a model:", options)

    if selected_model == 'CF-Bi-LSTM':
        def build_cf_bi_lstm_model(input_shape):
            inputs = Input(shape=input_shape)

            # Convolutional layers for feature extraction
            x = Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(inputs)
            x = MaxPooling1D(pool_size=2)(x)
            x = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(x)
            x = MaxPooling1D(pool_size=2)(x)

            # Bidirectional LSTM layers
            x = Bidirectional(LSTM(64, return_sequences=True))(x)
            x = Dropout(0.3)(x)
            x = Bidirectional(LSTM(32, return_sequences=True))(x)
            x = Dropout(0.3)(x)

            # Flatten and Dense layers for classification
            x = Flatten()(x)
            x = Dense(32, activation='relu')(x)
            outputs = Dense(1, activation='sigmoid')(x)

            model = Model(inputs, outputs)
            return model

        def train_model(model, x_train, y_train, x_test, y_test, save_to, epoch=150):
            opt_adam = Adam(learning_rate=0.001)
            mc = ModelCheckpoint(save_to + '_best_model.h5', monitor='val_accuracy', mode='max', verbose=1,
                                 save_best_only=True)
            lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 0.001 * np.exp(-epoch / 10.))

            model.compile(optimizer=opt_adam, loss='binary_crossentropy', metrics=['accuracy'])
            history = model.fit(x_train, y_train, batch_size=20, epochs=epoch, validation_data=(x_test, y_test),
                                callbacks=[mc, lr_schedule])

            return model, history


    if selected_model == 'Bi-LSTM':
        inputs = tf.keras.Input(shape=(5, 30))
        Dense1 = Dense(10, activation='relu', kernel_regularizer=keras.regularizers.l2())(inputs)

        lstm_1 = Bidirectional(LSTM(20, return_sequences=True))(Dense1)
        drop = Dropout(0.3)(lstm_1)
        lstm_3 = Bidirectional(LSTM(5, return_sequences=True))(drop)
        drop2 = Dropout(0.3)(lstm_3)

        flat = Flatten()(drop2)

        Dense_2 = Dense(5, activation='relu')(flat)
        outputs = Dense(1, activation='sigmoid')(Dense_2)

        model = tf.keras.Model(inputs, outputs)

        def train_model(model, x_train, y_train, x_test, y_test, save_to, epoch=300):
            opt_adam = keras.optimizers.Adam(learning_rate=0.0001)
            mc = ModelCheckpoint(save_to + '_best_model.h5', monitor='val_accuracy', mode='max', verbose=1,
                                 save_best_only=True)
            lr_schedule = LearningRateScheduler(lambda epoch: 0.001 * np.exp(-epoch / 10.))

            # Early Stopping
            es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)

            model.compile(optimizer=opt_adam, loss='binary_crossentropy', metrics=['accuracy'])

            history = model.fit(
                x_train, y_train,
                batch_size=30,
                epochs=epoch,
                validation_data=(x_test, y_test),
                callbacks=[mc, lr_schedule, es]  # Include early stopping here
            )

            return model, history


    if selected_model == 'Transformer':
        def transformer_encoder(inputs, num_heads, dim, dropout=0.1):
            # Attention and Normalization
            x = LayerNormalization(epsilon=1e-6)(inputs)
            x = MultiHeadAttention(num_heads=num_heads, key_dim=dim, dropout=dropout)(x, x)
            x = Dropout(dropout)(x)
            x = x + inputs  # Skip Connection

            # Feed Forward Network and Normalization
            x = LayerNormalization(epsilon=1e-6)(x)
            x = Dense(dim, activation='relu')(x)
            x = Dropout(dropout)(x)
            x = Dense(inputs.shape[-1])(x)
            x = x + inputs  # Skip Connection

            return x

        def build_transformer_model(input_shape, head_size, num_heads, ff_dim, num_transformer_blocks, mlp_units,
                                    dropout=0.1,
                                    mlp_dropout=0.1):
            inputs = Input(shape=input_shape)
            x = Dense(head_size, activation='relu', kernel_regularizer=l2())(inputs)

            for _ in range(num_transformer_blocks):
                x = transformer_encoder(x, num_heads, ff_dim, dropout)

            x = Flatten()(x)
            for units in mlp_units:
                x = Dense(units, activation='relu')(x)
                x = Dropout(mlp_dropout)(x)

            outputs = Dense(1, activation='sigmoid')(x)
            return Model(inputs, outputs)

        early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)

        input_shape = (5, 30)
        head_size = 5
        num_heads = 2
        ff_dim = 2
        num_transformer_blocks = 3
        mlp_units = [8, 4]

        model = build_transformer_model(input_shape, head_size, num_heads, ff_dim, num_transformer_blocks, mlp_units)

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])



    if st.button('Train Model') :
        with st.spinner('Training in progress...'):
            if selected_model == 'CF-Bi-LSTM':
                model = build_cf_bi_lstm_model((5, 30))
                model, history = train_model(model, x_train, y_train, x_test, y_test, save_to='./', epoch=150)
            if selected_model == 'Bi-LSTM':
                model, history = train_model(model, x_train, y_train, x_test, y_test, save_to='./', epoch=150)
            if selected_model == 'Transformer':
                history = model.fit(
                    x_train, y_train,
                    epochs=150,
                    batch_size=20,
                    validation_data=(x_test, y_test),
                    callbacks=[early_stopping]  # Add early stopping here
                )
            st.success('Training completed!')
            st.write(history.history)
            plt.figure(figsize=(10, 4))
            plt.plot(history.history['accuracy'])
            plt.plot(history.history['val_accuracy'])
            plt.title('Model Accuracy')
            plt.ylabel('Accuracy')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Test'], loc='upper left')
            st.pyplot(plt)

            # Plotting loss
            plt.figure(figsize=(10, 4))
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('Model Loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Test'], loc='upper left')
            st.pyplot(plt)
