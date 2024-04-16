import pandas as pd
import numpy as np
import tensorflow as tf
import google.protobuf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Concatenate, Dense, Embedding, Flatten, Input
from tensorflow.keras.models import Model

def load_data(file_path, column_names=None, delimiter=None, dtype=None):
    try:
        data = pd.read_csv(file_path, delimiter=delimiter, header=None, names=column_names, dtype=dtype)
        return data
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None

# Assuming data is already loaded and preprocessed from 'merged_data.csv'
data = pd.read_csv('merged_data.csv')
data['user_id'] = data['user_id'].astype('category').cat.codes
data['song_id'] = data['song_id'].astype('category').cat.codes

# Prepare input and output arrays
X_user = data['user_id'].values
X_song = data['song_id'].values
Y = data['play_count'].values

# Normalizing play counts to a 0-1 scale
max_play_count = Y.max()
Y = Y / max_play_count

# Neural Collaborative Filtering (NCF) model setup
num_users = data['user_id'].nunique()
num_items = data['song_id'].nunique()
latent_dim = 32

user_input = Input(shape=(1,), dtype='int32', name='user_input')
user_embedding = Embedding(input_dim=num_users, output_dim=latent_dim, name='user_embedding')(user_input)
user_vec = Flatten(name='FlattenUsers')(user_embedding)

item_input = Input(shape=(1,), dtype='int32', name='item_input')
item_embedding = Embedding(input_dim=num_items, output_dim=latent_dim, name='item_embedding')(item_input)
item_vec = Flatten(name='FlattenItems')(item_embedding)

concat_vec = Concatenate()([user_vec, item_vec])
layer_1 = Dense(64, activation='relu')(concat_vec)
layer_2 = Dense(32, activation='relu')(layer_1)
output = Dense(1, activation='linear')(layer_2)  # Changed to linear for a regression task

ncf_model = Model(inputs=[user_input, item_input], outputs=output)
ncf_model.compile(optimizer='adam', loss='mse', metrics=['mae'])  # Using MSE for regression

# Splitting data into train and test sets
X_user_train, X_user_test, X_song_train, X_song_test, Y_train, Y_test = train_test_split(X_user, X_song, Y, test_size=0.2, random_state=42)

# Training the model
ncf_model.fit([X_user_train, X_song_train], Y_train, epochs=10, validation_data=([X_user_test, X_song_test], Y_test))