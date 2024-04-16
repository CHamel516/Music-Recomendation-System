import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Concatenate, Dense, Embedding, Flatten, Input, dot
from tensorflow.keras.models import Model

# Function to load data
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

# Load and preprocess data NOTE: MOVED TO OTHER FILE
# unique_tracks_file = r'D:\Downloads D\unique_tracks.txt'
# tracks_data = load_data(unique_tracks_file, column_names=['track_id', 'song_id', 'artist', 'title'], delimiter='<SEP>')

# if tracks_data is None:
#     exit()

# taste_profile_file = r'small_taste_profile.csv'
# taste_profile = load_data(taste_profile_file, column_names=['user_id', 'song_id', 'play_count'], dtype={'user_id': object, 'song_id': object, 'play_count': float})

# if taste_profile is None:
#     exit()

# merged_data = pd.merge(taste_profile, tracks_data[['song_id', 'artist', 'title']], on='song_id', how='left')

data = pd.read_csv('merged_data.csv', header=0, column_names=['user_id','song_id','play_count','artist','title'], dtype={'user_id': object, 'song_id': object, 'play_count': float, 'artist':object, 'title':object})


# Encoding user IDs and song IDs to integer indices
data['user_id'] = data['user_id'].astype('category').cat.codes
data['song_id'] = data['song_id'].astype('category').cat.codes


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
output = Dense(1, activation='sigmoid')(layer_2)

ncf_model = Model(inputs=[user_input, item_input], outputs=output)
ncf_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Matrix Factorization model
# user_input = Input(shape=(1,), name='user_input')
# user_embedding = Embedding(num_users, latent_dim, name='user_embedding_mf')(user_input)
# user_vec = Flatten(name='FlattenUsers_mf')(user_embedding)

# item_input = Input(shape=(1,), name='item_input')
# item_embedding = Embedding(num_items, latent_dim, name='item_embedding_mf')(item_input)
# item_vec = Flatten(name='FlattenItems_mf')(item_embedding)

# prod = dot([user_vec, item_vec], axes=1)
# mf_model = Model([user_input, item_input], prod)
# mf_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])