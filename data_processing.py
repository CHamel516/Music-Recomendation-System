import pandas as pd
import numpy as np

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

def sample_users():
    taste_profile_file = r'D:\Downloads D\train_triplets.txt'
    taste_profile = load_data(taste_profile_file, column_names=['user_id', 'song_id', 'play_count'], dtype={'user_id': object, 'song_id': object, 'play_count': float})

    user_ids_kept = taste_profile['user_id'].sample(n=10000, random_state=3)
    filtered_taste_profile = taste_profile[taste_profile['user_id'].isin(user_ids_kept)]

    # the sep argument didn't work, had to manually adjust the file
    filtered_taste_profile.to_csv('small_taste_profile.csv', sep=',', index=False, lineterminator='\n')

def get_song_names():
    unique_tracks_file = r'D:\Downloads D\unique_tracks.txt'
    tracks_data = load_data(unique_tracks_file, column_names=['track_id', 'song_id', 'artist', 'title'], delimiter='<SEP>')

    if tracks_data is None:
        exit()

    taste_profile_file = r'small_taste_profile.csv'
    taste_profile = pd.read_csv(taste_profile_file, header=0, names=['user_id', 'song_id', 'play_count'], dtype={'user_id': object, 'song_id': object, 'play_count': float})

    if taste_profile is None:
        exit()

    merged_data = pd.merge(taste_profile, tracks_data[['song_id', 'artist', 'title']], on='song_id', how='left')
    merged_data.to_csv('merged_data.csv', index=False)

sample_users()