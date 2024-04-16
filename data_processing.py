import pandas as pd
import numpy as np

# Function to load data from a file
def load_data(file_path, column_names=None, delimiter=None, dtype=None):
    try:
        # Load the data with the specified options
        data = pd.read_csv(file_path, delimiter=delimiter, header=None, names=column_names, dtype=dtype)
        return data
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None

# Function to sample users and their interactions
def sample_users():
    taste_profile_file = r'D:\Downloads D\train_triplets.txt'
    # Load the data from the file, specifying tab as delimiter for .txt files
    taste_profile = load_data(taste_profile_file, column_names=['user_id', 'song_id', 'play_count'], delimiter='\t', dtype={'user_id': object, 'song_id': object, 'play_count': float})

    if taste_profile is None:
        return  # Exit if data loading fails

    # Sample 10,000 unique users
    user_ids_kept = taste_profile['user_id'].drop_duplicates().sample(n=10000, random_state=3)
    filtered_taste_profile = taste_profile[taste_profile['user_id'].isin(user_ids_kept)]

    # Debug information
    print(f"Total interactions in sampled data: {len(filtered_taste_profile)}")
    print(f"Unique users in sampled data: {filtered_taste_profile['user_id'].nunique()}")

    # Save the filtered data to CSV
    filtered_taste_profile.to_csv('small_taste_profile.csv', sep=',', index=False, lineterminator='\n')

# Function to load song names and merge with user data
def get_song_names():
    unique_tracks_file = r'D:\Downloads D\unique_tracks.txt'
    tracks_data = load_data(unique_tracks_file, column_names=['track_id', 'song_id', 'artist', 'title'], delimiter='<SEP>')

    if tracks_data is None:
        print("Failed to load tracks data.")
        return

    taste_profile_file = r'small_taste_profile.csv'
    taste_profile = pd.read_csv(taste_profile_file)

    if taste_profile is None:
        print("Failed to load taste profile data.")
        return

    # Merge the taste profile data with the track data on song_id
    merged_data = pd.merge(taste_profile, tracks_data[['song_id', 'artist', 'title']], on='song_id', how='left')
    merged_data.to_csv('merged_data.csv', index=False)

    # Debug print for verification
    print(f"Total records in merged data: {len(merged_data)}")

# Execute the functions to process data
sample_users()
get_song_names