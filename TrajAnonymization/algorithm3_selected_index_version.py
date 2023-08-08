from algorithm2v1 import calculate_transition_distance, update_values, generate_anonymous_points, \
    calculate_final_locations
import pandas as pd
import numpy as np
import random

from math import sin, cos, sqrt, atan2, radians


def calculate_lat_or_long_distance(start, end):
    radius = 6371.0
    start_rad = radians(start)
    end_rad = radians(end)
    difference = end_rad - start_rad
    a = sin(difference / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = radius * c
    return distance


def haversine_distance(lat1, lon1, lat2, lon2):
    # The radius of the Earth in kilometers
    earth_radius_km = 6371.0

    # Convert latitude and longitude from degrees to radians
    lat1_rad = radians(lat1)
    lon1_rad = radians(lon1)
    lat2_rad = radians(lat2)
    lon2_rad = radians(lon2)

    # Haversine formula
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = sin(dlat / 2) ** 2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    # Calculate the distance in kilometers
    distance_m = earth_radius_km * c * 1000
    return distance_m


def check_trajectory_leakage(updated_latitude, updated_longitude, original_latitude, original_longtitude):
    updated_latitude_len = len(updated_latitude)
    updated_longitude_len = len(updated_longitude)
    original_latitude_len = len(original_latitude)
    original_longtitude_len = len(original_longtitude)
    updated_distance = haversine_distance(updated_latitude[0], updated_longitude[0],
                                          updated_latitude[updated_latitude_len - 1],
                                          updated_longitude[updated_longitude_len - 1])
    # print("Distane starts")
    # print(updated_distance)
    original_distance = haversine_distance(original_latitude[0], original_longtitude[0],
                                           original_latitude[original_latitude_len - 1],
                                           original_longtitude[original_longtitude_len - 1])
    # print(original_distance)
    if original_distance==0.0:
        start_end_node_distance=0.0
    else:
        start_end_node_distance = abs(original_distance - updated_distance) / original_distance
    # print(start_end_node_distance)

    updated_lat = calculate_lat_or_long_distance(updated_latitude[0], updated_latitude[updated_latitude_len - 1])

    original_lat = calculate_lat_or_long_distance(original_latitude[0], original_latitude[original_latitude_len - 1])
    updated_long = calculate_lat_or_long_distance(updated_longitude[0], updated_longitude[updated_latitude_len - 1])

    original_long = calculate_lat_or_long_distance(original_longtitude[0],
                                                   original_longtitude[original_latitude_len - 1])
    if original_lat==0.0:
        lat_distance=0.0
    else:
        lat_distance = abs(original_lat - updated_lat) / original_lat
    if original_long==0.0:
        long_distance=0.
    else:
        long_distance = abs(original_long - updated_long) / original_long

    return start_end_node_distance, lat_distance, long_distance


def privacy_preservation(index, user_dataset, row, threshold,recursion_depth):

    print(index)
    user_df = user_dataset[user_dataset['user_id'] == row['user_id']]
    latitude_distance, longitude_distance = calculate_transition_distance(row['location_id'], row['timestamp'])
    # resulted_trajectory = generate_anonymous_points(user_df, row, 2*latitude_distance, 2*longitude_distance)
    resulted_trajectory = generate_anonymous_points(user_df, row, latitude_distance, longitude_distance)
    final_trajectory = []
    for i in range(len(resulted_trajectory)):
        # Flatten the array to find the minimum value
        flattened_array = resulted_trajectory[i].flatten()
        min_value = np.min(flattened_array)
        # Find the indices where the minimum value occurs
        indices = np.where(flattened_array == min_value)[0]
        # Reshape the indices to match the original array shape
        row_indices, col_indices = np.unravel_index(indices, resulted_trajectory[i].shape)
        # Print the minimum value and the corresponding indices
        # print("Minimum value:", min_value)
        # print("Indices:", list(zip(row_indices, col_indices)))
        result_value_list = list(zip(row_indices, col_indices))
        final_value = random.choice(result_value_list)
        # print(list(final_value))
        final_trajectory.append(final_value)

    updated_latitude, updated_longitude = calculate_final_locations(final_trajectory, latitude_distance,
                                                                    longitude_distance, row)
    given_latitude = eval(row['location_id'])
    given_longtitude = eval(row['timestamp'])
    start_end_node_distance, lat_distance, long_distance = check_trajectory_leakage(updated_latitude, updated_longitude,
                                                                                    given_latitude,
                                                                                    given_longtitude)
    if start_end_node_distance > threshold or lat_distance > threshold or long_distance > threshold:
        if recursion_depth<100:
            recursion_depth=recursion_depth+1
            return privacy_preservation(index, user_dataset, row, threshold,recursion_depth)
        else:
            return updated_latitude, updated_longitude

    else:
        return updated_latitude, updated_longitude


def main():
    # origianl_dataset = pd.read_csv('data/small_static.csv')
    # user_dataset = pd.read_csv('data/staticmap.csv')
    origianl_dataset = pd.read_csv('data/small_static.csv')
    user_dataset = pd.read_csv('data/small_static.csv')
    # origianl_dataset = origianl_dataset[:1]
    upLat = []
    upLon = []
    upUser = []
    for index, row in origianl_dataset.iterrows():
        updated_latitude, updated_longitude = privacy_preservation(index, user_dataset, row, threshold=1,recursion_depth=0)

        upLat.append(updated_latitude)
        upLon.append(updated_longitude)
        upUser.append(row['user_id'])
    final_df = pd.DataFrame({'user_id': upUser, 'location_id': upLat, 'timestamp': upLon})
    final_df.to_csv('Data/resultcsvV4.csv', index=False)


if __name__ == '__main__':
    main()
