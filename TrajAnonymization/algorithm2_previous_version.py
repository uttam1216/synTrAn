import pandas as pd
import numpy as np
import random
from algorithm1 import calculate_transition_distance


def generate_anonymous_points(user_df, given_trajectory, latitude_distance, longitude_distance):
    given_latitude = given_trajectory['location_id']
    given_latitude = eval(given_latitude)
    given_longtitude = given_trajectory['timestamp']
    given_longtitude = eval(given_longtitude)
    given_length = len(given_latitude)
    resulted_trajectory = []
    for i in range(given_length):
        resulted_trajectory.append(np.zeros((5, 5)))
    # print(resulted_trajectory[0][0, 1])

    for index, row in user_df.iterrows():
        current_latitude = row['location_id']
        current_latitude = eval(current_latitude)
        current_longtitude = row['timestamp']
        current_longtitude = eval(current_longtitude)
        current_length = len(current_latitude)
        looplength = 0
        if current_length <= given_length:
            looplength = current_length
        else:
            looplength = given_length
        for i in range(looplength):
            if (given_latitude[i] + latitude_distance) < current_latitude[i] <= (
                    given_latitude[i] + latitude_distance + latitude_distance) and given_longtitude[i] - (
                    longitude_distance + longitude_distance) <= current_longtitude[i] < (
                    given_longtitude[i] - longitude_distance):  # given_latitude[i] 2 given_longtitude[i] -2
                resulted_trajectory[i][0, 0] = resulted_trajectory[i][0, 0] + 1
                # print("given_latitude[i] 2 and given_longtitude[i] -2")
            elif (given_latitude[i] + latitude_distance) < current_latitude[i] <= (
                    given_latitude[i] + latitude_distance + latitude_distance) and given_longtitude[
                i] - longitude_distance <= current_longtitude[i] < given_longtitude[
                i]:  # given_latitude[i] 2 given_longtitude[i] -1
                resulted_trajectory[i][0, 1] = resulted_trajectory[i][0, 1] + 1
                # print("given_latitude[i] 2 and given_longtitude[i] -1")

            elif (given_latitude[i] + latitude_distance) < current_latitude[i] <= (
                    given_latitude[i] + latitude_distance + latitude_distance) and given_longtitude[i] < \
                    current_longtitude[i] <= (
                    given_longtitude[i] + longitude_distance):  # given_latitude[i] 2 given_longtitude[i] 1
                resulted_trajectory[i][0, 3] = resulted_trajectory[i][0, 3] + 1
                # print("given_latitude[i] 2 and given_longtitude[i] 1")
            elif (given_latitude[i] + latitude_distance) < current_latitude[i] <= (
                    given_latitude[i] + latitude_distance + latitude_distance) and (
                    given_longtitude[i] + longitude_distance) < current_longtitude[i] <= (
                    given_longtitude[
                        i] + longitude_distance + longitude_distance):  # given_latitude[i] 2 given_longtitude[i] 2
                resulted_trajectory[i][0, 4] = resulted_trajectory[i][0, 4] + 1
                # print("given_latitude[i] 2 and given_longtitude[i] 2")


            elif given_latitude[i] < current_latitude[i] <= (given_latitude[i] + latitude_distance) and \
                    given_longtitude[i] - (longitude_distance + longitude_distance) <= current_longtitude[i] < (
                    given_longtitude[i] - longitude_distance):  # given_latitude[i] 1 given_longtitude[i] -2
                resulted_trajectory[i][1, 0] = resulted_trajectory[i][1, 0] + 1
                # print("given_latitude[i] 1 and given_longtitude[i] -2")
            elif given_latitude[i] < current_latitude[i] <= (given_latitude[i] + latitude_distance) and \
                    given_longtitude[i] - longitude_distance <= current_longtitude[i] < given_longtitude[
                i]:  # given_latitude[i] 1 given_longtitude[i] -1
                resulted_trajectory[i][1, 1] = resulted_trajectory[i][1, 1] + 1
                # print("given_latitude[i] 1 and given_longtitude[i] -1")

            elif given_latitude[i] < current_latitude[i] <= (given_latitude[i] + latitude_distance) and \
                    given_longtitude[i] < current_longtitude[i] <= (
                    given_longtitude[i] + longitude_distance):  # given_latitude[i] 1 given_longtitude[i] 1
                resulted_trajectory[i][1, 3] = resulted_trajectory[i][1, 3] + 1
                # print("given_latitude[i] 1 and given_longtitude[i] 1")
            elif given_latitude[i] < current_latitude[i] <= (given_latitude[i] + latitude_distance) and (
                    given_longtitude[i] + longitude_distance) < current_longtitude[i] <= (given_longtitude[
                                                                                              i] + longitude_distance + longitude_distance):  # given_latitude[i] 1 given_longtitude[i] 2
                resulted_trajectory[i][1, 4] = resulted_trajectory[i][1, 4] + 1
                # print("given_latitude[i] 1 and given_longtitude[i] 2")


            elif given_latitude[i] - latitude_distance <= current_latitude[i] < given_latitude[i] and given_longtitude[
                i] - (longitude_distance + longitude_distance) <= current_longtitude[i] < (
                    given_longtitude[i] - longitude_distance):  # given_latitude[i] -1 given_longtitude[i] -2
                resulted_trajectory[i][3, 0] = resulted_trajectory[i][3, 0] + 1
                # print("given_latitude[i] -1 and given_longtitude[i] -2")
            elif given_latitude[i] - latitude_distance <= current_latitude[i] < given_latitude[i] and given_longtitude[
                i] - longitude_distance <= current_longtitude[i] < given_longtitude[
                i]:  # given_latitude[i] -1 given_longtitude[i] -1
                resulted_trajectory[i][3, 1] = resulted_trajectory[i][3, 1] + 1
                # print("given_latitude[i] -1 and given_longtitude[i] -1")
            elif given_latitude[i] - latitude_distance <= current_latitude[i] < given_latitude[i] and given_longtitude[
                i] < current_longtitude[i] <= (
                    given_longtitude[i] + longitude_distance):  # given_latitude[i] -1 given_longtitude[i] 1
                resulted_trajectory[i][3, 3] = resulted_trajectory[i][3, 3] + 1
                # print("given_latitude[i] -1 and given_longtitude[i] 1")
            elif given_latitude[i] - latitude_distance <= current_latitude[i] < given_latitude[i] and (
                    given_longtitude[i] + longitude_distance) < current_longtitude[i] <= (given_longtitude[
                                                                                              i] + longitude_distance + longitude_distance):  # given_latitude[i] -1 given_longtitude[i] 2
                resulted_trajectory[i][3, 4] = resulted_trajectory[i][3, 4] + 1
                # print("given_latitude[i] -1 and given_longtitude[i] 2")

            elif given_latitude[i] - (latitude_distance + latitude_distance) <= current_latitude[i] < (
                    given_latitude[i] - latitude_distance) and given_longtitude[i] - (
                    longitude_distance + longitude_distance) <= current_longtitude[i] < (
                    given_longtitude[i] - longitude_distance):  # given_latitude[i] -1 given_longtitude[i] -2
                resulted_trajectory[i][4, 0] = resulted_trajectory[i][4, 0] + 1
                # print("given_latitude[i] -1 and given_longtitude[i] -2")
            elif given_latitude[i] - (latitude_distance + latitude_distance) <= current_latitude[i] < (
                    given_latitude[i] - latitude_distance) and given_longtitude[i] - longitude_distance <= \
                    current_longtitude[i] < given_longtitude[i]:  # given_latitude[i] -1 given_longtitude[i] -1
                resulted_trajectory[i][4, 1] = resulted_trajectory[i][4, 1] + 1
                # print("given_latitude[i] -1 and given_longtitude[i] -1")
            elif given_latitude[i] - (latitude_distance + latitude_distance) <= current_latitude[i] < (
                    given_latitude[i] - latitude_distance) and given_longtitude[i] < current_longtitude[i] <= (
                    given_longtitude[i] + longitude_distance):  # given_latitude[i] -1 given_longtitude[i] 1
                resulted_trajectory[i][4, 3] = resulted_trajectory[i][4, 3] + 1
                # print("given_latitude[i] -1 and given_longtitude[i] 1")
            elif given_latitude[i] - (latitude_distance + latitude_distance) <= current_latitude[i] < (
                    given_latitude[i] - latitude_distance) and (given_longtitude[i] + longitude_distance) < \
                    current_longtitude[i] <= (
                    given_longtitude[
                        i] + longitude_distance + longitude_distance):  # given_latitude[i] -1 given_longtitude[i] 2
                resulted_trajectory[i][4, 4] = resulted_trajectory[i][4, 4] + 1
                # print("given_latitude[i] -1 and given_longtitude[i] 2")

            elif given_latitude[i] - latitude_distance <= current_latitude[i] < given_latitude[i] and \
                    current_longtitude[i] == given_longtitude[i]:  # given_latitude[i] -1
                resulted_trajectory[i][3, 2] = resulted_trajectory[i][3, 2] + 1
                # print("given_latitude[i] -1")
            elif given_latitude[i] - (latitude_distance + latitude_distance) <= current_latitude[i] < (
                    given_latitude[i] - latitude_distance) and current_longtitude[i] == given_longtitude[
                i]:  # given_latitude[i] -2
                resulted_trajectory[i][4, 2] = resulted_trajectory[i][4, 2] + 1
                # print("given_latitude[i] -2")

            elif given_latitude[i] < current_latitude[i] <= (given_latitude[i] + latitude_distance) and \
                    current_longtitude[i] == given_longtitude[i]:  # given_latitude[i] 1
                resulted_trajectory[i][1, 2] = resulted_trajectory[i][1, 2] + 1
                # print("given_latitude[i] 1")
            elif (given_latitude[i] + latitude_distance) < current_latitude[i] <= (
                    given_latitude[i] + latitude_distance + latitude_distance) and current_longtitude[i] == \
                    given_longtitude[i]:  # given_latitude[i] 2
                resulted_trajectory[i][0, 2] = resulted_trajectory[i][0, 2] + 1
                # print("given_latitude[i] 2")
            elif given_longtitude[i] < current_longtitude[i] <= (
                    given_longtitude[i] + longitude_distance):  # given_longtitude[i] 1
                resulted_trajectory[i][2, 3] = resulted_trajectory[i][2, 3] + 1
                # print("given_longtitude[i] 1")
            elif (given_longtitude[i] + longitude_distance) < current_longtitude[i] <= (
                    given_longtitude[i] + longitude_distance + longitude_distance) and current_latitude[i] == \
                    given_latitude[i]:  # given_longtitude[i] 2
                resulted_trajectory[i][2, 4] = resulted_trajectory[i][2, 4] + 1
                # print("given_longtitude[i] 2")
            elif given_longtitude[i] - longitude_distance <= current_longtitude[i] < given_longtitude[
                i]:  # given_longtitude[i] -1
                resulted_trajectory[i][2, 1] = resulted_trajectory[i][2, 1] + 1
                # print("given_longtitude[i] -1")
            elif given_longtitude[i] - (longitude_distance + longitude_distance) <= current_longtitude[i] < (
                    given_longtitude[i] - longitude_distance) and current_latitude[i] == given_latitude[
                i]:  # given_longtitude[i] -2
                resulted_trajectory[i][2, 0] = resulted_trajectory[i][2, 0] + 1
                # print("given_longtitude[i] -2")
            else:
                resulted_trajectory[i][2, 2] = resulted_trajectory[i][2, 2] + 1

    # print("akash")
    # print(resulted_trajectory)
    return resulted_trajectory


def update_values(location, distance, value, type):
    updated_location = location
    if type == 'lat':
        if value == 0:
            updated_location = location + (distance + distance)
        elif value == 1:
            updated_location = location + distance
        elif value == 3:
            updated_location = location - distance
        elif value == 4:
            updated_location = location - (distance + distance)
    elif type == 'lon':
        if value == 0:
            updated_location = location - (distance + distance)
        elif value == 1:
            updated_location = location - distance
        elif value == 3:
            updated_location = location + distance
        elif value == 4:
            updated_location = location + (distance + distance)

    return updated_location


def calculate_final_locations(final_trajectory, latitude_distance, longitude_distance, row):
    current_latitude = row['location_id']
    current_latitude = eval(current_latitude)
    current_longtitude = row['timestamp']
    current_longtitude = eval(current_longtitude)
    updated_latitude = []
    updated_longitude = []
    for i in range(len(final_trajectory)):
        lat = current_latitude[i]
        lon = current_longtitude[i]
        value = final_trajectory[i]
        updated_latitude.append(update_values(lat, latitude_distance, value[0], 'lat'))
        updated_longitude.append(update_values(lon, longitude_distance, value[1], 'lon'))
    # print(current_longtitude)
    # print(updated_longitude)
    # print(current_latitude)
    # print(updated_latitude)
    return updated_latitude, updated_longitude


def main():
    origianl_dataset = pd.read_csv('data/small_static.csv')
    user_dataset = pd.read_csv('data/staticmap.csv')
    # origianl_dataset = origianl_dataset[:5]
    upLat = []
    upLon = []
    upUser = []
    for index, row in origianl_dataset.iterrows():
        print(index)
        user_df = user_dataset[user_dataset['user_id'] == row['user_id']]
        latitude_distance, longitude_distance = calculate_transition_distance(row['location_id'], row['timestamp'])
        # resulted_trajectory = generate_anonymous_points(user_df, row, 2*latitude_distance, 2*longitude_distance)
        resulted_trajectory = generate_anonymous_points(user_df, row, latitude_distance, longitude_distance)
        final_trajectory = []
        for i in range(len(resulted_trajectory)):
            # Flatten the array to find the minimum value
            flattened_array = resulted_trajectory[i].flatten()
            # Find the minimum value in the flattened array
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
        # print(len(final_trajectory))
        updated_latitude, updated_longitude = calculate_final_locations(final_trajectory, latitude_distance,
                                                                        longitude_distance, row)

        upLat.append(updated_latitude)
        upLon.append(updated_longitude)
        upUser.append(row['user_id'])

    final_df = pd.DataFrame({'user_id': upUser, 'location_id': upLat, 'timestamp': upLon})
    final_df.to_csv('Data/resultcsvV2.csv', index=False)


if __name__ == '__main__':
    main()
