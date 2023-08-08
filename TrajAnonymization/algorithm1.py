import random

import pandas as pd
import ast


def calculate_transition_distance(latitude, longitude):
    latitude = eval(latitude)
    longitude = eval(longitude)
    latitude_distance = 0.0
    longitude_distance = 0.0
    for i in range(len(latitude) - 1):
        latitude_distance = latitude_distance + abs(latitude[i] - latitude[i + 1])
        longitude_distance = longitude_distance + abs(longitude[i] - longitude[i + 1])

    return round(latitude_distance / (len(latitude) - 1), 5), round(longitude_distance / (len(longitude) - 1), 5)


def main():
    origianl_dataset = pd.read_csv('data/staticmap.csv')
    print(origianl_dataset[:5])

    print(origianl_dataset.columns)

    latitude = origianl_dataset['location_id'][1]
    longitude = origianl_dataset['timestamp'][1]

    print(latitude)
    print(longitude)

    print(len(latitude))

    calculate_transition_distance(latitude, longitude)


if __name__ == '__main__':
    main()
