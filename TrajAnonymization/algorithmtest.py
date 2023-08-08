import numpy as np

lat = 0
lon = 0

la = 1
lo = 1

current_lat = 0
current_lon = 0

zero_array = np.zeros((5, 5))
print(zero_array)

# if
# zero_array[3, 2] = zero_array[3, 2] + 1
if (lat + la) < current_lat <= (lat + la + la) and lon - (lo + lo) <= current_lon < (lon - lo):  # lat 2 lon -2
    zero_array[0, 0] = zero_array[0, 0] + 1
    print("lat 2 and lon -2")
elif (lat + la) < current_lat <= (lat + la + la) and lon - lo <= current_lon < lon:  # lat 2 lon -1
    zero_array[0, 1] = zero_array[0, 1] + 1
    print("lat 2 and lon -1")

elif (lat + la) < current_lat <= (lat + la + la) and lon < current_lon <= (lon + lo):  # lat 2 lon 1
    zero_array[0, 3] = zero_array[0, 3] + 1
    print("lat 2 and lon 1")
elif (lat + la) < current_lat <= (lat + la + la) and (lon + lo) < current_lon <= (lon + lo + lo):  # lat 2 lon 2
    zero_array[0, 4] = zero_array[0, 4] + 1
    print("lat 2 and lon 2")


elif lat < current_lat <= (lat + la) and lon - (lo + lo) <= current_lon < (lon - lo):  # lat 1 lon -2
    zero_array[1, 0] = zero_array[1, 0] + 1
    print("lat 1 and lon -2")
elif lat < current_lat <= (lat + la) and lon - lo <= current_lon < lon:  # lat 1 lon -1
    zero_array[1, 1] = zero_array[1, 1] + 1
    print("lat 1 and lon -1")

elif lat < current_lat <= (lat + la) and lon < current_lon <= (lon + lo):  # lat 1 lon 1
    zero_array[1, 3] = zero_array[1, 3] + 1
    print("lat 1 and lon 1")
elif lat < current_lat <= (lat + la) and (lon + lo) < current_lon <= (lon + lo + lo):  # lat 1 lon 2
    zero_array[1, 4] = zero_array[1, 4] + 1
    print("lat 1 and lon 2")


elif lat - la <= current_lat < lat and lon - (lo + lo) <= current_lon < (lon - lo):  # lat -1 lon -2
    zero_array[3, 0] = zero_array[3, 0] + 1
    print("lat -1 and lon -2")
elif lat - la <= current_lat < lat and lon - lo <= current_lon < lon:  # lat -1 lon -1
    zero_array[3, 1] = zero_array[3, 1] + 1
    print("lat -1 and lon -1")
elif lat - la <= current_lat < lat and lon < current_lon <= (lon + lo):  # lat -1 lon 1
    zero_array[3, 3] = zero_array[3, 3] + 1
    print("lat -1 and lon 1")
elif lat - la <= current_lat < lat and (lon + lo) < current_lon <= (lon + lo + lo):  # lat -1 lon 2
    zero_array[3, 4] = zero_array[3, 4] + 1
    print("lat -1 and lon 2")

elif lat - (la + la) <= current_lat < (lat - la) and lon - (lo + lo) <= current_lon < (lon - lo):  # lat -1 lon -2
    zero_array[4, 0] = zero_array[4, 0] + 1
    print("lat -1 and lon -2")
elif lat - (la + la) <= current_lat < (lat - la) and lon - lo <= current_lon < lon:  # lat -1 lon -1
    zero_array[4, 1] = zero_array[4, 1] + 1
    print("lat -1 and lon -1")
elif lat - (la + la) <= current_lat < (lat - la) and lon < current_lon <= (lon + lo):  # lat -1 lon 1
    zero_array[4, 3] = zero_array[4, 3] + 1
    print("lat -1 and lon 1")
elif lat - (la + la) <= current_lat < (lat - la) and (lon + lo) < current_lon <= (lon + lo + lo):  # lat -1 lon 2
    zero_array[4, 4] = zero_array[4, 4] + 1
    print("lat -1 and lon 2")

elif lat - la <= current_lat < lat and current_lon == lon:  # lat -1
    zero_array[3, 2] = zero_array[3, 2] + 1
    print("lat -1")
elif lat - (la + la) <= current_lat < (lat - la) and current_lon == lon:  # lat -2
    zero_array[4, 2] = zero_array[4, 2] + 1
    print("lat -2")

elif lat < current_lat <= (lat + la) and current_lon == lon:  # lat 1
    zero_array[1, 2] = zero_array[1, 2] + 1
    print("lat 1")
elif (lat + la) < current_lat <= (lat + la + la) and current_lon == lon:  # lat 2
    zero_array[0, 2] = zero_array[0, 2] + 1
    print("lat 2")


elif lon < current_lon <= (lon + lo) :  # lon 1
    zero_array[2, 3] = zero_array[2, 3] + 1
    print("lon 1")
elif (lon + lo) < current_lon <= (lon + lo + lo) and current_lat == lat:  # lon 2
    zero_array[2, 4] = zero_array[2, 4] + 1
    print("lon 2")
elif lon - lo <= current_lon < lon :  # lon -1
    zero_array[2, 1] = zero_array[2, 1] + 1
    print("lon -1")
elif lon - (lo + lo) <= current_lon < (lon - lo) and current_lat == lat:  # lon -2
    zero_array[2, 0] = zero_array[2, 0] + 1
    print("lon -2")
else:
    zero_array[2, 2] = zero_array[2, 2] + 1

print(zero_array)
