# import overpy
#
# api = overpy.Overpass()
#
# # fetch all ways and nodes
# result = api.query("""
#     node(-8.6289522,41.1611859,-8.6287117,41.1609629) ;
#     (._;>;);
#     out body;
#     """)
#
# for way in result.ways:
#     print("Name: %s" % way.tags.get("name", "n/a"))
#     print("  Highway: %s" % way.tags.get("highway", "n/a"))
#     print("  Nodes:")
#     for node in way.nodes:
#         print(node.id)
#         print("    Lat: %f, Lon: %f" % (node.lat, node.lon))


import overpy
import pandas as pd

# df=pd.read_csv("result_quadtree_segments.csv",usecols= ['lon1','lat1','lon2','lat2'])
df = pd.read_csv("result_quadtree_segments.csv", header=None)
# df = pd.read_csv("result_quadtree_points.csv", header=None)

api = overpy.Overpass(url='https://maps.mail.ru/osm/tools/overpass/api/interpreter')


def segments_id(filename):
    df = pd.read_csv(filename, header=None)
    for i in range(len(df)):
        list = []
        max_lat = 0
        max_lon = 0
        min_lat = 0
        min_lon = 0
        if df[3][i] < df[1][i]:
            list.append(df[3][i])
            min_lat = df[3][i]
        else:
            list.append(df[1][i])
            min_lat = df[1][i]
            # For max Lon
        if df[2][i] > df[0][i]:
            list.append(df[2][i])
            max_lon = df[2][i]
        else:
            list.append(df[0][i])
            max_lon = df[0][i]

        if df[3][i] > df[1][i]:
            list.append(df[3][i])
            max_lat = df[3][i]
        else:
            list.append(df[1][i])
            max_lat = df[1][i]
            # For max Lon
        if df[2][i] < df[0][i]:
            list.append(df[2][i])
            min_lon = df[2][i]
        else:
            list.append(df[0][i])
            min_lon = df[0][i]
        # if df[3][i] < df[1][i]:
        #     list.append(df[3][i])
        #     min_lat = df[3][i]
        # else:
        #     list.append(df[1][i])
        #     min_lat = df[1][i]
        #     # For max Lon
        #
        #
        # if df[2][i] < df[0][i]:
        #     list.append(df[2][i])
        #     min_lon = df[2][i]
        # else:
        #     list.append(df[0][i])
        #     min_lon = df[0][i]
        #
        # if df[3][i] > df[1][i]:
        #     list.append(df[3][i])
        #     max_lat = df[3][i]
        # else:
        #     list.append(df[1][i])
        #     max_lat = df[1][i]
        # # For max Lon
        # if df[2][i] > df[0][i]:
        #     list.append(df[2][i])
        #     max_lon = df[2][i]
        # else:
        #     list.append(df[0][i])
        #     max_lon = df[0][i]

        data = tuple(list)
        print(data)
        # print(max_lon)
        # print(min_lon)

        # data=(41.1609629,-8.6287117,41.1611859,-8.6289522)
        # query = """area[name="Porto"]->.searchArea;(way['highway'](area.searchArea)""" + str(
        #     data) + """;);out body;>;out skel qt;"""
        query = """area[name="Porto"]->.searchArea;(node(area.searchArea)""" + str(
            data) + """;);out body;>;out skel qt;"""
        result = api.query(query)

        for i in range(len(result.nodes)):
            # print(result.nodes[i].lon)
            wayid = 0
            if min_lat <= result.nodes[i].lat <= max_lat:
                # print(result.nodes[i].lat)
                # print(result.nodes[i].lon)
                print("way Id is ")
                print(result.nodes[i].id)
                wayid = result.nodes[i].id
                break
            if wayid == 0:
                if min_lon <= result.nodes[i].lon <= max_lon:
                    print(result.nodes[i].lon)
                    print(result.nodes[i].lat)
                    print (result.nodes[i].id)
                    break

segments_id("result_quadtree_segments.csv")

# for i in range(len(df)):
#     list = []
#     max_lat = 0
#     max_lon = 0
#     min_lat = 0
#     min_lon = 0
#     if df[3][i] < df[1][i]:
#         list.append(df[3][i])
#         min_lat = df[3][i]
#     else:
#         list.append(df[1][i])
#         min_lat = df[1][i]
#         # For max Lon
#     if df[2][i] > df[0][i]:
#         list.append(df[2][i])
#         max_lon = df[2][i]
#     else:
#         list.append(df[0][i])
#         max_lon = df[0][i]
#
#     if df[3][i] > df[1][i]:
#         list.append(df[3][i])
#         max_lat = df[3][i]
#     else:
#         list.append(df[1][i])
#         max_lat = df[1][i]
#         # For max Lon
#     if df[2][i] < df[0][i]:
#         list.append(df[2][i])
#         min_lon = df[2][i]
#     else:
#         list.append(df[0][i])
#         min_lon = df[0][i]
#     # if df[3][i] < df[1][i]:
#     #     list.append(df[3][i])
#     #     min_lat = df[3][i]
#     # else:
#     #     list.append(df[1][i])
#     #     min_lat = df[1][i]
#     #     # For max Lon
#     #
#     #
#     # if df[2][i] < df[0][i]:
#     #     list.append(df[2][i])
#     #     min_lon = df[2][i]
#     # else:
#     #     list.append(df[0][i])
#     #     min_lon = df[0][i]
#     #
#     # if df[3][i] > df[1][i]:
#     #     list.append(df[3][i])
#     #     max_lat = df[3][i]
#     # else:
#     #     list.append(df[1][i])
#     #     max_lat = df[1][i]
#     # # For max Lon
#     # if df[2][i] > df[0][i]:
#     #     list.append(df[2][i])
#     #     max_lon = df[2][i]
#     # else:
#     #     list.append(df[0][i])
#     #     max_lon = df[0][i]
#
#     data = tuple(list)
#     print(data)
#     # print(max_lon)
#     # print(min_lon)
#
#     # data=(41.1609629,-8.6287117,41.1611859,-8.6289522)
#     # query = """area[name="Porto"]->.searchArea;(way['highway'](area.searchArea)""" + str(
#     #     data) + """;);out body;>;out skel qt;"""
#     query = """area[name="Porto"]->.searchArea;(node(area.searchArea)""" + str(
#         data) + """;);out body;>;out skel qt;"""
#     result = api.query(query)
#
#     for i in range(len(result.nodes)):
#         # print(result.nodes[i].lon)
#         wayid = 0
#         if min_lat <= result.nodes[i].lat <= max_lat:
#             # print(result.nodes[i].lat)
#             # print(result.nodes[i].lon)
#             print("way Id is ")
#             print(result.nodes[i].id)
#             wayid = result.nodes[i].id
#             break
#         if wayid == 0:
#             if min_lon <= result.nodes[i].lon <= max_lon:
#                 print(result.nodes[i].lon)
#                 print(result.nodes[i].lat)
#                 print (result.nodes[i].id)
#                 break
