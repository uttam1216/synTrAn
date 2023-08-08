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
# print(df[0][3])
# print(df[1][3])
# print(df[2][3])
# print(df[3][3])
# df = pd.read_csv("result_quadtree_points.csv", header=None)

api = overpy.Overpass(url='https://maps.mail.ru/osm/tools/overpass/api/interpreter')


# Final_csv_segments = []
# finallist_segments = []
# finallist_points = []
def way_ids(result):
    wayid = 0
    for i in range(len(result.nodes)):

        if(result.nodes[i].id>0):
            print(result.nodes[i].lat)
            print(result.nodes[i].lon)
            wayid=result.nodes[i].id
            break
    return wayid
        # print(result.nodes[i].lon)

    #     if min_lat <= result.nodes[i].lat <= max_lat:
    #         print(result.nodes[i].lat)
    #         print(result.nodes[i].lon)
    #         print("way Id is ")
    #         # print(result.nodes[i].id)
    #         wayid = result.nodes[i].id
    #         break
    #     if wayid == 0:
    #         if min_lon <= result.nodes[i].lon <= max_lon:
    #             print(result.nodes[i].lon)
    #             print(result.nodes[i].lat)
    #             # print (result.nodes[i].id)
    #             wayid = result.nodes[i].id
    #             break
    # return wayid


def segments_id(filename_segments, filename_points):
    try:

        df = pd.read_csv(filename_segments, header=None)
        df1 = pd.read_csv(filename_points, header=None)
        print(len(df))
        print(len(df1))
        for i in range(len(df)):
            list = []
            list_points = []
            max_lat = 0
            max_lon = 0
            min_lat = 0
            min_lon = 0
            max_lat_points = 0
            max_lon_points = 0
            min_lat_points = 0
            min_lon_points = 0
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

            # Data Preparation for Points
            if df1[3][i] < df1[1][i]:
                list_points.append(df1[3][i])
                min_lat_points = df1[3][i]
            else:
                list_points.append(df1[1][i])
                min_lat_points = df1[1][i]
                # For max Lon
            if df1[2][i] > df1[0][i]:
                list_points.append(df1[2][i])
                max_lon_points = df1[2][i]
            else:
                list_points.append(df1[0][i])
                max_lon_points = df1[0][i]

            if df1[3][i] > df1[1][i]:
                list_points.append(df1[3][i])
                max_lat_points = df1[3][i]
            else:
                list_points.append(df1[1][i])
                max_lat_points = df1[1][i]
                # For max Lon
            if df1[2][i] < df1[0][i]:
                list_points.append(df1[2][i])
                min_lon_points = df1[2][i]
            else:
                list_points.append(df1[0][i])
                min_lon_points = df1[0][i]

            data = tuple(list)
            data1 = tuple(list_points)
            print(data)
            # print(data1)
            query = """area[name="Porto"]->.searchArea;(way['highway'](area.searchArea)""" + str(
                data) + """;);out body;>;out skel qt;"""
            query1 = """area[name="Porto"]->.searchArea;(way['highway'](area.searchArea)""" + str(
                data1) + """;);out body;>;out skel qt;"""
            result = api.query(query)
            result1 = api.query(query1)
            wayid_segments = 0
            for i in range(len(result.nodes)):
                if min_lat <= result.nodes[i].lat <= max_lat:
                    wayid_segments = result.nodes[i].id
                    break
                if wayid_segments == 0:
                    if min_lon <= result.nodes[i].lon <= max_lon:
                        wayid_segments = result.nodes[i].id
                        break
            # Loop for Points
            wayid_points = 0
            for i in range(len(result1.nodes)):
                if min_lat_points <= result1.nodes[i].lat <= max_lat_points:
                    # print("way Id is ")
                    wayid_points = result1.nodes[i].id
                    break
                if wayid_points == 0:
                    if min_lon_points <= result1.nodes[i].lon <= max_lon_points:
                        wayid_points = result1.nodes[i].id
                        break
            # print("Final Way id is ")
            # print(len(result.nodes))
            # print(len(result1.nodes))
            # print(wayid_segments)
            # print(wayid_points)
            if wayid_segments > 0 and wayid_points > 0:
                finallist_segments.append(wayid_segments)
                finallist_points.append(wayid_points)
        print('Results are:')
        print(finallist_segments)
        print(finallist_points)
        print(len(finallist_segments))
        print(len(finallist_points))
        # list_of_tuples=list(zip(finallist_segments,finallist_points))

    except pd.errors.ParserError:
        print("Parse Error")
    except pd.errors.EmptyDataError:
        print("No data")
    except Exception:
        print("Some other exception")


# for i in range(1):
#     filename_segments = 'Segments/' + str(i) + '_result_quadtree_segments.csv'
#     filename_points = 'Quadpoints/' + str(i) + '_result_quadtree_points.csv'
#     segments_id(filename_segments, filename_points)
# segments_id("result_quadtree_segments.csv", "result_quadtree_points.csv")

# final_data = {'Points': finallist_points, 'Segments': finallist_segments}
# dataframe = pd.DataFrame(final_data)
# dataframe.to_csv("mobility_map_testing.csv")
# print(dataframe)
df = pd.read_csv("result_quadtree_segments.csv", header=None)
counter_first = 0
counter_second = 0
for i in range(len(df)):
    list = []
    max_lat = 0
    max_lon = 0
    min_lat = 0
    min_lon = 0
    list.append(df[1][i])
    list.append(df[0][i])
    list.append(df[3][i])
    list.append(df[2][i])
    # if df[3][i] < df[1][i]:
    #     list.append(df[3][i])
    #     min_lat = df[3][i]
    # else:
    #     list.append(df[1][i])
    #     min_lat = df[1][i]
    #     # For max Lon
    # if df[2][i] > df[0][i]:
    #     list.append(df[2][i])
    #     max_lon = df[2][i]
    # else:
    #     list.append(df[0][i])
    #     max_lon = df[0][i]
    #
    # if df[3][i] > df[1][i]:
    #     list.append(df[3][i])
    #     max_lat = df[3][i]
    # else:
    #     list.append(df[1][i])
    #     max_lat = df[1][i]
    #     # For max Lon
    # if df[2][i] < df[0][i]:
    #     list.append(df[2][i])
    #     min_lon = df[2][i]
    # else:
    #     list.append(df[0][i])
    #     min_lon = df[0][i]
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
    data1 = tuple(list[:2])
    # print(data1)
    print(data)
    # print(max_lon)
    # print(min_lon)
    # query="""node(around:200.00,"""+str(list[0])+""","""+str(list[1])+""")['highway']; node(around:200.00,"""+str(list[0])+""","""+str(list[1])+""","""+str(list[2])+""","""+str(list[3])+""")['highway'];out body; """
    # data=(41.1609629,-8.6287117,41.1611859,-8.6289522)
    # query = """area[name="Porto"]->.searchArea;(way['highway'](area.searchArea)""" + str(
    #     data) + """;);out body;>;out skel qt;"""
    # query = """area[name="Porto"]->.searchArea;(node(area.searchArea)""" + str(
    #     data) + """;);out body;>;out skel qt;"""
    # query1 = """area[name="Porto"]->.searchArea;(way['highway'](area.searchArea)""" + str(
    #     data) + """;);out body;>;out skel qt;"""
    query = """area[name="Porto"]->.searchArea;node(around:20.00,""" + str(list[0]) + """,""" + str(list[1]) + """)['highway'](area.searchArea);out body; """
    result = api.query(query)
    # result1=api.query(query1)
    print(len(result.nodes))
    wayid = way_ids(result)
    print(wayid)
    if wayid == 0:
        # counter_first=counter_first-1
        print(list[2])
        print(list[3])
        query = """area[name="Porto"]->.searchArea;node(around:20.00,""" + str(list[2]) + """,""" + str(list[3]) + """)['highway'](area.searchArea);out body; """
        result = api.query(query)
        print("Inside If")
        wayid = way_ids(result)

        print(wayid)
        if wayid == 0:
            counter_first = counter_first - 1


    # if len(result.nodes)>0:
    #     print(result.nodes[0].lat)
    #     print(result.nodes[0].lon)
    # print(len(result1.nodes))

    # for i in range(len(result1.nodes)):
    #     # print(result.nodes[i].lon)
    #     wayid = 0
    #     if min_lat <= result1.nodes[i].lat <= max_lat:
    #         # print(result.nodes[i].lat)
    #         # print(result.nodes[i].lon)
    #         print("way Id is ")
    #         print(result1.nodes[i].id)
    #         wayid = result1.nodes[i].id
    #         counter_first=counter_first+1
    #         break
    #     if wayid == 0:
    #         if min_lon <= result1.nodes[i].lon <= max_lon:
    #             print(result1.nodes[i].lon)
    #             print(result1.nodes[i].lat)
    #             print (result1.nodes[i].id)
    #             counter_first=counter_first+1
    #             break
print(counter_first)
# print(counter_second)
