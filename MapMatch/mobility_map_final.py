import overpy
import pandas as pd

# api = overpy.Overpass(url='https://maps.mail.ru/osm/tools/overpass/api/interpreter')
api = overpy.Overpass(url='https://overpass.kumi.systems/api/interpreter')


def way_ids(result):
    wayid = 0
    for i in range(len(result.nodes)):
        if result.nodes[i].id > 0:
            # print(result.nodes[i].lat)
            # print(result.nodes[i].lon)
            wayid = result.nodes[i].id
            # print(wayid)
            break
    return wayid


def way_ids_points(result):
    wayid = 0
    for i in range(len(result.nodes) - 1, -1, -1):
        if result.nodes[i].id > 0:
            # print(result.nodes[i].lat)
            # print(result.nodes[i].lon)
            wayid = result.nodes[i].id
            # print(wayid)
            break
    return wayid


def segments_id(filename_segments, filename_points):
    finallist_segments = []
    finallist_points = []
    finallist_points_first = []
    try:
        df = pd.read_csv(filename_segments, header=None)
        df1 = pd.read_csv(filename_points, header=None)
        # print("Start Length")
        # print(len(df))
        # print(len(df1))

        for i in range(len(df)):
            list = []
            list_points = []
            list.append(df[1][i])
            list.append(df[0][i])
            list.append(df[3][i])
            list.append(df[2][i])

            list_points.append(df1[1][i])
            list_points.append(df1[0][i])
            list_points.append(df1[3][i])
            list_points.append(df1[2][i])
            query_segments = """area[name="Porto"]->.searchArea;node(around:100.00,""" + str(list[0]) + """,""" + str(
                list[1]) + """)['highway'](area.searchArea);out body; """
            result_segments = api.query(query_segments)
            query_points = """area[name="Porto"]->.searchArea;node(around:50.00,""" + str(
                list_points[0]) + """,""" + str(
                list_points[1]) + """)['highway'](area.searchArea);out body; """
            result_points = api.query(query_points)
            print("Lenth of results points")
            print(len(result_points.nodes))
            wayid_segments = way_ids(result_segments)
            print("Lenth of results Segments")
            print(len(result_segments.nodes))
            wayid_points_first = way_ids(result_points)
            wayid_points_last = way_ids_points(result_points)
            if wayid_segments == 0:
                # print("Inside segments If")
                query_segments = """area[name="Porto"]->.searchArea;node(around:100.00,""" + str(
                    list[2]) + """,""" + str(
                    list[3]) + """)['highway'](area.searchArea);out body; """
                result_segments = api.query(query_segments)
                wayid_segments = way_ids(result_segments)
                # print("Inside segments If end")
            if wayid_points_last == 0:
                # print("inside points if")
                query_points = """area[name="Porto"]->.searchArea;node(around:100.00,""" + str(
                    list_points[2]) + """,""" + str(
                    list_points[3]) + """)['highway'](area.searchArea);out body; """
                result_points = api.query(query_points)
                wayid_points_last = way_ids_points(result_points)
                # print("inside points if End")
            if wayid_points_first == 0:
                # print("inside points if")
                query_points = """area[name="Porto"]->.searchArea;node(around:100.00,""" + str(
                    list_points[2]) + """,""" + str(
                    list_points[3]) + """)['highway'](area.searchArea);out body; """
                result_points = api.query(query_points)
                wayid_points_first = way_ids(result_points)
            if wayid_segments > 0 and wayid_points_last > 0 and wayid_points_first > 0:
                finallist_segments.append(wayid_segments)
                finallist_points.append(wayid_points_last)
                finallist_points_first.append(wayid_points_first)
        print('Results are:')
        print(finallist_segments)
        print(finallist_points)
        print(finallist_points_first)
        print(len(finallist_segments))
        print(len(finallist_points))
        print("Start Length")
        print(len(df))
        print(len(df1))
        return finallist_segments, finallist_points, finallist_points_first
    except pd.errors.ParserError:
        return finallist_segments, finallist_points, finallist_points_first
        print("Parse Error")
    except pd.errors.EmptyDataError:
        return finallist_segments, finallist_points, finallist_points_first
        print("No data")
    except Exception:
        return finallist_segments, finallist_points, finallist_points_first
        print("Some other exception")


for i in range(1):
    i = 1
    print(i)
    filename_segments = 'Segments/' + str(i) + '_result_quadtree_segments.csv'
    filename_points = 'Quadpoints/' + str(i) + '_result_quadtree_points.csv'
    finallist_segments, finallist_points, finallist_points_first = segments_id(filename_segments, filename_points)

    final_data = {'Points': finallist_points, 'PointsFirst': finallist_points_first, 'Segments': finallist_segments, }
    dataframe = pd.DataFrame(final_data)
    dataframe.to_csv("TrainDataset/" + str(i) + "_mobility_map.csv")

# print(dataframe)
