import overpy
import pandas as pd

api = overpy.Overpass(url='https://maps.mail.ru/osm/tools/overpass/api/interpreter')
# api = overpy.Overpass(url='https://overpass.kumi.systems/api/interpreter')


def segments_id(filename_points):
    way_ground=[]
    try:
        df1 = pd.read_csv(filename_points, header=None)
        # print(df1)
        for i in range(len(df1)):
            list_points = []
            list_points.append(df1[1][i])
            list_points.append(df1[0][i])
            list_points.append(df1[3][i])
            list_points.append(df1[2][i])
            # query_points="""area[name="Porto"]->.searchArea;(way(around:30.00,""""""41.150790 , -8.630613 ,41.150775,
            # -8.630611)["highway"](area.searchArea);>;);out body; """
            query_points = """area[name="Porto"]->.searchArea;(way(around:20.00,""" + str(list_points[0]) + """,""" + str(list_points[1]) + """ ,""" + str(list_points[2]) + """,""" + str(list_points[3]) + """)['highway'](area.searchArea); >;);out body;"""
            # print(query_points)
            result_points = api.query(query_points)
            # print(len(result_points.ways))
            for i in range(len(result_points.ways)):
                way_ground.append(result_points.ways[i].id)
        return way_ground
                # print()
    except pd.errors.ParserError:
        return way_ground
        # return finallist_segments, finallist_points, finallist_points_first
        print("Parse Error")
    except pd.errors.EmptyDataError:
        return way_ground
        # return finallist_segments, finallist_points, finallist_points_first
        print("No data")
    except Exception:
        return way_ground
        # return finallist_segments, finallist_points, finallist_points_first
        print("Some other exception")

final_list=[]
for i in range(300):
    # i = 1
    print("File No:"+str(i))
    filename_points = 'Quadpoints/' + str(i) + '_result_quadtree_points.csv'
    way_ground=segments_id(filename_points)
    way_input=sorted(way_ground)
    # way_input.sort()
    # print(way_ground)
    # print(way_input)
    temp=[]
    temp.append(way_input)
    temp.append(way_ground)
    final_list.append(temp)
# print(final_list)
# for data in final_list:
#     print(data[0])
#     print(data[1])
# import pickle
# with open('/trainpickel/train1.ob', 'wb') as fp:
#     pickle.dump(final_list, fp)

with open(r'trainpickel/train2.txt', 'w') as fp:
    for data in final_list:
        for input in data[0]:
            fp.write("%s " % input)
        fp.write(":")
        for output in data[1]:
            fp.write("%s " % output)
        fp.write("\n")

