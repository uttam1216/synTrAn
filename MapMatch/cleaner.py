import pandas as pd


def clean_segments():
    # for i in range(1500):
    for i in range(100):
        try:
            # filename = 'Segments/' + str(i) + '_result_quadtree_segments.csv'
            # filename1 = 'Quadpoints/' + str(i) + '_result_quadtree_points.csv'
            filename = 'test_segments/' + str(i) + '_test_result_quadtree_segments.csv'
            filename1 = 'Test_points/' + str(i) + '_test_result_quadtree_points.csv'
            df = pd.read_csv(filename, skiprows=2, header=None)
            df1 = pd.read_csv(filename1, skiprows=2, header=None)
            print(filename)
            print(filename1)
            print(df)
            print(df1)
            df.to_csv(filename, index=False)
            df1.to_csv(filename1, index=False)
        except pd.errors.ParserError:
            print("Parse Error")
        except pd.errors.EmptyDataError:
            print("No data")
        except Exception:
            print("Some other exception")


# def clean_points():
#     for i in range(1):
#         i = 48
#         filename = 'temp/' + str(i) + '_result_quadtree_points.csv'
#         df = pd.read_csv(filename,skiprows=2,header=None)
#         print(df)


clean_segments()
# clean_points()
