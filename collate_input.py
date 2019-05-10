import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def filter_for_most_valid(df):  #Only use the most recent data. May want to change a bit since the last two hours need to use the second-most recent data in reality.
    # rows = df.index
    cols = list(df.columns)
    cols[0], cols[1] = cols[1], cols[0]
    # df.sort_values("validtime", inplace=True)
    # df.index = rows
    df = df.reindex(columns=cols)
    filtered = {}
    for index, row in df.iterrows():
        this_validtime = row["validtime"]
        this_runtime = row["runtime"]
        if (this_validtime not in filtered) or (this_runtime > filtered[this_validtime]["runtime"]):
            filtered[this_validtime] = row
    fdf = pd.DataFrame(filtered).transpose()
    #fdf.index = range(0, len(fdf.index))
    return fdf

"""
def filter_for_most_valid(df):  #Tried to make it faster. May have ended up slower?
    cols = list(df.columns)
    cols[0], cols[1] = cols[1], cols[0]
    df = df.reindex(columns=cols)
    df.sort_values(["validtime", "runtime"], inplace=True)
    df.index = range(0, len(df.index))
    fdf = []
    for index, row in df.iterrows():
        if (index+1 >= len(df.index)) or (row["validtime"] != df.loc[index+1]["validtime"]):  #If the row is at the end or the next entry is for a different validtime
            fdf.append(row)
    print(len(fdf))
    return pd.DataFrame(fdf)
"""

load = pd.read_csv("data/train/load_1/load.csv")
gfs = pd.read_csv("data/train/load_1/gfs.csv")
nam = pd.read_csv("data/train/load_1/nam.csv")


if (False): #Limit data for faster testing
    limit = 100
    load = load.iloc[:limit]
    gfs = gfs.iloc[:limit]
    nam = nam.iloc[:limit]

load.index = load["validtime"]
gfs = filter_for_most_valid(gfs)
nam = filter_for_most_valid(nam)
#print(gfs)
#print("filtered")

gfs = gfs.drop(columns="validtime")   #Eliminate duplicate validtimes
nam = nam.drop(columns="validtime")

gfs = gfs.drop(columns="runtime")   #Let's ignore the runtime (for now?)
nam = nam.drop(columns="runtime")

gfs.columns = "GFS_"+gfs.columns
nam.columns = "NAM_"+nam.columns

all_data = load.merge(gfs, how="outer", left_index=True, right_index=True).merge(nam, how="outer", left_index=True, right_index=True)
all_data.dropna(inplace=True)   #Ignore instances where some of the data is missing
all_data.to_csv("collated_training_input.csv")