import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

def get_input():
    input = pd.read_csv("collated_training_input.csv")
    input = input.drop(columns=input.columns[0])
    input["validtime"] = pd.to_datetime(input["validtime"])
    input.insert(1, "day", [d.dayofyear for d in input["validtime"]])
    input.insert(2, "hour", [d.hour for d in input["validtime"]])
    input = input.drop(columns="validtime")
    return input

if __name__ == "__main__":
    input = get_input()
    print(input.columns)
    print(input.iloc[0])
    # input = input.iloc[:200]
    GFS_temp = input[["GFS_Temp."+str(n) for n in range(1, 13)]]
    NAM_temp = input[["NAM_Temp."+str(n) for n in range(1, 13)]]
    GFS_hum = input[["GFS_Relative_humidity."+str(n) for n in range(1, 13)]]
    NAM_dew = input[["NAM_DewPoint."+str(n) for n in range(1, 13)]]
    load_delayed = pd.concat([pd.Series([None]*(24*3)), input["target_load"][:len(input["target_load"])-24*3]])
    load_delayed.index = range(0, len(load_delayed))

    def scatterplot(data, label):
        plt.scatter(data, input["target_load"], s=1)
        plt.xlabel(label)
        plt.tight_layout()

    plt.figure(1)
    plt.plot(input.index, input["target_load"]-200, input.index, input["hour"]*4+100, input.index, (input["NAM_Temp.5"]-250)*10, linewidth=0.5)
    plt.figure(2, figsize=(5, 3))
    scatterplot(input["day"], "day")
    plt.figure(3, figsize=(5, 3))
    scatterplot(input["hour"], "hour")
    plt.figure(4, figsize=(5, 3))
    scatterplot(GFS_temp.mean(1), "GFS_temp")
    plt.figure(5, figsize=(5, 3))
    scatterplot(NAM_temp.mean(1), "NAM_temp")
    plt.figure(6, figsize=(5, 3))
    scatterplot(GFS_hum.mean(1), "GFS_hum")
    plt.figure(7, figsize=(5, 3))
    scatterplot(NAM_dew.mean(1), "NAM_dew")
    plt.figure(8, figsize=(5, 3))
    scatterplot(load_delayed, "load_delayed")
    plt.show()