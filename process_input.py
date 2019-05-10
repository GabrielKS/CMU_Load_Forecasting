import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import visualize_input

raw = visualize_input.get_input()
GFS_temp = raw[["GFS_Temp."+str(n) for n in range(1, 13)]].mean(1)
NAM_temp = raw[["NAM_Temp."+str(n) for n in range(1, 13)]].mean(1)
GFS_hum = raw[["GFS_Relative_humidity."+str(n) for n in range(1, 13)]].mean(1)
NAM_dew = raw[["NAM_DewPoint."+str(n) for n in range(1, 13)]].mean(1)
processed = pd.DataFrame({"load": raw["target_load"], "day": raw["day"], "hour": raw["hour"], "GFS_temp": GFS_temp, "NAM_temp": NAM_temp, "GFS_hum": GFS_hum, "NAM_dew": NAM_dew})
window_width = 4
step = 6
dataset_size = len(raw["target_load"])
for i in range(0, window_width):
    hours_before = 24*3+i*step
    delayed = pd.concat([pd.Series([None]*hours_before), raw["target_load"][:dataset_size-hours_before]])
    delayed.index = range(0, dataset_size)
    label = "load_t_"+str(hours_before)
    processed[label] = delayed
    print(i)
processed.dropna(inplace=True)
print(processed)
print(processed.columns)
processed.to_csv("processed_training_input.csv")