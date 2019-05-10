import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.ensemble import RandomForestRegressor

data = pd.read_csv("processed_training_input.csv")
data = data.drop(columns=data.columns[0])

data["day_sin"] = np.sin(data["day"]*2*np.pi/365)
data["day_cos"] = np.cos(data["day"]*2*np.pi/365)
data["hour_sin"] = np.sin(data["hour"]*2*np.pi/24)
data["hour_cos"] = np.cos(data["hour"]*2*np.pi/24)

dependent = np.array(data["load"])
data = data[["day_cos", "day_sin", "hour_cos", "hour_sin", "GFS_temp", "NAM_temp", "GFS_hum", "NAM_dew", "load_t_72", "load_t_78", "load_t_84", "load_t_90"]]
variables = list(data.columns)
independent = np.array(data)

rf = RandomForestRegressor(n_estimators=100, random_state=0)
rf.fit(independent, dependent)
print("Trained!")

#day_cos, day_sin, hour_cos, hour_sin, GFS_temp, NAM_temp, GFS_hum, NAM_dew, load_t_72, load_t_78, load_t_84, load_t_90
def predict(inputs):
    return rf.predict([inputs])

#print(data.iloc[0])
#print(predict(data.iloc[0]))