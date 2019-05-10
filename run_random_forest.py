import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from joblib import numpy_pickle

def train(dataset):
    print("Training with "+dataset)
    data = pd.read_csv("processed_training_input_"+dataset+".csv")
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
    print("\tTrained! Serializing.")
    numpy_pickle.dump(rf, "randomforest_"+dataset+".joblib")

rf = {}
def load(dataset):
    print("Loading model for "+dataset)
    rf[dataset] = numpy_pickle.load("randomforest_"+dataset+".joblib")

#inputs = [day, hour, GFS_temp, NAM_temp, GFS_hum, NAM_dew, load_t_72, load_t_78, load_t_84, load_t_90]
def predict(dataset, inputs):
    if not dataset in rf:
        load(dataset)
    day_sin = np.sin(inputs[0]*2*np.pi/365)
    day_cos = np.cos(inputs[0]*2*np.pi/365)
    hour_sin = np.sin(inputs[1]*2*np.pi/24)
    hour_cos = np.cos(inputs[1]*2*np.pi/24)
    return rf[dataset].predict([[day_cos, day_sin, hour_cos, hour_sin]+inputs[2:]])

def main():
    train("load_1")
    train("load_12")
    train("load_51")

if __name__ == "__main__":
    main()