import pandas as pd
import numpy as np
from keras.engine.saving import load_model
from keras.layers import Dense
from keras.models import Sequential

variables = ["day_cos", "day_sin", "hour_cos", "hour_sin", "GFS_temp", "NAM_temp", "GFS_hum", "NAM_dew", "load_t_72", "load_t_78", "load_t_84", "load_t_90"]

def train(dataset):
    print("Training with "+dataset)
    data = pd.read_csv("intermediate/processed_training_input_"+dataset+".csv")
    data = data.drop(columns=data.columns[0])

    data["day_sin"] = np.sin(data["day"]*2*np.pi/365)
    data["day_cos"] = np.cos(data["day"]*2*np.pi/365)
    data["hour_sin"] = np.sin(data["hour"]*2*np.pi/24)
    data["hour_cos"] = np.cos(data["hour"]*2*np.pi/24)

    dependent = np.array(data["load"])
    data = data[variables]
    independent = np.array(data)

    model = Sequential()
    model.add(Dense(500, activation = "relu", input_dim = len(variables)))
    model.add(Dense(1))
    model.compile(loss = "mse", optimizer = "adam")
    model.fit(independent, dependent, epochs = 100, batch_size = 100, verbose=False)
    print("\tTrained! Serializing.")
    model.save("models/MLP_"+dataset+".h5")

models = {}
def load(dataset):
    print("Loading model for "+dataset)
    models[dataset] = load_model("models/MLP_"+dataset+".h5")

def predict(dataset, day, hour, GFS_temp, NAM_temp, GFS_hum, NAM_dew, load_t_72, load_t_78, load_t_84, load_t_90):
    if not dataset in models:
        load(dataset)
    day_sin = np.sin(day*2*np.pi/365)
    day_cos = np.cos(day*2*np.pi/365)
    hour_sin = np.sin(hour*2*np.pi/24)
    hour_cos = np.cos(hour*2*np.pi/24)
    return models[dataset].predict(np.array([day_cos, day_sin, hour_cos, hour_sin, GFS_temp, NAM_temp, GFS_hum, NAM_dew, load_t_72, load_t_78, load_t_84, load_t_90]).reshape(1, len(variables)))[0][0]

def main():
    train("load_1")
    train("load_12")
    train("load_51")

if __name__ == "__main__":
    main()
    # print_importances("load_1")
    # print_importances("load_12")
    # print_importances("load_51")