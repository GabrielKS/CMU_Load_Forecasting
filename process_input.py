import pandas as pd

def process_input(dataset):
    print("Processing "+dataset)
    raw = pd.read_csv("intermediate/collated_training_input_"+dataset+".csv")

    #Make sense of the times
    raw["validtime"] = pd.to_datetime(raw["validtime"])
    raw.insert(1, "day", [d.dayofyear for d in raw["validtime"]])
    raw.insert(2, "hour", [d.hour for d in raw["validtime"]])
    raw = raw.drop(columns="validtime")

    #Average out the weather variables
    GFS_temp = raw[["GFS_Temp."+str(n) for n in range(1, 13)]].mean(1)
    NAM_temp = raw[["NAM_Temp."+str(n) for n in range(1, 13)]].mean(1)
    GFS_hum = raw[["GFS_Relative_humidity."+str(n) for n in range(1, 13)]].mean(1)
    NAM_dew = raw[["NAM_DewPoint."+str(n) for n in range(1, 13)]].mean(1)

    #Compute "sliding windows"
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

    processed.dropna(inplace=True)
    processed.to_csv("intermediate/processed_training_input_"+dataset+".csv")
    return processed


def main():
    process_input("load_1")
    process_input("load_12")
    process_input("load_51")

if __name__ == "__main__":
    main()