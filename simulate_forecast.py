import pandas as pd
import run_random_forest
import run_MLP
import run_ensemble

def control(dataset, day, hour, GFS_temp, NAM_temp, GFS_hum, NAM_dew, load_t_72, load_t_78, load_t_84, load_t_90):
    return load_t_72

def pi(dataset, day, hour, GFS_temp, NAM_temp, GFS_hum, NAM_dew, load_t_72, load_t_78, load_t_84, load_t_90):
    return 314.15

modeldefs = {
    "control": control,
    "pi": pi,
    "randomforest": run_random_forest.predict,
    "MLP": run_MLP.predict,
    "ensemble": run_ensemble.predict
}

def predict(dataset, model, day, hour, GFS_temp, NAM_temp, GFS_hum, NAM_dew, load_t_72, load_t_78, load_t_84, load_t_90):
    return modeldefs[model](dataset, day, hour, GFS_temp, NAM_temp, GFS_hum, NAM_dew, load_t_72, load_t_78, load_t_84, load_t_90)
    #load = run_random_forest.predict(dataset, [day, hour, GFS_temp, NAM_temp, GFS_hum, NAM_dew, load_t_72, load_t_78, load_t_84, load_t_90])[0]
    #load = load_t_72
    #load = 314.15
    #return load

def simulate(dataset, model):
    print("Simulating "+dataset+" with "+model)
    load = pd.read_csv("data/test/"+dataset+"/load.csv")
    gfs = pd.read_csv("data/test/"+dataset+"/gfs.csv")
    nam = pd.read_csv("data/test/"+dataset+"/nam.csv")

    load["validtime"] = pd.to_datetime(load["validtime"])
    gfs["validtime"] = pd.to_datetime(gfs["validtime"])
    nam["validtime"] = pd.to_datetime(nam["validtime"])
    gfs["runtime"] = pd.to_datetime(gfs["runtime"])
    nam["runtime"] = pd.to_datetime(nam["runtime"])

    load.index = load["validtime"]
    gfs.index = gfs["runtime"]
    nam.index = nam["runtime"]

    results = pd.DataFrame(columns=["runtime", "validtime", "prediction"])
    start = pd.to_datetime("2018-08-12 08:00:00")
    end = pd.to_datetime("2019-02-11 08:00:00")
    if dataset=="load_51":
        start += pd.DateOffset(days=1)
        #end += pd.DateOffset(days=1)
        end = pd.to_datetime("2018-11-19 08:00:00") #This is the last day I can simulate without running out of data.
    for runtime in pd.date_range(start=start, end=end, freq="D"):
        if (runtime.dayofyear % 5 == 0) : print("\tDay "+str(runtime.dayofyear)) #Progress bar
        gfs_today = gfs.loc[runtime-pd.DateOffset(hours=2)]
        nam_today = nam.loc[runtime-pd.DateOffset(hours=2)]
        gfs_today.index = gfs_today["validtime"]
        nam_today.index = nam_today["validtime"]
        for i in range(0, 24):
            validtime = runtime+pd.DateOffset(hours=i)
            day = validtime.dayofyear
            hour = validtime.hour
            gfs_now = gfs_today.loc[validtime]
            nam_now = nam_today.loc[validtime]
            # GFS_hum = raw[["GFS_Relative_humidity." + str(n) for n in range(1, 13)]].mean(1)
            GFS_temp = gfs_now[["Temp."+str(n) for n in range(1, 13)]].mean()
            NAM_temp = nam_now[["Temp."+str(n) for n in range(1, 13)]].mean()
            GFS_hum = gfs_now[["Relative_humidity."+str(n) for n in range(1, 13)]].mean()
            NAM_dew = nam_now[["DewPoint."+str(n) for n in range(1, 13)]].mean()
            load_t_72 = load["target_load"].loc[validtime-pd.DateOffset(hours=72)]
            load_t_78 = load["target_load"].loc[validtime-pd.DateOffset(hours=78)]
            load_t_84 = load["target_load"].loc[validtime-pd.DateOffset(hours=84)]
            load_t_90 = load["target_load"].loc[validtime-pd.DateOffset(hours=90)]

            prediction = predict(dataset, model, day, hour, GFS_temp, NAM_temp, GFS_hum, NAM_dew, load_t_72, load_t_78, load_t_84, load_t_90)
            results = results.append({"runtime": runtime, "validtime": validtime, "prediction": prediction}, ignore_index=True) #Possible performance improvement: append to list instead of DataFrame
    print("\tDone simulating!")

    results.index = results["validtime"]
    load.drop(columns="validtime", inplace=True)
    output = results.merge(load, how="outer", left_index=True, right_index=True)
    output.dropna(inplace=True)
    output.to_csv("results/results_"+dataset+"_"+model+".csv", index=False)

def main(models=("ensemble",)):
    datasets = ("load_1", "load_12", "load_51")
    #models = ("control", "randomforest", "MLP")
    for model in models:
        for dataset in datasets:
            simulate(dataset, model)

if __name__ == "__main__":
    main()