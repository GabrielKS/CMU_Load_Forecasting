import numpy as np

def predict(dataset, day, hour, GFS_temp, NAM_temp, GFS_hum, NAM_dew, load_t_72, load_t_78, load_t_84, load_t_90):
    modelnames = ["randomforest", "MLP"]
    predictions = []
    for modelname in modelnames:
        import simulate_forecast    #If I did this at the top of the file, it would complain about the run_ensemble.predict function being undefined
        predictions.append(simulate_forecast.modeldefs[modelname](dataset, day, hour, GFS_temp, NAM_temp, GFS_hum, NAM_dew, load_t_72, load_t_78, load_t_84, load_t_90))    #TODO: maybe read the files instead?
    return np.mean(predictions)

def main(): #Nothing really to do here; might as well test.
    print(predict("load_1", 100, 4, 290, 292, 60, 280, 300, 310, 320, 330))
    print(predict("load_12", 100, 4, 290, 292, 60, 280, 300, 310, 320, 330))
    print(predict("load_51", 100, 4, 290, 292, 60, 280, 300, 310, 320, 330))

if __name__ == "__main__":
    main()