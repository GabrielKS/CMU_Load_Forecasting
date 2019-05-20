import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def my_evaluate_forecast(dataset, model):
    print("Evaluation of "+model+" forecast for dataset "+dataset+":")
    results = pd.read_csv("results/results_"+dataset+"_"+model+".csv")
    results["validtime"] = pd.to_datetime(results["validtime"])
    results.index = [d.hour for d in results["validtime"]]
    mapes = []
    for i in range(0, 24):
        predictions = results.loc[(i+8)%24]["prediction"]
        actuals = results.loc[(i+8)%24]["target_load"]
        mapes.append(np.mean(100 * abs(predictions-actuals)/actuals))
    predictions = results["prediction"]
    actuals = results["target_load"]
    mape = np.mean(100 * abs(predictions-actuals)/actuals)
    rmse = np.sqrt(np.mean(np.power(predictions-actuals, 2)))
    # print("MAPEs by hour:")
    # print(mapes)
    # print("Total MAPE: "+str(mape))
    print("Total RMSE: "+str(rmse))

    #evaluation_metrics = evaluate_forecast.evaluate_forecast(results[["runtime", "validtime", "prediction"]], results[["validtime", "target_load"]])
    #print(evaluation_metrics["hourly_metrics"]) #Note: the hourly RMSE given to me is broken (it doesn't square the errors)

def main():
    datasets = ["load_1", "load_12", "load_51"]
    models = ["control", "randomforest", "MLP", "ensemble", "SVM"]
    for model in models:
        for dataset in datasets:
            my_evaluate_forecast(dataset, model)
            print()
    return
    figure = 1
    for dataset in datasets:
        plt.figure(figure)
        actuals = pd.read_csv("results/results_" + dataset + "_control.csv")
        actuals["validtime"] = pd.to_datetime(actuals["validtime"])
        plt.plot(actuals["validtime"], actuals["target_load"], zorder=100)
        for model in models:
            results = pd.read_csv("results/results_" + dataset + "_" + model + ".csv")
            results["validtime"] = pd.to_datetime(results["validtime"])
            results.rename(columns={"prediction": "prediction_"+model}, inplace=True)
            plt.plot(results["validtime"], results["prediction_"+model])
        figure += 1
        plt.legend()
        plt.show()

if __name__ == "__main__":
    main()