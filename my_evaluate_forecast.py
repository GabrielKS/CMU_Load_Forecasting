import pandas as pd
import numpy as np

from sample.src.test_harness import evaluate_forecast

results = pd.read_csv("results.csv")
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
print("MAPEs by hour:")
print(mapes)
print("Total MAPE: "+str(mape))
print("Total RMSE: "+str(rmse))

#evaluation_metrics = evaluate_forecast.evaluate_forecast(results[["runtime", "validtime", "prediction"]], results[["validtime", "target_load"]])
#print(evaluation_metrics["hourly_metrics"]) #Note: the hourly RMSE given to me is broken (it doesn't square the errors)