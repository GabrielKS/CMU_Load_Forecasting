#Mostly to make sure that everything runs error-free.
import collate_input
import process_input
import visualize_input
import run_random_forest
import run_MLP
import run_ensemble
import simulate_forecast
import my_evaluate_forecast

collate_input.main()
process_input.main()
visualize_input.main()
run_random_forest.main()
run_MLP.main(verbose=False)
run_ensemble.main()
simulate_forecast.main(models=("control", "randomforest", "MLP"))
my_evaluate_forecast.main()