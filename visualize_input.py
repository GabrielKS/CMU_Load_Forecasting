import pandas as pd
import matplotlib.pyplot as plt


def visualize_input(dataset):
    input = pd.read_csv("intermediate/processed_training_input_"+dataset+".csv")

    def scatterplot(label):
        plt.scatter(input[label], input["load"], s=1)
        plt.xlabel(label)
        plt.tight_layout()

    plt.figure(1)
    plt.plot(input.index, input["load"]-200, input.index, input["hour"]*4+100, input.index, (input["NAM_temp"]-250)*10, linewidth=0.5)
    plt.figure(2, figsize=(5, 3))
    scatterplot("day")
    plt.figure(3, figsize=(5, 3))
    scatterplot("hour")
    plt.figure(4, figsize=(5, 3))
    scatterplot("GFS_temp")
    plt.figure(5, figsize=(5, 3))
    scatterplot("NAM_temp")
    plt.figure(6, figsize=(5, 3))
    scatterplot("GFS_hum")
    plt.figure(7, figsize=(5, 3))
    scatterplot("NAM_dew")
    plt.figure(8, figsize=(5, 3))
    scatterplot("load_t_72")
    plt.show()

def main():
    visualize_input("load_1")
    visualize_input("load_12")
    visualize_input("load_51")

if __name__ == "__main__":
    main()