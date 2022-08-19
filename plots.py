import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":

    path = "data/mix_human_corr_2.csv"
    with open(path, 'r', encoding='utf-8') as f:
        next(f)
        lines = f.readlines()

    data_corr = {}
    for line in lines:
        line_split = line.split(",")
        if line_split[0].split("_")[-1] in data_corr:  # metric
            data_corr[line_split[0].split("_")[-1]].append(float(line_split[-1]))
        else:
            data_corr[line_split[0].split("_")[-1]] = [float(line_split[-1])]

    path = "data/mix_outputtab3.csv"
    with open(path, 'r', encoding='utf-8') as f:
        next(f)
        lines = f.readlines()

    data = {}
    for line in lines:
        line_split = line.split("(")
        if line_split[0] in data:  # metric
            data[line_split[0]].append(float(line.split(",")[-1]))
        else:
            data[line_split[0]] = [float(line.split(",")[-1])]

    for m, n in zip(data, data_corr):
        legend = []
        plt.plot(np.linspace(0, 1, 11), data[m])
        legend.append("Average accuracy")
        plt.plot(np.linspace(0, 1, 11), data_corr[n])
        legend.append("Average Correlation")
        plt.legend(legend)
        plt.title(m)
        plt.xlabel("Weight of NLI scores")
        save_path = "data/figures/{}.png".format(m)
        plt.savefig(save_path)
        plt.show()