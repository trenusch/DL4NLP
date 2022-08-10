import argparse
import csv
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_path', type=str, default="data/output.txt")
    parser.add_argument('--out_path', type=str, default="data/output.csv")
    args = parser.parse_args()
    in_path = args.in_path
    out_path = args.out_path
    with open(in_path, 'r', encoding='utf-8') as f:
        next(f)
        lines = f.readlines()

    data = {}
    for line in lines:
        line_split = line.split(",")
        if line_split[0] in data:  # metric
            if line_split[1] in data[line_split[0]]:  # dataset
                data[line_split[0]][line_split[1]][line_split[2]] = line_split[-1][:-2]
            else:
                data[line_split[0]][line_split[1]] = {line_split[2]: line_split[-1][:-2]}
        else:
            data[line_split[0]] = {line_split[1]: {line_split[2]: line_split[-1][:-2]}}

    tab1 = ["metric", "dataset", "antosub", "entrep", "numedit", "average"]
    tab2 = ["metric", "error", "faccte", "rankte", "qagsc", "faithfact", "average"]
    tab3 = ["metric", "average"]

    out_path_tab1 = out_path[:-4] + "tab1.csv"
    out_path_tab2 = out_path[:-4] + "tab2.csv"
    out_path_tab3 = out_path[:-4] + "tab3.csv"

    with open(out_path_tab1, 'w', newline='') as out:
        writer = csv.writer(out)
        writer.writerow(tab1)
        for metric in data:
            for dataset in data[metric]:
                antosub = data[metric][dataset]["AntoSub"]
                numedit = data[metric][dataset]["NumEdit"]
                entrep = data[metric][dataset]["EntRep"]
                row = [metric, dataset, antosub, entrep, numedit,
                       np.mean([float(antosub), float(numedit), float(entrep)])]
                writer.writerow(row)

    errors = ["AntoSub", "EntRep", "NumEdit"]
    with open(out_path_tab2, 'w', newline='') as out:
        writer = csv.writer(out)
        writer.writerow(tab2)
        for metric in data:
            for error in errors:
                faccte = data[metric]["FaccTe"][error]
                qagsc = data[metric]["QagsC"][error]
                faithfact = data[metric]["FaithFact"][error]
                rankte = data[metric]["RankTe"][error]
                row = [metric, error, faccte, qagsc, faithfact, rankte,
                       np.mean([float(faccte), float(faithfact), float(qagsc), float(rankte)])]
                writer.writerow(row)

    with open(out_path_tab3, 'w', newline='') as out:
        writer = csv.writer(out)
        writer.writerow(tab3)
        for metric in data:
            average = np.mean(
                [float(data[metric][dataset][error]) for dataset in data[metric] for error in data[metric][dataset]])
            row = [metric, average]
            writer.writerow(row)
