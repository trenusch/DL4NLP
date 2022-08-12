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
                data[line_split[0]][line_split[1]][line_split[2]] = [float(line_split[-1][:4]), int(line_split[-3])]
            else:
                data[line_split[0]][line_split[1]] = {line_split[2]: [float(line_split[-1][:4]), int(line_split[-3])]}
        else:
            data[line_split[0]] = {line_split[1]: {line_split[2]: [float(line_split[-1][:4]), int(line_split[-3])]}}

    tab1 = ["metric", "FaccTe", "QagsC", "FaithFact", "RankTe", "average"]
    tab2 = ["metric", "AntoSub", "NumEdit", "EntRep", "average"]
    tab3 = ["metric", "average"]

    out_path_tab1 = out_path[:-4] + "tab1.csv"
    out_path_tab2 = out_path[:-4] + "tab2.csv"
    out_path_tab3 = out_path[:-4] + "tab3.csv"

    with open(out_path_tab1, 'w', newline='') as out:
        writer = csv.writer(out)
        writer.writerow(tab1)
        for metric in data:
            row = [metric]
            totals = []
            for dataset in data[metric]:
                antosub = data[metric][dataset]["AntoSub"]
                numedit = data[metric][dataset]["NumEdit"]
                entrep = data[metric][dataset]["EntRep"]
                average = np.round((antosub[0] * antosub[1] + numedit[0] * numedit[1] + entrep[0] * entrep[1]) / \
                                   (antosub[1] + numedit[1] + entrep[1]), 2)
                row.append(average)
            average = np.round_(np.mean(row[1:]), 2)
            row.append(average)
            writer.writerow(row)

    errors = ["AntoSub", "EntRep", "NumEdit"]
    with open(out_path_tab2, 'w', newline='') as out:
        writer = csv.writer(out)
        writer.writerow(tab2)
        for metric in data:
            row = [metric]
            totals = []
            total_acc = []
            for error in errors:
                faccte = data[metric]["FaccTe"][error]
                qagsc = data[metric]["QagsC"][error]
                faithfact = data[metric]["FaithFact"][error]
                rankte = data[metric]["RankTe"][error]
                average = np.round_(
                    (faccte[0] * faccte[1] + qagsc[0] * qagsc[1] + faithfact[0] * faithfact[1] + rankte[0] *
                     rankte[1]) / (faccte[1] + qagsc[1] + faithfact[1] + rankte[1]), 2)
                row.append(average)
            average = np.round_(np.mean(row[1:]), 2)
            row.append(average)
            writer.writerow(row)

    with open(out_path_tab3, 'w', newline='') as out:
        writer = csv.writer(out)
        writer.writerow(tab3)
        for metric in data:
            average = np.round_(np.sum(
                [data[metric][dataset][error][0] * data[metric][dataset][error][1]
                 for dataset in data[metric] for error in data[metric][dataset]]) / np.sum(
                [data[metric][dataset][error][1] for dataset in data[metric] for error in data[metric][dataset]]), 2)
            row = [metric, average]
            writer.writerow(row)
