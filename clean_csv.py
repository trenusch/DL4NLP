import csv
import numpy as np

with open("data/human_corr_results.csv", 'r', encoding='utf-8') as f:
    next(f)
    lines = f.readlines()

first_row = ["Metric", "Coherence", "Consistency", "Fluency", "Relevance", "Average"]


with open("data/human_corr_results_2.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(first_row)
    for line in lines:
        split = line.split(",")
        if split[1] == "kendall":
            row = [split[0]]
            for value in split[3:]:
                row.append(np.round_(float(value), 4))
            writer.writerow(row)