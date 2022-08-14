import argparse
from evaluate_metrics import load_data, print_and_save, calculate_accuracy_and_kendall
from evaluate_correlation import load_data_summ, evaluate
import tqdm
import os
import datasets


def score_adv(scorer, error, dataset, refs, hyps, hyps_ad, sources):
    if scorer == "NLI_BartScore":
        pass

    elif scorer == "NLI_BertScore":
        pass

    elif scorer == "NLI_MoverScore2":
        pass

    elif scorer == "NLI_CHRF":
        from metrics.nli1_score import NLI1Scorer
        from metrics.chrf_score import ChrfppMetric

        metric = "NLI_CHRF"

        nli_scorer = NLI1Scorer()
        chrf_scorer = ChrfppMetric()

        nli_scores = nli_scorer.evaluate_batch(refs, hyps)
        nli_scores_ad = nli_scorer.evaluate_batch(refs, hyps_ad)
        chrf_scores = chrf_scorer.evaluate_batch(refs, hyps, aggregate=False)
        chrf_scores_ad = chrf_scorer.evaluate_batch(refs, hyps_ad, aggregate=False)
        chrf_scores = [s['chrf'] for s in chrf_scores]
        chrf_scores_ad = [s['chrf'] for s in chrf_scores_ad]

        for i in range(11):
            acc, kendall = {}, {}
            weight = i * 0.1
            metric_hash = "{}_nli_{}_chrf".format(weight, 1 - weight)
            scores = [0.5 * (weight * nli + (1 - weight) * chrf) for nli, chrf in zip(nli_scores[2], chrf_scores)]
            scores_ad = [0.5 * (weight * nli + (1 - weight) * chrf) for nli, chrf in
                         zip(nli_scores_ad[2], chrf_scores_ad)]
            acc[error], kendall[error] = calculate_accuracy_and_kendall(scores, scores_ad)

            print_and_save(metric, metric_hash, dataset, len(refs), [error], acc, kendall,
                           output_dir="data/mix_output.txt")

    elif scorer == "NLI_Meteor":
        pass

    else:
        raise NotImplementedError


def score_corr():
    if scorer == "NLI_BartScore":
        pass

    elif scorer == "NLI_BertScore":
        pass

    elif scorer == "NLI_MoverScore2":
        pass

    elif metric == "NLI_CHRF":
        from metrics.nli1_score import NLI1Scorer
        from metrics.chrf_score import ChrfppMetric

        nli_scorer = NLI1Scorer()
        chrf_scorer = ChrfppMetric()

        nli_scores = nli_scorer.evaluate_batch(refs, hyps)
        chrf_scores = chrf_scorer.evaluate_batch(refs, hyps, aggregate=False)
        chrf_scores = [s['chrf'] for s in chrf_scores]

        for i in range(11):
            weight = i * 0.1
            metric_hash = "{}_nli_{}_chrf".format(weight, 1 - weight)
            scores = [0.5 * (weight * nli + (1 - weight) * chrf) for nli, chrf in zip(nli_scores[2], chrf_scores)]
            evaluate(scores, data, metric, metric_hash, output_path="data/mix_human_correlation.csv")

    elif scorer == "NLI_Meteor":
        pass

    else:
        raise NotImplementedError


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--adv_path', type=str, default="data/adversarial_data/")
    parser.add_argument('--corr_path', type=str, default="data/model_annotations.aligned.scored.jsonl")
    args = parser.parse_args()
    adv_path = args.adv_path
    corr_path = args.corr_path

    metrics = ["BartScore", "BertScore", "MoverScore2", "CHRF", "Meteor"]
    dataset = datasets.load_dataset("cnn_dailymail", '3.0.0')
    data = load_data_summ(corr_path, dataset)

    for metric in metrics:
        scorer = "NLI_" + metric
        for file in tqdm.tqdm(os.listdir(adv_path)):
            hyps, hyps_ad, refs, sources = load_data(adv_path + file)
            error = file.split("_")[1][:-6]
            dataset = file.split("_")[0]
            score_adv(scorer, error, dataset, refs, hyps, hyps_ad, sources)
        score_corr(scorer, data)
