import argparse
from evaluate_metrics import load_data, print_and_save, calculate_accuracy_and_kendall
from evaluate_correlation import load_data_summ, evaluate
import tqdm
import os
import datasets
import numpy as np


def evaluate_mix(nli_scores, nli_scores_ad, metric_scores, metric_scores_ad, metric, dataset, error):
    for i in range(11):
        acc, kendall = {}, {}
        weight = np.round_(i * 0.1, 1)
        compl_weight = np.round_(1 - weight, 1)
        metric_hash = "{}_nli_{}_{}".format(weight, compl_weight, metric)
        scores = [0.5 * (weight * nli + compl_weight * m) for nli, m in zip(nli_scores[2], metric_scores)]
        scores_ad = [0.5 * (weight * nli + compl_weight * m) for nli, m in
                     zip(nli_scores_ad[2], metric_scores_ad)]
        acc[error], kendall[error] = calculate_accuracy_and_kendall(scores, scores_ad)

        print_and_save(metric, metric_hash, dataset, len(scores), [error], acc, kendall,
                       output_dir="data/mix_output.txt")


def score_adv(scorer, error, dataset, refs, hyps, hyps_ad, sources, nli_scores, nli_scores_ad):
    if scorer == "NLI_BartScore":
        metric = scorer
        from metrics.bart_score import BARTScorer
        scorer = BARTScorer()

        scores = scorer.evaluate_batch(refs, hyps)
        scores_ad = scorer.evaluate_batch(refs, hyps_ad)

        evaluate_mix(nli_scores, nli_scores_ad, scores, scores_ad, metric, dataset, error)

    elif scorer == "NLI_BertScore":
        metric = scorer
        from metrics.bert_score import BertScoreMetric
        scorer = BertScoreMetric()

        scores = scorer.evaluate_batch(refs, hyps, aggregate=False)
        scores_ad = scorer.evaluate_batch(refs, hyps_ad, aggregate=False)

        variants = ["bert_score_precision", "bert_score_recall", "bert_score_f1"]

        for v in variants:
            evaluate_mix(nli_scores, nli_scores_ad, scores, scores_ad, metric + v, dataset, error)

    elif scorer == "NLI_MoverScore2":
        metric = scorer
        from metrics.moverscore_v2_score import get_idf_dict, word_mover_score

        idf_dict_hyp = get_idf_dict(hyps)
        idf_dict_hyp_ad = get_idf_dict(hyps_ad)
        idf_dict_ref = get_idf_dict(refs)

        scores = word_mover_score(refs, hyps, idf_dict_ref, idf_dict_hyp, stop_words=[], n_gram=1, remove_subwords=True)
        scores_ad = word_mover_score(refs, hyps_ad, idf_dict_ref, idf_dict_hyp_ad, stop_words=[], n_gram=1,
                                     remove_subwords=True)

        evaluate_mix(nli_scores, nli_scores_ad, scores, scores_ad, metric, dataset, error)

    elif scorer == "NLI_CHRF":
        metric = scorer
        from metrics.chrf_score import ChrfppMetric

        chrf_scorer = ChrfppMetric()

        chrf_scores = chrf_scorer.evaluate_batch(refs, hyps, aggregate=False)
        chrf_scores_ad = chrf_scorer.evaluate_batch(refs, hyps_ad, aggregate=False)
        chrf_scores = [s['chrf'] for s in chrf_scores]
        chrf_scores_ad = [s['chrf'] for s in chrf_scores_ad]

        evaluate_mix(nli_scores, nli_scores_ad, chrf_scores, chrf_scores_ad, metric, dataset, error)

    elif scorer == "NLI_Meteor":
        metric = scorer
        from metrics.meteor_score import MeteorMetric
        meteor_scorer = MeteorMetric()

        meteor_scores = meteor_scorer.evaluate_batch(refs, hyps, aggregate=False)
        meteor_scores_ad = meteor_scorer.evaluate_batch(refs, hyps_ad, aggregate=False)
        meteor_scores = [b['meteor'] for b in meteor_scores]
        meteor_scores_ad = [b['meteor'] for b in meteor_scores_ad]

        evaluate_mix(nli_scores, nli_scores_ad, meteor_scores, meteor_scores_ad, metric, dataset, error)

    else:
        raise NotImplementedError


def evaluate_corr_mix(nli_scores, metric_scores, metric, data):
    for i in range(11):
        weight = np.round_(i * 0.1, 1)
        compl_weight = np.round_(1 - weight, 1)
        metric_hash = "{}_nli_{}_{}".format(weight, compl_weight, metric)
        scores = [0.5 * (weight * nli + compl_weight * m) for nli, m in zip(nli_scores[2], metric_scores)]
        evaluate(scores, data, metric, metric_hash, output_path="data/mix_human_correlation.csv")


def score_corr(scorer, data, nli_scores):
    hyps, refs, docs = [], [], []
    for h, rs, d in zip(data['hyp'], data['refs'], data['doc']):
        assert len(rs) == 11
        hyps += [h] * len(rs)
        docs += [d] * len(rs)
        refs += rs

    if scorer == "NLI_BartScore":
        metric = scorer
        from metrics.bart_score import BARTScorer
        scorer = BARTScorer()

        scores = scorer.evaluate_batch(refs, hyps)

        evaluate_corr_mix(nli_scores, scores, metric, data)

    elif scorer == "NLI_BertScore":
        metric = scorer
        from metrics.bert_score import BertScoreMetric
        scorer = BertScoreMetric()

        scores = scorer.evaluate_batch(refs, hyps)

        variants = ["bert_score_precision", "bert_score_recall", "bert_score_f1"]

        for v in variants:
            evaluate_corr_mix(nli_scores, scores, metric + v, data)

    elif scorer == "NLI_MoverScore2":
        metric = scorer
        from metrics.moverscore_v2_score import get_idf_dict, word_mover_score

        idf_dict_hyp = get_idf_dict(hyps)
        idf_dict_ref = get_idf_dict(refs)

        scores = word_mover_score(refs, hyps, idf_dict_ref, idf_dict_hyp, stop_words=[], n_gram=1, remove_subwords=True)

        evaluate_corr_mix(nli_scores, scores, metric, data)

    elif scorer == "NLI_CHRF":
        metric = scorer
        from metrics.chrf_score import ChrfppMetric
        chrf_scorer = ChrfppMetric()

        chrf_scores = chrf_scorer.evaluate_batch(refs, hyps, aggregate=False)
        chrf_scores = [s['chrf'] for s in chrf_scores]

        evaluate_corr_mix(nli_scores, chrf_scores, metric, data)

    elif scorer == "NLI_Meteor":
        metric = scorer
        from metrics.meteor_score import MeteorMetric
        meteor_scorer = MeteorMetric()

        meteor_scores = meteor_scorer.evaluate_batch(refs, hyps, aggregate=False)

        evaluate_corr_mix(nli_scores, meteor_scores, metric, data)

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

    hyps, refs, docs = [], [], []
    for h, rs, d in zip(data['hyp'], data['refs'], data['doc']):
        assert len(rs) == 11
        hyps += [h] * len(rs)
        docs += [d] * len(rs)
        refs += rs

    from metrics.nli1_score import NLI1Scorer

    nli_scorer = NLI1Scorer()
    nli_corr_scores = nli_scorer.evaluate_batch(refs, hyps)

    for metric in metrics:
        scorer = "NLI_" + metric
        for file in tqdm.tqdm(os.listdir(adv_path)):
            hyps, hyps_ad, refs, sources = load_data(adv_path + file)
            error = file.split("_")[1][:-6]
            dataset = file.split("_")[0]

            nli_scores = nli_scorer.evaluate_batch(refs, hyps)
            nli_scores_ad = nli_scorer.evaluate_batch(refs, hyps_ad)

            score_adv(scorer, error, dataset, refs, hyps, hyps_ad, sources, nli_scores, nli_scores_ad)

        score_corr(scorer, data, nli_corr_scores)
