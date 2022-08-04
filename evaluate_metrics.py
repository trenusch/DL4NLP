import numpy as np
import os
import argparse
from collections import defaultdict
import json

def load_data(path):
    hyps, hyps_ad, refs, sources = [], [], [], []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            jline = json.loads(line)
            hyps.append(jline["decoded"])
            hyps_ad.append(jline["adversarial"])
            refs.append(jline["reference"])
            sources.append(jline["text"])
    return hyps, hyps_ad, refs, sources


def calculate_accuracy_and_kendall(scores, scores_ad):
    num_hit = np.sum([scores[i] > scores_ad[i] for i in range(len(scores))])
    num_miss = np.sum([scores[i] < scores_ad[i] for i in range(len(scores))])
    accuracy = float(num_hit) / float(len(scores))
    kendall = float((num_hit - num_miss)) / float((num_hit + num_miss))  # - 1000 / 1000
    return accuracy, kendall


def print_and_save(metric, metric_hash, dataset, errors, acc, kendall, save=True, output_dir='data/output.txt'):
    cols = ['metric', 'setup', 'dataset', 'measurement'] + errors + ['average']
    cols = ','.join(cols) + '\n'

    accs = [str(acc[k]) for k in errors]

    values = ['{}({})'.format(metric, metric_hash), 'ref-based', dataset, 'accuracy'] + accs + \
             [str(np.mean(list(acc.values())))]

    values = ','.join(values) + '\n'
    if save:
        taus = [str(kendall[k]) for k in errors]
        values += ','.join(['{}({})'.format(metric, metric_hash), 'ref-based', dataset, 'kendall'] + taus + \
                           [str(np.mean(list(kendall.values())))]) + '\n'
        print(cols + values)

    if save:
        if not os.path.exists(output_dir):
            with open(output_dir, 'w') as f:
                f.write(cols + values)
        else:
            with open(output_dir, 'a') as f:
                f.write(values)


def evaluate(scorer, error, dataset, refs, hyps, hyps_ad, sources):
    # initialize metric
    if scorer == "BartScore":
        from metrics.bart_score import BARTScorer
        scorer = BARTScorer(device='cpu')
        metric = "BARTScore"
        metric_hash = scorer.hash

        acc, kendall = {}, {}
        scores = scorer.evaluate_batch(refs, hyps)
        scores_ad = scorer.evaluate_batch(refs, hyps_ad)


    elif scorer == "BertScore":
        from metrics.bert_score import BertScoreMetric
        scorer = BertScoreMetric()
        metric = "BertScore"
        metric_hash = scorer.model_type

        acc, kendall = defaultdict(dict), defaultdict(dict)
        scores = scorer.evaluate_batch(refs, hyps)
        scores_ad = scorer.evaluate_batch(refs, hyps_ad)
        acc['p'][error], kendall['p'][error] = calculate_accuracy_and_kendall([s["bert_score_precision"] for s in scores], [s["bert_score_precision"] for s in scores_ad])
        acc['r'][error], kendall['r'][error] = calculate_accuracy_and_kendall([s["bert_score_recall"] for s in scores], [s["bert_score_recall"] for s in scores_ad])
        acc['f1'][error], kendall['f1'][error] = calculate_accuracy_and_kendall([s["bert_score_f1"] for s in scores], [s["bert_score_f1"] for s in scores_ad])

        variants = acc.keys()
        for v in variants:
            v_hash = metric_hash + '_' + v
            print_and_save(metric, v_hash, dataset, [error], acc[v], kendall[v])

    elif scorer == "NLI1Score":
        from metrics.nli1_score import NLI1Scorer
        scorer = NLI1Scorer()
        metric = "NLI1Score"
        metric_hash = scorer.hash

        acc, kendall = defaultdict(dict), defaultdict(dict)
        scores = scorer.evaluate_batch(refs, hyps)
        scores_ad = scorer.evaluate_batch(refs, hyps_ad)

        acc['c'][error], kendall['c'][error] = calculate_accuracy_and_kendall(scores[0], scores_ad[0])
        acc['n'][error], kendall['n'][error] = calculate_accuracy_and_kendall(scores[1], scores_ad[1])
        acc['e'][error], kendall['e'][error] = calculate_accuracy_and_kendall(scores[2], scores_ad[2])

        variants = acc.keys()
        for v in variants:
            v_hash = metric_hash + '_' + v
            print_and_save(metric, v_hash, dataset, [error], acc[v], kendall[v])

    elif scorer == "NLI2Score":
        from metrics.nli2_score import NLI2Scorer
        scorer = NLI2Scorer()
        metric = "NLI2Score"
        metric_hash = scorer.hash

        acc, kendall = defaultdict(dict), defaultdict(dict)
        scores = scorer.evaluate_batch(refs, hyps)
        scores_ad = scorer.evaluate_batch(refs, hyps_ad)

        acc['c'][error], kendall['c'][error] = calculate_accuracy_and_kendall(scores[0], scores_ad[0])
        acc['n'][error], kendall['n'][error] = calculate_accuracy_and_kendall(scores[1], scores_ad[1])
        acc['e'][error], kendall['e'][error] = calculate_accuracy_and_kendall(scores[2], scores_ad[2])

        variants = acc.keys()
        for v in variants:
            v_hash = metric_hash + '_' + v
            print_and_save(metric, v_hash, dataset, [error], acc[v], kendall[v])

    elif scorer == "SummaCZS":
        from metrics.summaC_score import SummaCZS
        import nltk
        nltk.download('punkt')
        scorer = SummaCZS(granularity="sentence", model_name="vitc")
        metric = "SummaCZS"
        metric_hash = scorer.hash

        acc, kendall = {}, {}
        scores = scorer.evaluate_batch(sources, hyps)['scores']
        scores_ad = scorer.evaluate_batch(sources, hyps_ad)['scores']

        acc[error], kendall[error] = calculate_accuracy_and_kendall(scores, scores_ad)

        print_and_save(metric, metric_hash, dataset, [error], acc, kendall)

    elif scorer == "SummaCConv":
        from metrics.summaC_score import SummaCConv
        import nltk
        nltk.download('punkt')
        scorer = SummaCConv()
        metric = "SummaCConv"
        metric_hash = scorer.hash

        scores = scorer.score(sources, hyps)['scores']
        scores_ad = scorer.score(sources, hyps_ad)['scores']

        acc[error], kendall[error] = calculate_accuracy_and_kendall(scores, scores_ad)

        print_and_save(metric, metric_hash, dataset, [error], acc, kendall)

    elif scorer == "MoverScore":
        from metrics.nli2_score import NLI2Scorer
        scorer = NLI2Scorer()
        metric = "MoverScore"
        metric_hash = scorer.hash
        ref_based = True

    else:
        raise NotImplementedError


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--metric', type=str, default=None)
    parser.add_argument('--path', type=str, default=None)
    args = parser.parse_args()
    scorer = args.metric
    path = args.path

    scorer = "NLI1Score"
    path = "data/adjective_antonym_dataset.jsonl"

    hyps, hyps_ad, refs, sources = load_data(path)
    error = path.split("/")[1].split(".")[0]
    evaluate(scorer, error, "cnndm", refs[:3], hyps[:3], hyps_ad[:3], sources[:3])