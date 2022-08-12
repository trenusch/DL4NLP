import numpy as np
import os
import argparse
from collections import defaultdict
import json
import tqdm

def load_data(path):
    hyps, hyps_ad, refs, sources = [], [], [], []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            jline = json.loads(line)
            hyps.append(jline["hyp"])
            hyps_ad.append(jline["claim"])
            refs.append(jline["origin_claim"])
            sources.append(jline["text"])
    return hyps, hyps_ad, refs, sources


def calculate_accuracy_and_kendall(scores, scores_ad):
    num_hit = np.sum([scores[i] > scores_ad[i] for i in range(len(scores))])
    num_miss = np.sum([scores[i] < scores_ad[i] for i in range(len(scores))])
    accuracy = float(num_hit) / float(len(scores))
    kendall = float((num_hit - num_miss)) / float((num_hit + num_miss))  # - 1000 / 1000
    return accuracy, kendall


def print_and_save(metric, metric_hash, dataset, samples, errors, acc, kendall, save=True, output_dir='data/output.txt'):
    cols = ['metric', 'dataset', 'error', "#samples", 'measurement', 'score']
    cols = ','.join(cols) + '\n'

    accs = [str(acc[k]) for k in errors]

    values = ['{}({})'.format(metric, metric_hash), dataset, errors[0], str(samples), 'accuracy'] + accs

    values = ','.join(values) + '\n'
    #if save:
    #    taus = [str(kendall[k]) for k in errors]
    #    values += ','.join(['{}({})'.format(metric, metric_hash), 'ref-based', dataset, 'kendall'] + taus + \
    #                       [str(np.mean(list(kendall.values())))]) + '\n'
    #    print(cols + values)

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

        acc[error], kendall[error] = calculate_accuracy_and_kendall(scores, scores_ad)

        print_and_save(metric, metric_hash, dataset, len(refs), [error], acc, kendall)

    elif scorer == "BertScore":
        from metrics.bert_score import BertScoreMetric
        scorer = BertScoreMetric()
        metric = "BertScore"
        metric_hash = scorer.model_type

        acc, kendall = defaultdict(dict), defaultdict(dict)
        scores = scorer.evaluate_batch(refs, hyps, aggregate=False)
        scores_ad = scorer.evaluate_batch(refs, hyps_ad, aggregate=False)
        acc['p'][error], kendall['p'][error] = calculate_accuracy_and_kendall([s["bert_score_precision"] for s in scores], [s["bert_score_precision"] for s in scores_ad])
        acc['r'][error], kendall['r'][error] = calculate_accuracy_and_kendall([s["bert_score_recall"] for s in scores], [s["bert_score_recall"] for s in scores_ad])
        acc['f1'][error], kendall['f1'][error] = calculate_accuracy_and_kendall([s["bert_score_f1"] for s in scores], [s["bert_score_f1"] for s in scores_ad])

        variants = acc.keys()
        for v in variants:
            v_hash = metric_hash + '_' + v
            print_and_save(metric, v_hash, dataset, len(refs), [error], acc[v], kendall[v])

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
            print_and_save(metric, v_hash, dataset, len(refs), [error], acc[v], kendall[v])

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
            print_and_save(metric, v_hash, dataset, len(refs), [error], acc[v], kendall[v])

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

        print_and_save(metric, metric_hash, dataset, len(refs), [error], acc, kendall)

    elif scorer == "SummaCConv":
        from metrics.summaC_score import SummaCConv
        import nltk
        nltk.download('punkt')
        scorer = SummaCConv()
        metric = "SummaCConv"
        metric_hash = scorer.hash

        acc, kendall = {}, {}
        scores = scorer.score(sources, hyps)['scores']
        scores_ad = scorer.score(sources, hyps_ad)['scores']

        acc[error], kendall[error] = calculate_accuracy_and_kendall(scores, scores_ad)

        print_and_save(metric, metric_hash, dataset, len(refs), [error], acc, kendall)

    elif scorer == "SummaQA":
        from metrics.summaQA_score import SummaQAMetric
        scorer = SummaQAMetric()
        metric = "SummaQA"
        metric_hash = scorer.hash

        acc, kendall = defaultdict(dict), defaultdict(dict)
        scores = scorer.evaluate_batch(sources, hyps, aggregate=False)
        scores_ad = scorer.evaluate_batch(sources, hyps_ad, aggregate=False)

        acc['avg_prob'][error], kendall['avg_prob'][error] = calculate_accuracy_and_kendall([s["summaqa_avg_prob"] for s in scores], [s["summaqa_avg_prob"] for s in scores_ad])
        acc['avg_f1'][error], kendall['avg_f1'][error] = calculate_accuracy_and_kendall([s["summaqa_avg_fscore"] for s in scores], [s["summaqa_avg_fscore"] for s in scores_ad])

        variants = acc.keys()
        for v in variants:
            v_hash = metric_hash + '_' + v
            print_and_save(metric, v_hash, dataset, len(refs), [error], acc[v], kendall[v])

    elif scorer == "Blanc":
        from metrics.blanc_score import BlancMetric
        import nltk
        nltk.download('punkt')
        scorer = BlancMetric()
        metric = "Blanc"
        metric_hash = scorer.hash

        acc, kendall = {}, {}
        scores = scorer.evaluate_batch(sources, hyps, aggregate=False)
        scores_ad = scorer.evaluate_batch(sources, hyps_ad, aggregate=False)

        scores = [b['blanc'] for b in scores]
        scores_ad = [b['blanc'] for b in scores_ad]

        acc[error], kendall[error] = calculate_accuracy_and_kendall(scores, scores_ad)

        print_and_save(metric, metric_hash, dataset, len(refs), [error], acc, kendall)

    elif scorer == "MoverScore":
        from metrics.moverscore_score import get_idf_dict, word_mover_score
        metric = "MoverScore"
        metric_hash = "MoverScore"
        acc, kendall = {}, {}

        idf_dict_hyp = get_idf_dict(hyps)
        idf_dict_hyp_ad = get_idf_dict(hyps_ad)
        idf_dict_ref = get_idf_dict(refs)

        scores = word_mover_score(refs, hyps, idf_dict_ref, idf_dict_hyp, stop_words=[], n_gram=1, remove_subwords=True)
        scores_ad = word_mover_score(refs, hyps_ad, idf_dict_ref, idf_dict_hyp_ad, stop_words=[], n_gram=1, remove_subwords=True)

        acc[error], kendall[error] = calculate_accuracy_and_kendall(scores, scores_ad)

        print_and_save(metric, metric_hash, dataset, len(refs), [error], acc, kendall)

    elif scorer == "MoverScore2":
        from metrics.moverscore_v2_score import get_idf_dict, word_mover_score
        metric = "MoverScore2"
        metric_hash = "moverscorev2"
        acc, kendall = {}, {}

        idf_dict_hyp = get_idf_dict(hyps)
        idf_dict_hyp_ad = get_idf_dict(hyps_ad)
        idf_dict_ref = get_idf_dict(refs)

        scores = word_mover_score(refs, hyps, idf_dict_ref, idf_dict_hyp, stop_words=[], n_gram=1, remove_subwords=True)
        scores_ad = word_mover_score(refs, hyps_ad, idf_dict_ref, idf_dict_hyp_ad, stop_words=[], n_gram=1, remove_subwords=True)

        acc[error], kendall[error] = calculate_accuracy_and_kendall(scores, scores_ad)

        print_and_save(metric, metric_hash, dataset, len(refs), [error], acc, kendall)

    else:
        raise NotImplementedError


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--metric', type=str, default=None)
    parser.add_argument('--path', type=str, default="data/adversarial_data/")
    args = parser.parse_args()
    scorer = args.metric
    path = args.path

    for file in tqdm.tqdm(os.listdir(path)):
        hyps, hyps_ad, refs, sources = load_data(path + file)
        error = file.split("_")[1][:-6]
        dataset = file.split("_")[0]
        evaluate(scorer, error, dataset, refs, hyps, hyps_ad, sources)
