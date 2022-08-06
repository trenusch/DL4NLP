import pandas as pd
import os
import json
from scipy.stats import pearsonr, kendalltau, spearmanr
from collections import defaultdict
import numpy as np
import argparse
import datasets


def load_data_summ(data_path, dataset):
    with open(data_path, 'r', encoding='utf-8') as f:
        json_strings = f.readlines()

    lines = [json.loads(l) for l in json_strings]
    data = defaultdict(list)
    for l in lines:
        data['system'].append(l['model_id'])
        data['id'].append(l['id'])
        data_id = l['id'].split("-")[2]
        doc = dataset['test'][dataset['test'][:]['id'].index(data_id)]['article']
        data['doc'].append(doc)
        data['hyp'].append(l['decoded'])
        data['refs'].append(l['references'])

        for anno in ['expert', 'turker']:
            if anno == 'turker':
                continue
            for criterion in ['coherence', 'consistency', 'fluency', 'relevance']:
                data['{}_{}'.format(anno, criterion)].append(np.mean([i[criterion] for i in l[anno + "_annotations"]]))
    data = pd.DataFrame.from_dict(data).sort_values(by=['system', 'id'])
    return data


def print_and_save(corr_dict, metric_name):
    output_path = os.path.join("data/human_corr_results.csv")
    # first_raw = "metric,model,dataset,setup,aggregation,correlation,annotator,coherence,consistency,fluency,relevance,average\n"
    first_raw = "metric, correlation, annotator, coherence, consistency, fluency, relevance, average"
    for anno in ['expert', 'turker']:
        if anno == 'turker':
            continue
        s = ''
        for cor in corr_dict[anno].keys():
            coh = corr_dict[anno][cor]['coherence']
            con = corr_dict[anno][cor]['consistency']
            flu = corr_dict[anno][cor]['fluency']
            rel = corr_dict[anno][cor]['relevance']
            avg = np.mean([coh, con, flu, rel])
            s += f"{metric_name},{cor},{anno},{coh},{con},{flu},{rel},{avg}\n"
            # s += f"{metric_name},{self.args.model},{self.args.dataset},{'ref-free' if self.args.use_article else 'ref-based'}," \
            #     f"{self.args.aggregate},{cor},{anno},{coh},{con},{flu},{rel},{avg}\n"
        print(first_raw + s)
        mode = 'w' if not os.path.exists(output_path) else 'a'
        final_string = first_raw + s if not os.path.exists(output_path) else s
        with open(output_path, mode) as f:
            f.write(final_string)


def evaluate(scores, data, metric, metric_hash):
    scores = [np.max(scores[i * 11: i * 11 + 11]) for i in range(int(len(scores) / 11))]
    data['metric_scores'] = scores

    data = data.groupby('system').mean()
    cols = [col for col in data.columns if 'metric' in col]
    for col in cols:
        suffix = '' if len(col.split('_')) == 2 else '_' + col[-1]
        metric_name = metric + '_' + metric_hash + suffix
        corr_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        metric_scores = list(data[col])
        for anno in ['expert', 'turker']:
            if anno == 'turker':
                continue
            for c in ['coherence', 'consistency', 'fluency', 'relevance']:
                human_scores = list(data[anno + '_' + c])
                corr_dict[anno]['pearson'][c] = pearsonr(human_scores, metric_scores)[0]
                corr_dict[anno]['kendall'][c] = kendalltau(human_scores, metric_scores)[0]
        print_and_save(corr_dict, metric_name)


def score(metric, data):
    hyps, refs, docs = [], [], []
    for h, rs, d in zip(data['hyp'], data['refs'], data['doc']):
        assert len(rs) == 11
        hyps += [h] * len(rs)
        docs += [d] * len(rs)
        refs += rs

    assert len(refs) == len(hyps) == len(docs)

    # initialize metric
    if metric == "BartScore":
        from metrics.bart_score import BARTScorer
        scorer = BARTScorer(device='cpu')
        metric_hash = scorer.hash

    elif metric == "BertScore":
        from metrics.bert_score import BertScoreMetric
        scorer = BertScoreMetric()
        metric_hash = scorer.model_type

        scores = scorer.evaluate_batch(refs, hyps, aggregate=False)

        variants = ["bert_score_precision", "bert_score_recall", "bert_score_f1"]
        for v in variants:
            evaluate([s[v] for s in scores], data, v, metric_hash)

    elif metric == "NLI1Score":
        from metrics.nli1_score import NLI1Scorer
        scorer = NLI1Scorer()
        metric_hash = scorer.hash

        scores = scorer.evaluate_batch(refs, hyps)

        variants = ["c", "n", "e"]
        for score, v in zip(scores, variants):
            evaluate(score, data, metric + "_" + v, metric_hash)

    elif metric == "NLI2Score":
        from metrics.nli2_score import NLI2Scorer
        scorer = NLI2Scorer()
        metric_hash = scorer.hash

        scores = scorer.evaluate_batch(refs, hyps)

        variants = ["c", "n", "e"]
        for score, v in zip(scores, variants):
            evaluate(score, data, metric + "_" + v, metric_hash)

    elif metric == "SummaCZS":
        from metrics.summaC_score import SummaCZS
        import nltk
        nltk.download('punkt')
        scorer = SummaCZS(granularity="sentence", model_name="vitc")
        metric_hash = scorer.hash

        scores = scorer.evaluate_batch(docs, hyps)['scores']
        evaluate(scores, data, metric, metric_hash)

    elif metric == "SummaCConv":
        from metrics.summaC_score import SummaCConv
        import nltk
        nltk.download('punkt')
        scorer = SummaCConv()
        metric_hash = scorer.hash
        scores = scorer.score(docs, hyps)['scores']
        evaluate(scores, data, metric, metric_hash)

    elif metric == "SummaQA":
        from metrics.summaQA_score import SummaQAMetric
        scorer = SummaQAMetric()
        metric = "SummaQA"
        metric_hash = scorer.hash

        scores = scorer.evaluate_batch(docs, hyps, aggregate=False)

        variants = ["avg_prob", "avg_f1"]
        for v in variants:
            evaluate(scores, data, metric, metric_hash + "_" + v)

    elif metric == "Blanc":
        from metrics.blanc_score import BlancMetric
        scorer = BlancMetric()
        metric_hash = scorer.hash

        scores = scorer.evaluate_batch(docs, hyps, aggregate=False)
        evaluate(scores, data, metric, metric_hash)

    elif metric == "MoverScore":
        from metrics.moverscore_score import get_idf_dict, word_mover_score
        metric = "MoverScore"
        metric_hash = "MoverScore"

        idf_dict_hyp = get_idf_dict(hyps)
        idf_dict_ref = get_idf_dict(refs)

        scores = word_mover_score(refs, hyps, idf_dict_ref, idf_dict_hyp, stop_words=[], n_gram=1, remove_subwords=True)
        evaluate(scores, data, metric, metric_hash)

    elif metric == "MoverScore2":
        from metrics.moverscore_v2_score import get_idf_dict, word_mover_score
        metric = "MoverScore2"
        metric_hash = "MoverScore2"

        idf_dict_hyp = get_idf_dict(hyps)
        idf_dict_ref = get_idf_dict(refs)

        scores = word_mover_score(refs, hyps, idf_dict_ref, idf_dict_hyp, stop_words=[], n_gram=1, remove_subwords=True)
        evaluate(scores, data, metric, metric_hash)

    else:
        raise NotImplementedError


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--metric', type=str, default=None)
    parser.add_argument('--path', type=str, default=None)
    args = parser.parse_args()
    scorer = args.metric
    path = args.path

    # scorer = "SummaCConv"
    # path = "data/model_annotations.aligned.jsonl"
    dataset = datasets.load_dataset("cnn_dailymail", '3.0.0')
    data = load_data_summ(path, dataset)
    score(scorer, data)
