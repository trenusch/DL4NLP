import pandas as pd
import os
import json
from scipy.stats import pearsonr, kendalltau, spearmanr
from collections import defaultdict
import numpy as np


def find_source_document(doc_id):
    if "dm" not in doc_id:
        data_dir = "data/cnn/stories/"
        path = os.path.join(data_dir, doc_id.split('-')[-1] + '.story')
    else:
        data_dir = "data/dailymail/stories/"
        path = os.path.join(data_dir, doc_id.split('-')[-1] + '.story')
    with open(path, 'r') as f:
        doc = f.read()
    doc = doc.split('@highlight')[0].replace('\n', ' ').replace('(CNN)', '')
    # print(doc)
    return doc


def load_data_summ(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        json_strings = f.readlines()

    lines = [json.loads(l) for l in json_strings]
    data = defaultdict(list)
    for l in lines:
        data['system'].append(l['model_id'])
        data['id'].append(l['id'])
        doc = find_source_document(l['id'])
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
    #first_raw = "metric,model,dataset,setup,aggregation,correlation,annotator,coherence,consistency,fluency,relevance,average\n"
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
            #s += f"{metric_name},{self.args.model},{self.args.dataset},{'ref-free' if self.args.use_article else 'ref-based'}," \
            #     f"{self.args.aggregate},{cor},{anno},{coh},{con},{flu},{rel},{avg}\n"
        print(first_raw + s)
        mode = 'w' if not os.path.exists(output_path) else 'a'
        final_string = first_raw + s if not os.path.exists(output_path) else s
        with open(output_path, mode) as f:
            f.write(final_string)

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
        ref_based = True

    elif metric == "BertScore":
        from metrics.bert_score import BertScoreMetric
        scorer = BertScoreMetric()
        metric_hash = scorer.model_type
        ref_based = True

    else:
        raise NotImplementedError

    if ref_based:
        return scorer.evaluate_batch(refs, hyps)
    else:
        return scorer.evaluate_batch(sources, hyps)

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


if __name__ == "__main__":
    path = "data/model_annotations.aligned.jsonl"
    data = load_data_summ(path)

    scorer = "BartScore"
    score(scorer, data)

