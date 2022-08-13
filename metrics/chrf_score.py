
# pylint: disable=C0103
from multiprocessing import Pool
import sacrebleu
from .metric import Metric

class ChrfppMetric(Metric):
    def __init__(self, ncorder=6, beta=2, n_workers=24, remove_whitespace=True):
        """
        Chrf++ metric
        Wrapper around sacrebleu: https://github.com/mjpost/sacrebleu

        Args:
                :param ncorder: character n-gram order
                :param beta: beta parameter to balance precision and recall
                :param n_workers: number of processes to use if using multiprocessing

        """
        self.ncorder = ncorder
        self.beta = beta
        self.n_workers = n_workers
        self.remove_whitespace = remove_whitespace

    def evaluate_example(self, summary, reference):
        if not isinstance(reference, list):
            reference = [reference]
        score = sacrebleu.sentence_chrf(summary, reference, char_order=self.ncorder, word_order=0, \
            beta=self.beta, remove_whitespace=self.remove_whitespace)
        score_dict = {"chrf": score.score}
        return score_dict

    def evaluate_batch(self, references, summaries, aggregate=True):
        if aggregate:
            score = sacrebleu.corpus_chrf(summaries, [references], char_order=self.ncorder, \
                word_order=0, beta=self.beta, remove_whitespace=self.remove_whitespace)
            score_dict = {"chrf": score.score}
            return score_dict
        else:
            p = Pool(processes=self.n_workers)
            results = p.starmap(self.evaluate_example, zip(summaries, references))
            p.close()
            return results

    @property
    def supports_multi_ref(self):
        return True
