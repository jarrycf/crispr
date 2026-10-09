import os
import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             roc_auc_score, average_precision_score,
                             precision_recall_fscore_support)
from scipy import stats
from matplotlib import pyplot as plt


class ModelScore:
    """container of the performance of a model at a given epoch"""
    def __init__(self, best_epoch_indx, binary_f1, macro_f1, accuracy, auc, aupr):
        self.best_epoch_indx = best_epoch_indx
        self.binary_f1 = binary_f1
        self.macro_f1 = macro_f1
        self.accuracy = accuracy
        self.auc = auc
        self.aupr = aupr

    def __repr__(self):
        desc = " best_epoch_indx:{}\n binary_f1:{}\n macro_f1:{}\n accuracy:{}\n" \
               " auc:{}\n aupr:{}\n".format(self.best_epoch_indx, self.binary_f1,
                                            self.macro_f1, self.accuracy,
                                            self.auc, self.aupr)
        return desc


def compute_perfmetrics(true_class, pred_class, prob_scores):
    """classification metrics used to report the per-base editing performance

    Args:
        true_class: array, reference (i.e. true) class labels
        pred_class: array, predicted class labels
        prob_scores: array, predicted probability of the positive (i.e. edited) class

    Returns:
        dict of metrics; ``auc``/``aupr`` are ``numpy.nan`` when only one class is present
    """
    true_class = np.asarray(true_class)
    pred_class = np.asarray(pred_class)
    prob_scores = np.asarray(prob_scores)

    metrics = {'accuracy': accuracy_score(true_class, pred_class),
               'balanced_accuracy': balanced_accuracy_score(true_class, pred_class),
               'macro_f1': f1_score(true_class, pred_class, average='macro'),
               'num_samples': len(true_class)}
    if len(np.unique(true_class)) > 1:
        metrics['binary_f1'] = f1_score(true_class, pred_class, average='binary',
                                        zero_division=0)
        metrics['auc'] = roc_auc_score(true_class, prob_scores)
        metrics['aupr'] = average_precision_score(true_class, prob_scores)
    else:
        # a partition (or an epoch on a tiny subset) can hold one class only
        metrics['binary_f1'] = np.nan
        metrics['auc'] = np.nan
        metrics['aupr'] = np.nan
    return metrics


def perfmetric_report(pred_class, true_class, prob_scores, epoch, outlog=None):
    """compute, log and return the performance of one epoch

    Returns:
        instance of :class:`ModelScore`
    """
    metrics = compute_perfmetrics(true_class, pred_class, prob_scores)
    precision, recall, f1, __ = precision_recall_fscore_support(true_class, pred_class,
                                                                average='macro',
                                                                zero_division=0)
    report = ['-' * 30,
              f'epoch: {epoch}',
              'number of samples: {}'.format(metrics['num_samples']),
              'accuracy: {:.4f}'.format(metrics['accuracy']),
              'balanced accuracy: {:.4f}'.format(metrics['balanced_accuracy']),
              'macro f1: {:.4f}'.format(metrics['macro_f1']),
              'binary f1: {:.4f}'.format(metrics['binary_f1']),
              'macro precision: {:.4f}'.format(precision),
              'macro recall: {:.4f}'.format(recall),
              'AUC: {:.4f}'.format(metrics['auc']),
              'AUPR: {:.4f}'.format(metrics['aupr']),
              '-' * 30]
    report = '\n'.join(report) + '\n'
    if outlog:
        from .utilities import ReaderWriter
        ReaderWriter.write_log(report, outlog)
    else:
        print(report)

    return ModelScore(epoch,
                      metrics['binary_f1'],
                      metrics['macro_f1'],
                      metrics['accuracy'],
                      metrics['auc'],
                      metrics['aupr'])


def compute_correlation(true_score, pred_score):
    """Pearson/Spearman correlation used to report the bystander (haplotype) performance"""
    true_score = np.asarray(true_score, dtype=np.float64)
    pred_score = np.asarray(pred_score, dtype=np.float64)
    res = {'num_samples': len(true_score)}
    # correlation is undefined for constant input
    if len(true_score) < 2 or np.all(true_score == true_score[0]) or np.all(pred_score == pred_score[0]):
        res.update({'pearson_corr': np.nan, 'pearson_pvalue': np.nan,
                    'spearman_corr': np.nan, 'spearman_pvalue': np.nan})
    else:
        pearson = stats.pearsonr(true_score, pred_score)
        spearman = stats.spearmanr(true_score, pred_score)
        res.update({'pearson_corr': pearson[0], 'pearson_pvalue': pearson[1],
                    'spearman_corr': spearman[0], 'spearman_pvalue': spearman[1]})
    res['mae'] = float(np.mean(np.abs(true_score - pred_score)))
    res['rmse'] = float(np.sqrt(np.mean((true_score - pred_score) ** 2)))
    return res


def correlation_report(true_score, pred_score, epoch, outlog=None):
    res = compute_correlation(true_score, pred_score)
    report = ['-' * 30,
              f'epoch: {epoch}',
              'number of samples: {}'.format(res['num_samples']),
              'Pearson correlation: {:.4f}'.format(res['pearson_corr']),
              'Spearman correlation: {:.4f}'.format(res['spearman_corr']),
              'MAE: {:.4f}'.format(res['mae']),
              'RMSE: {:.4f}'.format(res['rmse']),
              '-' * 30]
    report = '\n'.join(report) + '\n'
    if outlog:
        from .utilities import ReaderWriter
        ReaderWriter.write_log(report, outlog)
    else:
        print(report)
    return res


def plot_loss(epoch_loss_avgbatch, wrk_dir, fname='loss', dsettypes=None):
    """plot the average epoch loss of the different dataset types"""
    if dsettypes is None:
        dsettypes = list(epoch_loss_avgbatch.keys())
    plt.figure(figsize=(9, 6))
    for dsettype in dsettypes:
        loss = epoch_loss_avgbatch[dsettype]
        if not len(loss):
            continue
        plt.plot(np.arange(1, len(loss) + 1), loss, label=dsettype)
    plt.xlabel('epoch')
    plt.ylabel('average batch loss')
    plt.legend()
    os.makedirs(wrk_dir, exist_ok=True)
    plt.savefig(os.path.join(wrk_dir, f'{fname}.pdf'), bbox_inches='tight')
    plt.close()
