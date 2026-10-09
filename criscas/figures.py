"""figure panels of the `BE-DICT (perbase)` model

Every function writes one panel (pdf + png) into `fig_dir` and returns the data frame it
plotted, so that the underlying numbers can be inspected or re-used.

Style conventions used across the panels:
    - one fixed, colorblind-safe color per base editor (never cycled, never reassigned)
    - a single-hue sequential colormap for magnitudes (same `YlOrRd` as the attention
      heatmaps of :meth:`criscas.predict_model.BEDICT_CriscasModel.highlight_attn_per_seq`)
    - recessive grid and spines, legend whenever more than one series is drawn, and never
      two y-scales in one axes
"""

import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve
from scipy import stats

from .evaluation import compute_perfmetrics


# fixed order and colors of the base editors reported in the paper
EDITOR_ORDER = ['ABEmax', 'ABE8e', 'BE4max', 'Target-AID']
EDITOR_COLORS = {'ABEmax': '#0072B2',
                 'ABE8e': '#D55E00',
                 'BE4max': '#009E73',
                 'Target-AID': '#785EF0'}
# single hue, light -> dark, for magnitude (attention scores, editing efficiency)
SEQUENTIAL_CMAP = 'YlOrRd'
NEUTRAL_INK = '#333333'
MUTED_INK = '#767676'


def get_editor_color(editor, default='#0072B2'):
    return EDITOR_COLORS.get(editor, default)


def order_editors(editors):
    """sort the editors in the fixed reporting order, unknown ones last (alphabetically)"""
    editors = list(editors)
    known = [e for e in EDITOR_ORDER if e in editors]
    unknown = sorted(e for e in editors if e not in EDITOR_ORDER)
    return known + unknown


def _style_axes(ax, grid_axis='both'):
    """recessive grid and spines so that the data marks dominate"""
    ax.grid(True, axis=grid_axis, color='#e6e6e6', linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color('#cccccc')
    ax.tick_params(colors=MUTED_INK, labelsize=9)
    ax.xaxis.label.set_color(NEUTRAL_INK)
    ax.yaxis.label.set_color(NEUTRAL_INK)
    return ax


def _savefig(fig, fig_dir, fname):
    if not fig_dir:
        return None
    os.makedirs(fig_dir, exist_ok=True)
    paths = []
    for ext in ('pdf', 'png'):
        pth = os.path.join(fig_dir, f'{fname}.{ext}')
        fig.savefig(pth, bbox_inches='tight', dpi=300)
        paths.append(pth)
    plt.close(fig)
    return paths


def plot_roc_curves(editor_pred_map, fig_dir=None, fname='perbase_roc_curves',
                    true_col='true_class', score_col='prob_score_class1'):
    """ROC curve of the per-base editing classification, one line per base editor

    Args:
        editor_pred_map: dict, ``{editor: predictions_df}`` where the data frame holds the
                         reference class and the predicted probability of the edited class
        fig_dir: string or None, directory where the panel is written
        fname: string, file name (without extension)
        true_col: string, column of the reference class
        score_col: string, column of the predicted probability of the positive class

    Returns:
        pandas.DataFrame with the AUC of every editor
    """
    fig, ax = plt.subplots(figsize=(5.2, 5))
    rows = []
    for editor in order_editors(editor_pred_map.keys()):
        df = editor_pred_map[editor]
        y_true = df[true_col].values
        y_score = df[score_col].values
        if len(np.unique(y_true)) < 2:
            print(f'{editor}: only one class present, ROC curve is skipped')
            continue
        fpr, tpr, __ = roc_curve(y_true, y_score)
        auc = compute_perfmetrics(y_true, (y_score >= 0.5).astype(int), y_score)['auc']
        ax.plot(fpr, tpr, color=get_editor_color(editor), linewidth=2,
                label=f'{editor} (AUC={auc:.3f})')
        rows.append({'model_name': editor, 'auc': auc, 'num_samples': len(y_true)})
    ax.plot([0, 1], [0, 1], color='#bbbbbb', linewidth=1, linestyle='--', zorder=1,
            label='random')
    ax.set(xlabel='false positive rate', ylabel='true positive rate',
           xlim=(-0.02, 1.02), ylim=(-0.02, 1.02))
    ax.set_title('per-base editing: ROC', color=NEUTRAL_INK, fontsize=11)
    _style_axes(ax)
    ax.legend(frameon=False, fontsize=9, loc='lower right')
    _savefig(fig, fig_dir, fname)
    return pd.DataFrame(rows)


def plot_pr_curves(editor_pred_map, fig_dir=None, fname='perbase_pr_curves',
                   true_col='true_class', score_col='prob_score_class1'):
    """precision-recall curve of the per-base editing classification

    Returns:
        pandas.DataFrame with the AUPR of every editor
    """
    fig, ax = plt.subplots(figsize=(5.2, 5))
    rows = []
    for editor in order_editors(editor_pred_map.keys()):
        df = editor_pred_map[editor]
        y_true = df[true_col].values
        y_score = df[score_col].values
        if len(np.unique(y_true)) < 2:
            print(f'{editor}: only one class present, PR curve is skipped')
            continue
        precision, recall, __ = precision_recall_curve(y_true, y_score)
        aupr = compute_perfmetrics(y_true, (y_score >= 0.5).astype(int), y_score)['aupr']
        ax.plot(recall, precision, color=get_editor_color(editor), linewidth=2,
                label=f'{editor} (AUPR={aupr:.3f})')
        rows.append({'model_name': editor, 'aupr': aupr,
                     'baseline': float(np.mean(y_true)), 'num_samples': len(y_true)})
    ax.set(xlabel='recall', ylabel='precision', xlim=(-0.02, 1.02), ylim=(-0.02, 1.02))
    ax.set_title('per-base editing: precision-recall', color=NEUTRAL_INK, fontsize=11)
    _style_axes(ax)
    ax.legend(frameon=False, fontsize=9, loc='lower left')
    _savefig(fig, fig_dir, fname)
    return pd.DataFrame(rows)


def plot_performance_bars(scores_df, fig_dir=None, fname='perbase_performance',
                          metrics=('auc', 'aupr'), editor_col='model_name'):
    """performance of the 5 runs per base editor, as mean +- standard deviation

    Args:
        scores_df: pandas.DataFrame with one row per (editor, run) and one column per
                   metric (i.e. the output of
                   :func:`criscas.train_val.summarize_scores` with an added editor column)
        metrics: sequence of strings, metrics to plot (one panel each)

    Returns:
        pandas.DataFrame of the aggregated statistics
    """
    metrics = [m for m in metrics if m in scores_df.columns]
    assert metrics, 'none of the requested metrics is present in the passed data frame'
    editors = order_editors(scores_df[editor_col].unique())
    agg = scores_df.groupby(editor_col)[metrics].agg(['mean', 'std', 'count'])

    fig, axes = plt.subplots(figsize=(3.6 * len(metrics), 4), ncols=len(metrics),
                             squeeze=False)
    for ax, metric in zip(axes[0], metrics):
        means = [agg.loc[e, (metric, 'mean')] for e in editors]
        stds = [0. if np.isnan(agg.loc[e, (metric, 'std')]) else agg.loc[e, (metric, 'std')]
                for e in editors]
        colors = [get_editor_color(e) for e in editors]
        xpos = np.arange(len(editors))
        ax.bar(xpos, means, yerr=stds, color=colors, width=0.62, capsize=4,
               error_kw={'elinewidth': 1.2, 'ecolor': MUTED_INK}, zorder=2)
        # direct value labels: the reader should not have to measure the bars
        for x, mean in zip(xpos, means):
            ax.text(x, mean + 0.02, f'{mean:.3f}', ha='center', va='bottom',
                    fontsize=8.5, color=NEUTRAL_INK)
        ax.set_xticks(xpos)
        ax.set_xticklabels(editors, rotation=20, ha='right')
        ax.set(ylabel=metric.upper(), ylim=(0, 1.08))
        _style_axes(ax, grid_axis='y')
    fig.suptitle('per-base editing performance across runs', color=NEUTRAL_INK, fontsize=11)
    fig.tight_layout()
    _savefig(fig, fig_dir, fname)

    agg.columns = ['_'.join(col) for col in agg.columns]
    return agg.reset_index()


def plot_efficiency_scatter(df, fig_dir=None, fname='perbase_efficiency_scatter',
                            obs_col='true_efficiency', pred_col='prob_score_class1',
                            editor_col='model_name'):
    """observed vs predicted per-base editing efficiency, one panel per base editor

    Args:
        df: pandas.DataFrame holding the observed efficiency and the predicted probability
        obs_col: string, column of the observed (i.e. measured) editing efficiency
        pred_col: string, column of the predicted editing probability

    Returns:
        pandas.DataFrame with the correlation of every editor
    """
    editors = order_editors(df[editor_col].unique())
    fig, axes = plt.subplots(figsize=(3.5 * len(editors), 3.6), ncols=len(editors),
                             squeeze=False)
    rows = []
    for ax, editor in zip(axes[0], editors):
        sub = df.loc[df[editor_col] == editor]
        x = sub[obs_col].values.astype(np.float64)
        y = sub[pred_col].values.astype(np.float64)
        ax.scatter(x, y, s=12, alpha=0.45, color=get_editor_color(editor),
                   edgecolors='none', zorder=2)
        pearson = spearman = np.nan
        if len(x) > 2 and np.ptp(x) > 0 and np.ptp(y) > 0:
            pearson = stats.pearsonr(x, y)[0]
            spearman = stats.spearmanr(x, y)[0]
            ax.text(0.04, 0.96, f'Pearson r={pearson:.3f}\nSpearman '+ r'$\rho$' + f'={spearman:.3f}',
                    transform=ax.transAxes, va='top', ha='left', fontsize=9,
                    color=NEUTRAL_INK)
        ax.set(xlabel='observed editing efficiency', ylabel='predicted edit probability')
        ax.set_title(editor, color=get_editor_color(editor), fontsize=11)
        _style_axes(ax)
        rows.append({'model_name': editor, 'pearson_corr': pearson,
                     'spearman_corr': spearman, 'num_samples': len(x)})
    fig.tight_layout()
    _savefig(fig, fig_dir, fname)
    return pd.DataFrame(rows)


def plot_position_profile(pred_w_attn_df, fig_dir=None, fname='perbase_position_profile',
                          editor_col='model_name', pred_col='prob_score_class1'):
    """editing probability and attention score per protospacer position

    Two stacked panels (never two y-scales in one axes): the predicted editing probability
    per target position, and the attention score the model puts on every position.

    Returns:
        pandas.DataFrame of the plotted profiles
    """
    attn_cols = [f'Attn{i}' for i in range(20)]
    editors = order_editors(pred_w_attn_df[editor_col].unique())
    fig, axes = plt.subplots(figsize=(7.5, 6), nrows=2, sharex=True)
    rows = []
    for editor in editors:
        sub = pred_w_attn_df.loc[pred_w_attn_df[editor_col] == editor]
        color = get_editor_color(editor)
        # mean predicted probability per target base position (1-based numbering)
        prof = sub.groupby('base_pos')[pred_col].agg(['mean', 'std', 'count'])
        positions = prof.index.values + 1
        axes[0].plot(positions, prof['mean'].values, color=color, linewidth=2,
                     marker='o', markersize=5, label=editor, zorder=2)
        # mean attention score per position
        attn_mean = sub[attn_cols].mean(axis=0).values
        axes[1].plot(np.arange(1, 21), attn_mean, color=color, linewidth=2,
                     marker='o', markersize=5, label=editor, zorder=2)
        for pos, mean, count in zip(positions, prof['mean'].values, prof['count'].values):
            rows.append({'model_name': editor, 'base_pos': int(pos),
                         'mean_pred': float(mean), 'num_samples': int(count)})
    axes[0].set(ylabel='mean predicted\nedit probability')
    axes[1].set(xlabel='protospacer position', ylabel='mean attention score')
    axes[1].axhline(1 / 20, color='#bbbbbb', linestyle='--', linewidth=1, zorder=1,
                    label='uniform (1/20)')
    for ax in axes:
        _style_axes(ax)
        ax.legend(frameon=False, fontsize=9)
    axes[1].set_xticks(np.arange(1, 21))
    fig.suptitle('position-wise editing probability and attention', color=NEUTRAL_INK,
                 fontsize=11)
    fig.tight_layout()
    _savefig(fig, fig_dir, fname)
    return pd.DataFrame(rows)


def plot_attention_matrix(pred_w_attn_df, editor, fig_dir=None, fname=None):
    """mean attention score per (target position, attended position) pair

    Returns:
        pandas.DataFrame, the 20x20 (target position x attended position) matrix
    """
    import seaborn as sns
    attn_cols = [f'Attn{i}' for i in range(20)]
    sub = pred_w_attn_df.loc[pred_w_attn_df['model_name'] == editor]
    mat = sub.groupby('base_pos')[attn_cols].mean()
    mat.index = [i + 1 for i in mat.index]
    mat.columns = list(range(1, 21))

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    sns.heatmap(mat, cmap=SEQUENTIAL_CMAP, linewidths=0.4, ax=ax,
                cbar_kws={'label': 'mean attention score'})
    ax.set(xlabel='attended position', ylabel='target base position')
    ax.set_title(f'{editor}: attention per target position', color=get_editor_color(editor),
                 fontsize=11)
    _savefig(fig, fig_dir, fname or f'perbase_attention_matrix_{editor}')
    return mat


def plot_motif_attention(pred_w_attn_df, proc_df, editor, fig_dir=None, fname=None,
                         offsets=range(-4, 5)):
    """mean attention score by the nucleotide identity around the target base

    For every prediction the attention score at a relative offset from the edited base is
    collected and averaged per nucleotide, which is the motif view of the attention
    analysis (i.e. which neighboring nucleotides the model attends to).

    Args:
        pred_w_attn_df: predictions with attention columns (`Attn0..Attn19`)
        proc_df: processed sequence data frame carrying the nucleotide letters (`L1..L20`)
        editor: string, base editor to plot
        offsets: iterable of ints, positions relative to the target base

    Returns:
        pandas.DataFrame, mean attention per (offset, nucleotide)
    """
    import seaborn as sns
    offsets = list(offsets)
    letter_vars = [f'L{i}' for i in range(1, 21)]
    sub = pred_w_attn_df.loc[pred_w_attn_df['model_name'] == editor].copy()
    merged = pd.merge(left=sub, right=proc_df[['ID'] + letter_vars], how='left',
                      left_on='id', right_on='ID')

    records = []
    attn_arr = merged[[f'Attn{i}' for i in range(20)]].values.astype(np.float64)
    letters_arr = merged[letter_vars].values
    base_pos_arr = merged['base_pos'].values.astype(int)
    for offset in offsets:
        tpos = base_pos_arr + offset
        valid = (tpos >= 0) & (tpos < 20)
        rows_idx = np.where(valid)[0]
        for r in rows_idx:
            records.append({'offset': offset,
                            'nucleotide': letters_arr[r, tpos[r]],
                            'attn_score': attn_arr[r, tpos[r]]})
    motif_df = pd.DataFrame(records)
    pivot = motif_df.pivot_table(index='nucleotide', columns='offset',
                                 values='attn_score', aggfunc='mean')
    pivot = pivot.reindex(index=[n for n in ['A', 'C', 'T', 'G'] if n in pivot.index])

    fig, ax = plt.subplots(figsize=(1.0 * len(offsets) + 2, 3.2))
    sns.heatmap(pivot, cmap=SEQUENTIAL_CMAP, annot=True, fmt='.3f', annot_kws={'fontsize': 8},
                linewidths=0.4, ax=ax, cbar_kws={'label': 'mean attention score'})
    ax.set(xlabel='position relative to the target base', ylabel='nucleotide')
    ax.set_title(f'{editor}: attention by neighboring nucleotide',
                 color=get_editor_color(editor), fontsize=11)
    _savefig(fig, fig_dir, fname or f'perbase_motif_attention_{editor}')
    return motif_df.groupby(['offset', 'nucleotide'])['attn_score'].mean().reset_index()
