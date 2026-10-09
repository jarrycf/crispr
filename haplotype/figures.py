"""figure panels of the `BE-DICT (bystander)` model

Every function writes one panel (pdf + png) into `fig_dir` and returns the data frame it
plotted. The color/style conventions are shared with :mod:`criscas.figures`.
"""

import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats


EDITOR_ORDER = ['ABEmax', 'ABE8e', 'BE4max', 'Target-AID']
EDITOR_COLORS = {'ABEmax': '#0072B2',
                 'ABE8e': '#D55E00',
                 'BE4max': '#009E73',
                 'Target-AID': '#785EF0'}
SEQUENTIAL_CMAP = 'YlOrRd'
NEUTRAL_INK = '#333333'
MUTED_INK = '#767676'


def get_editor_color(editor, default='#0072B2'):
    return EDITOR_COLORS.get(editor, default)


def order_editors(editors):
    editors = list(editors)
    known = [e for e in EDITOR_ORDER if e in editors]
    unknown = sorted(e for e in editors if e not in EDITOR_ORDER)
    return known + unknown


def _style_axes(ax, grid_axis='both'):
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


def plot_haplotype_scatter(pred_df, fig_dir=None, fname='bystander_haplotype_scatter',
                           true_col='true_score', pred_col='pred_score',
                           editor_col='model_name', use_hexbin_above=2000):
    """observed vs predicted haplotype (outcome) proportion, one panel per base editor

    Args:
        pred_df: pandas.DataFrame with the observed and predicted outcome proportions and
                 an editor column
        use_hexbin_above: int, number of points above which a hexbin density is drawn
                          instead of a scatter

    Returns:
        pandas.DataFrame with the correlation of every editor
    """
    if editor_col not in pred_df.columns:
        pred_df = pred_df.copy()
        pred_df[editor_col] = 'model'
    editors = order_editors(pred_df[editor_col].unique())
    fig, axes = plt.subplots(figsize=(3.6 * len(editors), 3.7), ncols=len(editors),
                             squeeze=False)
    rows = []
    for ax, editor in zip(axes[0], editors):
        sub = pred_df.loc[pred_df[editor_col] == editor]
        x = sub[true_col].values.astype(np.float64)
        y = sub[pred_col].values.astype(np.float64)
        if len(x) > use_hexbin_above:
            hb = ax.hexbin(x, y, gridsize=45, cmap=SEQUENTIAL_CMAP, mincnt=1, zorder=2)
            fig.colorbar(hb, ax=ax, label='number of haplotypes')
        else:
            ax.scatter(x, y, s=14, alpha=0.5, color=get_editor_color(editor),
                       edgecolors='none', zorder=2)
        lim = max(float(np.nanmax(x)) if len(x) else 1., float(np.nanmax(y)) if len(y) else 1.)
        lim = min(1.0, lim * 1.08)
        ax.plot([0, lim], [0, lim], color='#bbbbbb', linestyle='--', linewidth=1, zorder=1)
        pearson = spearman = np.nan
        if len(x) > 2 and np.ptp(x) > 0 and np.ptp(y) > 0:
            pearson = stats.pearsonr(x, y)[0]
            spearman = stats.spearmanr(x, y)[0]
            ax.text(0.04, 0.96,
                    f'Pearson r={pearson:.3f}\nSpearman ' + r'$\rho$' + f'={spearman:.3f}'
                    + f'\nn={len(x)}',
                    transform=ax.transAxes, va='top', ha='left', fontsize=9,
                    color=NEUTRAL_INK)
        ax.set(xlabel='observed outcome proportion', ylabel='predicted outcome proportion')
        ax.set_title(editor, color=get_editor_color(editor), fontsize=11)
        _style_axes(ax)
        rows.append({'model_name': editor, 'pearson_corr': pearson,
                     'spearman_corr': spearman, 'num_haplotypes': len(x)})
    fig.tight_layout()
    _savefig(fig, fig_dir, fname)
    return pd.DataFrame(rows)


def plot_correlation_bars(scores_df, fig_dir=None, fname='bystander_performance',
                          metrics=('pearson_corr', 'spearman_corr'),
                          editor_col='model_name'):
    """correlation of the 5 runs per base editor, as mean +- standard deviation

    Args:
        scores_df: pandas.DataFrame with one row per (editor, run), i.e. the output of
                   :func:`haplotype.train_val.summarize_scores` with an editor column

    Returns:
        pandas.DataFrame of the aggregated statistics
    """
    metrics = [m for m in metrics if m in scores_df.columns]
    assert metrics, 'none of the requested metrics is present in the passed data frame'
    editors = order_editors(scores_df[editor_col].unique())
    agg = scores_df.groupby(editor_col)[metrics].agg(['mean', 'std', 'count'])

    fig, axes = plt.subplots(figsize=(3.6 * len(metrics), 4), ncols=len(metrics),
                             squeeze=False)
    labels = {'pearson_corr': 'Pearson r', 'spearman_corr': r'Spearman $\rho$'}
    for ax, metric in zip(axes[0], metrics):
        means = [agg.loc[e, (metric, 'mean')] for e in editors]
        stds = [0. if np.isnan(agg.loc[e, (metric, 'std')]) else agg.loc[e, (metric, 'std')]
                for e in editors]
        xpos = np.arange(len(editors))
        ax.bar(xpos, means, yerr=stds, color=[get_editor_color(e) for e in editors],
               width=0.62, capsize=4, error_kw={'elinewidth': 1.2, 'ecolor': MUTED_INK},
               zorder=2)
        for x, mean in zip(xpos, means):
            ax.text(x, mean + 0.02, f'{mean:.3f}', ha='center', va='bottom', fontsize=8.5,
                    color=NEUTRAL_INK)
        ax.set_xticks(xpos)
        ax.set_xticklabels(editors, rotation=20, ha='right')
        ax.set(ylabel=labels.get(metric, metric), ylim=(0, 1.08))
        _style_axes(ax, grid_axis='y')
    fig.suptitle('bystander (haplotype) performance across runs', color=NEUTRAL_INK,
                 fontsize=11)
    fig.tight_layout()
    _savefig(fig, fig_dir, fname)

    agg.columns = ['_'.join(col) for col in agg.columns]
    return agg.reset_index()


def compute_perposition_conversion(pred_df, conv_nucl, seq_len=20, inpseq_col='Inp_seq',
                                   outpseq_col='Outp_seq', true_col='true_score',
                                   pred_col='pred_score'):
    """per-position conversion frequency implied by the haplotype proportions

    For every position the proportions of all haplotypes that carry a conversion at that
    position are summed, separately for the observed and the predicted proportions.

    Args:
        pred_df: pandas.DataFrame with input/outcome sequences and their proportions
        conv_nucl: tuple, (target nucleotide, conversion nucleotide) i.e. ('A', 'G')
        seq_len: int, sequence length

    Returns:
        pandas.DataFrame with one row per position
    """
    tb_nucl, cb_nucl = conv_nucl
    inp = np.array([list(s) for s in pred_df[inpseq_col].values])
    outp = np.array([list(s) for s in pred_df[outpseq_col].values])
    converted = (inp == tb_nucl) & (outp == cb_nucl)

    has_true = true_col in pred_df.columns
    pred_w = pred_df[pred_col].values.astype(np.float64)
    true_w = pred_df[true_col].values.astype(np.float64) if has_true else None

    rows = []
    for pos in range(seq_len):
        mask = converted[:, pos]
        # a position is only informative for the sequences that harbor the target base
        editable = (inp[:, pos] == tb_nucl)
        row = {'position': pos + 1,
               'num_sequences_with_target_base': int(editable.sum()),
               'pred_conversion': float(pred_w[mask].sum() / max(editable.sum(), 1))}
        if has_true:
            row['true_conversion'] = float(true_w[mask].sum() / max(editable.sum(), 1))
        rows.append(row)
    return pd.DataFrame(rows)


def plot_perposition_conversion(pred_df, conv_nucl, fig_dir=None,
                                fname='bystander_perposition_conversion', editor=None,
                                seq_len=20, **kwargs):
    """observed vs predicted per-position conversion frequency

    Returns:
        pandas.DataFrame, output of :func:`compute_perposition_conversion`
    """
    prof = compute_perposition_conversion(pred_df, conv_nucl, seq_len=seq_len, **kwargs)
    color = get_editor_color(editor) if editor else '#0072B2'

    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    xpos = prof['position'].values
    if 'true_conversion' in prof.columns:
        ax.plot(xpos, prof['true_conversion'].values, color=MUTED_INK, linewidth=2,
                marker='s', markersize=5, label='observed', zorder=2)
    ax.plot(xpos, prof['pred_conversion'].values, color=color, linewidth=2, marker='o',
            markersize=5, label='predicted', zorder=3)
    ax.set(xlabel='protospacer position', ylabel='conversion frequency')
    ax.set_xticks(xpos)
    title = f'{editor}: per-position conversion' if editor else 'per-position conversion'
    ax.set_title(title, color=color, fontsize=11)
    _style_axes(ax)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    _savefig(fig, fig_dir, fname)
    return prof


def plot_haplotype_rank_agreement(pred_df, fig_dir=None, fname='bystander_top_haplotype',
                                  true_col='true_score', pred_col='pred_score',
                                  seqid_col='seq_id', editor_col='model_name', top_k=3):
    """how often the highest predicted haplotype/s match the observed ranking

    For every target sequence the overlap between the `top_k` observed and the `top_k`
    predicted haplotypes is computed.

    Returns:
        pandas.DataFrame with the top-k agreement of every editor
    """
    if editor_col not in pred_df.columns:
        pred_df = pred_df.copy()
        pred_df[editor_col] = 'model'
    rows = []
    for editor, edf in pred_df.groupby(editor_col):
        hits = {k: [] for k in range(1, top_k + 1)}
        for __, gdf in edf.groupby(seqid_col):
            if len(gdf) < 2:
                continue
            true_rank = gdf.sort_values(true_col, ascending=False)['Outp_seq'].tolist()
            pred_rank = gdf.sort_values(pred_col, ascending=False)['Outp_seq'].tolist()
            for k in range(1, top_k + 1):
                kk = min(k, len(gdf))
                overlap = len(set(true_rank[:kk]) & set(pred_rank[:kk])) / kk
                hits[k].append(overlap)
        for k in range(1, top_k + 1):
            if hits[k]:
                rows.append({'model_name': editor, 'top_k': k,
                             'agreement': float(np.mean(hits[k])),
                             'num_sequences': len(hits[k])})
    agreement_df = pd.DataFrame(rows)
    if agreement_df.empty:
        print('not enough haplotypes per sequence to compute the top-k agreement')
        return agreement_df

    editors = order_editors(agreement_df['model_name'].unique())
    fig, ax = plt.subplots(figsize=(5.6, 4))
    width = 0.8 / max(len(editors), 1)
    for i, editor in enumerate(editors):
        sub = agreement_df.loc[agreement_df['model_name'] == editor].sort_values('top_k')
        xpos = np.arange(len(sub)) + i * width
        ax.bar(xpos, sub['agreement'].values, width=width * 0.92,
               color=get_editor_color(editor), label=editor, zorder=2)
    ax.set_xticks(np.arange(top_k) + width * (len(editors) - 1) / 2)
    ax.set_xticklabels([f'top-{k}' for k in range(1, top_k + 1)])
    ax.set(ylabel='mean overlap with the observed ranking', ylim=(0, 1.08))
    ax.set_title('agreement of the most probable haplotypes', color=NEUTRAL_INK, fontsize=11)
    _style_axes(ax, grid_axis='y')
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    _savefig(fig, fig_dir, fname)
    return agreement_df
