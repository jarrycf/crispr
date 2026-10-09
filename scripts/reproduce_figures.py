#!/usr/bin/env python
"""regenerate the `BE-DICT` figure panels from the trained models

The script runs the trained models over the sequences it is given and writes every panel
(pdf + png) plus the underlying numbers (csv) into `--fig-dir`.

Panel groups (``--panels``):
    perbase-attention   attention heatmap per sequence/position, attention matrix per
                        target position and attention by neighboring nucleotide
    perbase-profile     position-wise editing probability and attention profile
    perbase-performance ROC/PR curves, AUC/AUPR per editor and observed vs predicted
                        editing efficiency.
                        Needs reference labels: pass screen data with `--perbase-data`
                        holding [ID, seq, allCounts, V1..V20]
    bystander           observed vs predicted haplotype proportion, per-position
                        conversion frequency, top-k haplotype agreement and the haplotype
                        alignment tables (html)

Examples:
    # runs out of the box on the sequences shipped in sample_data
    python scripts/reproduce_figures.py --panels perbase-attention,perbase-profile,bystander

    # the performance panels need measured editing efficiencies
    python scripts/reproduce_figures.py --panels perbase-performance \
        --perbase-data data/ABEmax_screen.csv --editors ABEmax
"""

import argparse
import os
import sys

import pandas as pd
import matplotlib
matplotlib.use('Agg')

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, REPO_DIR)

from criscas import figures as perbase_figures
from criscas.data_preprocess import generate_perbase_df
from criscas.predict_model import BEDICT_CriscasModel
from criscas.train_val import get_target_base
from criscas.utilities import get_device
from haplotype import figures as bystander_figures
from haplotype.data_preprocess import (SeqProcessConfig, HaplotypeSeqProcessor,
                                       HaplotypeVizFile)
from haplotype.predict_model import BEDICT_HaplotypeModel
from haplotype.utilities import check_na

ALL_PANELS = ('perbase-attention', 'perbase-profile', 'perbase-performance', 'bystander')
EDITOR_CONV_NUCL = {'ABEmax': ('A', 'G'), 'ABE8e': ('A', 'G'),
                    'BE4max': ('C', 'T'), 'Target-AID': ('C', 'T')}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--panels', default='perbase-attention,perbase-profile,bystander',
                        help=f'comma separated list out of {",".join(ALL_PANELS)} or "all"')
    parser.add_argument('--editors', default='ABEmax,ABE8e,BE4max,Target-AID',
                        help='comma separated list of base editors')
    parser.add_argument('--trained-models-dir',
                        default=os.path.join(REPO_DIR, 'trained_models'),
                        help='directory holding the trained models')
    parser.add_argument('--fig-dir', default=os.path.join(REPO_DIR, 'figures'),
                        help='directory where the panels are written')
    parser.add_argument('--perbase-seqs',
                        default=os.path.join(REPO_DIR, 'sample_data', 'abemax_sampledata.csv'),
                        help='csv with [ID, seq] used for the per-base panels')
    parser.add_argument('--perbase-data', default=None,
                        help='csv with [ID, seq, allCounts, V1..V20]; required by the '
                             'perbase-performance panels')
    parser.add_argument('--bystander-data',
                        default=os.path.join(REPO_DIR, 'sample_data', 'bystander_sampledata.csv'),
                        help='csv with [seq_id, Inp_seq, Outp_seq] (+ true_score if available)')
    parser.add_argument('--bystander-outcome-col', default='true_score',
                        help='observed outcome proportion column; pass "" if unavailable')
    parser.add_argument('--num-runs', type=int, default=5,
                        help='number of trained models per editor (default: 5)')
    parser.add_argument('--pred-option', default='mean',
                        choices=['mean', 'median', 'min', 'max'],
                        help='how the predictions of the runs are combined')
    parser.add_argument('--cutoff-score', type=float, default=0.01,
                        help='editing efficiency threshold defining the edited class')
    parser.add_argument('--normalize-by-counts', action='store_true',
                        help='divide the per-base counts V{i} by allCounts')
    parser.add_argument('--attn-seqids', default=None,
                        help='comma separated "seqid:pos" pairs for the attention '
                             'heatmaps, i.e. "CTRL_HEKsiteNO1:5"; defaults to the first '
                             'two sequences of the input')
    parser.add_argument('--ewindow', default='3,10',
                        help='editable window of the bystander model (default: 3,10)')
    parser.add_argument('--seq-len', type=int, default=20, help='sequence length')
    parser.add_argument('--cpu', action='store_true', help='run on cpu even if a gpu exists')
    return parser.parse_args(argv)


def _available_editors(trained_models_dir, model_kind, editors):
    """keep the editors that actually have trained models on disk"""
    available = []
    for editor in editors:
        pth = os.path.join(trained_models_dir, model_kind, editor, 'train_val')
        if os.path.isdir(pth):
            available.append(editor)
        else:
            print(f'skipping {editor}: no trained {model_kind} model under {pth}')
    return available


def run_perbase_predictions(editors, args, device):
    """run the per-base models of every editor over the input sequences"""
    seq_df = pd.read_csv(args.perbase_seqs)
    assert {'ID', 'seq'}.issubset(seq_df.columns), \
        f'{args.perbase_seqs} must hold the columns [ID, seq]'
    pred_map = {}
    for editor in editors:
        print(f'=== per-base predictions: {editor} ===')
        bedict = BEDICT_CriscasModel(editor, device,
                                     trained_models_dir=args.trained_models_dir)
        pred_runs_df, proc_df = bedict.predict_from_dataframe(seq_df,
                                                              num_runs=args.num_runs)
        pred_map[editor] = {'model': bedict,
                            'pred_runs_df': pred_runs_df,
                            'proc_df': proc_df}
    return seq_df, pred_map


def panel_perbase_attention(editors, args, pred_map, fig_dir):
    """attention heatmaps per sequence/position + attention matrix + motif attention"""
    out_dir = os.path.join(fig_dir, 'perbase_attention')
    os.makedirs(out_dir, exist_ok=True)

    seqid_pos_map = None
    if args.attn_seqids:
        seqid_pos_map = {}
        for item in args.attn_seqids.split(','):
            seqid, __, pos = item.partition(':')
            seqid_pos_map.setdefault(seqid.strip(), [])
            if pos.strip():
                seqid_pos_map[seqid.strip()].append(int(pos))

    for editor, entry in pred_map.items():
        bedict = entry['model']
        pred_runs_df, proc_df = entry['pred_runs_df'], entry['proc_df']
        t_seqid_pos_map = seqid_pos_map
        if t_seqid_pos_map is None:
            # default: the first two sequences that carry a predicted target base
            seqids = pred_runs_df['id'].drop_duplicates().tolist()[:2]
            t_seqid_pos_map = {seqid: [] for seqid in seqids}
        editor_dir = os.path.join(out_dir, editor)
        os.makedirs(editor_dir, exist_ok=True)
        print(f'--- attention heatmaps: {editor} -> {editor_dir}')
        bedict.highlight_attn_per_seq(pred_runs_df, proc_df,
                                      seqid_pos_map=t_seqid_pos_map,
                                      pred_option=args.pred_option,
                                      fig_dir=editor_dir)

        pred_df = bedict.select_prediction(pred_runs_df, args.pred_option)
        mat = perbase_figures.plot_attention_matrix(pred_df, editor, fig_dir=out_dir)
        mat.to_csv(os.path.join(out_dir, f'attention_matrix_{editor}.csv'))
        motif_df = perbase_figures.plot_motif_attention(pred_df, proc_df, editor,
                                                        fig_dir=out_dir)
        motif_df.to_csv(os.path.join(out_dir, f'motif_attention_{editor}.csv'), index=False)


def panel_perbase_profile(editors, args, pred_map, fig_dir):
    """position-wise editing probability and attention profile across editors"""
    out_dir = os.path.join(fig_dir, 'perbase_profile')
    os.makedirs(out_dir, exist_ok=True)
    frames = []
    for editor, entry in pred_map.items():
        pred_df = entry['model'].select_prediction(entry['pred_runs_df'], args.pred_option)
        frames.append(pred_df)
    if not frames:
        return
    all_pred_df = pd.concat(frames, ignore_index=True)
    prof = perbase_figures.plot_position_profile(all_pred_df, fig_dir=out_dir)
    prof.to_csv(os.path.join(out_dir, 'position_profile.csv'), index=False)
    print(f'--- position profile written to {out_dir}')


def panel_perbase_performance(editors, args, device, fig_dir):
    """ROC/PR curves, AUC/AUPR bars and observed vs predicted editing efficiency"""
    if not args.perbase_data:
        print('skipping perbase-performance: --perbase-data with the measured editing '
              'efficiencies [ID, seq, allCounts, V1..V20] is required')
        return
    out_dir = os.path.join(fig_dir, 'perbase_performance')
    os.makedirs(out_dir, exist_ok=True)

    raw_df = pd.read_csv(args.perbase_data)
    editor_pred_map = {}
    scatter_frames = []
    for editor in editors:
        target_base = get_target_base(editor)
        print(f'=== per-base performance: {editor} ===')
        proc_df = generate_perbase_df(raw_df, target_base,
                                      cutoff_score=args.cutoff_score,
                                      normalize_by_counts=args.normalize_by_counts)
        bedict = BEDICT_CriscasModel(editor, device,
                                     trained_models_dir=args.trained_models_dir)
        pred_runs_df, __ = bedict.predict_from_dataframe(proc_df[['ID', 'seq']],
                                                         num_runs=args.num_runs,
                                                         target_base=target_base)
        pred_df = bedict.select_prediction(pred_runs_df, args.pred_option)
        # attach the measured per-base efficiency/class of every predicted position
        pred_df = _attach_reference_labels(pred_df, proc_df)
        editor_pred_map[editor] = pred_df
        scatter_frames.append(pred_df)
        pred_df.to_csv(os.path.join(out_dir, f'predictions_{editor}.csv'), index=False)

    if not editor_pred_map:
        return
    auc_df = perbase_figures.plot_roc_curves(editor_pred_map, fig_dir=out_dir)
    auc_df.to_csv(os.path.join(out_dir, 'roc_auc.csv'), index=False)
    aupr_df = perbase_figures.plot_pr_curves(editor_pred_map, fig_dir=out_dir)
    aupr_df.to_csv(os.path.join(out_dir, 'pr_aupr.csv'), index=False)

    scores_df = auc_df.merge(aupr_df[['model_name', 'aupr']], on='model_name', how='outer')
    perbase_figures.plot_performance_bars(scores_df, fig_dir=out_dir)

    scatter_df = pd.concat(scatter_frames, ignore_index=True)
    corr_df = perbase_figures.plot_efficiency_scatter(scatter_df, fig_dir=out_dir)
    corr_df.to_csv(os.path.join(out_dir, 'efficiency_correlation.csv'), index=False)
    print(f'--- performance panels written to {out_dir}')


def _attach_reference_labels(pred_df, proc_df):
    """join the measured per-base efficiency and class onto the predictions"""
    records = []
    es_cols = [f'ES{i}' for i in range(1, 21)]
    eci_cols = [f'ECi{i}' for i in range(1, 21)]
    ref = proc_df.set_index('ID')
    for __, row in pred_df.iterrows():
        seq_id, base_pos = row['id'], int(row['base_pos'])
        if seq_id not in ref.index:
            records.append((float('nan'), -1))
            continue
        ref_row = ref.loc[seq_id]
        records.append((float(ref_row[es_cols[base_pos]]), int(ref_row[eci_cols[base_pos]])))
    pred_df = pred_df.copy()
    pred_df['true_efficiency'] = [r[0] for r in records]
    pred_df['true_class'] = [r[1] for r in records]
    # positions without a reference label cannot be scored
    pred_df = pred_df.loc[pred_df['true_class'] >= 0].reset_index(drop=True)
    return pred_df


def panel_bystander(editors, args, device, fig_dir):
    """bystander panels: proportion scatter, per-position conversion, top-k agreement"""
    out_dir = os.path.join(fig_dir, 'bystander')
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(args.bystander_data)
    ewindow = tuple(int(v) for v in args.ewindow.split(','))
    outcome_col = args.bystander_outcome_col or None
    has_outcome = bool(outcome_col) and outcome_col in df.columns
    if outcome_col and not has_outcome:
        print(f'column {outcome_col} not found in {args.bystander_data}; '
              'the observed proportions are not plotted')

    pred_frames = []
    for editor in editors:
        conv_nucl = EDITOR_CONV_NUCL.get(editor)
        if conv_nucl is None:
            print(f'skipping {editor}: unknown conversion nucleotides')
            continue
        sub = df
        if 'Editor' in df.columns:
            sub = df.loc[df['Editor'] == editor]
            if not len(sub):
                print(f'skipping {editor}: no rows in {args.bystander_data}')
                continue
        print(f'=== bystander predictions: {editor} ===')
        seq_processor = HaplotypeSeqProcessor(editor, conv_nucl,
                                             SeqProcessConfig(args.seq_len,
                                                              (1, args.seq_len),
                                                              ewindow, 1))
        bedict_haplo = BEDICT_HaplotypeModel(seq_processor,
                                             SeqProcessConfig(args.seq_len,
                                                              (1, args.seq_len),
                                                              (1, args.seq_len), 1),
                                             device)
        # when the observed proportions are available the given haplotypes are scored,
        # otherwise all the combinatorial haplotypes are enumerated and scored
        dloader = bedict_haplo.prepare_data(
            sub,
            ['seq_id', 'Inp_seq'],
            outpseq_col='Outp_seq' if has_outcome else None,
            outcome_col=outcome_col if has_outcome else None,
            renormalize=has_outcome,
            batch_size=500)
        run_frames = []
        for run_num in range(args.num_runs):
            model_dir = os.path.join(args.trained_models_dir, 'bystander', editor,
                                     'train_val', f'run_{run_num}')
            if not os.path.isdir(model_dir):
                print(f'  run_{run_num} not found at {model_dir}, skipped')
                continue
            pred_df = bedict_haplo.predict_from_dloader(
                dloader, model_dir, outcome_col=outcome_col if has_outcome else None)
            pred_df['run_num'] = run_num
            run_frames.append(pred_df)
        if not run_frames:
            continue
        unif_df = pd.concat(run_frames, axis=0, ignore_index=True)
        check_na(unif_df)
        agg_df = bedict_haplo.compute_avg_predictions(unif_df)
        check_na(agg_df)
        agg_df['model_name'] = editor
        agg_df.to_csv(os.path.join(out_dir, f'predictions_{editor}.csv'), index=False)
        pred_frames.append(agg_df)

        # per-position conversion frequency
        prof = bystander_figures.plot_perposition_conversion(
            agg_df, conv_nucl, fig_dir=out_dir,
            fname=f'bystander_perposition_conversion_{editor}', editor=editor,
            seq_len=args.seq_len)
        prof.to_csv(os.path.join(out_dir, f'perposition_conversion_{editor}.csv'),
                    index=False)

        # haplotype alignment tables
        viz_dir = os.path.join(out_dir, 'haplotype_tables', editor)
        os.makedirs(viz_dir, exist_ok=True)
        # the alignment table reports the predicted outcome proportions
        html_map = bedict_haplo.visualize_haplotype(
            agg_df, agg_df['seq_id'].unique(), ['seq_id', 'Inp_seq'], 'Outp_seq',
            'pred_score', predscore_thr=0.)
        vf = HaplotypeVizFile(os.path.join(REPO_DIR, 'haplotype', 'viz_resources'))
        for seq_id in html_map:
            vf.create(html_map[seq_id], viz_dir, f'{editor}_{seq_id}_haplotype')
        print(f'--- haplotype tables written to {viz_dir}')

    if not pred_frames:
        return
    all_pred_df = pd.concat(pred_frames, ignore_index=True)
    if has_outcome:
        corr_df = bystander_figures.plot_haplotype_scatter(all_pred_df, fig_dir=out_dir)
        corr_df.to_csv(os.path.join(out_dir, 'haplotype_correlation.csv'), index=False)
        agreement_df = bystander_figures.plot_haplotype_rank_agreement(all_pred_df,
                                                                      fig_dir=out_dir)
        if not agreement_df.empty:
            agreement_df.to_csv(os.path.join(out_dir, 'topk_agreement.csv'), index=False)
    print(f'--- bystander panels written to {out_dir}')


def main(argv=None):
    args = parse_args(argv)
    panels = ALL_PANELS if args.panels.strip() == 'all' else \
        tuple(p.strip() for p in args.panels.split(',') if p.strip())
    unknown = [p for p in panels if p not in ALL_PANELS]
    assert not unknown, f'unknown panel/s: {unknown}; choose from {ALL_PANELS}'

    editors = [e.strip() for e in args.editors.split(',') if e.strip()]
    device = get_device(not args.cpu)
    fig_dir = os.path.abspath(args.fig_dir)
    os.makedirs(fig_dir, exist_ok=True)
    print(f'--- figures are written to: {fig_dir} ---')

    perbase_panels = [p for p in panels if p.startswith('perbase-') and p != 'perbase-performance']
    if perbase_panels:
        perbase_editors = _available_editors(args.trained_models_dir, 'perbase', editors)
        if perbase_editors:
            __, pred_map = run_perbase_predictions(perbase_editors, args, device)
            if 'perbase-attention' in panels:
                panel_perbase_attention(perbase_editors, args, pred_map, fig_dir)
            if 'perbase-profile' in panels:
                panel_perbase_profile(perbase_editors, args, pred_map, fig_dir)

    if 'perbase-performance' in panels:
        perbase_editors = _available_editors(args.trained_models_dir, 'perbase', editors)
        if perbase_editors:
            panel_perbase_performance(perbase_editors, args, device, fig_dir)

    if 'bystander' in panels:
        bystander_editors = _available_editors(args.trained_models_dir, 'bystander', editors)
        if bystander_editors:
            panel_bystander(bystander_editors, args, device, fig_dir)

    print('--- done ---')
    return 0


if __name__ == '__main__':
    sys.exit(main())
