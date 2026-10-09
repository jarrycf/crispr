#!/usr/bin/env python
"""train the `BE-DICT (perbase)` model on a high-throughput target library screen

Example:
    python scripts/train_perbase.py \
        --data data/ABEmax_screen.csv --editor ABEmax \
        --output-dir trained_models_rerun/perbase --num-runs 5 --num-epochs 500

The input csv is expected to hold the columns ``[ID, seq, allCounts, V1, ..., V20]``
(see `docs/TRAINING.md` for the description of the schema).
"""

import argparse
import os
import sys

import pandas as pd
import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

from criscas.hyperparam import get_hyperparam_options
from criscas.train_val import train_perbase_model, summarize_scores, get_target_base


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', required=True,
                        help='csv file of the screen data [ID, seq, allCounts, V1..V20]')
    parser.add_argument('--editor', required=True,
                        help='base editor name (i.e. ABEmax, ABE8e, BE4max, Target-AID)')
    parser.add_argument('--output-dir', default='trained_models_rerun/perbase',
                        help='directory where the trained models are written')
    parser.add_argument('--target-base', default=None,
                        help='targeted nucleotide; inferred from the editor name if omitted')
    parser.add_argument('--num-runs', type=int, default=5,
                        help='number of train/validation/test partitions (default: 5)')
    parser.add_argument('--num-epochs', type=int, default=None,
                        help='number of epochs (default: 500, the value used in the paper)')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='batch size (default: 3000, the value used in the paper)')
    parser.add_argument('--embed-dim', type=int, default=None, help='embedding dimension')
    parser.add_argument('--num-attn-heads', type=int, default=None,
                        help='number of attention heads')
    parser.add_argument('--num-transformer-units', type=int, default=None,
                        help='number of transformer units')
    parser.add_argument('--p-dropout', type=float, default=None, help='dropout probability')
    parser.add_argument('--l2-reg', type=float, default=None, help='weight decay')
    parser.add_argument('--cutoff-score', type=float, default=0.01,
                        help='editing efficiency threshold defining the edited class')
    parser.add_argument('--normalize-by-counts', action='store_true',
                        help='divide the per-base counts V{i} by allCounts')
    parser.add_argument('--val-set-portion', type=float, default=0.1,
                        help='portion of the training set held out for validation')
    parser.add_argument('--random-state', type=int, default=42,
                        help='seed of the data partitioning')
    parser.add_argument('--cpu', action='store_true', help='train on cpu even if a gpu exists')
    parser.add_argument('--skip-test', action='store_true',
                        help='do not evaluate on the held out test partitions')
    parser.add_argument('--sequence-level', action='store_true',
                        help='train the sequence level variant instead of the per-base one')
    return parser.parse_args(argv)


def build_hyperparam_config(args):
    """only override the hyperparameters that were explicitly passed"""
    kwargs = {}
    overrides = {'embed_dim': args.embed_dim,
                 'num_attn_heads': args.num_attn_heads,
                 'num_transformer_units': args.num_transformer_units,
                 'p_dropout': args.p_dropout,
                 'l2_reg': args.l2_reg,
                 'batch_size': args.batch_size,
                 'num_epochs': args.num_epochs}
    for name, value in overrides.items():
        if value is not None:
            kwargs[name] = (value,)
    return get_hyperparam_options(**kwargs)[0]


def main(argv=None):
    args = parse_args(argv)
    target_base = args.target_base or get_target_base(args.editor)
    print(f'--- base editor: {args.editor} | target base: {target_base} ---')

    data_df = pd.read_csv(args.data)
    print(f'read {len(data_df)} rows from {args.data}')

    hyperparam_config = build_hyperparam_config(args)
    print('hyperparameters:', hyperparam_config)

    train_val_scores, test_scores, __ = train_perbase_model(
        data_df,
        args.editor,
        args.output_dir,
        target_base=target_base,
        hyperparam_config=hyperparam_config,
        num_runs=args.num_runs,
        num_epochs=args.num_epochs,
        cutoff_score=args.cutoff_score,
        normalize_by_counts=args.normalize_by_counts,
        val_set_portion=args.val_set_portion,
        random_state=args.random_state,
        to_gpu=not args.cpu,
        run_test=not args.skip_test,
        per_base=not args.sequence_level)

    editor_dir = os.path.join(os.path.abspath(args.output_dir), args.editor)
    for dsettype, scores in (('validation', train_val_scores), ('test', test_scores)):
        if scores is None:
            continue
        summary = summarize_scores(scores, dsettype)
        if summary.empty:
            continue
        summary['model_name'] = args.editor
        out_pth = os.path.join(editor_dir, f'scores_{dsettype}.csv')
        summary.to_csv(out_pth, index=False)
        print(f'\n--- {dsettype} scores ---')
        print(summary.to_string(index=False))
        print(f'written to: {out_pth}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
