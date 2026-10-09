#!/usr/bin/env python
"""train the `BE-DICT (bystander)` model on observed haplotype (outcome) proportions

Example:
    python scripts/train_bystander.py \
        --data data/ABEmax_haplotypes.csv --editor ABEmax \
        --output-dir trained_models_rerun/bystander --num-runs 5 --num-epochs 150

The input csv is expected to hold the columns
``[seq_id, Inp_seq, Outp_seq, true_score]`` (see `docs/TRAINING.md`).
"""

import argparse
import os
import sys

import pandas as pd
import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

from haplotype.data_preprocess import SeqProcessConfig, HaplotypeSeqProcessor
from haplotype.hyperparam import get_hyperparam_options
from haplotype.train_val import train_bystander_model, summarize_scores


# target -> conversion nucleotide of the base editors reported in the paper
EDITOR_CONV_NUCL = {'ABEmax': ('A', 'G'),
                    'ABE8e': ('A', 'G'),
                    'BE4max': ('C', 'T'),
                    'Target-AID': ('C', 'T')}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', required=True,
                        help='csv file with [seq_id, Inp_seq, Outp_seq, true_score]')
    parser.add_argument('--editor', required=True,
                        help='base editor name (i.e. ABEmax, ABE8e, BE4max, Target-AID)')
    parser.add_argument('--output-dir', default='trained_models_rerun/bystander',
                        help='directory where the trained models are written')
    parser.add_argument('--conversion-nucl', default=None,
                        help='target and conversion nucleotide, i.e. "A,G"; inferred from '
                             'the editor name if omitted')
    parser.add_argument('--editor-col', default='Editor',
                        help='column used to select the rows of the editor (if present)')
    parser.add_argument('--seqid-col', default='seq_id', help='sequence id column')
    parser.add_argument('--inpseq-col', default='Inp_seq', help='input sequence column')
    parser.add_argument('--outpseq-col', default='Outp_seq', help='outcome sequence column')
    parser.add_argument('--outcome-col', default='true_score',
                        help='observed outcome proportion column')
    parser.add_argument('--seq-len', type=int, default=20, help='sequence length')
    parser.add_argument('--ewindow', default='3,10',
                        help='editable window (start,end) in the offset numbering of the '
                             'data processing config (default: 3,10)')
    parser.add_argument('--offset', type=int, default=1,
                        help='offset of the position numbering (default: 1)')
    parser.add_argument('--num-runs', type=int, default=5,
                        help='number of train/validation/test partitions (default: 5)')
    parser.add_argument('--num-epochs', type=int, default=None,
                        help='number of epochs (default: 150, the value used in the paper)')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='batch size (default: 300, the value used in the paper)')
    parser.add_argument('--embed-dim', type=int, default=None, help='embedding dimension')
    parser.add_argument('--num-attn-heads', type=int, default=None,
                        help='number of attention heads')
    parser.add_argument('--num-transformer-units', type=int, default=None,
                        help='number of encoder/decoder units')
    parser.add_argument('--p-dropout', type=float, default=None, help='dropout probability')
    parser.add_argument('--l2-reg', type=float, default=None, help='weight decay')
    parser.add_argument('--multihead-type', default=None, choices=['Wide', 'Narrow'],
                        help='multi-head attention type (default: Narrow)')
    parser.add_argument('--max-num-targets', type=int, default=12,
                        help='maximum number of target bases considered per sequence')
    parser.add_argument('--no-renormalize', action='store_true',
                        help='do not renormalize the observed outcome proportions')
    parser.add_argument('--no-normalize-pred', action='store_true',
                        help='do not renormalize the predicted haplotype probabilities '
                             'over the observed haplotype set')
    parser.add_argument('--val-set-portion', type=float, default=0.1,
                        help='portion of the training set held out for validation')
    parser.add_argument('--random-state', type=int, default=42,
                        help='seed of the data partitioning')
    parser.add_argument('--cpu', action='store_true', help='train on cpu even if a gpu exists')
    parser.add_argument('--skip-test', action='store_true',
                        help='do not evaluate on the held out test partitions')
    return parser.parse_args(argv)


def build_hyperparam_config(args):
    kwargs = {}
    overrides = {'embed_dim': args.embed_dim,
                 'num_attn_heads': args.num_attn_heads,
                 'num_transformer_units': args.num_transformer_units,
                 'p_dropout': args.p_dropout,
                 'multihead_type': args.multihead_type,
                 'l2_reg': args.l2_reg,
                 'batch_size': args.batch_size,
                 'num_epochs': args.num_epochs}
    for name, value in overrides.items():
        if value is not None:
            kwargs[name] = (value,)
    return get_hyperparam_options(**kwargs)[0]


def get_conversion_nucl(args):
    if args.conversion_nucl:
        parts = [p.strip().upper() for p in args.conversion_nucl.split(',')]
        assert len(parts) == 2, '--conversion-nucl expects "target,conversion" i.e. "A,G"'
        return tuple(parts)
    if args.editor in EDITOR_CONV_NUCL:
        return EDITOR_CONV_NUCL[args.editor]
    raise ValueError(f'unknown base editor: {args.editor}; '
                     'please pass --conversion-nucl explicitly')


def main(argv=None):
    args = parse_args(argv)
    conv_nucl = get_conversion_nucl(args)
    ewindow = tuple(int(v) for v in args.ewindow.split(','))
    assert len(ewindow) == 2, '--ewindow expects "start,end" i.e. "3,10"'

    data_df = pd.read_csv(args.data)
    if args.editor_col in data_df.columns:
        data_df = data_df.loc[data_df[args.editor_col] == args.editor].copy()
    print(f'read {len(data_df)} rows for editor {args.editor} from {args.data}')
    assert len(data_df), f'no rows left for editor {args.editor}'

    # config used to identify the editable window during data processing
    seqconfig_dataproc = SeqProcessConfig(args.seq_len, (1, args.seq_len), ewindow,
                                          args.offset)
    seq_processor = HaplotypeSeqProcessor(args.editor, conv_nucl, seqconfig_dataproc,
                                          max_num_targets=args.max_num_targets)
    # config used for the tensor generation (the whole sequence is decoded)
    seqconfig_tensgen = SeqProcessConfig(args.seq_len, (1, args.seq_len),
                                         (1, args.seq_len), args.offset)

    hyperparam_config = build_hyperparam_config(args)
    print('hyperparameters:', hyperparam_config)

    train_val_scores, test_scores, __ = train_bystander_model(
        data_df,
        args.editor,
        args.output_dir,
        seq_processor,
        seqconfig_tensgen,
        inpseq_cols=(args.seqid_col, args.inpseq_col),
        outpseq_col=args.outpseq_col,
        outcome_col=args.outcome_col,
        hyperparam_config=hyperparam_config,
        num_runs=args.num_runs,
        num_epochs=args.num_epochs,
        renormalize=not args.no_renormalize,
        val_set_portion=args.val_set_portion,
        random_state=args.random_state,
        to_gpu=not args.cpu,
        run_test=not args.skip_test,
        normalize_pred=not args.no_normalize_pred)

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
