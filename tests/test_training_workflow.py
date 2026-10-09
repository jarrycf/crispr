"""smoke tests of the training and figure reproduction workflows

The tests train tiny models on synthetic data; they check that the workflows run, that
the produced artifacts have the layout expected by the predict/inference code, and that
the trained models are architecturally identical to the ones shipped in this repo.

Run with:
    python -m pytest tests/ -v
"""

import itertools
import os
import sys

import numpy as np
import pandas as pd
import pytest
import torch
import matplotlib
matplotlib.use('Agg')

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, REPO_DIR)

from criscas.data_preprocess import generate_perbase_df, process_perbase_df
from criscas.data_partition import get_stratified_partitions, generate_partition_datatensor
from criscas.dataset import create_datatensor
from criscas.hyperparam import get_hyperparam_options, get_saved_config, build_config_map
from criscas.train_val import (build_perbase_model, train_perbase_model, summarize_scores,
                               get_target_base)
from haplotype.data_preprocess import SeqProcessConfig, HaplotypeSeqProcessor
from haplotype.hyperparam import (get_default_hyperparam_config as hap_default_config,
                                  get_saved_config as hap_get_saved_config,
                                  build_config_map as hap_build_config_map,
                                  get_hyperparam_options as hap_hyperparam_options)
from haplotype.train_val import build_haplotype_model, train_bystander_model

TRAINED_MODELS_DIR = os.path.join(REPO_DIR, 'trained_models')
SEQ_LEN = 20


def make_perbase_data(n=200, target='A', seed=0):
    """synthetic screen data in the raw schema [ID, seq, allCounts, V1..V20]"""
    rng = np.random.default_rng(seed)
    bases = np.array(list('ACTG'))
    pos_eff = np.exp(-0.5 * ((np.arange(SEQ_LEN) - 6) / 2.5) ** 2)
    rows = []
    for i in range(n):
        seq = ''.join(rng.choice(bases, SEQ_LEN))
        scores = [float(rng.beta(2, 5) * pos_eff[p]) if ch == target else 0.
                  for p, ch in enumerate(seq)]
        rows.append(dict(ID=f'seq_{i}', seq=seq, allCounts=int(rng.integers(500, 5000)),
                         **{f'V{j + 1}': scores[j] for j in range(SEQ_LEN)}))
    return pd.DataFrame(rows)


def make_haplotype_data(n_seqs=30, editor='ABEmax', target='A', conv='G', seed=1):
    """synthetic bystander data: input sequences, their haplotypes and proportions"""
    rng = np.random.default_rng(seed)
    bases = np.array(list('ACTG'))
    pos_eff = np.exp(-0.5 * ((np.arange(SEQ_LEN) - 6) / 2.5) ** 2)
    rows = []
    for i in range(n_seqs):
        seq = ''.join(rng.choice(bases, SEQ_LEN))
        tpos = [p for p in range(2, 10) if seq[p] == target][:4]
        if not tpos:
            continue
        p_edit = {p: float(np.clip(rng.beta(2, 3) * pos_eff[p], 0.01, 0.95)) for p in tpos}
        haps = []
        for combo in itertools.product([0, 1], repeat=len(tpos)):
            prob = 1.
            outp = list(seq)
            for p, edited in zip(tpos, combo):
                prob *= p_edit[p] if edited else (1 - p_edit[p])
                if edited:
                    outp[p] = conv
            haps.append((''.join(outp), prob))
        total = sum(h[1] for h in haps)
        for outp, prob in haps:
            rows.append(dict(Editor=editor, seq_id=f'seq_{i}', Inp_seq=seq,
                             Outp_seq=outp, true_score=prob / total))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# data preprocessing
# --------------------------------------------------------------------------------------

def test_process_perbase_df_yields_numeric_bases():
    """the nucleotide encoding must stay numeric, otherwise tensorization fails"""
    df = make_perbase_data(n=20)
    proc_df = process_perbase_df(df, 'A')
    base_cols = [f'B{i}' for i in range(1, SEQ_LEN + 1)]
    assert proc_df[base_cols].to_numpy().dtype.kind in 'iu'
    assert proc_df[base_cols].to_numpy().max() <= 3
    assert proc_df[base_cols].to_numpy().min() >= 0


def test_generate_perbase_df_labels():
    df = make_perbase_data(n=50)
    proc_df = generate_perbase_df(df, 'A', cutoff_score=0.05)
    for prefix in ('B', 'L', 'M', 'ES', 'ECi', 'ESpr'):
        assert all(f'{prefix}{i}' in proc_df.columns for i in range(1, SEQ_LEN + 1))
    assert {'efficiency_score', 'edited_seq_categ'}.issubset(proc_df.columns)
    # a non target base position is never labeled as edited
    mask = proc_df[[f'M{i}' for i in range(1, SEQ_LEN + 1)]].values.astype(bool)
    categ = proc_df[[f'ECi{i}' for i in range(1, SEQ_LEN + 1)]].values
    assert (categ[~mask] == 0).all()
    # the sequence level class is consistent with the cutoff
    expected = (proc_df['efficiency_score'] >= 0.05).astype(int)
    assert (proc_df['edited_seq_categ'] == expected).all()


def test_generate_perbase_df_normalize_by_counts():
    df = make_perbase_data(n=20)
    proc = generate_perbase_df(df, 'A', normalize_by_counts=True)
    raw = generate_perbase_df(df, 'A', normalize_by_counts=False)
    es_cols = [f'ES{i}' for i in range(1, SEQ_LEN + 1)]
    # dividing by the read counts can only shrink the (positive) scores
    assert (proc[es_cols].values <= raw[es_cols].values + 1e-9).all()


def test_partitions_do_not_leak():
    df = make_perbase_data(n=120)
    proc_df = generate_perbase_df(df, 'A', cutoff_score=0.05)
    partitions = get_stratified_partitions(proc_df, num_splits=5, report=False)
    assert len(partitions) == 5
    all_test = set()
    for run_num, part in partitions.items():
        train, val, test = (set(part[k].tolist()) for k in ('train', 'validation', 'test'))
        assert not (train & test) and not (train & val) and not (val & test)
        assert len(train | val | test) == len(proc_df)
        all_test |= test
    # across the folds every sequence is tested exactly once
    assert all_test == set(range(len(proc_df)))


# --------------------------------------------------------------------------------------
# model construction / compatibility with the shipped models
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize('editor', ['ABEmax', 'ABE8e', 'BE4max', 'Target-AID'])
def test_shipped_perbase_models_load_from_their_config(editor):
    """the saved configs must be readable and must describe the saved weights"""
    run_dir = os.path.join(TRAINED_MODELS_DIR, 'perbase', editor, 'train_val', 'run_0')
    if not os.path.isdir(run_dir):
        pytest.skip(f'no shipped model for {editor}')
    mconfig, options = get_saved_config(os.path.join(run_dir, 'config'))
    model = build_perbase_model(mconfig, options)
    state_dict = torch.load(os.path.join(run_dir, 'model_statedict', 'Transformer.pkl'),
                            map_location='cpu')
    model.load_state_dict(state_dict)  # raises if the architecture does not match


@pytest.mark.parametrize('editor', ['ABEmax', 'ABE8e', 'BE4max', 'Target-AID'])
def test_shipped_bystander_models_load_from_their_config(editor):
    run_dir = os.path.join(TRAINED_MODELS_DIR, 'bystander', editor, 'train_val', 'run_0')
    if not os.path.isdir(run_dir):
        pytest.skip(f'no shipped model for {editor}')
    mconfig, options = hap_get_saved_config(os.path.join(run_dir, 'config'))
    model = build_haplotype_model(mconfig, options)
    state_dict = torch.load(
        os.path.join(run_dir, 'model_statedict', 'HaplotypeTransformer.pkl'),
        map_location='cpu')
    model.load_state_dict(state_dict)


def test_default_hyperparams_match_shipped_perbase_config():
    """training with the defaults must rebuild the published architecture"""
    run_dir = os.path.join(TRAINED_MODELS_DIR, 'perbase', 'ABEmax', 'train_val', 'run_0')
    if not os.path.isdir(run_dir):
        pytest.skip('no shipped model')
    shipped_mconfig, shipped_options = get_saved_config(os.path.join(run_dir, 'config'))
    shipped = shipped_mconfig['model_config']
    mine = get_hyperparam_options()[0]
    for attr in ('embed_dim', 'num_attn_heads', 'num_transformer_units', 'p_dropout',
                 'mlp_embed_factor', 'l2_reg', 'batch_size', 'num_epochs'):
        assert getattr(mine, attr) == getattr(shipped, attr), attr

    cfg, opts = build_config_map(mine, 'test')
    built = build_perbase_model(cfg, opts).state_dict()
    shipped_sd = torch.load(os.path.join(run_dir, 'model_statedict', 'Transformer.pkl'),
                            map_location='cpu')
    assert list(built.keys()) == list(shipped_sd.keys())
    assert all(built[k].shape == shipped_sd[k].shape for k in shipped_sd)


def test_default_hyperparams_match_shipped_bystander_config():
    run_dir = os.path.join(TRAINED_MODELS_DIR, 'bystander', 'ABEmax', 'train_val', 'run_0')
    if not os.path.isdir(run_dir):
        pytest.skip('no shipped model')
    shipped_mconfig, __ = hap_get_saved_config(os.path.join(run_dir, 'config'))
    shipped = shipped_mconfig['model_config']
    mine = hap_default_config()
    for attr in ('embed_dim', 'num_attn_heads', 'num_transformer_units', 'p_dropout',
                 'mlp_embed_factor', 'multihead_type', 'l2_reg', 'batch_size',
                 'num_epochs'):
        assert getattr(mine, attr) == getattr(shipped, attr), attr

    cfg, opts = hap_build_config_map(mine, 'test')
    built = build_haplotype_model(cfg, opts).state_dict()
    shipped_sd = torch.load(
        os.path.join(run_dir, 'model_statedict', 'HaplotypeTransformer.pkl'),
        map_location='cpu')
    assert list(built.keys()) == list(shipped_sd.keys())
    assert all(built[k].shape == shipped_sd[k].shape for k in shipped_sd)


def test_get_target_base():
    assert get_target_base('ABEmax') == 'A'
    assert get_target_base('ABE8e') == 'A'
    assert get_target_base('BE4max') == 'C'
    assert get_target_base('Target-AID') == 'C'
    with pytest.raises(ValueError):
        get_target_base('unknown-editor')


# --------------------------------------------------------------------------------------
# training workflows
# --------------------------------------------------------------------------------------

def test_train_perbase_model_writes_expected_layout(tmp_path):
    df = make_perbase_data(n=120)
    hyperparam = get_hyperparam_options(batch_size=(64,), num_epochs=(2,))[0]
    train_scores, test_scores, proc_df = train_perbase_model(
        df, 'ABEmax', str(tmp_path), hyperparam_config=hyperparam, num_runs=2,
        num_epochs=2, cutoff_score=0.05, to_gpu=False, run_test=True)

    for run_num in range(2):
        run_dir = tmp_path / 'ABEmax' / 'train_val' / f'run_{run_num}'
        # the layout the predict code expects
        assert (run_dir / 'model_statedict' / 'Transformer.pkl').is_file()
        assert (run_dir / 'config' / 'mconfig.pkl').is_file()
        assert (run_dir / 'config' / 'exp_options.pkl').is_file()
        # the dumped config must rebuild a model that loads the dumped weights
        mconfig, options = get_saved_config(str(run_dir / 'config'))
        model = build_perbase_model(mconfig, options)
        model.load_state_dict(torch.load(str(run_dir / 'model_statedict' / 'Transformer.pkl'),
                                         map_location='cpu'))
        assert (tmp_path / 'ABEmax' / 'test' / f'run_{run_num}' / 'predictions_test.csv').is_file()

    summary = summarize_scores(test_scores, 'test')
    assert len(summary) == 2
    assert summary['accuracy'].between(0, 1).all()


def test_train_perbase_model_runs_sequence_level(tmp_path):
    df = make_perbase_data(n=80)
    hyperparam = get_hyperparam_options(batch_size=(32,), num_epochs=(1,))[0]
    train_scores, __, __ = train_perbase_model(
        df, 'ABEmax', str(tmp_path), hyperparam_config=hyperparam, num_runs=2,
        num_epochs=1, cutoff_score=0.05, to_gpu=False, run_test=False, per_base=False)
    assert (tmp_path / 'ABEmax' / 'train_val' / 'run_0' / 'model_statedict' /
            'Transformer.pkl').is_file()


def test_train_bystander_model_writes_expected_layout(tmp_path):
    df = make_haplotype_data(n_seqs=40)
    seq_processor = HaplotypeSeqProcessor('ABEmax', ('A', 'G'),
                                          SeqProcessConfig(SEQ_LEN, (1, SEQ_LEN), (3, 10), 1))
    seqconfig = SeqProcessConfig(SEQ_LEN, (1, SEQ_LEN), (1, SEQ_LEN), 1)
    hyperparam = hap_hyperparam_options(embed_dim=(32,), num_attn_heads=(4,),
                                        batch_size=(8,), num_epochs=(2,))[0]
    train_scores, test_scores, __ = train_bystander_model(
        df, 'ABEmax', str(tmp_path), seq_processor, seqconfig, outcome_col='true_score',
        hyperparam_config=hyperparam, num_runs=2, num_epochs=2, to_gpu=False,
        run_test=True)

    for run_num in range(2):
        run_dir = tmp_path / 'ABEmax' / 'train_val' / f'run_{run_num}'
        assert (run_dir / 'model_statedict' / 'HaplotypeTransformer.pkl').is_file()
        assert (run_dir / 'config' / 'mconfig.pkl').is_file()
        mconfig, options = hap_get_saved_config(str(run_dir / 'config'))
        model = build_haplotype_model(mconfig, options)
        model.load_state_dict(torch.load(
            str(run_dir / 'model_statedict' / 'HaplotypeTransformer.pkl'), map_location='cpu'))

    for run_num, score in test_scores.items():
        assert score['test']['loss'] >= 0


def test_bystander_predictions_are_a_distribution(tmp_path):
    """the renormalized haplotype probabilities of one sequence must sum to one"""
    from haplotype.train_val import compute_haplotype_logprob
    from haplotype.dataset import HaplotypeDataTensor
    from haplotype.predict_model import BEDICT_HaplotypeModel

    df = make_haplotype_data(n_seqs=10)
    seq_processor = HaplotypeSeqProcessor('ABEmax', ('A', 'G'),
                                          SeqProcessConfig(SEQ_LEN, (1, SEQ_LEN), (3, 10), 1))
    seqconfig = SeqProcessConfig(SEQ_LEN, (1, SEQ_LEN), (1, SEQ_LEN), 1)
    device = torch.device('cpu')
    bedict = BEDICT_HaplotypeModel(seq_processor, seqconfig, device)
    proc_df = bedict._process_df(df, ['seq_id', 'Inp_seq'], 'Outp_seq', 'true_score',
                                 renormalize=True)
    dtensor = HaplotypeDataTensor(seqconfig)
    dtensor.generate_tensor_from_df(proc_df, ('A', 'G'), 'true_score')

    hyperparam = hap_hyperparam_options(embed_dim=(32,), num_attn_heads=(4,))[0]
    cfg, opts = hap_build_config_map(hyperparam, 'test')
    model = build_haplotype_model(cfg, opts).eval()
    with torch.no_grad():
        logprob, target_prob, __, __ = compute_haplotype_logprob(model, dtensor[0], device,
                                                                 normalize_pred=True)
    assert float(torch.exp(logprob).sum()) == pytest.approx(1.0, abs=1e-5)
    # the renormalized reference proportions sum to one as well
    assert float(target_prob.sum()) == pytest.approx(1.0, abs=1e-4)


# --------------------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------------------

def test_perbase_figures(tmp_path):
    from criscas import figures as perbase_figures
    rng = np.random.default_rng(0)
    n = 200
    pred_df = pd.DataFrame({
        'id': [f'seq_{i // 4}' for i in range(n)],
        'base_pos': rng.integers(0, SEQ_LEN, n),
        'model_name': 'ABEmax',
        'prob_score_class1': rng.random(n),
        'true_efficiency': rng.random(n),
    })
    pred_df['true_class'] = (pred_df['true_efficiency'] > 0.5).astype(int)
    for i in range(SEQ_LEN):
        pred_df[f'Attn{i}'] = rng.random(n)

    auc_df = perbase_figures.plot_roc_curves({'ABEmax': pred_df}, fig_dir=str(tmp_path))
    assert (tmp_path / 'perbase_roc_curves.pdf').is_file()
    assert len(auc_df) == 1

    perbase_figures.plot_pr_curves({'ABEmax': pred_df}, fig_dir=str(tmp_path))
    assert (tmp_path / 'perbase_pr_curves.pdf').is_file()

    corr_df = perbase_figures.plot_efficiency_scatter(pred_df, fig_dir=str(tmp_path))
    assert (tmp_path / 'perbase_efficiency_scatter.pdf').is_file()
    assert len(corr_df) == 1

    prof = perbase_figures.plot_position_profile(pred_df, fig_dir=str(tmp_path))
    assert (tmp_path / 'perbase_position_profile.pdf').is_file()
    assert len(prof)

    mat = perbase_figures.plot_attention_matrix(pred_df, 'ABEmax', fig_dir=str(tmp_path))
    assert mat.shape[1] == SEQ_LEN


def test_perbase_figures_editor_colors_are_fixed():
    """a figure must not repaint an editor depending on how many are plotted"""
    from criscas import figures as perbase_figures
    from haplotype import figures as bystander_figures
    assert perbase_figures.EDITOR_COLORS == bystander_figures.EDITOR_COLORS
    assert perbase_figures.order_editors(['Target-AID', 'ABEmax']) == ['ABEmax', 'Target-AID']
    # unknown editors are appended, known ones keep their slot
    assert perbase_figures.order_editors(['zzz', 'ABEmax']) == ['ABEmax', 'zzz']


def test_bystander_figures(tmp_path):
    from haplotype import figures as bystander_figures
    df = make_haplotype_data(n_seqs=12)
    rng = np.random.default_rng(0)
    # a noisy version of the reference proportions stands in for the predictions
    df['pred_score'] = np.clip(df['true_score'] + rng.normal(0, 0.03, len(df)), 0, 1)
    df['model_name'] = 'ABEmax'

    corr_df = bystander_figures.plot_haplotype_scatter(df, fig_dir=str(tmp_path))
    assert (tmp_path / 'bystander_haplotype_scatter.pdf').is_file()
    assert corr_df.loc[0, 'pearson_corr'] > 0.8

    prof = bystander_figures.plot_perposition_conversion(df, ('A', 'G'),
                                                         fig_dir=str(tmp_path),
                                                         editor='ABEmax')
    assert (tmp_path / 'bystander_perposition_conversion.pdf').is_file()
    assert {'true_conversion', 'pred_conversion'}.issubset(prof.columns)
    assert len(prof) == SEQ_LEN

    agreement = bystander_figures.plot_haplotype_rank_agreement(df, fig_dir=str(tmp_path))
    assert not agreement.empty
    assert agreement['agreement'].between(0, 1).all()
