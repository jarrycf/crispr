"""training/validation workflow of the `BE-DICT (perbase)` model

The workflow reproduces the setup of the models shipped under ``trained_models/perbase``:
a 2-unit transformer encoder (embedding dim 64, 8 attention heads) trained for 500 epochs
with Adam (weight decay 1e-4) and a negative log-likelihood loss computed on the positions
harboring the target nucleotide.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from .model import Categ_CrisCasTransformer
from .dataset import create_datatensor
from .data_preprocess import generate_perbase_df
from .data_partition import (get_stratified_partitions, generate_partition_datatensor,
                             dump_data_partitions)
from .hyperparam import (get_default_hyperparam_config, build_config_map, dump_config,
                         get_saved_config)
from .evaluation import perfmetric_report, plot_loss
from .utilities import get_device, dump_dict_content, build_probscores_df


def create_directory(path):
    """create (if needed) and return `path`

    Note:
        unlike :func:`criscas.utilities.create_directory`, a relative path is resolved
        against the current working directory and not against the package directory.
    """
    path = os.path.abspath(path)
    os.makedirs(path, exist_ok=True)
    return path


def build_perbase_model(mconfig, options):
    """instantiate :class:`Categ_CrisCasTransformer` from a saved/!generated config

    Args:
        mconfig: dict, model config (i.e. `mconfig.pkl`)
        options: dict, experiment options (i.e. `exp_options.pkl`)
    """
    model_config = mconfig['model_config']
    seq_length = options.get('seq_length', 20)
    num_classes = options.get('num_classes', 2)
    model = Categ_CrisCasTransformer(input_size=model_config.embed_dim,
                                     num_nucleotides=options.get('num_nucleotides', 4),
                                     seq_length=seq_length,
                                     num_attn_heads=model_config.num_attn_heads,
                                     mlp_embed_factor=model_config.mlp_embed_factor,
                                     nonlin_func=model_config.nonlin_func,
                                     pdropout=model_config.p_dropout,
                                     num_transformer_units=model_config.num_transformer_units,
                                     pooling_mode=options.get('pooling_mode', 'attn'),
                                     num_classes=num_classes,
                                     per_base=options.get('per_base', True))
    return model


def _build_dataloaders(datatensor_partition, dsettypes, config):
    dataloader_config = config['dataloader_config']
    data_loaders = {}
    epoch_loss_avgbatch = {}
    flog_out = {}
    for dsettype in dsettypes:
        if dsettype not in datatensor_partition:
            continue
        shuffle = (dsettype == 'train')
        data_loaders[dsettype] = DataLoader(datatensor_partition[dsettype],
                                            batch_size=dataloader_config['batch_size'],
                                            shuffle=shuffle,
                                            num_workers=dataloader_config.get('num_workers', 0),
                                            sampler=None)
        epoch_loss_avgbatch[dsettype] = []
        flog_out[dsettype] = None
    return data_loaders, epoch_loss_avgbatch, flog_out


def run_perbase_trf(datatensor_partition, dsettypes, config, options, wrk_dir,
                    state_dict_dir=None, to_gpu=True, gpu_index=0):
    """train (and/or evaluate) the per-base model on one data partition

    Args:
        datatensor_partition: dict, ``{dsettype: PartitionDataTensor}`` for one run
        dsettypes: list of strings, dataset types to process (i.e. ['train', 'validation'])
        config: dict, model config (i.e. `mconfig`)
        options: dict, experiment options (i.e. `exp_options`)
        wrk_dir: string or None, directory where the model, config, logs, predictions and
                 loss curves are written. ``None`` runs without dumping anything.
        state_dict_dir: string or None, directory of a model state_dict to warm start from
        to_gpu: boolean, use a gpu when available
        gpu_index: int, index of the gpu to use

    Returns:
        dict, ``{dsettype: ModelScore}`` best score reached per dataset type
    """
    pid = "{}".format(os.getpid())
    device = get_device(to_gpu, gpu_index)
    fdtype = options.get('fdtype', torch.float32)
    num_epochs = options.get('num_epochs', 500)
    per_base = options.get('per_base', True)
    model_name = options.get('model_name', 'Transformer')

    data_loaders, epoch_loss_avgbatch, flog_out = _build_dataloaders(datatensor_partition,
                                                                     dsettypes, config)
    if wrk_dir:
        for dsettype in data_loaders:
            flog_out[dsettype] = os.path.join(wrk_dir, f'{dsettype}.log')

    model = build_perbase_model(config, options)
    model.type(fdtype).to(device)

    if state_dict_dir:  # warm start from a trained model
        model.load_state_dict(torch.load(os.path.join(state_dict_dir, f'{model_name}.pkl'),
                                         map_location=device))

    loss_func = nn.NLLLoss(reduction='mean')

    if 'train' in data_loaders:
        weight_decay = options.get('weight_decay', 1e-4)
        base_lr = options.get('base_lr', 3e-4)
        max_lr_mult = options.get('max_lr_mult', 5)
        optimizer = torch.optim.Adam(model.parameters(), weight_decay=weight_decay,
                                     lr=base_lr)
        num_iter = max(len(data_loaders['train']), 1)
        c_step_size = int(np.ceil(5 * num_iter))
        cyc_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer,
                                                          base_lr,
                                                          max_lr_mult * base_lr,
                                                          step_size_up=c_step_size,
                                                          mode='triangular',
                                                          cycle_momentum=False)

    # tracks the best score reached on every dataset type
    best_score = {dsettype: None for dsettype in data_loaders}
    # the checkpoint is selected on the validation set when available
    sel_dsettype = 'validation' if 'validation' in data_loaders else 'train'

    for epoch in range(num_epochs):
        for dsettype in dsettypes:
            if dsettype not in data_loaders:
                continue
            if dsettype == 'train':
                model.train()
            else:
                model.eval()

            epoch_loss = 0.
            num_batches = 0
            pred_class = []
            ref_class = []
            prob_scores = []
            seqs_ids_lst = []
            base_pos_lst = []

            for samples_batch in data_loaders[dsettype]:
                if per_base:
                    X_batch, y_score, y_categ, mask, b_seqs_indx, b_seqs_id = samples_batch
                    mask = mask.to(device)
                else:
                    X_batch, y_score, y_categ, b_seqs_indx, b_seqs_id = samples_batch
                    mask = None
                X_batch = X_batch.to(device)
                y_categ = y_categ.type(torch.int64).to(device)

                with torch.set_grad_enabled(dsettype == 'train'):
                    logsoftmax_scores, __, __ = model(X_batch)
                    if per_base:
                        # only the positions harboring the target nucleotide are editable,
                        # hence only those contribute to the loss and to the evaluation
                        tindx = torch.where(mask.type(torch.bool))
                        t_logsoftmax = logsoftmax_scores[tindx]
                        t_ref = y_categ[tindx]
                    else:
                        t_logsoftmax = logsoftmax_scores
                        t_ref = y_categ

                    loss = loss_func(t_logsoftmax, t_ref)

                    if dsettype == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        cyc_scheduler.step()

                    __, y_pred_clss = torch.max(t_logsoftmax, -1)
                    pred_class.extend(y_pred_clss.view(-1).tolist())
                    ref_class.extend(t_ref.view(-1).tolist())
                    prob_scores.append(torch.exp(t_logsoftmax.detach().cpu()).numpy())
                    if per_base:
                        seqs_ids_lst.extend([b_seqs_id[i] for i in tindx[0].tolist()])
                        base_pos_lst.extend(tindx[1].tolist())
                    else:
                        seqs_ids_lst.extend(list(b_seqs_id))

                epoch_loss += loss.item()
                num_batches += 1

            epoch_loss_avgbatch[dsettype].append(epoch_loss / max(num_batches, 1))
            prob_scores_arr = np.concatenate(prob_scores, axis=0)
            modelscore = perfmetric_report(pred_class, ref_class, prob_scores_arr[:, 1],
                                           epoch + 1, flog_out[dsettype])

            # keep track of the best performing epoch
            if _is_better_score(modelscore, best_score[dsettype]):
                best_score[dsettype] = modelscore
                if dsettype == sel_dsettype and wrk_dir:
                    # checkpoint + predictions of the best epoch so far
                    m_state_dict_dir = create_directory(os.path.join(wrk_dir, 'model_statedict'))
                    torch.save(model.state_dict(),
                               os.path.join(m_state_dict_dir, f'{model_name}.pkl'))
                if wrk_dir:
                    predictions_df = build_probscores_df(seqs_ids_lst, prob_scores_arr,
                                                         base_pos_lst if per_base else None)
                    predictions_df['true_class'] = ref_class
                    predictions_df['pred_class'] = pred_class
                    predictions_df.to_csv(os.path.join(wrk_dir,
                                                       f'predictions_{dsettype}.csv'),
                                          index=False)

    if wrk_dir:
        plot_loss(epoch_loss_avgbatch, wrk_dir, fname=f'loss_{pid}')
        dump_dict_content(best_score, list(best_score.keys()), 'score', wrk_dir)
        dump_config(config, options, create_directory(os.path.join(wrk_dir, 'config')))

    return best_score


def _is_better_score(modelscore, best):
    """selection criterion of the best epoch: AUC first, then AUPR, then accuracy"""
    if best is None:
        return True
    for attr in ('auc', 'aupr', 'accuracy'):
        new_val = getattr(modelscore, attr)
        best_val = getattr(best, attr)
        if np.isnan(new_val) and np.isnan(best_val):
            continue
        if np.isnan(best_val):
            return True
        if np.isnan(new_val):
            return False
        if new_val != best_val:
            return new_val > best_val
    return False


def train_val_run(datatensor_partitions, config_map, train_val_dir, run_gpu_map=None,
                  num_epochs=None, target_runs=None):
    """train the model on every data partition (i.e. run)

    Args:
        datatensor_partitions: dict, output of
            :func:`criscas.data_partition.generate_partition_datatensor`
        config_map: tuple, ``(mconfig, options)`` as built by
            :func:`criscas.hyperparam.build_config_map`
        train_val_dir: string, directory where ``run_{num}`` folders are created
        run_gpu_map: dict or None, ``{run_num: gpu_index}``
        num_epochs: int or None, overrides the number of epochs of the config
        target_runs: list or None, restrict the training to these run numbers

    Returns:
        dict, ``{run_num: {dsettype: ModelScore}}``
    """
    dsettypes = ['train', 'validation']
    mconfig, options = config_map
    options = dict(options)
    if num_epochs is not None:
        options['num_epochs'] = num_epochs
    scores = {}
    for run_num in datatensor_partitions:
        if target_runs is not None and run_num not in target_runs:
            continue
        run_options = dict(options)
        run_options['run_num'] = run_num
        wrk_dir = create_directory(os.path.join(train_val_dir, f'run_{run_num}'))
        print(f'--- training run_{run_num} | wrk_dir: {wrk_dir} ---')
        gpu_index = run_gpu_map.get(run_num, 0) if run_gpu_map else 0
        scores[run_num] = run_perbase_trf(datatensor_partitions[run_num],
                                          dsettypes,
                                          mconfig,
                                          run_options,
                                          wrk_dir,
                                          to_gpu=run_options.get('to_gpu', True),
                                          gpu_index=gpu_index)
    return scores


def test_run(datatensor_partitions, train_val_dir, test_dir, run_gpu_map=None,
             target_runs=None):
    """evaluate the trained models of every run on their held out test partition

    Args:
        datatensor_partitions: dict, output of
            :func:`criscas.data_partition.generate_partition_datatensor`
        train_val_dir: string, directory holding the ``run_{num}`` trained models
        test_dir: string, directory where the test results are written
        run_gpu_map: dict or None, ``{run_num: gpu_index}``
        target_runs: list or None, restrict the evaluation to these run numbers

    Returns:
        dict, ``{run_num: {'test': ModelScore}}``
    """
    scores = {}
    for run_num in datatensor_partitions:
        if target_runs is not None and run_num not in target_runs:
            continue
        train_dir = os.path.join(train_val_dir, f'run_{run_num}')
        state_dict_dir = os.path.join(train_dir, 'model_statedict')
        mconfig, options = get_saved_config(os.path.join(train_dir, 'config'))
        options = dict(options)
        options['num_epochs'] = 1  # single pass over the test partition
        options['train_flag'] = False
        wrk_dir = create_directory(os.path.join(test_dir, f'run_{run_num}'))
        print(f'--- testing run_{run_num} | wrk_dir: {wrk_dir} ---')
        gpu_index = run_gpu_map.get(run_num, 0) if run_gpu_map else 0
        scores[run_num] = run_perbase_trf(datatensor_partitions[run_num],
                                          ['test'],
                                          mconfig,
                                          options,
                                          wrk_dir,
                                          state_dict_dir=state_dict_dir,
                                          to_gpu=options.get('to_gpu', True),
                                          gpu_index=gpu_index)
    return scores


def train_perbase_model(data_df, base_editor, output_dir, target_base=None,
                        hyperparam_config=None, num_runs=5, num_epochs=None,
                        cutoff_score=0.01, normalize_by_counts=False, val_set_portion=0.1,
                        random_state=42, to_gpu=True, run_test=True, per_base=True):
    """end-to-end training of the `BE-DICT (perbase)` models of one base editor

    The produced folder structure matches the one of the shipped models, so that the
    resulting models can be used with :class:`criscas.predict_model.BEDICT_CriscasModel`::

        {output_dir}/{base_editor}/train_val/run_{num}/model_statedict/Transformer.pkl
        {output_dir}/{base_editor}/train_val/run_{num}/config/{mconfig,exp_options}.pkl
        {output_dir}/{base_editor}/test/run_{num}/predictions_test.csv

    Args:
        data_df: pandas.DataFrame, raw screen data with columns [ID, seq, allCounts, V1..V20]
        base_editor: string, name of the base editor (i.e. `ABEmax`)
        output_dir: string, directory where the models are written
        target_base: string or None, targeted nucleotide; inferred from `base_editor`
                     for the editors reported in the paper
        hyperparam_config: :class:`criscas.hyperparam.TrfHyperparamConfig` or None
        num_runs: int, number of data partitions (5 in the paper)
        num_epochs: int or None, overrides the number of epochs of the config
        cutoff_score: float, editing efficiency threshold defining the `edited` class
        normalize_by_counts: boolean, divide `V{i}` by `allCounts` (see
                             :func:`criscas.data_preprocess.generate_perbase_df`)
        val_set_portion: float, portion of the training set used for validation
        random_state: int, seed of the data partitioning
        to_gpu: boolean, use a gpu when available
        run_test: boolean, evaluate the trained models on the held out test partitions
        per_base: boolean, train the per-base model (``False`` trains the sequence
                  level variant)

    Returns:
        tuple, ``(train_val_scores, test_scores, proc_df)``
    """
    if target_base is None:
        target_base = get_target_base(base_editor)
    if hyperparam_config is None:
        hyperparam_config = get_default_hyperparam_config()

    print(f'--- processing data of base editor: {base_editor} (target base: {target_base}) ---')
    proc_df = generate_perbase_df(data_df, target_base, cutoff_score=cutoff_score,
                                  normalize_by_counts=normalize_by_counts)
    dtensor = create_datatensor(proc_df, per_base=per_base, refscore_available=True)

    data_partitions = get_stratified_partitions(proc_df, num_splits=num_runs,
                                                val_set_portion=val_set_portion,
                                                random_state=random_state)
    datatensor_partitions = generate_partition_datatensor(dtensor, data_partitions)

    editor_dir = create_directory(os.path.join(output_dir, base_editor))
    dump_data_partitions(data_partitions, editor_dir)
    proc_df.to_csv(os.path.join(editor_dir, 'processed_data.csv'), index=False)

    experiment_desc = f'{base_editor}_perbase'
    config_map = build_config_map(hyperparam_config, experiment_desc,
                                  model_name='Transformer', per_base=per_base,
                                  to_gpu=to_gpu)
    train_val_dir = create_directory(os.path.join(editor_dir, 'train_val'))
    train_val_scores = train_val_run(datatensor_partitions, config_map, train_val_dir,
                                     num_epochs=num_epochs)

    test_scores = None
    if run_test:
        test_dir = create_directory(os.path.join(editor_dir, 'test'))
        test_scores = test_run(datatensor_partitions, train_val_dir, test_dir)

    return train_val_scores, test_scores, proc_df


def get_target_base(base_editor):
    """targeted nucleotide of the base editors reported in the paper"""
    if base_editor in {'ABEmax', 'ABE8e'}:
        return 'A'
    elif base_editor in {'BE4max', 'Target-AID'}:
        return 'C'
    raise ValueError(f'unknown base editor: {base_editor}; please pass `target_base` explicitly')


def summarize_scores(scores, dsettype='test'):
    """collect the :class:`ModelScore` of the different runs into one data frame"""
    rows = []
    for run_num, run_scores in scores.items():
        if dsettype not in run_scores or run_scores[dsettype] is None:
            continue
        mscore = run_scores[dsettype]
        rows.append({'run_num': run_num,
                     'dsettype': dsettype,
                     'best_epoch': mscore.best_epoch_indx,
                     'accuracy': mscore.accuracy,
                     'macro_f1': mscore.macro_f1,
                     'binary_f1': mscore.binary_f1,
                     'auc': mscore.auc,
                     'aupr': mscore.aupr})
    return pd.DataFrame(rows)
