"""training/validation workflow of the `BE-DICT (bystander)` model

The workflow reproduces the setup of the models shipped under ``trained_models/bystander``:
an encoder-decoder transformer (embedding dim 128, 8 narrow attention heads, 1 unit,
dropout 0.25) trained for 150 epochs with Adam (weight decay 1e-4).

The model scores a haplotype (i.e. an outcome sequence) by summing the per-position
log-probabilities of the observed conversion states over the positions harboring the
target nucleotide. The training objective is the Kullback-Leibler divergence between the
predicted distribution over the haplotypes of a target sequence and the observed
(renormalized) outcome proportions.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .model import HaplotypeEncoderDecoder, Encoder, Decoder
from .dataset import HaplotypeDataTensor, hap_collate
from .data_partition import get_partitions, generate_partition_datatensor, dump_data_partitions
from .hyperparam import (get_default_hyperparam_config, build_config_map, dump_config,
                         get_saved_config)
from .utilities import ReaderWriter, get_device, build_predictions_df


def _safe_corr(pred_score, ref_score, method='pearson'):
    """correlation that returns ``numpy.nan`` instead of failing on degenerate input

    :func:`haplotype.utilities.compute_pearson_corr` (and its spearman counterpart) return
    the full scipy result object and are undefined for constant input, which happens on
    the first epochs and on tiny partitions.
    """
    from scipy import stats
    pred_score = np.asarray(pred_score, dtype=np.float64)
    ref_score = np.asarray(ref_score, dtype=np.float64)
    if (len(pred_score) < 2 or np.all(pred_score == pred_score[0])
            or np.all(ref_score == ref_score[0])):
        return np.nan
    if method == 'pearson':
        return float(stats.pearsonr(pred_score, ref_score)[0])
    return float(stats.spearmanr(pred_score, ref_score)[0])


def create_directory(path):
    """create (if needed) and return `path` (relative to the current working directory)"""
    path = os.path.abspath(path)
    os.makedirs(path, exist_ok=True)
    return path


def build_haplotype_model(mconfig, options):
    """instantiate :class:`HaplotypeEncoderDecoder` from a saved/generated config"""
    model_config = mconfig['model_config']
    inp_seqlen = options.get('inp_seqlen')
    outp_seqlen = options.get('outp_seqlen')
    encoder = Encoder(model_config.embed_dim,
                      num_nucleotides=4,
                      seq_length=inp_seqlen,
                      num_attn_heads=model_config.num_attn_heads,
                      mlp_embed_factor=model_config.mlp_embed_factor,
                      nonlin_func=model_config.nonlin_func,
                      pdropout=model_config.p_dropout,
                      num_encoder_units=model_config.num_transformer_units,
                      pooling_mode='attn',
                      multihead_type=model_config.multihead_type)
    decoder = Decoder(model_config.embed_dim,
                      num_nucleotides=4,
                      seq_length=outp_seqlen,
                      num_attn_heads=model_config.num_attn_heads,
                      mlp_embed_factor=model_config.mlp_embed_factor,
                      nonlin_func=model_config.nonlin_func,
                      pdropout=model_config.p_dropout,
                      num_decoder_units=model_config.num_transformer_units,
                      pooling_mode='attn',
                      multihead_type=model_config.multihead_type,
                      num_classes=2)
    return HaplotypeEncoderDecoder(encoder, decoder)


def compute_haplotype_logprob(model, dexample, device, normalize_pred=True):
    """score every haplotype of one target sequence

    Args:
        model: instance of :class:`HaplotypeEncoderDecoder`
        dexample: one item of :class:`haplotype.dataset.HaplotypeDataTensor`
        device: torch device
        normalize_pred: boolean, renormalize the haplotype log-probabilities over the
                        haplotype set of the target sequence. This matches the
                        renormalization applied to the observed outcome proportions by
                        :meth:`HaplotypeSeqProcessor.renormalize_outcome_prop`.

    Returns:
        tuple, ``(haplotype_logprob, target_prob)`` where `haplotype_logprob` is a tensor
        of shape (num_haplotypes,) and `target_prob` is ``None`` when the outcome
        proportions are not available
    """
    (Xinp_enc, Xinp_dec, mask_enc, mask_dec, mask_encdec, mask_targetbase_enc,
     target_conv_onehot, target_prob, indx, seqid, outpseqs_lst) = dexample

    num_haplotypes = Xinp_dec.shape[0]
    hap_seqlen = Xinp_dec.shape[1]

    Xinp_enc = Xinp_enc.to(device)
    Xinp_dec = Xinp_dec.to(device)
    mask_dec = mask_dec.to(device)
    mask_targetbase_enc = mask_targetbase_enc.to(device)
    target_conv_onehot = target_conv_onehot.to(device)
    if target_prob is not None:
        target_prob = target_prob.to(device)

    pred_logprob, __, __ = model(Xinp_enc, Xinp_dec, mask_enc, mask_dec, mask_encdec)
    # (num_haplotypes, seqlen, 1, 2) x (num_haplotypes, seqlen, 2, 1)
    # selects the log-probability of the observed conversion state at every position
    out = pred_logprob.unsqueeze(2).matmul(target_conv_onehot.unsqueeze(3).type(torch.float32))
    out = out.reshape(num_haplotypes, hap_seqlen)
    # only the positions harboring the target nucleotide are editable
    tindx = torch.where(mask_targetbase_enc.type(torch.bool))
    haplotype_logprob = out[tindx].view(num_haplotypes, -1).sum(axis=1)

    if normalize_pred:
        haplotype_logprob = haplotype_logprob - torch.logsumexp(haplotype_logprob, dim=0)

    return haplotype_logprob, target_prob, seqid, outpseqs_lst


def run_haplotype_trf(datatensor_partition, dsettypes, config, options, wrk_dir,
                      state_dict_dir=None, to_gpu=True, gpu_index=0, normalize_pred=True):
    """train (and/or evaluate) the bystander model on one data partition

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
        normalize_pred: boolean, see :func:`compute_haplotype_logprob`

    Returns:
        dict, ``{dsettype: dict}`` best correlation report reached per dataset type
    """
    device = get_device(to_gpu, gpu_index)
    fdtype = options.get('fdtype', torch.float32)
    num_epochs = options.get('num_epochs', 150)
    model_name = options.get('model_name', 'HaplotypeTransformer')
    dataloader_config = config['dataloader_config']

    data_loaders = {}
    epoch_loss_avgbatch = {}
    flog_out = {}
    for dsettype in dsettypes:
        if dsettype not in datatensor_partition:
            continue
        data_loaders[dsettype] = DataLoader(datatensor_partition[dsettype],
                                            batch_size=dataloader_config['batch_size'],
                                            shuffle=(dsettype == 'train'),
                                            num_workers=dataloader_config.get('num_workers', 0),
                                            collate_fn=hap_collate,
                                            sampler=None)
        epoch_loss_avgbatch[dsettype] = []
        flog_out[dsettype] = os.path.join(wrk_dir, f'{dsettype}.log') if wrk_dir else None

    model = build_haplotype_model(config, options)
    model.type(fdtype).to(device)

    if state_dict_dir:
        model.load_state_dict(torch.load(os.path.join(state_dict_dir, f'{model_name}.pkl'),
                                         map_location=device))

    # the model outputs log-probabilities, hence KLDivLoss is fed directly
    loss_func = nn.KLDivLoss(reduction='batchmean')

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

    best_score = {dsettype: None for dsettype in data_loaders}
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
            pred_score = []
            ref_score = []
            seqs_ids_lst = []
            outpseqs_ids_lst = []

            for sbatch in data_loaders[dsettype]:
                batch_loss = 0.
                num_examples = 0
                with torch.set_grad_enabled(dsettype == 'train'):
                    for dexample in sbatch:
                        hap_logprob, target_prob, seqid, outpseqs_lst = \
                            compute_haplotype_logprob(model, dexample, device,
                                                      normalize_pred=normalize_pred)
                        if target_prob is None:
                            raise ValueError('outcome proportions are required for training; '
                                             'pass `outcome_col` when preparing the data')
                        # one KL divergence per target sequence
                        loss = loss_func(hap_logprob.view(1, -1), target_prob.view(1, -1))
                        batch_loss = batch_loss + loss
                        num_examples += 1

                        pred_score.extend(torch.exp(hap_logprob.detach().cpu()).view(-1).tolist())
                        ref_score.extend(target_prob.detach().cpu().view(-1).tolist())
                        seqs_ids_lst.extend([seqid] * len(outpseqs_lst))
                        outpseqs_ids_lst.extend(outpseqs_lst)

                    batch_loss = batch_loss / max(num_examples, 1)

                    if dsettype == 'train':
                        optimizer.zero_grad()
                        batch_loss.backward()
                        optimizer.step()
                        cyc_scheduler.step()

                epoch_loss += batch_loss.item()
                num_batches += 1

            avg_loss = epoch_loss / max(num_batches, 1)
            epoch_loss_avgbatch[dsettype].append(avg_loss)

            modelscore = {'epoch': epoch + 1,
                          'loss': avg_loss,
                          'pearson_corr': _safe_corr(pred_score, ref_score, 'pearson'),
                          'spearman_corr': _safe_corr(pred_score, ref_score, 'spearman'),
                          'num_haplotypes': len(pred_score)}
            report = ('-' * 30 + '\n'
                      + 'epoch: {}\n'.format(modelscore['epoch'])
                      + 'average batch loss: {:.6f}\n'.format(modelscore['loss'])
                      + 'Pearson correlation: {:.4f}\n'.format(modelscore['pearson_corr'])
                      + 'Spearman correlation: {:.4f}\n'.format(modelscore['spearman_corr'])
                      + 'number of haplotypes: {}\n'.format(modelscore['num_haplotypes'])
                      + '-' * 30 + '\n')
            if flog_out[dsettype]:
                ReaderWriter.write_log(report, flog_out[dsettype])
            else:
                print(report)

            if _is_better_hap_score(modelscore, best_score[dsettype]):
                best_score[dsettype] = modelscore
                if dsettype == sel_dsettype and wrk_dir:
                    m_state_dict_dir = create_directory(os.path.join(wrk_dir, 'model_statedict'))
                    torch.save(model.state_dict(),
                               os.path.join(m_state_dict_dir, f'{model_name}.pkl'))
                if wrk_dir:
                    predictions_df = build_predictions_df(seqs_ids_lst, outpseqs_ids_lst,
                                                          ref_score, pred_score)
                    predictions_df.to_csv(os.path.join(wrk_dir, f'predictions_{dsettype}.csv'),
                                          index=False)

    if wrk_dir:
        _plot_loss(epoch_loss_avgbatch, wrk_dir)
        ReaderWriter.dump_data(best_score, os.path.join(wrk_dir, 'best_score.pkl'))
        dump_config(config, options, create_directory(os.path.join(wrk_dir, 'config')))

    return best_score


def _is_better_hap_score(modelscore, best):
    """selection criterion of the best epoch: Pearson correlation, then the loss"""
    if best is None:
        return True
    new_corr, best_corr = modelscore['pearson_corr'], best['pearson_corr']
    if np.isnan(new_corr) and np.isnan(best_corr):
        return modelscore['loss'] < best['loss']
    if np.isnan(best_corr):
        return True
    if np.isnan(new_corr):
        return False
    if new_corr != best_corr:
        return new_corr > best_corr
    return modelscore['loss'] < best['loss']


def _plot_loss(epoch_loss_avgbatch, wrk_dir, fname='loss'):
    from matplotlib import pyplot as plt
    plt.figure(figsize=(9, 6))
    for dsettype, loss in epoch_loss_avgbatch.items():
        if not len(loss):
            continue
        plt.plot(np.arange(1, len(loss) + 1), loss, label=dsettype)
    plt.xlabel('epoch')
    plt.ylabel('average batch loss (KL divergence)')
    plt.legend()
    plt.savefig(os.path.join(wrk_dir, f'{fname}.pdf'), bbox_inches='tight')
    plt.close()


def train_val_run(datatensor_partitions, config_map, train_val_dir, run_gpu_map=None,
                  num_epochs=None, target_runs=None, normalize_pred=True):
    """train the bystander model on every data partition (i.e. run)"""
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
        scores[run_num] = run_haplotype_trf(datatensor_partitions[run_num],
                                            dsettypes,
                                            mconfig,
                                            run_options,
                                            wrk_dir,
                                            to_gpu=run_options.get('to_gpu', True),
                                            gpu_index=gpu_index,
                                            normalize_pred=normalize_pred)
    return scores


def test_run(datatensor_partitions, train_val_dir, test_dir, run_gpu_map=None,
             target_runs=None, normalize_pred=True):
    """evaluate the trained bystander models on their held out test partition"""
    scores = {}
    for run_num in datatensor_partitions:
        if target_runs is not None and run_num not in target_runs:
            continue
        train_dir = os.path.join(train_val_dir, f'run_{run_num}')
        state_dict_dir = os.path.join(train_dir, 'model_statedict')
        mconfig, options = get_saved_config(os.path.join(train_dir, 'config'))
        options = dict(options)
        options['num_epochs'] = 1
        options['train_flag'] = False
        wrk_dir = create_directory(os.path.join(test_dir, f'run_{run_num}'))
        print(f'--- testing run_{run_num} | wrk_dir: {wrk_dir} ---')
        gpu_index = run_gpu_map.get(run_num, 0) if run_gpu_map else 0
        scores[run_num] = run_haplotype_trf(datatensor_partitions[run_num],
                                            ['test'],
                                            mconfig,
                                            options,
                                            wrk_dir,
                                            state_dict_dir=state_dict_dir,
                                            to_gpu=options.get('to_gpu', True),
                                            gpu_index=gpu_index,
                                            normalize_pred=normalize_pred)
    return scores


def train_bystander_model(data_df, base_editor, output_dir, seq_processor, seqconfig,
                          inpseq_cols=('seq_id', 'Inp_seq'), outpseq_col='Outp_seq',
                          outcome_col='true_score', hyperparam_config=None, num_runs=5,
                          num_epochs=None, renormalize=True, val_set_portion=0.1,
                          random_state=42, to_gpu=True, run_test=True,
                          normalize_pred=True):
    """end-to-end training of the `BE-DICT (bystander)` models of one base editor

    The produced folder structure matches the one of the shipped models, so that the
    resulting models can be used with
    :class:`haplotype.predict_model.BEDICT_HaplotypeModel`::

        {output_dir}/{base_editor}/train_val/run_{num}/model_statedict/HaplotypeTransformer.pkl
        {output_dir}/{base_editor}/train_val/run_{num}/config/{mconfig,exp_options}.pkl

    Args:
        data_df: pandas.DataFrame with the input sequences, their outcome sequences and
                 the observed outcome proportions
        base_editor: string, name of the base editor (i.e. `ABEmax`)
        output_dir: string, directory where the models are written
        seq_processor: instance of
            :class:`haplotype.data_preprocess.HaplotypeSeqProcessor`
        seqconfig: instance of :class:`haplotype.data_preprocess.SeqProcessConfig`,
                   sequence config used for the tensor generation
        inpseq_cols: sequence of strings, `sequence id` and `input sequence` column names
        outpseq_col: string, column name of the outcome (haplotype) sequence
        outcome_col: string, column name of the observed outcome proportion
        hyperparam_config: :class:`haplotype.hyperparam.Haplotype_Trf_HyperparamConfig`
                           or None
        num_runs: int, number of data partitions (5 in the paper)
        num_epochs: int or None, overrides the number of epochs of the config
        renormalize: boolean, renormalize the observed outcome proportions
        val_set_portion: float, portion of the training set used for validation
        random_state: int, seed of the data partitioning
        to_gpu: boolean, use a gpu when available
        run_test: boolean, evaluate the trained models on the held out test partitions
        normalize_pred: boolean, see :func:`compute_haplotype_logprob`

    Returns:
        tuple, ``(train_val_scores, test_scores, proc_df)``
    """
    from .predict_model import BEDICT_HaplotypeModel

    if hyperparam_config is None:
        hyperparam_config = get_default_hyperparam_config()
    inpseq_cols = list(inpseq_cols)

    print(f'--- processing data of base editor: {base_editor} ---')
    bedict_haplo = BEDICT_HaplotypeModel(seq_processor, seqconfig, get_device(to_gpu))
    proc_df = bedict_haplo._process_df(data_df, inpseq_cols, outpseq_col, outcome_col,
                                       renormalize=renormalize)
    dtensor = HaplotypeDataTensor(seqconfig)
    dtensor.generate_tensor_from_df(proc_df, seq_processor.conversion_nucl, outcome_col)

    data_partitions = get_partitions(len(dtensor), num_splits=num_runs,
                                     val_set_portion=val_set_portion,
                                     random_state=random_state)
    datatensor_partitions = generate_partition_datatensor(dtensor, data_partitions)

    editor_dir = create_directory(os.path.join(output_dir, base_editor))
    dump_data_partitions(data_partitions, editor_dir)

    experiment_desc = f'{base_editor}_haplotype'
    config_map = build_config_map(hyperparam_config, experiment_desc,
                                  model_name='HaplotypeTransformer',
                                  inp_seqlen=seqconfig.seq_len,
                                  outp_seqlen=seqconfig.ewindow_end + 1,
                                  to_gpu=to_gpu)
    train_val_dir = create_directory(os.path.join(editor_dir, 'train_val'))
    train_val_scores = train_val_run(datatensor_partitions, config_map, train_val_dir,
                                     num_epochs=num_epochs, normalize_pred=normalize_pred)

    test_scores = None
    if run_test:
        test_dir = create_directory(os.path.join(editor_dir, 'test'))
        test_scores = test_run(datatensor_partitions, train_val_dir, test_dir,
                               normalize_pred=normalize_pred)

    return train_val_scores, test_scores, proc_df


def summarize_scores(scores, dsettype='test'):
    """collect the scores of the different runs into one data frame"""
    rows = []
    for run_num, run_scores in scores.items():
        if dsettype not in run_scores or run_scores[dsettype] is None:
            continue
        mscore = dict(run_scores[dsettype])
        mscore.update({'run_num': run_num, 'dsettype': dsettype})
        rows.append(mscore)
    return pd.DataFrame(rows)
