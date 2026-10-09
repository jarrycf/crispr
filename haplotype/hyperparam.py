import os
import itertools
import torch
from torch import nn
from .utilities import ReaderWriter

class Haplotype_Trf_HyperparamConfig:
    def __init__(self, embed_dim, num_attn_heads, num_transformer_units, 
                p_dropout, nonlin_func, mlp_embed_factor, multihead_type, 
                l2_reg, batch_size, num_epochs):
        self.embed_dim = embed_dim
        self.num_attn_heads = num_attn_heads
        self.num_transformer_units = num_transformer_units
        self.p_dropout = p_dropout
        self.nonlin_func = nonlin_func
        self.mlp_embed_factor = mlp_embed_factor
        self.multihead_type = multihead_type
        self.l2_reg = l2_reg
        self.batch_size = batch_size
        self.num_epochs = num_epochs


    def __repr__(self):
        desc = " embed_dim:{}\n num_attn_heads:{}\n num_transformer_units:{}\n p_dropout:{} \n " \
               "nonlin_func:{} \n mlp_embed_factor:{} \n multihead_type:{} \n" \
               "l2_reg:{} \n batch_size:{} \n num_epochs: {}".format(self.embed_dim,
                                                                     self.num_attn_heads,
                                                                     self.num_transformer_units,
                                                                     self.p_dropout, 
                                                                     self.nonlin_func,
                                                                     self.mlp_embed_factor,
                                                                     self.multihead_type,
                                                                     self.l2_reg, 
                                                                     self.batch_size,
                                                                     self.num_epochs)
        return desc



def get_saved_config(config_dir):
    options = ReaderWriter.read_data(os.path.join(config_dir, 'exp_options.pkl'))
    mconfig = ReaderWriter.read_data(os.path.join(config_dir, 'mconfig.pkl'))
    return mconfig, options


def get_hyperparam_options(embed_dim=(128,), num_attn_heads=(8,), num_transformer_units=(1,),
                           p_dropout=(0.25,), nonlin_func=(nn.ReLU(),), mlp_embed_factor=(2,),
                           multihead_type=('Narrow',), l2_reg=(1e-4,), batch_size=(300,),
                           num_epochs=(150,)):
    """generate the list of :class:`Haplotype_Trf_HyperparamConfig` spanned by the options

    The defaults are the hyperparameters of the trained models shipped with this repo
    (see ``trained_models/bystander/{editor}/train_val/run_{n}/config/mconfig.pkl``).
    """
    hyperparam_space = itertools.product(embed_dim, num_attn_heads, num_transformer_units,
                                         p_dropout, nonlin_func, mlp_embed_factor,
                                         multihead_type, l2_reg, batch_size, num_epochs)
    return [Haplotype_Trf_HyperparamConfig(*hyperparam) for hyperparam in hyperparam_space]


def get_default_hyperparam_config():
    """hyperparameter config reproducing the trained `BE-DICT (bystander)` models"""
    return get_hyperparam_options()[0]


def build_config_map(hyperparam_config, experiment_desc, model_name='HaplotypeTransformer',
                     run_num=-1, fdtype=None, loss_func='klloss', inp_seqlen=20,
                     outp_seqlen=20, to_gpu=True, train_flag=True, num_workers=0):
    """assemble the ``(mconfig, options)`` pair that is serialized next to a trained model

    Args:
        hyperparam_config: instance of :class:`Haplotype_Trf_HyperparamConfig`
        experiment_desc: string, short description of the experiment
        model_name: string, name used for the state_dict file (`HaplotypeTransformer.pkl`)
        run_num: int, run number (i.e. data partition index)
        fdtype: torch dtype used for the float tensors (default ``torch.float32``)
        loss_func: string, loss function identifier
        inp_seqlen: int, length of the encoder (input) sequence
        outp_seqlen: int, length of the decoder (haplotype) sequence
        to_gpu: boolean, move the model to gpu (if available)
        train_flag: boolean, whether the config describes a training run
        num_workers: int, number of dataloader workers
    """
    if fdtype is None:
        fdtype = torch.float32
    dataloader_config = {'batch_size': hyperparam_config.batch_size,
                         'num_workers': num_workers}
    config = {'dataloader_config': dataloader_config,
              'model_config': hyperparam_config,
              'optimizer_config': {}}
    options = {'experiment_desc': experiment_desc,
               'run_num': run_num,
               'model_name': model_name,
               'num_epochs': hyperparam_config.num_epochs,
               'weight_decay': hyperparam_config.l2_reg,
               'fdtype': fdtype,
               'to_gpu': to_gpu,
               'loss_func': loss_func,
               'inp_seqlen': inp_seqlen,
               'outp_seqlen': outp_seqlen,
               'train_flag': train_flag}
    return config, options


def dump_config(config, options, config_dir):
    """serialize ``mconfig``/``exp_options`` in the layout used by :func:`get_saved_config`"""
    os.makedirs(config_dir, exist_ok=True)
    ReaderWriter.dump_data(config, os.path.join(config_dir, 'mconfig.pkl'))
    ReaderWriter.dump_data(options, os.path.join(config_dir, 'exp_options.pkl'))






