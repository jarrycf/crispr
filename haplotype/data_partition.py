import os
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from .dataset import PartitionDataTensor
from .utilities import ReaderWriter


def get_partitions(num_sequences, num_splits=5, val_set_portion=0.1, random_state=42,
                   report=True):
    """split the *input* sequences into `num_splits` train/validation/test partitions

    The split is done at the level of the input (target) sequences, so that all the
    haplotypes (i.e. outcome sequences) of one input sequence stay within the same
    partition and no outcome of a test sequence is seen during training.

    Args:
        num_sequences: int, number of input sequences (i.e. ``len(datatensor)``)
        num_splits: int, number of train/validation/test partitions (i.e. runs)
        val_set_portion: float, portion of the training set held out for validation
        random_state: int, seed of the random number generator
        report: boolean, print the size of every partition

    Returns:
        dict, ``{run_num: {'train': ids, 'validation': ids, 'test': ids}}``
    """
    sequence_ids = np.arange(num_sequences)
    kf = KFold(n_splits=num_splits, random_state=random_state, shuffle=True)
    data_partitions = {}
    for run_num, (train_index, test_index) in enumerate(kf.split(sequence_ids)):
        # a validation set can only be carved out if the training fold is big enough
        can_split_val = val_set_portion and int(np.floor(len(train_index) * val_set_portion)) >= 1
        if val_set_portion and not can_split_val:
            print(f'run_{run_num}: training fold holds {len(train_index)} sequence/s, '
                  'which is too few to hold out a validation set; '
                  'the checkpoint will be selected on the training set instead.')
        if can_split_val:
            train_index, val_index = train_test_split(train_index,
                                                      test_size=val_set_portion,
                                                      random_state=random_state,
                                                      shuffle=True)
        else:
            val_index = np.array([], dtype=np.int64)
        data_partitions[run_num] = {'train': train_index,
                                    'validation': val_index,
                                    'test': test_index}
    if report:
        for run_num in data_partitions:
            sizes = {dsettype: len(ids) for dsettype, ids in data_partitions[run_num].items()}
            print(f'run_{run_num}: {sizes}')
    validate_partitions(data_partitions, sequence_ids)
    return data_partitions


def validate_partitions(data_partitions, sequence_ids):
    """sanity checks: partitions do not leak sequences and cover the whole dataset"""
    num_sequences = len(sequence_ids)
    seen_test_ids = set()
    for run_num, partition in data_partitions.items():
        train_ids = set(np.asarray(partition['train']).tolist())
        val_ids = set(np.asarray(partition['validation']).tolist())
        test_ids = set(np.asarray(partition['test']).tolist())
        assert not (train_ids & val_ids), f'run_{run_num}: train and validation sets overlap!'
        assert not (train_ids & test_ids), f'run_{run_num}: train and test sets overlap!'
        assert not (val_ids & test_ids), f'run_{run_num}: validation and test sets overlap!'
        assert len(train_ids | val_ids | test_ids) == num_sequences, \
            f'run_{run_num}: partitions do not cover all sequences!'
        seen_test_ids |= test_ids
    assert len(seen_test_ids) == num_sequences, 'test folds do not cover all sequences!'
    print('partitions validated!')


def generate_partition_datatensor(datatensor, data_partitions):
    """wrap a datatensor into :class:`PartitionDataTensor` for every run and dataset type

    Args:
        datatensor: instance of :class:`haplotype.dataset.HaplotypeDataTensor`
        data_partitions: dict, output of :func:`get_partitions`
    """
    datatensor_partitions = {}
    for run_num, partition in data_partitions.items():
        datatensor_partitions[run_num] = {}
        for dsettype, target_ids in partition.items():
            if not len(target_ids):
                continue
            datatensor_partitions[run_num][dsettype] = PartitionDataTensor(datatensor,
                                                                           target_ids,
                                                                           dsettype,
                                                                           run_num)
    return datatensor_partitions


def dump_data_partitions(data_partitions, wrk_dir, fname='data_partitions.pkl'):
    os.makedirs(wrk_dir, exist_ok=True)
    ReaderWriter.dump_data(data_partitions, os.path.join(wrk_dir, fname))


def read_data_partitions(wrk_dir, fname='data_partitions.pkl'):
    return ReaderWriter.read_data(os.path.join(wrk_dir, fname))
