import pandas as pd
import numpy as np
from scipy import stats

# function to extend each concatenated sequence into its bases
def get_char(seq):
    """split string int sequence of chars returned in pandas.Series"""
    chars = list(seq)
    return pd.Series(chars)
    
def process_perbase_df(df, target_base):
    """cleans a data frame representing sequences and their edit info obtained from crispr experiment
    
    Args:
        df: pandas.DataFrame
        cutoff_score: float, threshold for creating two categories from edit score
    
    Note:
        assumed columns in the dataframe are: [ID, seq, allCounts, V1, V2, ..., V20]
    """
    target_cols = ['ID', 'seq']
    df = df[target_cols].copy()
    # harmonize sequence string representation to capitalized form
    df['seq'] = df['seq'].str.upper()
    # allocate sequences to train vs. val/test
    df['seq_type'] = 1

    baseseq_df = df['seq'].apply(get_char)
    baseseq_df.columns = [f'B{i}' for  i in range(1, 21)]
    base_mask = (baseseq_df == target_base) * 1
    base_mask.columns = [f'M{i}' for  i in range(1, 21)]
    
    baseseq_letters_df = baseseq_df.copy()
    baseseq_letters_df.columns = [f'L{i}' for  i in range(1, 21)]
    # replace base letters with numbers
    # Note: the explicit cast keeps the columns numeric (i.e. int64) across pandas versions,
    # as `replace` on object columns no longer downcasts the dtype automatically
    baseseq_df = baseseq_df.replace(['A', 'C', 'T', 'G'], [0,1,2,3]).astype(np.int64)
    base_df = pd.concat([base_mask,
                         df,
                         baseseq_letters_df,
                         baseseq_df], axis=1)
    # keep sequences that have target base
    # M1..M20 is mask for target base occurrence
    base_df = base_df[(base_df[[f'M{i}' for  i in range(1, 21)]].sum(axis=1) != 0)].copy()
    base_df.reset_index(inplace=True)


    return base_df

def validate_df(df):
    mask_cols = [f'M{i}' for  i in range(1, 21)]
    print('number of NA:', df.isna().any().sum())
    print('number of sequences without target base:', (df[mask_cols].sum(axis=1) == 0).sum())

def compute_rank(s, eff_scores=None):
    return stats.percentileofscore(eff_scores, s, kind='weak')


def generate_perbase_df(df, target_base, cutoff_score=0.01, normalize_by_counts=False,
                        counts_col='allCounts', overall_agg='max', compute_prank=True,
                        seq_len=20):
    """builds the training data frame (sequences + per-base editing labels) from screen data

    Args:
        df: pandas.DataFrame, raw data of a high-throughput target library screen
        target_base: string, targeted nucleotide (i.e. `A` for ABE and `C` for CBE editors)
        cutoff_score: float, threshold on the editing efficiency above which a base
                      (resp. a sequence) is labeled as `edited`
        normalize_by_counts: boolean, divide the per-base editing counts `V{i}` by the
                             total read counts `allCounts` to obtain proportions.
                             Use ``False`` when `V{i}` already holds proportions.
        counts_col: string, column name of the total read counts
        overall_agg: string, how the per-base scores of a sequence are aggregated into one
                     sequence level efficiency score, one of {`max`, `mean`}
        compute_prank: boolean, additionally compute the percentile rank of the per-base
                       efficiency scores (`ESpr{i}` columns)
        seq_len: int, length of the protospacer sequences

    Returns:
        pandas.DataFrame with the columns consumed by
        :func:`criscas.dataset.create_datatensor` (with ``per_base=True``):

        - ``B{i}``   numeric encoding of the nucleotide at position `i` (A,C,T,G -> 0,1,2,3)
        - ``L{i}``   nucleotide letter at position `i`
        - ``M{i}``   mask flagging the occurrence of `target_base` at position `i`
        - ``ES{i}``  per-base editing efficiency score at position `i`
        - ``ECi{i}`` per-base editing category at position `i` (0/1 w.r.t `cutoff_score`)
        - ``ESpr{i}`` percentile rank of ``ES{i}`` (only if `compute_prank` is ``True``)
        - ``efficiency_score``  sequence level efficiency score
        - ``edited_seq_categ``  sequence level editing category (0/1)

    Note:
        assumed columns in the dataframe are: [ID, seq, allCounts, V1, V2, ..., V20]
        Positions without the target base carry a 0 score/category; they are excluded from
        the loss and from the evaluation through the ``M{i}`` mask.
    """
    assert overall_agg in {'max', 'mean'}, "overall_agg should be in {max, mean}!"
    score_cols = [f'V{i}' for i in range(1, seq_len+1)]
    missing = [col for col in ['ID', 'seq'] + score_cols if col not in df.columns]
    assert not missing, f'missing columns in the passed data frame: {missing}'

    # sequence features (B{i}, L{i}, M{i}) are built by the shared routine
    base_df = process_perbase_df(df, target_base)
    mask_cols = [f'M{i}' for i in range(1, seq_len+1)]
    mask_arr = base_df[mask_cols].values.astype(bool)

    # per-base editing scores, aligned on the sequences kept by :func:`process_perbase_df`
    scores_df = df.loc[base_df['index'].values, score_cols].astype(np.float64)
    if normalize_by_counts:
        assert counts_col in df.columns, f'missing counts column: {counts_col}'
        counts = df.loc[base_df['index'].values, counts_col].astype(np.float64).values
        assert (counts > 0).all(), f'{counts_col} has non-positive entries!'
        scores_df = scores_df.divide(counts, axis=0)
    scores_arr = scores_df.values
    # positions that do not harbor the target base are not editable
    scores_arr = np.where(mask_arr, scores_arr, 0.)

    es_df = pd.DataFrame(scores_arr.astype(np.float32),
                         columns=[f'ES{i}' for i in range(1, seq_len+1)])
    categ_arr = ((scores_arr >= cutoff_score) & mask_arr).astype(np.int64)
    eci_df = pd.DataFrame(categ_arr, columns=[f'ECi{i}' for i in range(1, seq_len+1)])

    res_df = pd.concat([base_df, es_df, eci_df], axis=1)

    # sequence level score aggregated across the editable positions only
    masked_scores = np.where(mask_arr, scores_arr, np.nan)
    if overall_agg == 'max':
        overall_score = np.nanmax(masked_scores, axis=1)
    else:
        overall_score = np.nanmean(masked_scores, axis=1)
    res_df['efficiency_score'] = overall_score.astype(np.float32)
    res_df['edited_seq_categ'] = (res_df['efficiency_score'] >= cutoff_score).astype(np.int64)

    if compute_prank:
        # percentile rank computed on the pool of editable positions
        eff_scores = scores_arr[mask_arr]
        prank_arr = np.zeros_like(scores_arr, dtype=np.float32)
        prank_arr[mask_arr] = stats.rankdata(eff_scores, method='max') / len(eff_scores) * 100
        prank_df = pd.DataFrame(prank_arr, columns=[f'ESpr{i}' for i in range(1, seq_len+1)])
        res_df = pd.concat([res_df, prank_df], axis=1)

    validate_df(res_df)
    return res_df