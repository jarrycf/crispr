# Training the BE-DICT models & reproducing the figures

This document describes the training code of the two `BE-DICT` models and the figure
panels that can be regenerated from them.

- `BE-DICT (perbase)` — predicts, for every target nucleotide of a protospacer, the
  probability that it is edited (`criscas` package).
- `BE-DICT (bystander)` — predicts the proportion of every outcome sequence (haplotype)
  of a protospacer (`haplotype` package).

Everything below runs on cpu; a gpu is used automatically when available (drop `--cpu`).

---

## 1. Quick start

```shell
# install the package and its dependencies
pip install -e .

# regenerate the figure panels that only need the shipped models + sample sequences
python scripts/reproduce_figures.py \
    --panels perbase-attention,perbase-profile,bystander \
    --fig-dir figures

# train the per-base models of one editor on your screen data (5 runs, as in the paper)
python scripts/train_perbase.py --data data/ABEmax_screen.csv --editor ABEmax \
    --output-dir trained_models_rerun/perbase

# train the bystander models of one editor
python scripts/train_bystander.py --data data/ABEmax_haplotypes.csv --editor ABEmax \
    --output-dir trained_models_rerun/bystander

# run the test suite (trains tiny models on synthetic data)
python -m pytest tests/ -v
```

---

## 2. Input data schemas

### 2.1 Per-base model

One row per protospacer, measured in a high-throughput target library screen:

| column | description |
|---|---|
| `ID` | sequence identifier |
| `seq` | 20 nt protospacer sequence (A/C/T/G) |
| `allCounts` | total read count of the sequence (only used with `--normalize-by-counts`) |
| `V1` … `V20` | editing efficiency measured at protospacer position 1 … 20 |

`V{i}` is expected to be a **proportion** (0–1). If your table holds raw counts instead,
pass `--normalize-by-counts` and they are divided by `allCounts`.

`criscas.data_preprocess.generate_perbase_df` turns this into the training frame:

| columns | description |
|---|---|
| `B1..B20` | numeric nucleotide encoding (A,C,T,G → 0,1,2,3) |
| `L1..L20` | nucleotide letters |
| `M1..M20` | mask of the target nucleotide (1 where the editor can act) |
| `ES1..ES20` | per-base editing efficiency |
| `ECi1..ECi20` | per-base class, `ES{i} >= cutoff_score` |
| `ESpr1..ESpr20` | percentile rank of the per-base efficiency |
| `efficiency_score`, `edited_seq_categ` | sequence level score and class |

Positions that do not harbor the target nucleotide get a zero score/class and are
**excluded from the loss and from every metric** through the `M{i}` mask.

### 2.2 Bystander model

One row per (input sequence, observed outcome sequence) pair:

| column | description |
|---|---|
| `seq_id` | sequence identifier |
| `Inp_seq` | 20 nt input (target) sequence |
| `Outp_seq` | observed outcome sequence (same length) |
| `true_score` | observed proportion of that outcome |
| `Editor` | *(optional)* used to select the rows of one editor |

`sample_data/bystander_sampledata.csv` is an example of this schema. The proportions of
one input sequence are renormalized to sum to 1 (disable with `--no-renormalize`).

---

## 3. What the training code does

### 3.1 Per-base model (`criscas/train_val.py`)

1. `generate_perbase_df` builds the sequence features and the per-base labels.
2. `get_stratified_partitions` splits the sequences into 5 train/validation/test
   partitions, stratified on the sequence level class. Across the 5 folds every sequence
   is in the test set exactly once; the partitions are asserted to be disjoint.
3. For every run a `Categ_CrisCasTransformer` is trained with a **negative
   log-likelihood** loss computed **only on the positions harboring the target
   nucleotide**, Adam (weight decay 1e-4) and a cyclic learning rate schedule.
4. The epoch with the best validation AUC is checkpointed, and the model, its config, the
   predictions, the loss curve and the logs are written to:

```
{output_dir}/{editor}/train_val/run_{num}/model_statedict/Transformer.pkl
{output_dir}/{editor}/train_val/run_{num}/config/{mconfig,exp_options}.pkl
{output_dir}/{editor}/train_val/run_{num}/predictions_{train,validation}.csv
{output_dir}/{editor}/test/run_{num}/predictions_test.csv
```

This is exactly the layout of `trained_models/perbase`, so a freshly trained model can be
used directly:

```python
bedict = BEDICT_CriscasModel('ABEmax', device,
                             trained_models_dir='trained_models_rerun')
```

### 3.2 Bystander model (`haplotype/train_val.py`)

1. The input/outcome sequences are processed with `HaplotypeSeqProcessor` and tensorized
   with `HaplotypeDataTensor`.
2. The sequences are split into 5 partitions **at the level of the input sequence**, so
   no outcome of a test sequence is seen during training.
3. A haplotype is scored by summing the per-position log-probabilities of its observed
   conversion states over the target positions. Those log-probabilities are renormalized
   over the haplotype set of the input sequence (matching the renormalization applied to
   the observed proportions) and compared to the observed proportions with a
   **Kullback-Leibler divergence**, one per input sequence.
4. The epoch with the best validation Pearson correlation is checkpointed into the layout
   of `trained_models/bystander` (state dict name: `HaplotypeTransformer.pkl`).

### 3.3 Hyperparameters

The defaults of `get_hyperparam_options()` are the ones of the shipped models — they are
read back out of `trained_models/*/{editor}/train_val/run_{n}/config/mconfig.pkl` and
asserted against the defaults in `tests/test_training_workflow.py`:

| | per-base | bystander |
|---|---|---|
| embedding dim | 64 | 128 |
| attention heads | 8 | 8 (narrow) |
| transformer units | 2 | 1 |
| dropout | 0.1 | 0.25 |
| mlp embed factor | 2 | 2 |
| weight decay (l2) | 1e-4 | 1e-4 |
| batch size | 3000 | 300 |
| epochs | 500 | 150 |
| loss | NLL | KL divergence |

Any of them can be overridden on the command line (`--embed-dim`, `--num-epochs`, …).

---

## 4. Reproducing the figure panels

```shell
python scripts/reproduce_figures.py --panels all --fig-dir figures \
    --perbase-data data/ABEmax_screen.csv
```

Each panel is written as **pdf + png**, together with a **csv of the plotted numbers**.

| `--panels` value | panels produced | data needed |
|---|---|---|
| `perbase-attention` | attention heatmap per sequence/position; mean attention per (target position × attended position); attention by neighboring nucleotide (motif view) | sequences only (`--perbase-seqs`, defaults to the shipped sample data) |
| `perbase-profile` | position-wise editing probability and attention profile, all editors overlaid | sequences only |
| `perbase-performance` | ROC and precision-recall curves, AUC/AUPR per editor, observed vs predicted efficiency scatter | **measured efficiencies** (`--perbase-data`, schema 2.1) |
| `bystander` | observed vs predicted outcome proportion scatter, per-position conversion frequency, top-k haplotype agreement, haplotype alignment tables (html) | `--bystander-data` (schema 2.2); the observed proportions are optional, without them only the predicted quantities are plotted |

Figure conventions: one fixed colorblind-safe color per base editor (never cycled), a
single-hue sequential colormap for magnitudes, and no chart with two y-scales.

### What you need to supply for the published numbers

The screen data the models were trained on is **not part of this repository** (`data/` is
git-ignored). Consequently:

- Panels that only need sequences (`perbase-attention`, `perbase-profile`, `bystander`)
  run out of the box on `sample_data/` and reproduce the **published models'** behavior,
  since they use the shipped trained weights.
- Panels that compare against measured values (`perbase-performance`, and the
  observed-vs-predicted parts of `bystander`) need the corresponding screen data in the
  schemas above. Point `--perbase-data` / `--bystander-data` at it and the same panels are
  produced with the published models.
- Retraining from scratch and reproducing the paper's exact numbers additionally requires
  the original screen libraries; with those in the schemas above, `scripts/train_*.py`
  rebuild the models with the published hyperparameters.

---

## 5. Module map of the added code

| file | content |
|---|---|
| `criscas/hyperparam.py` | `TrfHyperparamConfig` (required to unpickle the shipped per-base configs), config generation and serialization |
| `criscas/data_preprocess.py` | `generate_perbase_df` — per-base labels from screen data |
| `criscas/data_partition.py` | stratified 5-fold partitioning + leakage assertions |
| `criscas/train_val.py` | per-base training/validation/test workflow |
| `criscas/evaluation.py` | `ModelScore`, classification metrics, correlation, loss curves |
| `criscas/figures.py` | per-base figure panels |
| `haplotype/hyperparam.py` | bystander config generation and serialization |
| `haplotype/data_partition.py` | partitioning at the input sequence level |
| `haplotype/train_val.py` | bystander training/validation/test workflow (KL loss) |
| `haplotype/figures.py` | bystander figure panels |
| `scripts/train_perbase.py`, `scripts/train_bystander.py` | training CLIs |
| `scripts/reproduce_figures.py` | figure reproduction CLI |
| `tests/test_training_workflow.py` | smoke tests of all of the above |
