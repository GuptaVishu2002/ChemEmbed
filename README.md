# ChemEmbed

Code to train and run [ChemEmbed](https://github.com/massspecdl/ChemEmbed) on our benchmark splits. Adapted from the original repo.

ChemEmbed is a 1D CNN that predicts a 300-d Mol2vec embedding from an MS/MS spectrum. Candidates are ranked by cosine similarity between the predicted embedding and each candidate's Mol2vec embedding.

## Setup

```bash
conda create -n chemembed python=3.10 -y
conda activate chemembed
pip install torch pytorch-lightning numpy pandas tqdm matchms gensim mol2vec
conda install -c conda-forge rdkit

wget https://raw.githubusercontent.com/samoturk/mol2vec/master/examples/models/model_300dim.pkl
```

`prepare_data.py` reads the Mol2vec model from `MOL2VEC_PATH` at the top of the file, so point that at your copy of `model_300dim.pkl`.

## Running

Preprocess spectra. The MGF needs SMILES, InChIKey, ion mode and precursor m/z. The fold CSV needs `TITLE` and `FOLD` columns.

```bash
python prepare_data.py --mgf spectra.mgf --fold_csv folds.csv.gz \
    --val_fold 9 --test_fold 10 --mode merged_nl --out_dir data
```

`--mode` picks the model variant:
- `individual`: CNN_A, one example per spectrum
- `merged`: CNN_B, spectra merged by InChIKey (first block) and ion mode
- `merged_nl`: CNN_C, merged spectra plus a neutral-loss channel (default)

Train. This also writes test-set predictions to `<out_dir>/CNN_X/CNN_X_test_preds.csv`.

```bash
python train_model.py --data_file data/split_10/merged_nl/processed.pt --out_dir output/split_10
```

To rerun a saved checkpoint:

```bash
python predict.py --data_file data/split_10/merged_nl/processed.pt \
    --checkpoint output/split_10/CNN_C/CNN_C_best.ckpt --out_csv preds.csv
```

Get Mol2vec embeddings for the candidate sets. The input is a JSON of `{query_smiles: [candidate_smiles, ...]}`.

```bash
python prepare_candidate_mol2vec.py -i candidates.json -m model_300dim.pkl \
    -o candidates_chemembed.json --skip_failures
```

Score and rank candidates. Arguments are positional:

```bash
python retreival.py metadata.csv.gz 10 preds.csv candidates_chemembed.json scores.txt ranks.txt
```

Each line of the output files is a spectrum title followed by the scores (or ranks) for its candidates. Ties are broken at random.
