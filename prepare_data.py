import argparse
import os
from collections import defaultdict
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import torch
from matchms.importing import load_from_mgf
from matchms.filtering import normalize_intensities
from tqdm import tqdm
from rdkit import Chem
from gensim.models import word2vec
from mol2vec.features import mol2alt_sentence

BIN_SIZE = 0.01
MAX_MZ = 1000.0
NUM_BINS = int(MAX_MZ / BIN_SIZE)
NUM_ION_MODES = 2
MOL2VEC_PATH = '/Genomics/argo/users/vg8892/ChemEmbed/model_300dim.pkl'
_mol2vec_model = None

def _get_mol2vec():
    global _mol2vec_model
    if _mol2vec_model is None:
        _mol2vec_model = word2vec.Word2Vec.load(MOL2VEC_PATH)

    return _mol2vec_model

def smiles_to_mol2vec(smiles):
    m = _get_mol2vec()
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        raise ValueError(f'RDKit cannot parse SMILES: {smiles}')

    sentence = mol2alt_sentence(mol, radius=1)
    vecs = [m.wv[t] for t in sentence if t in m.wv]

    if not vecs:
        raise ValueError(f'No Mol2vec tokens found for SMILES: {smiles}')

    return np.mean(vecs, axis=0).astype(np.float32)

def filter_peaks(mz, intensity, precursor_mz):
    mask = mz <= precursor_mz + 0.5
    mz = mz[mask]
    intensity = intensity[mask]

    if len(intensity) == 0:
        return (mz, intensity)
        
    thresh = 0.01 * intensity.max()
    mask2 = intensity >= thresh

    return (mz[mask2], intensity[mask2])

def mz_to_bins(mz_values):
    vec = np.zeros(NUM_BINS, dtype=np.float32)

    if len(mz_values) == 0:
        return vec

    idx = (mz_values / BIN_SIZE).astype(int)
    idx = idx[(idx >= 0) & (idx < NUM_BINS)]
    vec[idx] = 1.0

    return vec

def spectrum_to_frag_and_nl(mz, intensity, precursor_mz):
    mz, intensity = filter_peaks(mz, intensity, precursor_mz)
    frag_vec = mz_to_bins(mz)
    nl_mz = precursor_mz - mz
    nl_mz = nl_mz[(nl_mz >= 0.0) & (nl_mz <= MAX_MZ)]
    nl_vec = mz_to_bins(nl_mz)

    return (frag_vec, nl_vec)

def parse_mgf(path):
    records = []
    n_skip_mode = 0
    n_skip_meta = 0
    for spec in tqdm(load_from_mgf(path), desc='Reading MGF'):
        spec = normalize_intensities(spec)
        ionmode = (spec.get('ionmode') or '').lower().strip()
        if ionmode == 'positive':
            ion_mode = 1
        elif ionmode == 'negative':
            ion_mode = 0
        else:
            n_skip_mode += 1
            continue

        inchikey = (spec.get('inchikey') or spec.get('inchi_key') or '').strip()
        smiles = (spec.get('smiles') or '').strip()
        title = (spec.get('title') or '').strip()
        prec = spec.get('precursor_mz') or spec.get('pepmass')

        if not inchikey or not smiles or prec is None:
            n_skip_meta += 1
            continue

        records.append({'title': title, 'inchikey14': inchikey[:14], 'smiles': smiles, 'precursor_mz': float(prec), 'mz': np.array(spec.mz, dtype=np.float32), 'intensity': np.array(spec.intensities, dtype=np.float32), 'ion_mode': ion_mode})

    return records

def load_fold_map(path):
    df = pd.read_csv(path, compression='infer')
    df = df[['TITLE', 'FOLD']].dropna()
    df['TITLE'] = df['TITLE'].astype(str).str.strip()
    df['FOLD'] = df['FOLD'].astype(int)

    return dict(zip(df['TITLE'], df['FOLD']))

def split_records_by_fold(records, fold_map, val_fold, test_fold):
    train, val, test = ([], [], [])
    for r in records:
        f = fold_map.get(r['title'])
        if f == val_fold:
            val.append(r)
        elif f == test_fold:
            test.append(r)
        else:
            train.append(r)

    return (train, val, test)

def make_individual_items(records, split_name):
    items = []
    for r in tqdm(records, desc=f'Individual [{split_name}]'):
        frag_vec, _ = spectrum_to_frag_and_nl(r['mz'], r['intensity'], r['precursor_mz'])
        items.append({'inchikey14': r['inchikey14'], 'smiles': r['smiles'], 'ion_mode': r['ion_mode'], 'frag_vec': frag_vec, 'nl_vec': None, 'titles': [r['title']] if r['title'] else []})

    return items

def merge_by_inchikey(records, split_name):
    groups = defaultdict(list)
    for r in records:
        groups[r['inchikey14'], r['ion_mode']].append(r)

    merged = []
    for (inchikey14, ion_mode), group in tqdm(groups.items(), desc=f'Merging [{split_name}]'):
        frag_vec = np.zeros(NUM_BINS, dtype=np.float32)
        nl_vec = np.zeros(NUM_BINS, dtype=np.float32)
        smiles = group[0]['smiles']
        titles = []

        for r in group:
            fv, nv = spectrum_to_frag_and_nl(r['mz'], r['intensity'], r['precursor_mz'])
            frag_vec = np.maximum(frag_vec, fv)
            nl_vec = np.maximum(nl_vec, nv)
            if r['title']:
                titles.append(r['title'])

        merged.append({'inchikey14': inchikey14, 'smiles': smiles, 'ion_mode': ion_mode, 'frag_vec': frag_vec, 'nl_vec': nl_vec, 'titles': titles})

    return merged

def build_mol2vec_cache(records):
    unique = {r['smiles'] for r in records}
    cache = {}
    for s in tqdm(unique, desc='Mol2vec embeddings'):
        cache[s] = smiles_to_mol2vec(s)

    return cache

def build_items(records, emb_cache, mode, split_name):
    items = []
    for r in tqdm(records, desc=f'Building {split_name} items'):
        item = {'inchikey14': r['inchikey14'], 'smiles': r['smiles'], 'ion': r['ion_mode'], 'titles': r['titles'], 'spec': torch.tensor(r['frag_vec'], dtype=torch.float32), 'emb': torch.tensor(emb_cache[r['smiles']], dtype=torch.float32)}

        if mode == 'merged_nl':
            item['nl'] = torch.tensor(r['nl_vec'], dtype=torch.float32)
        items.append(item)

    return items

def ion_mode_counts(items):
    c = {0: 0, 1: 0}
    for it in items:
        c[it['ion']] += 1

    return c

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mgf', required=True)
    ap.add_argument('--fold_csv', required=True)
    ap.add_argument('--val_fold', type=int, default=9)
    ap.add_argument('--test_fold', type=int, default=10)
    ap.add_argument('--out_dir', default='data')
    ap.add_argument('--mode', choices=['individual', 'merged', 'merged_nl'], default='merged_nl')
    args = ap.parse_args()

    split_dir = os.path.join(args.out_dir, f'split_{args.test_fold}', args.mode)
    os.makedirs(split_dir, exist_ok=True)
    out_file = os.path.join(split_dir, 'processed.pt')
    all_records = parse_mgf(args.mgf)

    if not all_records:
        raise RuntimeError('No valid spectra parsed. Aborting.')

    fold_map = load_fold_map(args.fold_csv)
    train_raw, val_raw, test_raw = split_records_by_fold(all_records, fold_map, args.val_fold, args.test_fold)
    mgf_titles = {r['title'] for r in all_records}
    csv_titles = set(fold_map.keys())
    missing_from_mgf = csv_titles - mgf_titles

    if missing_from_mgf:
        pass
    unassigned = [r for r in all_records if r['title'] not in fold_map]

    if unassigned:
        pass
    
    if args.mode == 'individual':
        train_records = make_individual_items(train_raw, 'train')
        val_records = make_individual_items(val_raw, 'val')
        test_records = make_individual_items(test_raw, 'test')
    else:
        train_records = merge_by_inchikey(train_raw, 'train')
        val_records = merge_by_inchikey(val_raw, 'val')
        test_records = merge_by_inchikey(test_raw, 'test')
        all_raw_titles = {r['title'] for r in all_records if r['title']}
        all_merged_titles = set()
        for m in train_records + val_records + test_records:
            all_merged_titles.update(m['titles'])

        lost_titles = all_raw_titles - all_merged_titles
        if lost_titles:
            pass
        
    all_records_flat = train_records + val_records + test_records
    emb_cache = build_mol2vec_cache(all_records_flat)
    emb_dim = 300

    train_items = build_items(train_records, emb_cache, args.mode, 'train')
    val_items = build_items(val_records, emb_cache, args.mode, 'val')
    test_items = build_items(test_records, emb_cache, args.mode, 'test')

    meta = {'num_bins': NUM_BINS, 'num_ion_modes': NUM_ION_MODES, 'val_fold': args.val_fold, 'test_fold': args.test_fold, 'label_type': 'mol2vec', 'emb_dim': emb_dim, 'data_mode': args.mode}
    torch.save({'meta': meta, 'splits': {'train': train_items, 'val': val_items, 'test': test_items}}, out_file)
    
if __name__ == '__main__':
    main()
