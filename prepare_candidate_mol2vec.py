import argparse
import json
import sys
from typing import Dict, List, Optional
import numpy as np
from rdkit import Chem
from rdkit import RDLogger
from gensim.models import word2vec
from mol2vec.features import mol2alt_sentence
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
RDLogger.DisableLog('rdApp.*')

def load_mol2vec_model(model_path):
    model = word2vec.Word2Vec.load(model_path)
    return model

def compute_mol2vec(smiles, model):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    sentence = mol2alt_sentence(mol, radius=1)
    vecs = [model.wv[t] for t in sentence if t in model.wv]

    if not vecs:
        return None

    return np.mean(vecs, axis=0).astype(np.float32)

_worker_model = None
_worker_model_path = None

def _init_worker(model_path):
    global _worker_model, _worker_model_path
    RDLogger.DisableLog('rdApp.*')
    _worker_model_path = model_path
    _worker_model = word2vec.Word2Vec.load(model_path)

def _worker(cand_smiles):
    vec = compute_mol2vec(cand_smiles, _worker_model)
    if vec is None:
        raise ValueError(f'compute_mol2vec returned None for SMILES: {cand_smiles}')

    return vec.tolist()

def _worker_query(args_tuple):
    query_smiles, cand_list, skip_failures = args_tuple
    results = []
    for cand_smiles in cand_list:
        try:
            vec = compute_mol2vec(cand_smiles, _worker_model)
            if vec is None:
                raise ValueError(f'compute_mol2vec returned None for SMILES: {cand_smiles}')
            results.append(vec.tolist())
        except Exception as e:
            if skip_failures:
                results.append(None)
            else:
                raise RuntimeError(f'Mol2vec failed for query={query_smiles}, cand={cand_smiles}') from e

    return (query_smiles, results)

def compute_mol2vec_for_candidates(candidates, model_path, *, verbose=True, n_workers=None, skip_failures=False, timeout=60):
    if n_workers is None:
        n_workers = multiprocessing.cpu_count()
    total_queries = len(candidates)
    total_candidates = sum((len(v) for v in candidates.values()))
    result = {}
    n_ok = 0
    n_fail = 0
    tasks = [(query_smiles, cand_list, skip_failures) for query_smiles, cand_list in candidates.items()]
    with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker, initargs=(model_path,)) as executor:
        futures = {executor.submit(_worker_query, task): task[0] for task in tasks}
        with tqdm(total=total_queries, desc='Queries', unit='query', disable=not verbose) as pbar:
            for future in as_completed(futures):
                query_smiles = futures[future]
                try:
                    q, vecs = future.result()
                except Exception as e:
                    if skip_failures:
                        result[query_smiles] = [None] * len(candidates[query_smiles])
                        n_fail += len(candidates[query_smiles])
                        pbar.update(1)
                        continue
                    raise
                result[q] = vecs
                n_ok += sum((1 for v in vecs if v is not None))
                n_fail += sum((1 for v in vecs if v is None))
                pbar.update(1)
                
    if verbose:
        pass

    return result

def _parse_args():
    parser = argparse.ArgumentParser(description='Compute 300-dim Mol2vec', formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input', '-i', required=True)
    parser.add_argument('--model', '-m', required=True)
    parser.add_argument('--output', '-o', required=True)
    parser.add_argument('--skip_failures', action='store_true')
    parser.add_argument('--quiet', action='store_true')
    parser.add_argument('--n_workers', type=int, default=None)
    return parser.parse_args()

def main():
    args = _parse_args()
    with open(args.input, 'r') as f:
        candidates = json.load(f)

    if not isinstance(candidates, dict):
        sys.exit(1)

    n_queries = len(candidates)
    n_candidates = sum((len(v) for v in candidates.values()))
    model = load_mol2vec_model(args.model)
    probe_smiles = 'CC(=O)Oc1ccccc1C(=O)O'
    probe_vec = compute_mol2vec(probe_smiles, model)

    if probe_vec is None or len(probe_vec) != model.wv.vector_size:
        sys.exit(1)

    result = compute_mol2vec_for_candidates(candidates, args.model, verbose=not args.quiet, n_workers=args.n_workers, skip_failures=args.skip_failures)
    with open(args.output, 'w') as f:
        json.dump(result, f)

    n_computed = sum((sum((1 for v in vecs if v is not None)) for vecs in result.values()))
    n_failed = n_candidates - n_computed

    
if __name__ == '__main__':
    main()
