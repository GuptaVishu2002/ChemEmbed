import pandas as pd
import json
import numpy as np
import torch
import torch.nn.functional as F
import sys, os
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

data_path = sys.argv[1]
fold_no = int(sys.argv[2])
csv_path = sys.argv[3]
json_path = sys.argv[4]
scores_txt = sys.argv[5]
ranks_txt = sys.argv[6]

dir_scores = os.path.dirname(scores_txt)
if dir_scores:
    os.makedirs(dir_scores, exist_ok=True)

dir_ranks = os.path.dirname(ranks_txt)
if dir_ranks:
    os.makedirs(dir_ranks, exist_ok=True)

df_data = pd.read_csv(data_path)
df_data = df_data[df_data['FOLD'] == fold_no].copy()
df_data['IONMODE'] = df_data['IONMODE'].map({'positive': 1, 'negative': 0}).fillna(df_data['IONMODE']).astype(int)
df_data['inchikey14'] = df_data['INCHIKEY'].str[:14]
df_csv = pd.read_csv(csv_path)

if 'ion' not in df_csv.columns:
    raise ValueError("csv_path must have an 'ion' column (0/1 for negative/positive)")

df_csv['ion'] = df_csv['ion'].astype(int)
with open(json_path, 'r') as f:
    fp_dict = json.load(f)

smiles_col = 'smiles'
ion_col = 'ion'
embedding_col = 'embedding'

def random_tiebreak_ranking(scores):
    scores = np.array(scores)
    ranks = np.empty_like(scores, dtype=int)
    unique_scores = np.unique(scores)
    current_rank = 1
    for val in sorted(unique_scores, reverse=True):
        idx = np.where(scores == val)[0]
        if len(idx) > 1:
            np.random.shuffle(idx)
        for i in idx:
            ranks[i] = current_rank
            current_rank += 1

    return ranks

def process_row(args):
    row, spectrum_id = args
    smiles = row[smiles_col]
    emb_str = row[embedding_col]
    csv_fp = torch.tensor([float(x) for x in emb_str.split(',')], dtype=torch.float32)
    json_fps = fp_dict.get(smiles, [])
    if not json_fps:
        scores = []
        ranks = []
    else:
        cands = torch.tensor(np.array(json_fps).astype(float))
        fp_pred_repeated = csv_fp.repeat(cands.size(0), 1)
        similarities = F.cosine_similarity(fp_pred_repeated, cands, dim=1)
        scores = similarities.cpu().numpy()
        ranks = random_tiebreak_ranking(scores)

    return (spectrum_id, scores, ranks)

rows_to_process = []
for _, r in df_data.iterrows():
    inchikey14 = r['inchikey14']
    ion = r['IONMODE']
    spectrum_id = r['TITLE']
    smiles = r['SMILES']
    match = df_csv[(df_csv['inchikey14'] == inchikey14) & (df_csv[ion_col] == ion)]
    if match.empty:
        continue

    row = match.iloc[0].copy()
    row[smiles_col] = smiles
    rows_to_process.append((row, spectrum_id))

with ThreadPoolExecutor() as executor:
    results = list(tqdm(executor.map(process_row, rows_to_process), total=len(rows_to_process), desc='Processing rows'))
    
with open(scores_txt, 'w') as score_f, open(ranks_txt, 'w') as rank_f:
    for spectrum_id, scores, ranks in tqdm(results, desc='Writing output'):
        score_f.write(f"{spectrum_id} {' '.join(map(str, scores))}\n")
        rank_f.write(f"{spectrum_id} {' '.join(map(str, ranks))}\n")
