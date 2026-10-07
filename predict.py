import argparse
import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from train_model import CNN_A, CNN_B, CNN_C, LitSingle, LitDual, SingleInputDataset, DualInputDataset, resolve_model

def run_predictions(model, dataloader, test_items, is_dual, device):
    model.eval()
    model.to(device)
    preds = []

    with torch.no_grad():
        for batch in dataloader:
            if is_dual:
                spec, nl, emb, ion = [x.to(device) for x in batch]
                out = model(spec, nl, ion)
            else:
                spec, emb, ion = [x.to(device) for x in batch]
                out = model(spec, ion)
            preds.append(out.cpu())
    preds = torch.cat(preds, dim=0).numpy()

    rows = []
    for item, pred in zip(test_items, preds):
        rows.append({'smiles': item.get('smiles', ''), 'inchikey14': item.get('inchikey14', ''), 'ion': item.get('ion', ''), 'titles': '|'.join(item.get('titles', [])), 'embedding': ','.join(map(str, pred))})

    return rows

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--out_csv', required=True)
    parser.add_argument('--model', choices=['A', 'B', 'C'], default=None)
    parser.add_argument('--split', choices=['train', 'val', 'test'], default='test')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    obj = torch.load(args.data_file, weights_only=False)
    meta = obj['meta']
    num_bins = meta['num_bins']
    data_mode = meta.get('data_mode', 'merged_nl')
    items = obj['splits'][args.split]

    dataset_cls, model_cls, lit_cls, is_dual = resolve_model(data_mode, args.model)
    model_instance = model_cls(num_bins)

    lit = lit_cls.load_from_checkpoint(args.checkpoint, model=model_instance)
    lit.model.eval()

    dataloader = DataLoader(dataset_cls(items), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    rows = run_predictions(lit.model, dataloader, items, is_dual, device)

    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out_csv, index=False)
    
if __name__ == '__main__':
    main()
