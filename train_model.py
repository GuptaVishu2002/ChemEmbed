import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

class SingleInputDataset(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        spec = it['spec'].float()
        emb = it['emb'].float()
        onehot = torch.zeros(2, dtype=torch.float32)
        onehot[it['ion']] = 1.0
        return (spec, emb, onehot)

class DualInputDataset(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        spec = it['spec'].float()
        nl = it['nl'].float()
        emb = it['emb'].float()
        onehot = torch.zeros(2, dtype=torch.float32)
        onehot[it['ion']] = 1.0
        return (spec, nl, emb, onehot)

def build_cnn_branch():
    return nn.Sequential(nn.Conv1d(1, 32, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2), nn.Conv1d(32, 32, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2), nn.Conv1d(32, 64, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2), nn.Conv1d(64, 64, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2), nn.Conv1d(64, 128, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2), nn.Conv1d(128, 128, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2))

class CNN_A(nn.Module):
    def __init__(self, num_bins):
        super().__init__()
        self.conv = build_cnn_branch()
        flat = num_bins // 2 ** 6 * 128
        self.fc = nn.Sequential(nn.Linear(flat + 2, 1024), nn.ReLU(), nn.Dropout(0.3), nn.Linear(1024, 300))

    def forward(self, x, ion):
        x = self.conv(x.unsqueeze(1)).flatten(1)
        x = torch.cat([x, ion], dim=1)
        return self.fc(x)

class CNN_B(nn.Module):
    def __init__(self, num_bins):
        super().__init__()
        self.conv = build_cnn_branch()
        flat = num_bins // 2 ** 6 * 128
        self.fc = nn.Sequential(nn.Linear(flat + 2, 1024), nn.ReLU(), nn.Dropout(0.3), nn.Linear(1024, 300))

    def forward(self, x, ion):
        x = self.conv(x.unsqueeze(1)).flatten(1)
        x = torch.cat([x, ion], dim=1)
        return self.fc(x)

class CNN_C(nn.Module):
    def __init__(self, num_bins):
        super().__init__()
        self.spec_branch = build_cnn_branch()
        self.nl_branch = build_cnn_branch()
        flat = num_bins // 2 ** 6 * 128
        self.fc = nn.Sequential(nn.Linear(2 * flat + 2, 1024), nn.ReLU(), nn.Dropout(0.3), nn.Linear(1024, 300))

    def forward(self, xs, xn, ion):
        xs = self.spec_branch(xs.unsqueeze(1)).flatten(1)
        xn = self.nl_branch(xn.unsqueeze(1)).flatten(1)
        x = torch.cat([xs, xn, ion], dim=1)
        return self.fc(x)

class LitSingle(pl.LightningModule):
    def __init__(self, model, lr=0.0001):
        super().__init__()
        self.model = model
        self.loss = nn.MSELoss()
        self.lr = lr

    def forward(self, x, ion):
        return self.model(x, ion)

    def training_step(self, batch, _):
        x, y, ion = batch
        loss = self.loss(self(x, ion), y)
        self.log('train_loss', loss, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, _):
        x, y, ion = batch
        yhat = self(x, ion)
        loss = self.loss(yhat, y)
        cos = nn.functional.cosine_similarity(yhat, y, dim=1).mean()
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_cos', cos, prog_bar=True)

    def test_step(self, batch, _):
        x, y, ion = batch
        yhat = self(x, ion)
        loss = self.loss(yhat, y)
        cos = nn.functional.cosine_similarity(yhat, y, dim=1).mean()
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_cos', cos, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

class LitDual(pl.LightningModule):
    def __init__(self, model, lr=0.0001):
        super().__init__()
        self.model = model
        self.loss = nn.MSELoss()
        self.lr = lr

    def forward(self, xs, xn, ion):
        return self.model(xs, xn, ion)

    def training_step(self, batch, _):
        xs, xn, y, ion = batch
        loss = self.loss(self(xs, xn, ion), y)
        self.log('train_loss', loss, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, _):
        xs, xn, y, ion = batch
        yhat = self(xs, xn, ion)
        loss = self.loss(yhat, y)
        cos = nn.functional.cosine_similarity(yhat, y, dim=1).mean()
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_cos', cos, prog_bar=True)

    def test_step(self, batch, _):
        xs, xn, y, ion = batch
        yhat = self(xs, xn, ion)
        loss = self.loss(yhat, y)
        cos = nn.functional.cosine_similarity(yhat, y, dim=1).mean()
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_cos', cos, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

_MODE_DEFAULTS = {'individual': (SingleInputDataset, CNN_A, LitSingle, False), 'merged': (SingleInputDataset, CNN_B, LitSingle, False), 'merged_nl': (DualInputDataset, CNN_C, LitDual, True)}

def resolve_model(data_mode, model_override):
    if model_override is None:
        return _MODE_DEFAULTS[data_mode]

    if model_override == 'C':
        if data_mode != 'merged_nl':
            raise ValueError("CNN_C requires neutral-loss vectors. Use data_mode='merged_nl' (prepare_data.py --mode merged_nl).")
        return (DualInputDataset, CNN_C, LitDual, True)

    if data_mode == 'merged_nl':
        raise ValueError(f"CNN_{model_override} is a single-input model but data_mode='merged_nl' was prepared for CNN_C. Re-run prepare_data.py with --mode merged or --mode individual.")

    model_cls = CNN_A if model_override == 'A' else CNN_B

    return (SingleInputDataset, model_cls, LitSingle, False)

def make_trainer(epochs, patience, accelerator, out_dir, model_name):
    ckpt_dir = os.path.join(out_dir, model_name)
    callbacks = [EarlyStopping(monitor='val_loss', patience=patience, mode='min', verbose=True), ModelCheckpoint(dirpath=ckpt_dir, filename=f'{model_name}_best', monitor='val_loss', mode='min', save_top_k=1, verbose=True)]

    return pl.Trainer(max_epochs=epochs, accelerator=accelerator, callbacks=callbacks, log_every_n_steps=1)

def make_dataloaders(dataset_cls, train_items, val_items, test_items, batch_size, num_workers):
    dl_tr = DataLoader(dataset_cls(train_items), batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    dl_va = DataLoader(dataset_cls(val_items), batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True) if val_items else None
    dl_te = DataLoader(dataset_cls(test_items), batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True) if test_items else None

    return (dl_tr, dl_va, dl_te)

def save_test_predictions(model, dataloader, test_items, out_csv, is_dual, device='cpu'):
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

    pd.DataFrame(rows).to_csv(out_csv, index=False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_file', required=True)
    ap.add_argument('--model', choices=['A', 'B', 'C'], default=None)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--epochs', type=int, default=70)
    ap.add_argument('--patience', type=int, default=10)
    ap.add_argument('--accelerator', default='auto')
    ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--out_dir', default='checkpoints')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    obj = torch.load(args.data_file, weights_only=False)

    meta = obj['meta']
    splits = obj['splits']
    num_bins = meta['num_bins']
    data_mode = meta.get('data_mode', 'merged_nl')
    train_items = splits['train']
    val_items = splits['val']
    test_items = splits['test']

    dataset_cls, model_cls, lit_cls, is_dual = resolve_model(data_mode, args.model)

    model_name = model_cls.__name__
    dl_tr, dl_va, dl_te = make_dataloaders(dataset_cls, train_items, val_items, test_items, args.batch_size, args.num_workers)
    lit = lit_cls(model_cls(num_bins))

    trainer = make_trainer(args.epochs, args.patience, args.accelerator, args.out_dir, model_name)
    trainer.fit(lit, dl_tr, dl_va)

    if dl_te:
        trainer.test(lit, dl_te, ckpt_path='best')

    ckpt_path = os.path.join(args.out_dir, model_name, f'{model_name}_best.ckpt')
    lit_loaded = lit_cls.load_from_checkpoint(ckpt_path, model=model_cls(num_bins))
    pred_csv = os.path.join(args.out_dir, model_name, f'{model_name}_test_preds.csv')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    save_test_predictions(lit_loaded.model, dl_te, test_items, pred_csv, is_dual, device)
    
if __name__ == '__main__':
    main()
