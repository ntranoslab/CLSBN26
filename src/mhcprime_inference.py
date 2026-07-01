from functools import partial
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from .model_utils import PeptideMHCDataset, collate_fn

def run_mhcprime(model, test_df, tokenizer, batch_size=1024, num_workers=16, score_col="score", device="cuda", return_embeddings=False, include_hardness_score=False, include_domain_id=False, include_prior_score=False):
  
    model.eval()
    model.to(device)

    test_dataset = PeptideMHCDataset(test_df, tokenizer, include_hardness_score=include_hardness_score, include_domain_id=include_domain_id, include_prior_score=include_prior_score)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True
    )

    logit_scores = []
    bos_embeddings_dict = {}
    batch_indices = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Running inference")):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}  
            _ = batch.pop("label")

            if return_embeddings:
                logits, bos_embeddings = model(batch, return_embeddings=True)
                
                batch_start = batch_idx * batch_size
                batch_end = min(batch_start + batch_size, len(test_df))
                batch_indices.extend(range(batch_start, batch_end))
                
                for i, embedding in enumerate(bos_embeddings):
                    if batch_start + i < len(test_df):
                        bos_embeddings_dict[batch_start + i] = embedding.cpu().numpy()
            else:
                if model.transformer_type=="switch":
                    outputs = model(batch, return_embeddings=False)
                    logits=outputs[0]
                else:
                    logits = model(batch, return_embeddings=False)

            if model.fc_output_dim==1:
                logit_scores.extend(logits.cpu().numpy())
            elif model.fc_output_dim==2:
                logit_scores.extend(logits[:, 1].detach().cpu().numpy())  

    test_df = test_df.copy()
    test_df[score_col] = logit_scores
    
    if return_embeddings:
        keys = test_df['seq'] + '_' + test_df['allele']
        
        peptide_mhc_to_bos = {
            key: embedding 
            for key, embedding, idx in zip(keys, bos_embeddings_dict.values(), bos_embeddings_dict.keys())
            if idx < len(test_df)  # Safety check
        }
        
        return test_df, peptide_mhc_to_bos
    
    return test_df


"""
Cached version
"""

def build_token_cache(df, # csv_in: str, # input df directly
                      cache_out: str,
                      tokenizer,
                      pep_len: int = 34,
                      mhc_len: int = 34,
                      pad_id: int = 1,
                      use_domain_id=False):
    """
    Read a CSV with columns n_flank, seq, c_flank, allele, label, mhc_*.
    Write a torch file holding three stacked tensors:
        pep [N, pep_len]   int64
        mhc [N, mhc_len]   int64
        lab [N]            float32
    """
    # df = pd.read_csv(csv_in)
    N  = len(df)

    pep  = torch.full((N, pep_len), pad_id, dtype=torch.long)
    mhc  = torch.full((N, mhc_len), pad_id, dtype=torch.long)
    lab  = torch.tensor(df["label"].values, dtype=torch.float32)

    if use_domain_id:
        did  = torch.tensor(df["domain_id"].values, dtype=torch.long)

    for i, row in tqdm(enumerate(df.itertuples(index=False))):
        pep_ids = tokenizer.encode(row.n_flank + row.seq + row.c_flank)[:pep_len]
        pep[i, :len(pep_ids)] = torch.as_tensor(pep_ids, dtype=torch.long)

        mhc_col = f"mhc_{row.allele[0].lower()}_1"
        mhc_ids = tokenizer.encode(getattr(row, mhc_col))[:mhc_len]
        mhc[i, :len(mhc_ids)] = torch.as_tensor(mhc_ids, dtype=torch.long)

    if use_domain_id:
        torch.save({"pep": pep, "mhc": mhc, "lab": lab, "domain_id": did}, cache_out)
    else:
        torch.save({"pep": pep, "mhc": mhc, "lab": lab}, cache_out)
    print(f"wrote cached tensors → {cache_out}  ({pep.nbytes/1e6:.1f} MB)")

class CachedDataset(Dataset):
    def __init__(self, cache_pt: str, use_domain_id=False):
        data = torch.load(cache_pt, map_location="cpu")
        self.pep   = data["pep"] # [N, 34] int64
        self.mhc   = data["mhc"] # [N, 34] int64
        self.label = data["lab"] # [N] float32
        self.use_domain_id=use_domain_id

        if use_domain_id:
            self.domain_id = data["domain_id"]

    def __len__(self):  return self.pep.size(0)

    def __getitem__(self, idx):

        if self.use_domain_id:
            return self.pep[idx], self.mhc[idx], self.label[idx], self.domain_id[idx]
        else:
            return self.pep[idx], self.mhc[idx], self.label[idx]

def collate(batch, pad_id=1, use_domain_id=False):

    if use_domain_id:
        pep, mhc, lab, did = map(torch.stack, zip(*batch)) # each [B, …]
        pep_mask = pep.ne(pad_id)
        mhc_mask = mhc.ne(pad_id).unsqueeze(1) # [B,1,34]
        return {
            "peptide": pep,
            "peptide_mask": pep_mask,
            "mhc_list": mhc.unsqueeze(1), # [B,1,34]
            "mhc_mask_list": mhc_mask,
            "label": lab,
            "domain_id": did,
        }

    else:
        pep, mhc, lab = map(torch.stack, zip(*batch)) # each [B, …]
        pep_mask = pep.ne(pad_id)
        mhc_mask = mhc.ne(pad_id).unsqueeze(1) # [B,1,34]
        return {
            "peptide": pep,
            "peptide_mask": pep_mask,
            "mhc_list": mhc.unsqueeze(1), # [B,1,34]
            "mhc_mask_list": mhc_mask,
            "label": lab,
        }



def run_mhcprime_fast(model, cache_pt: str, batch_size: int = 3072, num_workers: int = 8, use_bf16: bool = True, compile_model: bool = True, use_domain_id=False):
    print("Getting data")
    ds = CachedDataset(cache_pt, use_domain_id=use_domain_id)

    print("Loading data")
    loader = DataLoader(ds,
                        batch_size=batch_size,
                        shuffle=False,
                        # collate_fn=collate,
                        collate_fn=partial(collate, use_domain_id=use_domain_id),
                        pin_memory=True,
                        persistent_workers=True,
                        num_workers=num_workers)

    print("Setting up model")
    model = model.eval().to("cuda")
    if use_bf16:
        model = model.to(torch.bfloat16)
    if compile_model:
        model = torch.compile(model, mode="reduce-overhead") # PyTorch ≥2.0

    all_logits = []
    torch.backends.cuda.enable_flash_sdp(True) # fused attention kernels (2.1+)

    print("Running inference")
    with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.bfloat16 if use_bf16 else torch.float32):
        for batch in tqdm(loader):
            batch = {k: v.cuda(non_blocking=True)
                     if torch.is_tensor(v) else v
                     for k, v in batch.items()}
            logits = model(batch, return_embeddings=False) # shape [B,1] or [B]
            all_logits.append(logits.float().cpu()) # keep fp32 on host

    return torch.cat(all_logits)
