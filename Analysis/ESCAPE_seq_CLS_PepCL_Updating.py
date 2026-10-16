"""
ESCAPE-seq classification updating:
    FT
    PepCL
    FT + copied MS negatives in domain 1
    PepCL + copied MS negatives in domain 1

Modes:
    run_mode="seed" -> fixed cleaned ESCAPE train/val split, multiple seeds
    run_mode="cv"   -> cleaned CV train/val splits, seed 0 per split

Run:
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
    python ESCAPE_seq_CLS_PepCL_Updating.py -g 0
"""
import sys, os, copy, time
from pathlib import Path
import pandas as pd
import torch

PROJECT_ROOT=Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0,str(PROJECT_ROOT))

DATA_DIR=PROJECT_ROOT/"Data"
MODEL_CHECKPOINTS_DIR=PROJECT_ROOT/"Model_Checkpoints"

from Data.mhc_pseudo_seqs import mhc_seq_dict
from src.Model.mhcprime import *
from src.model_utils import *
from src.general_utils import *
from src.get_feature_data import *
from src.mhcprime_trainer import *
from src.mhcprime_inference import *
from src.pepcl import *
from src.pepcl_logging import *

import warnings
warnings.filterwarnings("ignore")

# ============================================================
# GPU
# ============================================================
import argparse
parser=argparse.ArgumentParser(description="Select specific GPU.")
parser.add_argument("-g","--gpu",type=int,default=0)
args=parser.parse_args()
GPU_ID=args.gpu
torch.cuda.set_device(GPU_ID)
device=torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")
print(f"Using GPU: {torch.cuda.get_device_name(GPU_ID)} (ID: {torch.cuda.current_device()})")

# ============================================================
# USER CONFIG
# ============================================================

# "seed" or "cv"
# run_mode="seed"
run_mode="cv"

# Seed mode: all requested seeds use the fixed cleaned train/val split.
seeds=[0,1,2,3,4]

# CV mode: each datasplit uses seed 0.
cv_seed=0
cv_splits=[1,2,3,4]

# Classification update hyperparameters.
steps=300
learning_rate=2e-4
dr_lambda=1e-5
num_workers=4
batch_size=3072

# Optional intermediate checkpoint capture through PepCL_Logging_Ext.
# Usually unnecessary here because we only need final models.
save_every_n_steps=None

# Output roots.
data_output_root=DATA_DIR/"ESCAPE_seq_CLS_Updating"
model_output_root=MODEL_CHECKPOINTS_DIR/"ESCAPE_seq_CLS_Updating"

# Inference.
inference_batch_size=3072
inference_num_workers=4

if run_mode not in {"seed","cv"}: raise ValueError("run_mode must be 'seed' or 'cv'.")

# ============================================================
# HELPERS
# ============================================================

def copy_domain0_negs_to_domain1(df,allele_col="allele",label_col="label",seq_col="seq",domain_col="domain_id"):
    d0_negs=df[(df[domain_col]==0)&(df[label_col]==0)].copy()
    d0_negs[domain_col]=1
    return pd.concat([df,d0_negs],ignore_index=True)

def add_existing_columns(df,path):
    """Preserve model-score columns from an existing output CSV."""
    path=Path(path)
    if not path.exists(): return df
    old=pd.read_csv(path,index_col=0)
    if len(old)!=len(df): raise ValueError(f"Existing output length mismatch: {path}: {len(old)} vs {len(df)}")
    for col in ["seq","allele"]:
        if col in old.columns and col in df.columns and not old[col].astype(str).reset_index(drop=True).equals(df[col].astype(str).reset_index(drop=True)):
            raise ValueError(f"Existing output row order/content mismatch for '{col}': {path}")
    for col in old.columns:
        if col not in df.columns: df[col]=old[col].to_numpy()
    return df

def save_model(model,path):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    torch.save(model.state_dict(),path)
    print("Saved model:",path)

def run_inference_models(models,df,cache_path):
    for model_name,model in models.items():
        print("Inference:",model_name)
        model=model.to(device).eval()
        with torch.no_grad():
            scores=run_mhcprime_fast(model,cache_path,batch_size=inference_batch_size,
                                     num_workers=inference_num_workers,compile_model=False,use_domain_id=False)
        df[model_name]=scores.detach().cpu().tolist()
    return df

def process_eval_df(df):
    df=full_process_data(df,mhc_seq_dict,remove_flank=True,add_pseudo_label=False)
    return create_seq_allele_column(df)

# ============================================================
# COMMON SETUP
# ============================================================
mhc_seq_dict={k.removeprefix("HLA-").replace("*","").replace(":",""):v for k,v in mhc_seq_dict.items()}

print("Loading feature data...")
aa_features_table_ext,descriptor_groups=load_feature_data()
processed_feature_table,corr_info=process_aa_feature_table(aa_features_table_ext)
feature_names=list(processed_feature_table.columns)

print("Loading MS data...")
ms_dir=DATA_DIR/"Training_Datasets/Processed/MS_Train_Test_Sets"
ms_train_base=pd.read_csv(ms_dir/"ms_train_data_remaining_minimal.csv",index_col=0)
ms_val_base=pd.read_csv(ms_dir/"ms_val_data.csv",index_col=0)
ms_test_base=pd.read_csv(ms_dir/"ms_test_data.csv",index_col=0)

escape_dir=DATA_DIR/"Training_Datasets/Processed/ESCAPE_seq"
escape_test_base=pd.read_csv(escape_dir/"escape_con_test_set_cleaned.csv",index_col=0)

# ============================================================
# MHCPrime
# ============================================================
tokenizer=AminoAcidTokenizer()

base_model_params={
    'vocab_size':tokenizer.vocab_size,'embed_dim':480,'fc_hidden_dims':[240,120],
    'fc_dropout':0.1,'n_heads':16,'max_peptide_len':34,'max_mhc_len':34,
    'n_peptide_layers':2,'n_mhc_layers':1,'n_concat_layers':3,'add_pos_enc':True,
    'use_segment_embedder':True,'dim_feedforward_multiplier':1,'debugging':False,
    'fc_output_dim':1,'pooling':'cls','ffnn_use_layernorm':False,'use_feature_table':True,
    'feature_table':processed_feature_table,'sequence_embedder_feature_names':feature_names,
    'sequence_embedder_feature_mode':'concat','sequence_embedder_tokenizer':tokenizer,
    'peptide_start_pos':1,'mhc_start_pos':35,'pos_enc_type':'sincos','seed':42,
    'add_peptide_conformer':True,'peptide_conformer_num_layers':2,
    'peptide_conformer_n_heads':16,'peptide_conformer_conv_kernel':5,'peptide_conformer_dropout':0.0,
    'add_mhc_conformer':False,'mhc_conformer_num_layers':2,'mhc_conformer_n_heads':16,
    'mhc_conformer_conv_kernel':5,'mhc_conformer_dropout':0.0,
    'add_fusion_conformer':True,'fusion_conformer_num_layers':1,'fusion_conformer_n_heads':16,
    'fusion_conformer_conv_kernel':7,'fusion_conformer_dropout':0.0,
    'fusion_conformer_mask_bos':True,'ff_pre_post_swiglu':False,'legacy_init_compat':True,
}

base_model_dir=MODEL_CHECKPOINTS_DIR/"MHCPrime_Base_for_Ablations"/"Checkpoints"
ckpt_path=base_model_dir/"model_final.pt"

clear_all_gpu_memory()
mhcprime=init_model(base_model_params,print_check=False,print_params=True)
mhcprime.load_state_dict(torch.load(ckpt_path,map_location="cpu"),strict=True)
mhcprime.to(device).eval()
print("Loaded MHCPrime:",ckpt_path)

# ============================================================
# BASE CLASSIFICATION PARAMETERS
# ============================================================
pepcl_base_params_cls=dict(
    tokenizer=tokenizer,device=device,steps=steps,output_dir=None,
    save_every_n_steps=save_every_n_steps,seed=42,num_workers=num_workers,collate_fn=collate_fn,
    num_ms_pos=2000,num_nonms_pos=2000,ms_neg_ratio=1.0,nonms_neg_ratio=1.0,
    auto_calibrate=True,calibration_steps=1,ms_batch_size=batch_size,nonms_batch_size=batch_size,
    encoder_lr=learning_rate,decoder_lr=learning_rate,weight_decay=0.01,warmup_ratio=0.1,
    update_domain_embeddings=False,primary_task_id=1,sequential_backprop=True,
    ms_loss_type="logsmoothap",ms_loss_alpha=10.0,ms_lambda=1.0,
    nonms_loss_type="logsmoothap",nonms_loss_alpha=10.0,ft_lambda=1.0,
    dr_lambda=dr_lambda,dr_loss_type="mse",
)

filtered_columns=[
    'seq','allele','label','seq_allele','n_flank','c_flank',
    'mhc_a_1','mhc_b_1','mhc_c_1','mhc_a_2','mhc_b_2','mhc_c_2',
    'sa_ma','mhcprime','escore','ba_value','name2','name','gene','mutation','mut_type'
]

# ============================================================
# ONE TRAIN/VAL SPLIT RUNNER
# ============================================================

def run_split(escape_train_remaining,escape_val,run_seeds,data_output_dir,model_output_dir,split_name=None):
    data_output_dir=Path(data_output_dir); model_output_dir=Path(model_output_dir)
    log_output_dir=data_output_dir/"logs"
    data_output_dir.mkdir(parents=True,exist_ok=True)
    model_output_dir.mkdir(parents=True,exist_ok=True)
    log_output_dir.mkdir(parents=True,exist_ok=True)

    print("\n"+"="*80)
    print("RUN:",split_name if split_name is not None else "fixed seed split")
    print("Data output:",data_output_dir)
    print("Model output:",model_output_dir)
    print("="*80)

    # Fresh copies because this function mutates domain_id/seq_len.
    ms_train=ms_train_base.copy()
    ms_val=ms_val_base.copy()
    ms_test=ms_test_base.copy()
    escape_train=escape_train_remaining.copy()
    escape_val=escape_val.copy()
    escape_test=escape_test_base.copy()

    # --------------------------------------------------------
    # IDENTICAL MS/ESCAPE merge + allele/length matching.
    # --------------------------------------------------------
    print("Merging MS/ESCAPE training data...")
    ms_train["domain_id"]=0
    escape_train["domain_id"]=1
    ms_escape_train_set=pd.concat([ms_train,escape_train],ignore_index=True)
    print_data_stats(ms_escape_train_set)

    ms_escape_train_set["seq_len"]=ms_escape_train_set.seq.str.replace("$","").apply(len)
    ms_escape_train_set_matched_allele_length=filter_domain0_to_domain1_pairs(
        ms_escape_train_set,allele_col="allele",seq_len_col="seq_len",domain_col="domain_id"
    ).reset_index(drop=True)
    print_data_stats(ms_escape_train_set_matched_allele_length)

    # Extra classification variant: copy domain-0 negatives into domain 1.
    ms_escape_train_set_more_negs=copy_domain0_negs_to_domain1(
        ms_escape_train_set_matched_allele_length,
        allele_col="allele",label_col="label",seq_col="seq",domain_col="domain_id"
    )
    print("\nTraining data with copied domain-0 negatives in domain 1:")
    print_data_stats(ms_escape_train_set_more_negs)

    # --------------------------------------------------------
    # IDENTICAL test/val allele-length matching.
    # --------------------------------------------------------
    print("Matching MS evaluation allele/length support...")
    ms_test["seq_len"]=ms_test.seq.str.replace("$","").apply(len)
    ms_val["seq_len"]=ms_val.seq.str.replace("$","").apply(len)
    escape_test["seq_len"]=escape_test.seq.str.replace("$","").apply(len)
    escape_val["seq_len"]=escape_val.seq.str.replace("$","").apply(len)

    ms_test_matched=filter_to_shared_pairs(ms_test,escape_test,allele_col="allele",seq_len_col="seq_len")
    ms_val_matched=filter_to_shared_pairs(ms_val,escape_val,allele_col="allele",seq_len_col="seq_len")

    ms_test_matched=process_eval_df(ms_test_matched)
    ms_val_matched=process_eval_df(ms_val_matched)
    escape_test=process_eval_df(escape_test)
    escape_val=process_eval_df(escape_val)

    ms_test_matched=filter_single_df_to_cols(ms_test_matched,filtered_columns)
    ms_val_matched=filter_single_df_to_cols(ms_val_matched,filtered_columns)
    escape_test=filter_single_df_to_cols(escape_test,filtered_columns)
    escape_val=filter_single_df_to_cols(escape_val,filtered_columns)

    # Preserve previously generated seed columns if this directory already exists.
    test_sets={
        "escape_test":add_existing_columns(escape_test,data_output_dir/"escape_test.csv"),
        "ms_test":add_existing_columns(ms_test_matched,data_output_dir/"ms_test.csv"),
        "ms_val":add_existing_columns(ms_val_matched,data_output_dir/"ms_val.csv"),
    }
    val_sets={"escape_val":add_existing_columns(escape_val,data_output_dir/"escape_val.csv")}

    # --------------------------------------------------------
    # Caches.
    # --------------------------------------------------------
    cache_tag=f"{GPU_ID}_{'fixed' if split_name is None else split_name}"
    test_cache_paths={}
    val_cache_paths={}

    for name,df in test_sets.items():
        path=DATA_DIR/f"{name}_{cache_tag}_cls_cache.pt"
        build_token_cache(df,path,tokenizer,use_domain_id=False)
        test_cache_paths[name]=path

    for name,df in val_sets.items():
        path=DATA_DIR/f"{name}_{cache_tag}_cls_cache.pt"
        build_token_cache(df,path,tokenizer,use_domain_id=False)
        val_cache_paths[name]=path

    base_model=copy.deepcopy(mhcprime)

    # Add base MHCPrime only if not already present.
    for name,df in test_sets.items():
        with torch.no_grad():
            scores=run_mhcprime_fast(base_model,test_cache_paths[name],batch_size=inference_batch_size,
                                        num_workers=inference_num_workers,compile_model=False,use_domain_id=False)
        df["mhcprime"]=scores.detach().cpu().tolist()

    for name,df in val_sets.items():
        with torch.no_grad():
            scores=run_mhcprime_fast(base_model,val_cache_paths[name],batch_size=inference_batch_size,
                                        num_workers=inference_num_workers,compile_model=False,use_domain_id=False)
        df["mhcprime"]=scores.detach().cpu().tolist()

    # ========================================================
    # RUN SEEDS
    # ========================================================
    for pepcl_seed in run_seeds:
        print("\n"+"="*80)
        print(f"SEED {pepcl_seed}")
        print("="*80)

        params=pepcl_base_params_cls.copy()
        params["seed"]=pepcl_seed

        # Four independent copies of the same incoming MHCPrime.
        escape_seq_neoant_ft_cls=copy.deepcopy(mhcprime)
        escape_seq_neoant_pepcl_cls=copy.deepcopy(mhcprime)
        escape_seq_neoant_ft_w_more_negs_cls=copy.deepcopy(mhcprime)
        escape_seq_neoant_pepcl_w_more_negs_cls=copy.deepcopy(mhcprime)
        clear_all_gpu_memory()

        # Exact requested model names + seed suffix.
        names={
            "ft":f"escape_seq_neoant_ft_cls_seed_{pepcl_seed}",
            "pepcl":f"escape_seq_neoant_pepcl_cls_seed_{pepcl_seed}",
            "ft_more":f"escape_seq_neoant_ft_w_more_negs_cls_seed_{pepcl_seed}",
            "pepcl_more":f"escape_seq_neoant_pepcl_w_more_negs_cls_seed_{pepcl_seed}",
        }

        # Per-seed model directory. For CV, caller already supplies datasplit_N root.
        seed_model_dir=model_output_dir/f"seed_{pepcl_seed}"
        seed_model_dir.mkdir(parents=True,exist_ok=True)

        # ----------------------------------------------------
        # 1. Standard FT
        # ----------------------------------------------------
        print("\n1/4 FT classification")
        ft_params=params.copy()
        ft_params["dr_lambda"]=0.0
        trainer=PepCL_Logging_Ext(
            model=escape_seq_neoant_ft_cls,
            train_df=ms_escape_train_set_matched_allele_length.query("domain_id == 1").copy(),
            use_replay=False,mhc_seq_dict=mhc_seq_dict,enable_logging=True,**ft_params
        )
        training_stats=trainer.train()
        logs=trainer.get_logs(as_dataframes=True)
        logs["training_stats"]=training_stats
        save_log_dict(logs,log_output_dir/f"{names['ft']}.pkl")
        save_model(escape_seq_neoant_ft_cls,seed_model_dir/f"{names['ft']}.pt")

        # ----------------------------------------------------
        # 2. Standard PepCL
        # ----------------------------------------------------
        print("\n2/4 PepCL classification")
        pepcl_params=params.copy()
        pepcl_params["dr_lambda"]=dr_lambda
        trainer=PepCL_Logging_Ext(
            model=escape_seq_neoant_pepcl_cls,
            train_df=ms_escape_train_set_matched_allele_length,
            use_replay=True,mhc_seq_dict=mhc_seq_dict,enable_logging=True,**pepcl_params
        )
        training_stats=trainer.train()
        logs=trainer.get_logs(as_dataframes=True)
        logs["training_stats"]=training_stats
        save_log_dict(logs,log_output_dir/f"{names['pepcl']}.pkl")
        save_model(escape_seq_neoant_pepcl_cls,seed_model_dir/f"{names['pepcl']}.pt")

        # ----------------------------------------------------
        # 3. FT + copied MS negatives in domain 1
        # ----------------------------------------------------
        print("\n3/4 FT classification + more negatives")
        ft_more_params=params.copy()
        ft_more_params["dr_lambda"]=0.0
        trainer=PepCL_Logging_Ext(
            model=escape_seq_neoant_ft_w_more_negs_cls,
            train_df=ms_escape_train_set_more_negs.query("domain_id == 1").copy(),
            use_replay=False,mhc_seq_dict=mhc_seq_dict,enable_logging=True,**ft_more_params
        )
        training_stats=trainer.train()
        logs=trainer.get_logs(as_dataframes=True)
        logs["training_stats"]=training_stats
        save_log_dict(logs,log_output_dir/f"{names['ft_more']}.pkl")
        save_model(escape_seq_neoant_ft_w_more_negs_cls,seed_model_dir/f"{names['ft_more']}.pt")

        # ----------------------------------------------------
        # 4. PepCL + copied MS negatives in domain 1
        # ----------------------------------------------------
        print("\n4/4 PepCL classification + more negatives")
        pepcl_more_params=params.copy()
        pepcl_more_params["dr_lambda"]=dr_lambda
        trainer=PepCL_Logging_Ext(
            model=escape_seq_neoant_pepcl_w_more_negs_cls,
            train_df=ms_escape_train_set_more_negs,
            use_replay=True,mhc_seq_dict=mhc_seq_dict,enable_logging=True,**pepcl_more_params
        )
        training_stats=trainer.train()
        logs=trainer.get_logs(as_dataframes=True)
        logs["training_stats"]=training_stats
        save_log_dict(logs,log_output_dir/f"{names['pepcl_more']}.pkl")
        save_model(escape_seq_neoant_pepcl_w_more_negs_cls,seed_model_dir/f"{names['pepcl_more']}.pt")

        clear_all_gpu_memory()

        final_models={
            names["ft"]:escape_seq_neoant_ft_cls,
            names["pepcl"]:escape_seq_neoant_pepcl_cls,
            names["ft_more"]:escape_seq_neoant_ft_w_more_negs_cls,
            names["pepcl_more"]:escape_seq_neoant_pepcl_w_more_negs_cls,
        }

        # ----------------------------------------------------
        # Final-model inference.
        # All seed columns accumulate in the same dataframes.
        # ----------------------------------------------------
        print("\nRunning test/MS inference...")
        for set_name,df in test_sets.items():
            test_sets[set_name]=run_inference_models(final_models,df,test_cache_paths[set_name])

        print("\nRunning validation inference...")
        for set_name,df in val_sets.items():
            val_sets[set_name]=run_inference_models(final_models,df,val_cache_paths[set_name])

        # Save after every seed so completed work is preserved.
        print("\nSaving output dataframes...")
        for name,df in test_sets.items():
            df.to_csv(data_output_dir/f"{name}.csv")
            print("Saved:",data_output_dir/f"{name}.csv")
        for name,df in val_sets.items():
            df.to_csv(data_output_dir/f"{name}.csv")
            print("Saved:",data_output_dir/f"{name}.csv")

    # Remove caches for this split.
    for path in list(test_cache_paths.values())+list(val_cache_paths.values()):
        if os.path.exists(path): os.remove(path)

# ============================================================
# SEED MODE
# ============================================================
if run_mode=="seed":
    print("\nRUNNING FIXED-SPLIT SEED MODE")

    escape_train=pd.read_csv(escape_dir/"escape_con_train_set_remaining_cleaned.csv",index_col=0)
    escape_val=pd.read_csv(escape_dir/"escape_con_val_set_cleaned.csv",index_col=0)

    run_split(
        escape_train_remaining=escape_train,
        escape_val=escape_val,
        run_seeds=seeds,
        data_output_dir=data_output_root,
        model_output_dir=model_output_root,
        split_name=None,
    )

# ============================================================
# CV MODE
# ============================================================
else:
    print("\nRUNNING CV MODE")

    escape_con_train_val_sets=load_dict_pickle(
        escape_dir/"escape_con_train_val_sets_cleaned.pkl"
    )

    for ds_split in cv_splits:
        if ds_split not in escape_con_train_val_sets:
            raise KeyError(
                f"Requested CV datasplit {ds_split} not found. "
                f"Available: {list(escape_con_train_val_sets)}"
            )

        print(f"\n{'#'*80}\nDATA SPLIT {ds_split}\n{'#'*80}")
        split_dict=escape_con_train_val_sets[ds_split]

        run_split(
            escape_train_remaining=split_dict["train"],
            escape_val=split_dict["val"],
            run_seeds=[cv_seed],
            data_output_dir=data_output_root/f"datasplit_{ds_split}",
            model_output_dir=model_output_root/f"datasplit_{ds_split}",
            split_name=f"datasplit_{ds_split}",
        )

print("\nAll requested ESCAPE-seq classification runs complete.")