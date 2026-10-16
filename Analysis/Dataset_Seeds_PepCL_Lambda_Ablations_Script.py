"""
Performs lambda variations across the default or assay-specific LR and step parameters and also does DR only optionally.

Run cmd: python Dataset_Seeds_PepCL_Lambda_Ablations_Script.py -g gpu_id
"""


import sys
from pathlib import Path
import shutil

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "Data"
SRC_DIR = PROJECT_ROOT / "src"
ANALYSIS_DIR = PROJECT_ROOT / "Analysis"

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

import argparse
parser = argparse.ArgumentParser(description="Select specific GPU to run model on.")
parser.add_argument(
    "-g", "--gpu",
    type=int,
    default=0,
    help="Which GPU ID to use (e.g. 0, 1, 2...)"
)
args = parser.parse_args()
gpu_id = args.gpu

# set gpu id
GPU_ID = gpu_id
torch.cuda.set_device(GPU_ID)
print(f"Using GPU: {torch.cuda.get_device_name(GPU_ID)} (ID: {torch.cuda.current_device()})")

# set device
device = f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu'

# mhc pseudosequences (loaded above)
mhc_seq_dict={k.removeprefix("HLA-").replace("*","").replace(":",""):v for k,v in mhc_seq_dict.items()}

# get processed feature data
print("Loading feature data...")
aa_features_table_ext, descriptor_groups=load_feature_data()
processed_feature_table, corr_info = process_aa_feature_table(aa_features_table_ext)
feature_names = list(processed_feature_table.columns) # pass to FeatureProjectionEncoder
print(end="\n")

# ============================================================
# SELECT DATASET
# ============================================================
valid_datasets_to_path_prefix={"yd": "YD", "episcan_r9": "EpiScan_R9", "episcan_r9rt": "EpiScan_R9_RT", "ba_con": "BA_CON", "escape_con": "ESCAPE_seq_CON"}

default_lr_steps={"lr": 2e-4, "steps": 300}
default_lr=default_lr_steps["lr"]
default_steps=default_lr_steps["steps"]

# use either default or custom config
step_lr_configs = {
    "default": {"lr": 2e-4, "steps": 300},
    "yd": {"lr": 2e-4, "steps": 200},
    "episcan_r9": {"lr": 2e-4, "steps": 800},
    "episcan_r9rt": {"lr": 2e-4, "steps": 400},
    "ba_con": {"lr": 2e-4, "steps": 800},
    "escape_con": {"lr": 2e-4, "steps": 300},
}


# pepcl and dr
run_pepcl_r_dr=True
# run_pepcl_r_dr=False

 # dr only
run_pepcl_dr=True
# run_pepcl_dr=False

# which step/lr config to use
step_lr_config_to_use="default"
# step_lr_config_to_use="custom"

# "yd", "episcan_r9", "episcan_r9rt", "ba_con", "escape_con"
# dataset_to_run="yd" 
# dataset_to_run="episcan_r9"
# dataset_to_run="episcan_r9rt"
# dataset_to_run="ba_con"
dataset_to_run="escape_con"

path_prefix=valid_datasets_to_path_prefix[dataset_to_run]

# for pepcl_r_dr
# full_output_dir=f"{DATA_DIR}/{path_prefix}_Lambda_Ablations_w_Alt_MHCPrime_Default_Params_1_Seed/"

# for pepcl_dr (dr only)
# full_output_dir=f"{DATA_DIR}/{path_prefix}_Lambda_Ablations_w_Alt_MHCPrime_DR_Only_Default_Params_1_Seed/"

# for pepcl and pepcl_dr across 4 remaining seeds
# full_output_dir=f"{DATA_DIR}/{path_prefix}_Lambda_Ablations_w_Alt_MHCPrime_Remaining_4_Seeds/"

# for pepcl and dr across all 5 seeds but with each assays custom config
# full_output_dir=f"{DATA_DIR}/{path_prefix}_Lambda_Ablations_w_Alt_MHCPrime_Full_5_Seeds_Custom_LR_Steps/"

# full_output_dir=f"{DATA_DIR}/{path_prefix}_Lambda_Ablations_w_Alt_MHCPrime_Default_Params_Full_5_Seeds_Cleaned/"

# for escape dr only and ft ablations
# full_output_dir=f"{DATA_DIR}/{path_prefix}_Lambda_Ablations_w_Alt_MHCPrime_Default_Params_Full_5_Seeds_Cleaned_FT_DR_Only/"

# for getting test set performance on final config
full_output_dir=f"{DATA_DIR}/{path_prefix}_Lambda_Ablations_w_Alt_MHCPrime_Default_Params_Full_5_Seeds_w_Test/"

print("Output dir.:", full_output_dir)

# set default or custom step and lr to use

if step_lr_config_to_use == "custom":
    default_lr_steps=step_lr_configs[dataset_to_run]
else:
    default_lr_steps=step_lr_configs[step_lr_config_to_use]
default_lr=default_lr_steps["lr"]
default_steps=default_lr_steps["steps"]


valid_datasets={"yd","episcan_r9","episcan_r9rt","ba_con","escape_con"}
if dataset_to_run not in valid_datasets:
    raise ValueError(f"dataset_to_run must be one of {sorted(valid_datasets)}, got '{dataset_to_run}'.")

print(f"\n{'='*80}")
print(f"LOADING DATASET: {dataset_to_run}")
print(f"{'='*80}\n")

full_log_output_dir=f"{full_output_dir}/logs"
# check before creating directories
if os.path.exists(full_output_dir):
    response = input(
        f"\nWARNING: Output directory already exists:\n{full_output_dir}\n"
        "Continuing may overwrite existing outputs.\n"
        "Continue? [y/n]: "
    ).strip().lower()

    if response not in {"y", "yes"}:
        raise SystemExit("Stopped by user. Existing output directory was not modified.")

    print("Continuing with existing output directory. Existing files may be overwritten.\n")

print("Output directory for everything:", full_output_dir)
os.makedirs(full_output_dir, exist_ok=True)
print("Output directory for logs:", full_log_output_dir)
os.makedirs(full_log_output_dir, exist_ok=True)


# ============================================================
# LOAD COMMON MS DATA
# ============================================================

print("Loading MS data...")
ms_train_test_dir=f"{DATA_DIR}/Training_Datasets/Processed/MS_Train_Test_Sets"
ms_train_data=pd.read_csv(f"{ms_train_test_dir}/ms_train_data_remaining_minimal.csv",index_col=0)
ms_val_data=pd.read_csv(f"{ms_train_test_dir}/ms_val_data.csv",index_col=0)
ms_test_data=pd.read_csv(f"{ms_train_test_dir}/ms_test_data.csv",index_col=0)
ms_train_data["domain_id"]=0

# ============================================================
# YD / HOLEC
# ============================================================

if dataset_to_run=="yd":
    dataset_prefix="holec"
    task_type="classification"

    print("Loading YD data...")
    data_dir=f"{DATA_DIR}/Training_Datasets/Processed/Holec_et_al_2025"
    ft_train=pd.read_csv(f"{data_dir}/holec_train_set_remaining.csv",index_col=0)
    ft_val=pd.read_csv(f"{data_dir}/holec_val_set.csv",index_col=0)
    ft_test=pd.read_csv(f"{data_dir}/holec_test_set.csv",index_col=0)
    ft_train["domain_id"]=1

    # MS + FT train
    train_set=pd.concat([ms_train_data,ft_train],ignore_index=True)
    train_set["seq_len"]=train_set.seq.str.replace("$","").apply(len)
    train_set=train_set[train_set.allele.isin(ft_train.allele.unique())].query("seq_len == 9").reset_index(drop=True)

    # Match MS val/test to assay allele/length support
    ms_test_data["seq_len"]=ms_test_data.seq.str.replace("$","").apply(len)
    ms_val_data["seq_len"]=ms_val_data.seq.str.replace("$","").apply(len)
    ms_test_matched=ms_test_data[ms_test_data.allele.isin(ft_test.allele.unique())].query("seq_len == 9").copy()
    ms_val_matched=ms_val_data[ms_val_data.allele.isin(ft_val.allele.unique())].query("seq_len == 9").copy()

    for name,df in [("ms_test",ms_test_matched),("ms_val",ms_val_matched),("ft_test",ft_test),("ft_val",ft_val)]:
        df=full_process_data(df,mhc_seq_dict,remove_flank=True,add_pseudo_label=False)
        df=create_seq_allele_column(df)
        if name=="ms_test": ms_test_matched=df
        elif name=="ms_val": ms_val_matched=df
        elif name=="ft_test": ft_test=df
        else: ft_val=df

    filtered_columns=['seq','allele','label','seq_allele','n_flank','c_flank',
        'mhc_a_1','mhc_b_1','mhc_c_1','mhc_a_2','mhc_b_2','mhc_c_2',
        'sa_ma','mhcprime','group']

    ms_test_matched=filter_single_df_to_cols(ms_test_matched,filtered_columns)
    ms_val_matched=filter_single_df_to_cols(ms_val_matched,filtered_columns)
    ft_test=filter_single_df_to_cols(ft_test,filtered_columns)
    ft_val=filter_single_df_to_cols(ft_val,filtered_columns)

    test_sets={"holec_test": ft_test, "ms_val":ms_val_matched, "ms_test": ms_test_matched}
    val_sets={"holec_val":ft_val}


# ============================================================
# EPISCAN R9 / R9RT
# ============================================================

elif dataset_to_run in {"episcan_r9","episcan_r9rt"}:
    dataset_prefix=dataset_to_run
    task_type="classification"

    print("Loading EpiScan data...")
    data_dir=f"{DATA_DIR}/Training_Datasets/Processed/EpiScan"

    r9_train=pd.read_csv(f"{data_dir}/episcan_r9_train_set_remaining.csv",index_col=0)
    rt_train=pd.read_csv(f"{data_dir}/episcan_rt_train_set_remaining.csv",index_col=0)
    r9_val=pd.read_csv(f"{data_dir}/episcan_r9_val_set.csv",index_col=0)
    rt_val=pd.read_csv(f"{data_dir}/episcan_rt_val_set.csv",index_col=0)
    r9_test=pd.read_csv(f"{data_dir}/episcan_r9_test_set.csv",index_col=0)
    rt_test=pd.read_csv(f"{data_dir}/episcan_rt_test_set.csv",index_col=0)
    sd3_formatted=pd.read_csv(f"{data_dir}/sd3_formatted.csv",index_col=0)

    r9_train["domain_id"]=1
    rt_train["domain_id"]=1

    # R9: R9 +/- plus RT positives only
    if dataset_to_run=="episcan_r9":
        ft_train=pd.concat([r9_train,rt_train.query("label == 1")],ignore_index=True)

    # R9RT: R9 +/- plus RT +/-
    else:
        ft_train=pd.concat([r9_train,rt_train],ignore_index=True)

    train_set=pd.concat([ms_train_data,ft_train],ignore_index=True)
    train_set["seq_len"]=train_set.seq.apply(len)

    iso_alleles=['A0201','A0301','B0801','B5701']
    train_set=train_set[train_set.allele.isin(iso_alleles)].query("seq_len == 9").reset_index(drop=True)

    # Combined R9/RT validation and test sets
    r9_val["dataset_source"]="r9"; rt_val["dataset_source"]="rt"
    ft_val=pd.concat([r9_val,rt_val],ignore_index=True)

    r9_test["dataset_source"]="r9"; rt_test["dataset_source"]="rt"
    ft_test=pd.concat([r9_test,rt_test],ignore_index=True)

    # Match MS support
    ms_test_data["seq_len"]=ms_test_data.seq.str.replace("$","").apply(len)
    ms_val_data["seq_len"]=ms_val_data.seq.str.replace("$","").apply(len)
    ms_test_matched=ms_test_data[ms_test_data.allele.isin(iso_alleles)].query("seq_len == 9").copy()
    ms_val_matched=ms_val_data[ms_val_data.allele.isin(iso_alleles)].query("seq_len == 9").copy()

    for name,df in [("ms_test",ms_test_matched),("ms_val",ms_val_matched),("ft_test",ft_test),("ft_val",ft_val),("sd3",sd3_formatted)]:
        df=full_process_data(df,mhc_seq_dict,remove_flank=True,add_pseudo_label=False)
        df=create_seq_allele_column(df)
        if name=="ms_test": ms_test_matched=df
        elif name=="ms_val": ms_val_matched=df
        elif name=="ft_test": ft_test=df
        elif name=="ft_val": ft_val=df
        else: sd3_formatted=df

    filtered_columns=['seq','allele','label','seq_allele','n_flank','c_flank',
        'mhc_a_1','mhc_b_1','mhc_c_1','mhc_a_2','mhc_b_2','mhc_c_2',
        'sa_ma','mhcprime','dataset_source']

    ms_test_matched=filter_single_df_to_cols(ms_test_matched,filtered_columns)
    ms_val_matched=filter_single_df_to_cols(ms_val_matched,filtered_columns)
    ft_test=filter_single_df_to_cols(ft_test,filtered_columns)
    ft_val=filter_single_df_to_cols(ft_val,filtered_columns)
    sd3_formatted=filter_single_df_to_cols(sd3_formatted,filtered_columns)

    test_sets={
        "episcan_r9_rt_test":ft_test,
        "cov_test":sd3_formatted,
        "ms_val":ms_val_matched,
        "ms_test": ms_test_matched,
    }
    val_sets={"episcan_r9_rt_val":ft_val}


# ============================================================
# BA CONTINUOUS
# ============================================================

elif dataset_to_run=="ba_con":
    dataset_prefix="ba_con"
    task_type="continuous"

    print("Loading BA data...")
    data_dir=f"{DATA_DIR}/Training_Datasets/Processed/BA"
    ft_train=pd.read_csv(f"{data_dir}/train_df_ba_for_updating_remaining.csv",index_col=0)
    ft_val=pd.read_csv(f"{data_dir}/ba_val_set.csv",index_col=0)
    ft_test=pd.read_csv(f"{data_dir}/test_df_ba_for_updating.csv",index_col=0)
    ft_train["domain_id"]=1

    # MS + FT train, then match domain 0 to domain 1 allele/length support
    train_set=pd.concat([ms_train_data,ft_train],ignore_index=True)
    train_set["seq_len"]=train_set.seq.str.replace("$","").apply(len)
    train_set=filter_domain0_to_domain1_pairs(
        train_set,allele_col="allele",seq_len_col="seq_len",domain_col="domain_id"
    ).reset_index(drop=True)

    ms_test_data["seq_len"]=ms_test_data.seq.str.replace("$","").apply(len)
    ms_val_data["seq_len"]=ms_val_data.seq.str.replace("$","").apply(len)
    ft_test["seq_len"]=ft_test.seq.str.replace("$","").apply(len)
    ft_val["seq_len"]=ft_val.seq.str.replace("$","").apply(len)

    ms_test_matched=filter_to_shared_pairs(ms_test_data,ft_test,allele_col="allele",seq_len_col="seq_len")
    ms_val_matched=filter_to_shared_pairs(ms_val_data,ft_val,allele_col="allele",seq_len_col="seq_len")

    for name,df in [("ms_test",ms_test_matched),("ms_val",ms_val_matched),("ft_test",ft_test),("ft_val",ft_val)]:
        df=full_process_data(df,mhc_seq_dict,remove_flank=True,add_pseudo_label=False)
        df=create_seq_allele_column(df)
        if name=="ms_test": ms_test_matched=df
        elif name=="ms_val": ms_val_matched=df
        elif name=="ft_test": ft_test=df
        else: ft_val=df

    filtered_columns=['seq','allele','label','seq_allele','n_flank','c_flank',
        'mhc_a_1','mhc_b_1','mhc_c_1','mhc_a_2','mhc_b_2','mhc_c_2',
        'sa_ma','mhcprime','ba_value']

    ms_test_matched=filter_single_df_to_cols(ms_test_matched,filtered_columns)
    ms_val_matched=filter_single_df_to_cols(ms_val_matched,filtered_columns)
    ft_test=filter_single_df_to_cols(ft_test,filtered_columns)
    ft_val=filter_single_df_to_cols(ft_val,filtered_columns)

    test_sets={"ba_test":ft_test,"ms_val":ms_val_matched, "ms_test": ms_test_matched}
    val_sets={"ba_val":ft_val}


# ============================================================
# ESCAPE-SEQ CONTINUOUS
# ============================================================

elif dataset_to_run=="escape_con":
    dataset_prefix="escape_con"
    task_type="continuous"

    print("Loading ESCAPE-seq data...")
    data_dir=f"{DATA_DIR}/Training_Datasets/Processed/ESCAPE_seq"

    ft_train=pd.read_csv(f"{DATA_DIR}/Training_Datasets/Processed/ESCAPE_seq/escape_con_train_set_remaining_cleaned.csv", index_col=0)
    ft_val=pd.read_csv(f"{DATA_DIR}/Training_Datasets/Processed/ESCAPE_seq/escape_con_val_set_cleaned.csv", index_col=0)
    ft_test=pd.read_csv(f"{DATA_DIR}/Training_Datasets/Processed/ESCAPE_seq/escape_con_test_set_cleaned.csv", index_col=0)

    ft_train["domain_id"]=1

    # MS + FT train, then match domain 0 to domain 1 allele/length support
    train_set=pd.concat([ms_train_data,ft_train],ignore_index=True)
    train_set["seq_len"]=train_set.seq.str.replace("$","").apply(len)
    train_set=filter_domain0_to_domain1_pairs(
        train_set,allele_col="allele",seq_len_col="seq_len",domain_col="domain_id"
    ).reset_index(drop=True)

    ms_test_data["seq_len"]=ms_test_data.seq.str.replace("$","").apply(len)
    ms_val_data["seq_len"]=ms_val_data.seq.str.replace("$","").apply(len)
    ft_test["seq_len"]=ft_test.seq.str.replace("$","").apply(len)
    ft_val["seq_len"]=ft_val.seq.str.replace("$","").apply(len)

    ms_test_matched=filter_to_shared_pairs(ms_test_data,ft_test,allele_col="allele",seq_len_col="seq_len")
    ms_val_matched=filter_to_shared_pairs(ms_val_data,ft_val,allele_col="allele",seq_len_col="seq_len")

    for name,df in [("ms_test",ms_test_matched),("ms_val",ms_val_matched),("ft_test",ft_test),("ft_val",ft_val)]:
        df=full_process_data(df,mhc_seq_dict,remove_flank=True,add_pseudo_label=False)
        df=create_seq_allele_column(df)
        if name=="ms_test": ms_test_matched=df
        elif name=="ms_val": ms_val_matched=df
        elif name=="ft_test": ft_test=df
        else: ft_val=df

    filtered_columns=['seq','allele','label','seq_allele','n_flank','c_flank',
        'mhc_a_1','mhc_b_1','mhc_c_1','mhc_a_2','mhc_b_2','mhc_c_2',
        'sa_ma','mhcprime','escore','ba_value','name2','name','gene','mutation','mut_type']

    ms_test_matched=filter_single_df_to_cols(ms_test_matched,filtered_columns)
    ms_val_matched=filter_single_df_to_cols(ms_val_matched,filtered_columns)
    ft_test=filter_single_df_to_cols(ft_test,filtered_columns)
    ft_val=filter_single_df_to_cols(ft_val,filtered_columns)

    test_sets={"escape_test":ft_test,"ms_val":ms_val_matched, "ms_test": ms_test_matched}
    val_sets={"escape_val":ft_val}


# ============================================================
# COMMON OUTPUTS FOR DOWNSTREAM TRAINING
# ============================================================

# train_set always contains domain 0 = MS and domain 1 = update assay.
# FT gets only domain 1; PepCL gets the complete train_set.
ft_train_set=train_set.query("domain_id == 1").copy()
pepcl_train_set=train_set

print(f"\nDataset: {dataset_to_run}")
print(f"Task type: {task_type}")
print(f"Dataset prefix: {dataset_prefix}")
print(f"FT train rows: {len(ft_train_set):,}")
print(f"PepCL train rows: {len(pepcl_train_set):,}")
print(f"Test sets: {list(test_sets)}")
print(f"Validation sets: {list(val_sets)}")
print_data_stats(pepcl_train_set)


"""
Load model
"""
print("Load MHCPrime for ablations...")

# setup tokenizer
tokenizer = AminoAcidTokenizer()

# set base model params
base_model_params={
        'vocab_size': tokenizer.vocab_size,
        'embed_dim': 480,
        'fc_hidden_dims': [240, 120], 
        'fc_dropout': 0.1,
        'n_heads': 16,
        'max_peptide_len': 34,
        'max_mhc_len': 34,
        "n_peptide_layers": 2,
        "n_mhc_layers": 1,
        'n_concat_layers': 3,
        'add_pos_enc': True,
        'use_segment_embedder': True,
        'dim_feedforward_multiplier': 1,
        'debugging': False,
        'fc_output_dim': 1,
        'pooling': 'cls', # cls, mean, min, max, attn, for sequence ffnn head
        "ffnn_use_layernorm": False, 
        'use_feature_table': True,
        "feature_table": processed_feature_table,
        "sequence_embedder_feature_names": feature_names,
        'sequence_embedder_feature_mode': "concat",  # replace or concat
        'sequence_embedder_tokenizer': tokenizer,
        "peptide_start_pos": 1, 
        "mhc_start_pos": 35, 
        "pos_enc_type": "sincos", 
        "seed": 42,

        # peptide conformer
        "add_peptide_conformer": True, 
        "peptide_conformer_num_layers": 2,
        "peptide_conformer_n_heads": 16,
        "peptide_conformer_conv_kernel": 5,
        "peptide_conformer_dropout": 0.0,

        # mhc conformer
        "add_mhc_conformer": False, 
        "mhc_conformer_num_layers": 2,
        "mhc_conformer_n_heads": 16,
        "mhc_conformer_conv_kernel": 5,
        "mhc_conformer_dropout": 0.0,

        # fusion conformer parameters
        "add_fusion_conformer": True, 
        "fusion_conformer_num_layers": 1,
        "fusion_conformer_n_heads": 16,
        "fusion_conformer_conv_kernel": 7,
        "fusion_conformer_dropout": 0.0,
        "fusion_conformer_mask_bos": True,
        "ff_pre_post_swiglu": False,

        # legacy init
        "legacy_init_compat": True,
}

# load base model (optionally, can load model_params directly)
MODEL_CHECKPOINTS_DIR = PROJECT_ROOT / "Model_Checkpoints"
base_model_dir = MODEL_CHECKPOINTS_DIR / "MHCPrime_Base_for_Ablations" / "Checkpoints"
# base_model_dir = MODEL_CHECKPOINTS_DIR / "MHCPrime_Base" / "Checkpoints"
ckpt_path = base_model_dir / "model_final.pt"

device = torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")

clear_all_gpu_memory()

mhcprime = init_model(
    base_model_params,
    print_check=False,
    print_params=True,
)

state_dict = torch.load(
    ckpt_path,
    map_location="cpu",
)

mhcprime.load_state_dict(state_dict, strict=True)
mhcprime.to(device)
mhcprime.eval()

print("Model loaded from checkpoint...")
print(f"Checkpoint: {ckpt_path}")
print(f"Model device: {next(mhcprime.parameters()).device}")

print(end="\n")

"""
Create test set caches for inference
"""
print("Creating test and val set caches for inference...")
test_set_cache_paths={}

for test_set_name, test_data_set in test_sets.items():
    print(test_set_name)
    data_to_run_path=f"{DATA_DIR}/{test_set_name}_{str(gpu_id)}_cache.pt"
    build_token_cache(test_data_set, data_to_run_path, tokenizer, use_domain_id=False)
    test_set_cache_paths[test_set_name]=data_to_run_path

val_set_cache_paths={}

for val_set_name, val_data_set in val_sets.items():
    print(val_set_name)
    data_to_run_path=f"{DATA_DIR}/{val_set_name}_{str(gpu_id)}_cache.pt"
    build_token_cache(val_data_set, data_to_run_path, tokenizer, use_domain_id=False)
    val_set_cache_paths[val_set_name]=data_to_run_path

print(end="\n")

"""
Run model inference for the base MHCPrime model
"""

print("Run base MHCPrime inference once on all test sets...")

# get models
base_model=copy.deepcopy(mhcprime)

base_and_ft_models={
    # base model
    "mhcprime": base_model,
}
print(base_and_ft_models.keys())

for test_set_name, test_data_set in test_sets.items():
    
    data_to_run_path=test_set_cache_paths[test_set_name]
    print(test_set_name, data_to_run_path)

    for model_name, model in base_and_ft_models.items():
        model = model.to(device)
        model.eval()
        with torch.no_grad():
            output_scores=run_mhcprime_fast(model, data_to_run_path, batch_size=3072, num_workers=16, compile_model=False, use_domain_id=False)

        test_data_set[model_name]=output_scores.detach().cpu().tolist()

    # add df back to dictionary
    test_sets[test_set_name]=test_data_set

for val_set_name, val_data_set in val_sets.items():
    
    data_to_run_path=val_set_cache_paths[val_set_name]
    print(val_set_name, data_to_run_path)

    for model_name, model in base_and_ft_models.items():
        model = model.to(device)
        model.eval()
        with torch.no_grad():
            output_scores=run_mhcprime_fast(model, data_to_run_path, batch_size=3072, num_workers=16, compile_model=False, use_domain_id=False)

        val_data_set[model_name]=output_scores.detach().cpu().tolist()

    # add df back to dictionary
    val_sets[val_set_name]=val_data_set

print(end="\n")

print("Saving initial test sets...")
for test_set_name, test_data_set in test_sets.items():
    print(test_set_name)
    test_data_set.to_csv(f"{full_output_dir}/{test_set_name}.csv")
os.listdir(full_output_dir)
print(end="\n")

print("Saving initial val sets...")
for val_set_name, val_data_set in val_sets.items():
    print(val_set_name)
    val_data_set.to_csv(f"{full_output_dir}/{val_set_name}.csv")
os.listdir(full_output_dir)
print(end="\n")


"""
Set base pepcl params
"""

print("Set base parameters for PepCL...")
pepcl_base_params_cls=dict(
            tokenizer=tokenizer,
            device=device,

            # steps
            steps = 200,
            
            output_dir = None,
            # save_every_n_steps = 2000,
            seed = 42,
            num_workers = 16,
            collate_fn = collate_fn,

            # sampling
            num_ms_pos = 2000,
            num_nonms_pos = 2000,
            ms_neg_ratio = 1.0,
            nonms_neg_ratio = 1.0,

            auto_calibrate = True,
            calibration_steps = 1,

            ms_batch_size = 3072,
            nonms_batch_size = 3072,

            # optimization
            encoder_lr = 2e-4,
            decoder_lr = 2e-4,
            weight_decay = 0.01,
            warmup_ratio = 0.1,
            update_domain_embeddings = False,

            # continual learning parameters
            primary_task_id = 1,
            sequential_backprop = True,

            # losses
            ms_loss_type = "logsmoothap",
            ms_loss_alpha = 10.0,
            ms_lambda = 1.0,
            nonms_loss_type = "logsmoothap",
            nonms_loss_alpha = 10.0,
            ft_lambda = 1.0,

            # drift regularization
            dr_lambda = 1e-5,
            dr_loss_type = "mse"
)

# for continuous value training
pepcl_base_params_con=dict(
            tokenizer=tokenizer,
            device=device,
            steps = 200,
            output_dir = None,
            save_every_n_steps = 2000,
            seed = 42,
            num_workers = 16,
            collate_fn = collate_fn,
            num_ms_pos = 2000,
            num_nonms_pos = 2000,
            ms_neg_ratio = 1.0,
            nonms_neg_ratio = 1.0,
            auto_calibrate = True,
            calibration_steps = 1,
            ms_batch_size = 3072,
            nonms_batch_size = 3072,
            encoder_lr = 2e-4,
            decoder_lr = 2e-4,
            weight_decay = 0.01,
            warmup_ratio = 0.1,
            update_domain_embeddings = False,
            primary_task_id = 1,
            sequential_backprop = True,
            ms_loss_type = "logsmoothap",
            ms_loss_alpha = 10.0,
            ms_lambda = 1.0,
            nonms_loss_type = "spearman",
            nonms_loss_alpha = 0.01,
            ft_lambda = 1.0,
            dr_lambda = 1e-5,
            dr_loss_type = "mse",
            continuous_target_col="ba_value",
            continuous_ft_sampling_mode="allele_aware",
)

if dataset_to_run in ["ba_con", "escape_con"]:
    print("Using CON PepCL parameters")
    pepcl_base_params=pepcl_base_params_con.copy()

else:
    print("Using CLS PepCL parameters")
    pepcl_base_params=pepcl_base_params_cls.copy()

print("Setting default LR and Steps...")
pepcl_base_params["steps"]=default_steps
pepcl_base_params["encoder_lr"]=default_lr
pepcl_base_params["decoder_lr"]=default_lr
print("LR and steps being used: ", pepcl_base_params["encoder_lr"], pepcl_base_params["decoder_lr"], pepcl_base_params["steps"])


# set seeds and lambda list
# seeds = [0]
# seeds = [1, 2, 3, 4] # run remaining seeds
seeds = [0, 1, 2, 3, 4]
# dr_lambda_list=[0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
dr_lambda_list=[0, 1e-5]

# dr_lambda_strings=lr_to_strings(dr_lambda_list)

# NOTE: set num workers and make sure this is significant
# num_workers=2
num_workers=4
# num_workers=8
# num_workers=16
pepcl_base_params["num_workers"] = num_workers


for pepcl_seed in seeds: 

    for dr_lambda in dr_lambda_list:

        if dr_lambda == 0:
            dr_lambda_string="0"
        else:
            dr_lambda_string=lr_to_string(dr_lambda)
        
        learning_rate_string=lr_to_string(default_lr)
        pepcl_base_params["seed"] = pepcl_seed
        pepcl_base_params["save_every_n_steps"] = None # no logging here
        pepcl_base_params["dr_lambda"]=dr_lambda

        print(f"Seed: {str(pepcl_seed)}")
        print(f"Steps: {str(default_steps)}")
        print(f"LR: {str(default_lr)}")
        print(f"DR Lambda: {str(dr_lambda)}")

        final_models_to_run={}

        pepcl_r_dr_model_name=f"{dataset_prefix}_pepcl_r_dr_steps_{str(default_steps)}_lr_{learning_rate_string}_dr_lambda_{dr_lambda_string}_seed_{str(pepcl_seed)}"

        pepcl_dr_model_name=f"{dataset_prefix}_pepcl_dr_steps_{str(default_steps)}_lr_{learning_rate_string}_dr_lambda_{dr_lambda_string}_seed_{str(pepcl_seed)}"

        if run_pepcl_r_dr:
            print("PepCL w/ replay and drift regularization")
            model_pepcl_r_dr = copy.deepcopy(mhcprime)

            pepcl=PepCL_Logging_Ext(
                        model=model_pepcl_r_dr,
                        train_df=pepcl_train_set, # contains both domain ids
                        use_replay = True,
                        mhc_seq_dict=mhc_seq_dict, # must now be provided
                        enable_logging=False,
                        **pepcl_base_params,
            )
            training_stats=pepcl.train() # store pepcl model states

            final_models_to_run[pepcl_r_dr_model_name]=model_pepcl_r_dr

        if run_pepcl_dr:
            print("PepCL w/ DR only")
            model_pepcl_dr = copy.deepcopy(mhcprime)

            # run updated pepcl function
            pepcl=PepCL_Logging_Ext(
                        model=model_pepcl_dr,
                        train_df=pepcl_train_set, # contains both domain ids
                        use_replay = False, # NOTE: no replay for DR only (0 is basically fine-tuning)
                        mhc_seq_dict=mhc_seq_dict, # must now be provided
                        enable_logging=False,
                        **pepcl_base_params,
            )
            training_stats=pepcl.train() # store pepcl model states

            final_models_to_run[pepcl_dr_model_name]=model_pepcl_dr

        clear_all_gpu_memory()

        # final_models_to_run={pepcl_r_dr_model_name: model_pepcl_r_dr}
        
        for test_set_name, test_data_set in test_sets.items():
            
            data_to_run_path=test_set_cache_paths[test_set_name]
            print(test_set_name, data_to_run_path)

            for model_name, model in final_models_to_run.items(): # NOTE: CHANGED DICT HERE
                model = model.to(device)
                model.eval()
                with torch.no_grad():
                    output_scores=run_mhcprime_fast(model, data_to_run_path, batch_size=3072, num_workers=num_workers, compile_model=False, use_domain_id=False)

                test_data_set[model_name]=output_scores.detach().cpu().tolist()

            # add df back to dictionary
            test_sets[test_set_name]=test_data_set

        for val_set_name, val_data_set in val_sets.items():
            
            data_to_run_path=val_set_cache_paths[val_set_name]
            print(val_set_name, data_to_run_path)

            for model_name, model in final_models_to_run.items(): # NOTE: CHANGED DICT HERE
                model = model.to(device)
                model.eval()
                with torch.no_grad():
                    output_scores=run_mhcprime_fast(model, data_to_run_path, batch_size=3072, num_workers=num_workers, compile_model=False, use_domain_id=False)

                val_data_set[model_name]=output_scores.detach().cpu().tolist()

            # add df back to dictionary
            val_sets[val_set_name]=val_data_set

        print(end="\n")

        print("Saving test sets...")
        for test_set_name, test_data_set in test_sets.items():
            print(test_set_name)
            test_data_set.to_csv(f"{full_output_dir}/{test_set_name}.csv")
        os.listdir(full_output_dir)
        print(end="\n")

        print("Saving val sets...")
        for val_set_name, val_data_set in val_sets.items():
            print(val_set_name)
            val_data_set.to_csv(f"{full_output_dir}/{val_set_name}.csv")
        os.listdir(full_output_dir)
        print(end="\n")

# NOTE: remove caches at the end
for test_name, test_path in test_set_cache_paths.items():
    os.remove(test_path)

for val_name, val_path in val_set_cache_paths.items():
    os.remove(val_path)
