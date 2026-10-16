"""
Run validation-selected FT and PepCL configurations for one dataset across 5 seeds.

Each method may have either one config dict or a list of config dicts. For each selected dataset:
- preserves that dataset's original train/validation/test preprocessing,
- runs every requested FT and PepCL configuration independently for every seed,
- evaluates final FT/PepCL models on assay + matched-MS test sets,
- evaluates logged checkpoints + final models on assay + matched-MS validation sets,
- saves FT/PepCL loss logs,
- optionally saves final per-seed FT/PepCL state_dict checkpoints.

Run cmd:
    python Dataset_Seeds_PepCL_Step_LR_Ablations_Final_Model_Testing.py -g gpu_id
"""
import sys
from pathlib import Path
import os
import copy
import argparse
import warnings
import pandas as pd
import torch

PROJECT_ROOT=Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0,str(PROJECT_ROOT))

DATA_DIR=PROJECT_ROOT/"Data"
SRC_DIR=PROJECT_ROOT/"src"
ANALYSIS_DIR=PROJECT_ROOT/"Analysis"
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

warnings.filterwarnings("ignore")

parser=argparse.ArgumentParser(description="Run final validation-selected FT/PepCL configurations on one dataset.")
parser.add_argument("-g","--gpu",type=int,default=0,help="GPU ID (e.g. 0, 1, 2...).")
args=parser.parse_args()
gpu_id=args.gpu
GPU_ID=gpu_id
torch.cuda.set_device(GPU_ID)
print(f"Using GPU: {torch.cuda.get_device_name(GPU_ID)} (ID: {torch.cuda.current_device()})")
device=torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")

# ============================================================
# USER SETTINGS
# ============================================================

# "yd", "episcan_r9", "episcan_r9rt", "ba_con", "escape_con"

# dataset_to_run="yd"
dataset_to_run="episcan_r9"
# dataset_to_run="episcan_r9rt"
# dataset_to_run="ba_con"
# dataset_to_run="escape_con"


# Validation-selected assay-specific operating points.
# Each method can be either a single config dict or a list of config dicts.
# Examples:
#   "ft":{"lr":2e-4,"steps":300}
#   "pepcl_r_dr":[{"lr":2e-4,"steps":200},{"lr":2e-4,"steps":300}]
selected_configs={
    "yd":{
        "ft":{"lr":2e-4,"steps":300},
        "pepcl_r_dr":[{"lr":2e-4,"steps":200}, {"lr":2e-4,"steps":300}],
    },
    "episcan_r9":{
        # "ft":{"lr":2e-4,"steps":400},
        "ft":{"lr":2e-4,"steps":300},
        # "pepcl_r_dr":[{"lr":2e-4,"steps":800}, {"lr":2e-4,"steps":300}],
        "pepcl_r_dr":[{}],
    },
    "episcan_r9rt":{
        # "ft":{"lr":2e-4,"steps":800},
        "ft":{"lr":2e-4,"steps":400},
        # "pepcl_r_dr":[{"lr":2e-4,"steps":300}], # default and optimal are same
        # "pepcl_r_dr":[{"lr":2e-4,"steps":400}, {"lr":2e-4,"steps":300}], # default and optimal are same
        "pepcl_r_dr":[{"lr":2e-4,"steps":400}], # default and optimal are same
    },
    "ba_con":{
        "ft":{"lr":2e-4,"steps":200},
        "pepcl_r_dr":[{"lr":2e-4,"steps":800}, {"lr":2e-4,"steps":300}],
    },
    "escape_con":{
        "ft":{"lr":2e-4,"steps":200},
        # "pepcl_r_dr":[{"lr":2e-4,"steps":400}, {"lr":2e-4,"steps":300}], # UPDATE FOR ESCAPE-SEQ
        "pepcl_r_dr":[{"lr":2e-4,"steps":300}], # UPDATE FOR ESCAPE-SEQ
    },
}

seeds=[0,1,2,3,4]
num_workers=4
save_log_data=True
run_logged_models=True

# Folder WITHIN Model_Checkpoints for final per-seed FT/PepCL state_dicts.
# Example: model_save_subdir="PepCL_Final_Selected_Models"
# Set to None to save no model checkpoints. CSV outputs and logs are still saved.

# save yd models
model_save_subdir=None

if dataset_to_run=="yd":
    model_save_subdir="YD_Validation_Selected_Models_5_Seeds"
elif dataset_to_run=="episcan_r9":
    # model_save_subdir="EpiScan_R9_Validation_Selected_Models_5_Seeds"
    model_save_subdir="EpiScan_R9_Validation_Selected_Models_5_Seeds_FT_Final"
elif dataset_to_run=="episcan_r9rt":
    # model_save_subdir="EpiScan_R9_RT_Validation_Selected_Models_5_Seeds"
    model_save_subdir="EpiScan_R9_RT_Validation_Selected_Models_5_Seeds_FT_PepCL_Optimal"
elif dataset_to_run=="ba_con":
    model_save_subdir="BA_CON_Validation_Selected_Models_5_Seeds"
elif dataset_to_run=="escape_con":
    # model_save_subdir="ESCAPE_seq_CON_Validation_Selected_Models_5_Seeds"
    model_save_subdir="ESCAPE_seq_CON_Validation_Selected_Models_5_Seeds_Cleaned"
else:
    model_save_subdir=None


valid_datasets_to_path_prefix={
    "yd":"YD",
    "episcan_r9":"EpiScan_R9",
    "episcan_r9rt":"EpiScan_R9_RT",
    "ba_con":"BA_CON",
    "escape_con":"ESCAPE_seq_CON",
}
valid_datasets=set(valid_datasets_to_path_prefix)
if dataset_to_run not in valid_datasets:
    raise ValueError(f"dataset_to_run must be one of {sorted(valid_datasets)}, got '{dataset_to_run}'.")

path_prefix=valid_datasets_to_path_prefix[dataset_to_run]

# full_output_dir=DATA_DIR/f"{path_prefix}_Validation_Selected_Final_5_Seeds" # for yd
# full_output_dir=DATA_DIR/f"{path_prefix}_Validation_Selected_Final_5_Seeds_Cleaned" for escape-seq cleaned
full_output_dir=DATA_DIR/f"{path_prefix}_Validation_Selected_Final_5_Seeds_Updated" # for other datasets


full_log_output_dir=full_output_dir/"logs"
model_output_dir=None if model_save_subdir is None else MODEL_CHECKPOINTS_DIR/model_save_subdir

print(f"\n{'='*90}")
print(f"DATASET: {dataset_to_run}")
print(f"FT config(s): {selected_configs[dataset_to_run]['ft']}")
print(f"PepCL config(s): {selected_configs[dataset_to_run]['pepcl_r_dr']}")
print(f"Output directory: {full_output_dir}")
print(f"Model checkpoint directory: {model_output_dir if model_output_dir is not None else 'None (models will not be saved)'}")
print(f"{'='*90}\n")

if os.path.exists(full_output_dir):
    response=input(
        f"\nWARNING: Output directory already exists:\n{full_output_dir}\n"
        "Continuing may overwrite existing outputs.\n"
        "Continue? [y/n]: "
    ).strip().lower()
    if response not in {"y","yes"}:
        raise SystemExit("Stopped by user. Existing output directory was not modified.")
    print("Continuing with existing output directory. Existing files may be overwritten.\n")

os.makedirs(full_output_dir,exist_ok=True)
os.makedirs(full_log_output_dir,exist_ok=True)
if model_output_dir is not None:
    os.makedirs(model_output_dir,exist_ok=True)

# ============================================================
# COMMON SETUP
# ============================================================

mhc_seq_dict={k.removeprefix("HLA-").replace("*","").replace(":",""):v for k,v in mhc_seq_dict.items()}

print("Loading feature data...")
aa_features_table_ext,descriptor_groups=load_feature_data()
processed_feature_table,corr_info=process_aa_feature_table(aa_features_table_ext)
feature_names=list(processed_feature_table.columns)
print(end="\n")

print("Loading MS data...")
ms_train_test_dir=DATA_DIR/"Training_Datasets"/"Processed"/"MS_Train_Test_Sets"
ms_train_data=pd.read_csv(ms_train_test_dir/"ms_train_data_remaining_minimal.csv",index_col=0)
ms_val_data=pd.read_csv(ms_train_test_dir/"ms_val_data.csv",index_col=0)
ms_test_data=pd.read_csv(ms_train_test_dir/"ms_test_data.csv",index_col=0)
ms_train_data["domain_id"]=0

# ============================================================
# YD / HOLEC
# ============================================================

if dataset_to_run=="yd":
    dataset_prefix="holec"
    task_type="classification"

    print("Loading YD data...")
    holec_train_test_dir=DATA_DIR/"Training_Datasets"/"Processed"/"Holec_et_al_2025"
    holec_train_set_remaining=pd.read_csv(holec_train_test_dir/"holec_train_set_remaining.csv",index_col=0)
    holec_val_set=pd.read_csv(holec_train_test_dir/"holec_val_set.csv",index_col=0)
    holec_test_set=pd.read_csv(holec_train_test_dir/"holec_test_set.csv",index_col=0)

    print("Merging MS/YD data...")
    holec_train_set_remaining["domain_id"]=1
    ms_holec_train_set=pd.concat([ms_train_data,holec_train_set_remaining])
    ms_holec_train_set.reset_index(inplace=True,drop=True)
    print_data_stats(ms_holec_train_set)

    ms_holec_train_set["seq_len"]=ms_holec_train_set.seq.str.replace("$","").apply(len)
    ms_holec_train_set_matched_allele_length=ms_holec_train_set[
        ms_holec_train_set.allele.isin(holec_train_set_remaining.allele.unique().tolist())
    ].query("seq_len == 9")
    ms_holec_train_set_matched_allele_length.reset_index(inplace=True,drop=True)
    print_data_stats(ms_holec_train_set_matched_allele_length)

    print("Matching allele- and length distributions for MS/YD data...")
    ms_test_data["seq_len"]=ms_test_data.seq.str.replace("$","").apply(len)
    ms_val_data["seq_len"]=ms_val_data.seq.str.replace("$","").apply(len)
    ms_test_data_matched_allele_length=ms_test_data[
        ms_test_data.allele.isin(holec_test_set.allele.unique().tolist())
    ].query("seq_len == 9")
    ms_val_data_matched_allele_length=ms_val_data[
        ms_val_data.allele.isin(holec_val_set.allele.unique().tolist())
    ].query("seq_len == 9")

    ms_test_data_matched_allele_length=full_process_data(ms_test_data_matched_allele_length,mhc_seq_dict,remove_flank=True,add_pseudo_label=False)
    ms_test_data_matched_allele_length=create_seq_allele_column(ms_test_data_matched_allele_length)
    ms_val_data_matched_allele_length=full_process_data(ms_val_data_matched_allele_length,mhc_seq_dict,remove_flank=True,add_pseudo_label=False)
    ms_val_data_matched_allele_length=create_seq_allele_column(ms_val_data_matched_allele_length)

    ms_test_data=full_process_data(ms_test_data,mhc_seq_dict,remove_flank=True,add_pseudo_label=False)
    ms_test_data=create_seq_allele_column(ms_test_data)
    ms_val_data=full_process_data(ms_val_data,mhc_seq_dict,remove_flank=True,add_pseudo_label=False)
    ms_val_data=create_seq_allele_column(ms_val_data)

    holec_test_set=full_process_data(holec_test_set,mhc_seq_dict,remove_flank=True,add_pseudo_label=False)
    holec_test_set=create_seq_allele_column(holec_test_set)
    holec_val_set=full_process_data(holec_val_set,mhc_seq_dict,remove_flank=True,add_pseudo_label=False)
    holec_val_set=create_seq_allele_column(holec_val_set)

    filtered_columns=['seq','allele','label','seq_allele','n_flank','c_flank','mhc_a_1','mhc_b_1','mhc_c_1','mhc_a_2','mhc_b_2','mhc_c_2','sa_ma','mhcprime','group']
    ms_test_data_matched_allele_length=filter_single_df_to_cols(ms_test_data_matched_allele_length,filtered_columns)
    ms_val_data_matched_allele_length=filter_single_df_to_cols(ms_val_data_matched_allele_length,filtered_columns)
    holec_test_set=filter_single_df_to_cols(holec_test_set,filtered_columns)
    holec_val_set=filter_single_df_to_cols(holec_val_set,filtered_columns)
    ms_test_data=filter_single_df_to_cols(ms_test_data,filtered_columns)
    ms_val_data=filter_single_df_to_cols(ms_val_data,filtered_columns)

    # Preserve the original YD-specific pathogen test-set processing.
    pathogen_set_datasets_dir=ms_train_test_dir/"Pathogen_Sets"
    ms_cov_set=pd.read_csv(pathogen_set_datasets_dir/"ms_cov_set.csv",index_col=0)
    ms_zika_set=pd.read_csv(pathogen_set_datasets_dir/"ms_zika_set.csv",index_col=0)
    ms_mtb_set=pd.read_csv(pathogen_set_datasets_dir/"ms_mtb_set.csv",index_col=0)

    ms_cov_set=full_process_data(ms_cov_set,mhc_seq_dict,remove_flank=True,add_pseudo_label=True)
    ms_cov_set=create_seq_allele_column(ms_cov_set)
    ms_zika_set=full_process_data(ms_zika_set,mhc_seq_dict,remove_flank=True,add_pseudo_label=True)
    ms_zika_set=create_seq_allele_column(ms_zika_set)
    ms_mtb_set=full_process_data(ms_mtb_set,mhc_seq_dict,remove_flank=True,add_pseudo_label=True)
    ms_mtb_set=create_seq_allele_column(ms_mtb_set)

    ms_cov_set.rename(columns={"cell_type":"group"},inplace=True)
    ms_zika_set.rename(columns={"cell_line":"group"},inplace=True)
    ms_mtb_set.rename(columns={"donor":"group"},inplace=True)
    ms_cov_set=filter_single_df_to_cols(ms_cov_set,filtered_columns)
    ms_zika_set=filter_single_df_to_cols(ms_zika_set,filtered_columns)
    ms_mtb_set=filter_single_df_to_cols(ms_mtb_set,filtered_columns)
    ms_cov_set["dataset_name"]="ms_cov_set"
    ms_zika_set["dataset_name"]="ms_zika_set"
    ms_mtb_set["dataset_name"]="ms_mtb_set"
    ms_full_pathogen_test_set=pd.concat([ms_cov_set,ms_zika_set,ms_mtb_set])

    train_set=ms_holec_train_set_matched_allele_length
    test_sets={
        "ms_test":ms_test_data_matched_allele_length,
        "holec_test":holec_test_set,
        "ms_full_pathogen_test":ms_full_pathogen_test_set,
    }
    val_sets={
        "ms_val":ms_val_data_matched_allele_length,
        "holec_val":holec_val_set,
    }

# ============================================================
# EPISCAN R9 / R9RT
# ============================================================

elif dataset_to_run in {"episcan_r9","episcan_r9rt"}:
    dataset_prefix=dataset_to_run
    task_type="classification"

    print("Loading EpiScan data...")
    episcan_train_test_dir=DATA_DIR/"Training_Datasets"/"Processed"/"EpiScan"
    episcan_r9_train_set_remaining=pd.read_csv(episcan_train_test_dir/"episcan_r9_train_set_remaining.csv",index_col=0)
    episcan_rt_train_set_remaining=pd.read_csv(episcan_train_test_dir/"episcan_rt_train_set_remaining.csv",index_col=0)
    episcan_r9_val_set=pd.read_csv(episcan_train_test_dir/"episcan_r9_val_set.csv",index_col=0)
    episcan_rt_val_set=pd.read_csv(episcan_train_test_dir/"episcan_rt_val_set.csv",index_col=0)
    episcan_r9_test_set=pd.read_csv(episcan_train_test_dir/"episcan_r9_test_set.csv",index_col=0)
    episcan_rt_test_set=pd.read_csv(episcan_train_test_dir/"episcan_rt_test_set.csv",index_col=0)
    sd3_formatted=pd.read_csv(episcan_train_test_dir/"sd3_formatted.csv",index_col=0)

    print("Merging MS/FT data...")
    episcan_r9_train_set_remaining["domain_id"]=1
    episcan_rt_train_set_remaining["domain_id"]=1

    if dataset_to_run=="episcan_r9":
        train_set=pd.concat([ms_train_data,episcan_r9_train_set_remaining,episcan_rt_train_set_remaining.query("label == 1")])
    else:
        train_set=pd.concat([ms_train_data,episcan_r9_train_set_remaining,episcan_rt_train_set_remaining])
    train_set.reset_index(inplace=True,drop=True)
    print_data_stats(train_set)

    train_set["seq_len"]=train_set.seq.apply(len)
    iso_alleles=['A0201','A0301','B0801','B5701']
    train_set=train_set[train_set.allele.isin(iso_alleles)].query("seq_len == 9")
    train_set.reset_index(inplace=True,drop=True)
    print_data_stats(train_set)

    print("Merging R9/RT val sets and test sets separately...")
    episcan_r9_val_set["dataset_source"]="r9"
    episcan_rt_val_set["dataset_source"]="rt"
    episcan_r9_rt_val_sets=pd.concat([episcan_r9_val_set,episcan_rt_val_set])
    episcan_r9_test_set["dataset_source"]="r9"
    episcan_rt_test_set["dataset_source"]="rt"
    episcan_r9_rt_test_sets=pd.concat([episcan_r9_test_set,episcan_rt_test_set])

    print("Matching allele- and length distributions for MS/FT data...")
    ms_test_data["seq_len"]=ms_test_data.seq.str.replace("$","").apply(len)
    ms_val_data["seq_len"]=ms_val_data.seq.str.replace("$","").apply(len)
    ms_test_data_matched_allele_length=ms_test_data[ms_test_data.allele.isin(iso_alleles)].query("seq_len == 9")
    ms_val_data_matched_allele_length=ms_val_data[ms_val_data.allele.isin(iso_alleles)].query("seq_len == 9")

    ms_test_data_matched_allele_length=full_process_data(ms_test_data_matched_allele_length,mhc_seq_dict,remove_flank=True,add_pseudo_label=False)
    ms_test_data_matched_allele_length=create_seq_allele_column(ms_test_data_matched_allele_length)
    ms_val_data_matched_allele_length=full_process_data(ms_val_data_matched_allele_length,mhc_seq_dict,remove_flank=True,add_pseudo_label=False)
    ms_val_data_matched_allele_length=create_seq_allele_column(ms_val_data_matched_allele_length)

    ms_test_data=full_process_data(ms_test_data,mhc_seq_dict,remove_flank=True,add_pseudo_label=False)
    ms_test_data=create_seq_allele_column(ms_test_data)
    ms_val_data=full_process_data(ms_val_data,mhc_seq_dict,remove_flank=True,add_pseudo_label=False)
    ms_val_data=create_seq_allele_column(ms_val_data)

    episcan_r9_rt_test_sets=full_process_data(episcan_r9_rt_test_sets,mhc_seq_dict,remove_flank=True,add_pseudo_label=False)
    episcan_r9_rt_test_sets=create_seq_allele_column(episcan_r9_rt_test_sets)
    episcan_r9_rt_val_sets=full_process_data(episcan_r9_rt_val_sets,mhc_seq_dict,remove_flank=True,add_pseudo_label=False)
    episcan_r9_rt_val_sets=create_seq_allele_column(episcan_r9_rt_val_sets)
    sd3_formatted=full_process_data(sd3_formatted,mhc_seq_dict,remove_flank=True,add_pseudo_label=False)
    sd3_formatted=create_seq_allele_column(sd3_formatted)

    filtered_columns=['seq','allele','label','seq_allele','n_flank','c_flank','mhc_a_1','mhc_b_1','mhc_c_1','mhc_a_2','mhc_b_2','mhc_c_2','sa_ma','mhcprime','dataset_source']
    ms_test_data_matched_allele_length=filter_single_df_to_cols(ms_test_data_matched_allele_length,filtered_columns)
    ms_val_data_matched_allele_length=filter_single_df_to_cols(ms_val_data_matched_allele_length,filtered_columns)
    episcan_r9_rt_test_sets=filter_single_df_to_cols(episcan_r9_rt_test_sets,filtered_columns)
    episcan_r9_rt_val_sets=filter_single_df_to_cols(episcan_r9_rt_val_sets,filtered_columns)
    sd3_formatted=filter_single_df_to_cols(sd3_formatted,filtered_columns)
    ms_test_data=filter_single_df_to_cols(ms_test_data,filtered_columns)
    ms_val_data=filter_single_df_to_cols(ms_val_data,filtered_columns)

    test_sets={
        "ms_test":ms_test_data_matched_allele_length,
        "episcan_r9_rt_test":episcan_r9_rt_test_sets,
        "cov_test":sd3_formatted,
    }
    val_sets={
        "ms_val":ms_val_data_matched_allele_length,
        "episcan_r9_rt_val":episcan_r9_rt_val_sets,
    }

# ============================================================
# BA CONTINUOUS
# ============================================================

elif dataset_to_run=="ba_con":
    dataset_prefix="ba_con"
    task_type="continuous"

    print("Loading BA data...")
    ba_train_test_dir=DATA_DIR/"Training_Datasets"/"Processed"/"BA"
    train_df_ba_for_updating_remaining=pd.read_csv(ba_train_test_dir/"train_df_ba_for_updating_remaining.csv",index_col=0)
    ba_val_set=pd.read_csv(ba_train_test_dir/"ba_val_set.csv",index_col=0)
    test_df_ba_for_updating=pd.read_csv(ba_train_test_dir/"test_df_ba_for_updating.csv",index_col=0)

    print("Merging MS/FT data...")
    train_df_ba_for_updating_remaining["domain_id"]=1
    ms_ba_train_set=pd.concat([ms_train_data,train_df_ba_for_updating_remaining])
    ms_ba_train_set.reset_index(inplace=True,drop=True)
    print_data_stats(ms_ba_train_set)

    ms_ba_train_set["seq_len"]=ms_ba_train_set.seq.str.replace("$","").apply(len)
    ms_ba_train_set_matched_allele_length=filter_domain0_to_domain1_pairs(ms_ba_train_set,allele_col='allele',seq_len_col='seq_len',domain_col='domain_id')
    ms_ba_train_set_matched_allele_length.reset_index(inplace=True,drop=True)
    print_data_stats(ms_ba_train_set_matched_allele_length)

    print("Matching allele- and length distributions for MS/FT data...")
    ms_test_data["seq_len"]=ms_test_data.seq.str.replace("$","").apply(len)
    ms_val_data["seq_len"]=ms_val_data.seq.str.replace("$","").apply(len)
    test_df_ba_for_updating["seq_len"]=test_df_ba_for_updating.seq.str.replace("$","").apply(len)
    ba_val_set["seq_len"]=ba_val_set.seq.str.replace("$","").apply(len)
    ms_test_data_matched_allele_length=filter_to_shared_pairs(ms_test_data,test_df_ba_for_updating,allele_col='allele',seq_len_col='seq_len')
    ms_val_data_matched_allele_length=filter_to_shared_pairs(ms_val_data,ba_val_set,allele_col='allele',seq_len_col='seq_len')
    print_data_stats(ms_test_data_matched_allele_length)

    ms_test_data_matched_allele_length=full_process_data(ms_test_data_matched_allele_length,mhc_seq_dict,remove_flank=True,add_pseudo_label=False)
    ms_test_data_matched_allele_length=create_seq_allele_column(ms_test_data_matched_allele_length)
    ms_val_data_matched_allele_length=full_process_data(ms_val_data_matched_allele_length,mhc_seq_dict,remove_flank=True,add_pseudo_label=False)
    ms_val_data_matched_allele_length=create_seq_allele_column(ms_val_data_matched_allele_length)
    test_df_ba_for_updating=full_process_data(test_df_ba_for_updating,mhc_seq_dict,remove_flank=True,add_pseudo_label=False)
    test_df_ba_for_updating=create_seq_allele_column(test_df_ba_for_updating)
    ba_val_set=full_process_data(ba_val_set,mhc_seq_dict,remove_flank=True,add_pseudo_label=False)
    ba_val_set=create_seq_allele_column(ba_val_set)
    ms_test_data=full_process_data(ms_test_data,mhc_seq_dict,remove_flank=True,add_pseudo_label=False)
    ms_test_data=create_seq_allele_column(ms_test_data)
    ms_val_data=full_process_data(ms_val_data,mhc_seq_dict,remove_flank=True,add_pseudo_label=False)
    ms_val_data=create_seq_allele_column(ms_val_data)

    filtered_columns=['seq','allele','label','seq_allele','n_flank','c_flank','mhc_a_1','mhc_b_1','mhc_c_1','mhc_a_2','mhc_b_2','mhc_c_2','sa_ma','mhcprime','ba_value']
    ms_test_data_matched_allele_length=filter_single_df_to_cols(ms_test_data_matched_allele_length,filtered_columns)
    ms_val_data_matched_allele_length=filter_single_df_to_cols(ms_val_data_matched_allele_length,filtered_columns)
    test_df_ba_for_updating=filter_single_df_to_cols(test_df_ba_for_updating,filtered_columns)
    ba_val_set=filter_single_df_to_cols(ba_val_set,filtered_columns)
    ms_test_data=filter_single_df_to_cols(ms_test_data,filtered_columns)
    ms_val_data=filter_single_df_to_cols(ms_val_data,filtered_columns)

    train_set=ms_ba_train_set_matched_allele_length
    test_sets={"ms_test":ms_test_data_matched_allele_length,"ba_test":test_df_ba_for_updating}
    val_sets={"ms_val":ms_val_data_matched_allele_length,"ba_val":ba_val_set}

# ============================================================
# ESCAPE-SEQ CONTINUOUS
# ============================================================

elif dataset_to_run=="escape_con":
    dataset_prefix="escape_con"
    task_type="continuous"

    print("Loading ESCAPE-seq data...")
    escape_train_test_dir=DATA_DIR/"Training_Datasets"/"Processed"/"ESCAPE_seq"
    # escape_train_remaining=pd.read_csv(escape_train_test_dir/"oncogene_mut_wt_df_dedup_noX_noA68_train_remaining.csv",index_col=0)
    # escape_val=pd.read_csv(escape_train_test_dir/"oncogene_mut_wt_df_dedup_noX_noA68_val.csv",index_col=0)
    # escape_test=pd.read_csv(escape_train_test_dir/"oncogene_mut_wt_df_dedup_noX_noA68_test.csv",index_col=0)

    escape_train_remaining=pd.read_csv(f"{DATA_DIR}/Training_Datasets/Processed/ESCAPE_seq/escape_con_train_set_remaining_cleaned.csv", index_col=0)
    escape_val=pd.read_csv(f"{DATA_DIR}/Training_Datasets/Processed/ESCAPE_seq/escape_con_val_set_cleaned.csv", index_col=0)
    escape_test=pd.read_csv(f"{DATA_DIR}/Training_Datasets/Processed/ESCAPE_seq/escape_con_test_set_cleaned.csv", index_col=0)


    print("Merging MS/FT data...")
    escape_train_remaining["domain_id"]=1
    ms_escape_train_set=pd.concat([ms_train_data,escape_train_remaining])
    ms_escape_train_set.reset_index(inplace=True,drop=True)
    print_data_stats(ms_escape_train_set)

    ms_escape_train_set["seq_len"]=ms_escape_train_set.seq.str.replace("$","").apply(len)
    ms_escape_train_set_matched_allele_length=filter_domain0_to_domain1_pairs(ms_escape_train_set,allele_col='allele',seq_len_col='seq_len',domain_col='domain_id')
    ms_escape_train_set_matched_allele_length.reset_index(inplace=True,drop=True)
    print_data_stats(ms_escape_train_set_matched_allele_length)

    print("Matching allele- and length distributions for MS/FT data...")
    ms_test_data["seq_len"]=ms_test_data.seq.str.replace("$","").apply(len)
    ms_val_data["seq_len"]=ms_val_data.seq.str.replace("$","").apply(len)
    escape_test["seq_len"]=escape_test.seq.str.replace("$","").apply(len)
    escape_val["seq_len"]=escape_val.seq.str.replace("$","").apply(len)
    ms_test_data_matched_allele_length=filter_to_shared_pairs(ms_test_data,escape_test,allele_col='allele',seq_len_col='seq_len')
    ms_val_data_matched_allele_length=filter_to_shared_pairs(ms_val_data,escape_val,allele_col='allele',seq_len_col='seq_len')
    print_data_stats(ms_test_data_matched_allele_length)

    ms_test_data_matched_allele_length=full_process_data(ms_test_data_matched_allele_length,mhc_seq_dict,remove_flank=True,add_pseudo_label=False)
    ms_test_data_matched_allele_length=create_seq_allele_column(ms_test_data_matched_allele_length)
    ms_val_data_matched_allele_length=full_process_data(ms_val_data_matched_allele_length,mhc_seq_dict,remove_flank=True,add_pseudo_label=False)
    ms_val_data_matched_allele_length=create_seq_allele_column(ms_val_data_matched_allele_length)
    escape_test=full_process_data(escape_test,mhc_seq_dict,remove_flank=True,add_pseudo_label=False)
    escape_test=create_seq_allele_column(escape_test)
    escape_val=full_process_data(escape_val,mhc_seq_dict,remove_flank=True,add_pseudo_label=False)
    escape_val=create_seq_allele_column(escape_val)
    ms_test_data=full_process_data(ms_test_data,mhc_seq_dict,remove_flank=True,add_pseudo_label=False)
    ms_test_data=create_seq_allele_column(ms_test_data)
    ms_val_data=full_process_data(ms_val_data,mhc_seq_dict,remove_flank=True,add_pseudo_label=False)
    ms_val_data=create_seq_allele_column(ms_val_data)

    filtered_columns=['seq','allele','label','seq_allele','n_flank','c_flank','mhc_a_1','mhc_b_1','mhc_c_1','mhc_a_2','mhc_b_2','mhc_c_2','sa_ma','mhcprime','escore','ba_value','name2','name','gene','mutation','mut_type']
    ms_test_data_matched_allele_length=filter_single_df_to_cols(ms_test_data_matched_allele_length,filtered_columns)
    ms_val_data_matched_allele_length=filter_single_df_to_cols(ms_val_data_matched_allele_length,filtered_columns)
    escape_test=filter_single_df_to_cols(escape_test,filtered_columns)
    escape_val=filter_single_df_to_cols(escape_val,filtered_columns)
    ms_test_data=filter_single_df_to_cols(ms_test_data,filtered_columns)
    ms_val_data=filter_single_df_to_cols(ms_val_data,filtered_columns)

    train_set=ms_escape_train_set_matched_allele_length
    test_sets={"ms_test":ms_test_data_matched_allele_length,"escape_test":escape_test}
    val_sets={"ms_val":ms_val_data_matched_allele_length,"escape_val":escape_val}

# ============================================================
# COMMON TRAINING DATA
# ============================================================

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

# ============================================================
# LOAD BASE MHCPrime
# ============================================================

print("\nLoading MHCPrime...")
tokenizer=AminoAcidTokenizer()
base_model_params={
    'vocab_size':tokenizer.vocab_size,
    'embed_dim':480,
    'fc_hidden_dims':[240,120],
    'fc_dropout':0.1,
    'n_heads':16,
    'max_peptide_len':34,
    'max_mhc_len':34,
    'n_peptide_layers':2,
    'n_mhc_layers':1,
    'n_concat_layers':3,
    'add_pos_enc':True,
    'use_segment_embedder':True,
    'dim_feedforward_multiplier':1,
    'debugging':False,
    'fc_output_dim':1,
    'pooling':'cls',
    'ffnn_use_layernorm':False,
    'use_feature_table':True,
    'feature_table':processed_feature_table,
    'sequence_embedder_feature_names':feature_names,
    'sequence_embedder_feature_mode':'concat',
    'sequence_embedder_tokenizer':tokenizer,
    'peptide_start_pos':1,
    'mhc_start_pos':35,
    'pos_enc_type':'sincos',
    'seed':42,
    'add_peptide_conformer':True,
    'peptide_conformer_num_layers':2,
    'peptide_conformer_n_heads':16,
    'peptide_conformer_conv_kernel':5,
    'peptide_conformer_dropout':0.0,
    'add_mhc_conformer':False,
    'mhc_conformer_num_layers':2,
    'mhc_conformer_n_heads':16,
    'mhc_conformer_conv_kernel':5,
    'mhc_conformer_dropout':0.0,
    'add_fusion_conformer':True,
    'fusion_conformer_num_layers':1,
    'fusion_conformer_n_heads':16,
    'fusion_conformer_conv_kernel':7,
    'fusion_conformer_dropout':0.0,
    'fusion_conformer_mask_bos':True,
    'ff_pre_post_swiglu':False,
    'legacy_init_compat':True,
}

base_model_dir=MODEL_CHECKPOINTS_DIR/"MHCPrime_Base_for_Ablations"/"Checkpoints"
ckpt_path=base_model_dir/"model_final.pt"
clear_all_gpu_memory()
mhcprime=init_model(base_model_params,print_check=False,print_params=True)
state_dict=torch.load(ckpt_path,map_location="cpu")
mhcprime.load_state_dict(state_dict,strict=True)
mhcprime.to(device)
mhcprime.eval()
print(f"Model loaded from checkpoint: {ckpt_path}")
print(f"Model device: {next(mhcprime.parameters()).device}\n")

# ============================================================
# BUILD INFERENCE CACHES
# ============================================================

print("Creating test and validation set caches for inference...")
test_set_cache_paths={}
for test_set_name,test_data_set in test_sets.items():
    data_to_run_path=DATA_DIR/f"{dataset_prefix}_{test_set_name}_{gpu_id}_cache.pt"
    print(test_set_name,data_to_run_path)
    build_token_cache(test_data_set,data_to_run_path,tokenizer,use_domain_id=False)
    test_set_cache_paths[test_set_name]=data_to_run_path

val_set_cache_paths={}
for val_set_name,val_data_set in val_sets.items():
    data_to_run_path=DATA_DIR/f"{dataset_prefix}_{val_set_name}_{gpu_id}_cache.pt"
    print(val_set_name,data_to_run_path)
    build_token_cache(val_data_set,data_to_run_path,tokenizer,use_domain_id=False)
    val_set_cache_paths[val_set_name]=data_to_run_path
print(end="\n")

# ============================================================
# BASE MHCPrime INFERENCE
# ============================================================

def run_models_on_sets(data_sets,cache_paths,models,workers):
    for set_name,data_set in data_sets.items():
        data_to_run_path=cache_paths[set_name]
        print(set_name,data_to_run_path)
        for model_name,model in models.items():
            model=model.to(device)
            model.eval()
            with torch.no_grad():
                output_scores=run_mhcprime_fast(model,data_to_run_path,batch_size=3072,num_workers=workers,compile_model=False,use_domain_id=False)
            data_set[model_name]=output_scores.detach().cpu().tolist()
        data_sets[set_name]=data_set

print("Run base MHCPrime inference once on all test and validation sets...")
run_models_on_sets(test_sets,test_set_cache_paths,{"mhcprime":copy.deepcopy(mhcprime)},16)
run_models_on_sets(val_sets,val_set_cache_paths,{"mhcprime":copy.deepcopy(mhcprime)},16)

print("Saving initial test sets...")
for set_name,df in test_sets.items():
    print(set_name)
    df.to_csv(full_output_dir/f"{set_name}.csv")

print("Saving initial validation sets...")
for set_name,df in val_sets.items():
    print(set_name)
    df.to_csv(full_output_dir/f"{set_name}.csv")
print(end="\n")

# ============================================================
# PEPCL PARAMETERS
# ============================================================

pepcl_base_params_cls=dict(
    tokenizer=tokenizer,
    device=device,
    steps=200,
    output_dir=None,
    seed=42,
    num_workers=16,
    collate_fn=collate_fn,
    num_ms_pos=2000,
    num_nonms_pos=2000,
    ms_neg_ratio=1.0,
    nonms_neg_ratio=1.0,
    auto_calibrate=True,
    calibration_steps=1,
    ms_batch_size=3072,
    nonms_batch_size=3072,
    encoder_lr=2e-4,
    decoder_lr=2e-4,
    weight_decay=0.01,
    warmup_ratio=0.1,
    update_domain_embeddings=False,
    primary_task_id=1,
    sequential_backprop=True,
    ms_loss_type="logsmoothap",
    ms_loss_alpha=10.0,
    ms_lambda=1.0,
    nonms_loss_type="logsmoothap",
    nonms_loss_alpha=10.0,
    ft_lambda=1.0,
    dr_lambda=1e-5,
    dr_loss_type="mse",
)

pepcl_base_params_con=dict(
    tokenizer=tokenizer,
    device=device,
    steps=200,
    output_dir=None,
    seed=42,
    num_workers=16,
    collate_fn=collate_fn,
    num_ms_pos=2000,
    num_nonms_pos=2000,
    ms_neg_ratio=1.0,
    nonms_neg_ratio=1.0,
    auto_calibrate=True,
    calibration_steps=1,
    ms_batch_size=3072,
    nonms_batch_size=3072,
    encoder_lr=2e-4,
    decoder_lr=2e-4,
    weight_decay=0.01,
    warmup_ratio=0.1,
    update_domain_embeddings=False,
    primary_task_id=1,
    sequential_backprop=True,
    ms_loss_type="logsmoothap",
    ms_loss_alpha=10.0,
    ms_lambda=1.0,
    nonms_loss_type="spearman",
    nonms_loss_alpha=0.01,
    ft_lambda=1.0,
    dr_lambda=1e-5,
    dr_loss_type="mse",
    continuous_target_col="ba_value",
    continuous_ft_sampling_mode="allele_aware",
)

pepcl_base_params=pepcl_base_params_con.copy() if task_type=="continuous" else pepcl_base_params_cls.copy()
pepcl_base_params["num_workers"]=num_workers

# Same checkpoint schedules used in the step/LR runs for loss/performance tracking.
ckpt_to_save_per_budget={
    25:[5,10,15,20,25],
    50:[5,10,20,30,40,50],
    100:[10,20,40,60,80,100],
    200:[10,20,25,50,75,100,150,200],
    300:[10,25,50,75,100,150,200,250,300],
    400:[10,25,50,100,150,200,250,300,350,400],
    600:[10,25,50,100,150,200,250,300,350,400,450,500,550,600],
    800:[10,25,50,100,150,200,250,300,350,400,450,500,550,600,650,700,750,800],
}

# ============================================================
# HELPERS
# ============================================================

# def normalize_method_configs(configs,method):
#     configs=[configs] if isinstance(configs,dict) else configs
#     if not isinstance(configs,list) or not configs:
#         raise ValueError(f"selected_configs['{dataset_to_run}']['{method}'] must be a config dict or non-empty list of config dicts.")
#     for i,cfg in enumerate(configs):
#         if not isinstance(cfg,dict) or "lr" not in cfg or "steps" not in cfg:
#             raise ValueError(f"Invalid {method} config at index {i}: expected {{'lr': ..., 'steps': ...}}, got {cfg}.")
#     return configs

def normalize_method_configs(configs,method):
    # Empty config means skip this method.
    if configs is None or configs=={} or configs==[] or configs==[{}]:
        return []

    configs=[configs] if isinstance(configs,dict) else configs

    if not isinstance(configs,list):
        raise ValueError(
            f"selected_configs['{dataset_to_run}']['{method}'] must be "
            f"a config dict, list of config dicts, or empty to skip."
        )

    for i,cfg in enumerate(configs):
        if not isinstance(cfg,dict) or "lr" not in cfg or "steps" not in cfg:
            raise ValueError(
                f"Invalid {method} config at index {i}: "
                f"expected {{'lr': ..., 'steps': ...}}, got {cfg}."
            )

    return configs

def add_prefix_to_keys(d,prefix):
    return {f"{prefix}{k}":v for k,v in d.items()}

def save_model_state_dict(model,model_name):
    if model_output_dir is None:
        return
    save_path=model_output_dir/f"{model_name}.pt"
    cpu_state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()}
    torch.save(cpu_state_dict,save_path)
    print(f"Saved model checkpoint: {save_path}")

ft_configs=normalize_method_configs(selected_configs[dataset_to_run]["ft"],"ft")
pepcl_configs=normalize_method_configs(selected_configs[dataset_to_run]["pepcl_r_dr"],"pepcl_r_dr")

for method,configs in [("ft",ft_configs),("pepcl_r_dr",pepcl_configs)]:
    for cfg in configs:
        steps=int(cfg["steps"])
        if steps not in ckpt_to_save_per_budget:
            raise ValueError(f"No checkpoint logging schedule defined for selected {method} steps={steps}.")

# ============================================================
# RUN SELECTED FT + PEPCL CONFIGURATIONS
# ============================================================

for pepcl_seed in seeds:
    print(f"\n{'='*90}")
    print(f"Seed: {pepcl_seed}")
    print(f"FT configs: {ft_configs}")
    print(f"PepCL configs: {pepcl_configs}")
    print(f"{'='*90}\n")

    final_models_to_run={}
    model_checkpoints_for_trajectories={}
    models_to_delete=[]

    # ---------------- FT ----------------
    for config_idx,ft_cfg in enumerate(ft_configs,1):
        ft_lr=float(ft_cfg["lr"])
        ft_steps=int(ft_cfg["steps"])
        ft_lr_string=lr_to_string(ft_lr)

        print(f"FT config {config_idx}/{len(ft_configs)}: LR={ft_lr}, steps={ft_steps}")
        model_ft=copy.deepcopy(mhcprime)
        ft_params=pepcl_base_params.copy()
        ft_params["seed"]=pepcl_seed
        ft_params["steps"]=ft_steps
        ft_params["save_every_n_steps"]=ckpt_to_save_per_budget[ft_steps]
        ft_params["encoder_lr"]=ft_lr
        ft_params["decoder_lr"]=ft_lr
        ft_params["dr_lambda"]=0

        pepcl=PepCL_Logging_Ext(
            model=model_ft,
            train_df=ft_train_set,
            use_replay=False,
            mhc_seq_dict=mhc_seq_dict,
            enable_logging=True,
            **ft_params,
        )
        training_stats_ft,model_states_ft=pepcl.train()
        ft_logs=pepcl.get_logs(as_dataframes=True)
        ft_logs["training_stats"]=training_stats_ft

        ft_model_prefix=f"{dataset_prefix}_ft_steps_{ft_steps}_lr_{ft_lr_string}_seed_{pepcl_seed}_step_"
        model_states_ft=add_prefix_to_keys(model_states_ft,ft_model_prefix)
        ft_model_name=ft_model_prefix.removesuffix("_step_")
        print("FT logged checkpoints:",model_states_ft.keys())

        if save_log_data:
            print("Saving FT logs...")
            save_log_dict(ft_logs,full_log_output_dir/f"{ft_model_name}.pkl")

        save_model_state_dict(model_ft,ft_model_name)
        final_models_to_run[ft_model_name]=model_ft
        model_checkpoints_for_trajectories.update(model_states_ft)
        models_to_delete.append(model_ft)

    # ---------------- PepCL ----------------
    for config_idx,pepcl_cfg in enumerate(pepcl_configs,1):
        pepcl_lr=float(pepcl_cfg["lr"])
        pepcl_steps=int(pepcl_cfg["steps"])
        pepcl_lr_string=lr_to_string(pepcl_lr)

        print(f"PepCL config {config_idx}/{len(pepcl_configs)}: LR={pepcl_lr}, steps={pepcl_steps}")
        model_pepcl_r_dr=copy.deepcopy(mhcprime)
        pepcl_params=pepcl_base_params.copy()
        pepcl_params["seed"]=pepcl_seed
        pepcl_params["steps"]=pepcl_steps
        pepcl_params["save_every_n_steps"]=ckpt_to_save_per_budget[pepcl_steps]
        pepcl_params["encoder_lr"]=pepcl_lr
        pepcl_params["decoder_lr"]=pepcl_lr
        pepcl_params["dr_lambda"]=1e-5

        pepcl=PepCL_Logging_Ext(
            model=model_pepcl_r_dr,
            train_df=pepcl_train_set,
            use_replay=True,
            mhc_seq_dict=mhc_seq_dict,
            enable_logging=True,
            **pepcl_params,
        )
        training_stats_pepcl_r_dr,model_states_pepcl_r_dr=pepcl.train()
        pepcl_r_dr_logs=pepcl.get_logs(as_dataframes=True)
        pepcl_r_dr_logs["training_stats"]=training_stats_pepcl_r_dr

        pepcl_r_dr_model_prefix=f"{dataset_prefix}_pepcl_r_dr_steps_{pepcl_steps}_lr_{pepcl_lr_string}_seed_{pepcl_seed}_step_"
        model_states_pepcl_r_dr=add_prefix_to_keys(model_states_pepcl_r_dr,pepcl_r_dr_model_prefix)
        pepcl_r_dr_model_name=pepcl_r_dr_model_prefix.removesuffix("_step_")
        print("PepCL logged checkpoints:",model_states_pepcl_r_dr.keys())

        if save_log_data:
            print("Saving PepCL logs...")
            save_log_dict(pepcl_r_dr_logs,full_log_output_dir/f"{pepcl_r_dr_model_name}.pkl")

        save_model_state_dict(model_pepcl_r_dr,pepcl_r_dr_model_name)
        final_models_to_run[pepcl_r_dr_model_name]=model_pepcl_r_dr
        model_checkpoints_for_trajectories.update(model_states_pepcl_r_dr)
        models_to_delete.append(model_pepcl_r_dr)

    clear_all_gpu_memory()

    if not run_logged_models:
        model_checkpoints_for_trajectories={}
    all_models_to_run_for_val_set=model_checkpoints_for_trajectories | final_models_to_run

    # Test sets: final models only.
    print("\nRunning final models on test sets...")
    run_models_on_sets(test_sets,test_set_cache_paths,final_models_to_run,num_workers)

    # Validation sets: logged checkpoints + final models.
    print("\nRunning logged checkpoints + final models on validation sets...")
    run_models_on_sets(val_sets,val_set_cache_paths,all_models_to_run_for_val_set,num_workers)

    print("\nSaving test sets...")
    for set_name,df in test_sets.items():
        print(set_name)
        df.to_csv(full_output_dir/f"{set_name}.csv")

    print("Saving validation sets...")
    for set_name,df in val_sets.items():
        print(set_name)
        df.to_csv(full_output_dir/f"{set_name}.csv")

    del final_models_to_run,model_checkpoints_for_trajectories,all_models_to_run_for_val_set,models_to_delete
    clear_all_gpu_memory()

# ============================================================
# CLEAN UP CACHES
# ============================================================

for cache_path in test_set_cache_paths.values():
    if os.path.exists(cache_path):
        os.remove(cache_path)
for cache_path in val_set_cache_paths.values():
    if os.path.exists(cache_path):
        os.remove(cache_path)

print(f"\nFinished dataset: {dataset_to_run}")
print(f"Outputs: {full_output_dir}")
if model_output_dir is not None:
    print(f"Model checkpoints: {model_output_dir}")
