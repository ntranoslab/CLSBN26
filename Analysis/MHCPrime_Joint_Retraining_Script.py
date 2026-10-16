"""
Joint MHCPrime retraining from scratch on:
    full MS training data + one new assay training dataset.

Uses PepCL_Retraining as the training engine:
    - blank/random MHCPrime initialization
    - no pretrained checkpoint
    - full MS training set
    - use_replay=True so domain 0 + domain 1 are jointly trained
    - dr_lambda=0: no teacher / no drift regularization
    - classification:
          MS = LogSmoothAP
          assay = LogSmoothAP
    - continuous:
          MS = LogSmoothAP
          assay = Soft Spearman
    - optional merged classification loss

Outputs:
    Models:
        Model_Checkpoints/Joint_Retraining/<dataset>/
        Model_Checkpoints/Joint_Retraining_MS_Matched/<dataset>/ if ms_matched=True

    Data / inference / logs:
        Data/Joint_Retraining/<dataset>/
        Data/Joint_Retraining_MS_Matched/<dataset>/ if ms_matched=True

Run:
    python MHCPrime_Joint_Retraining_Script.py -g GPU_ID
"""
import sys
import os
import json
import time
import pickle
from pathlib import Path

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

import warnings
warnings.filterwarnings("ignore")


# ============================================================
# GPU
# ============================================================

import argparse

parser=argparse.ArgumentParser(
    description="Select specific GPU to run model on."
)

parser.add_argument(
    "-g","--gpu",
    type=int,
    default=0,
    help="Which GPU ID to use (e.g. 0, 1, 2...)",
)

parser.add_argument(
    "--dataset",
    type=str,
    default=None,
    choices=["yd","episcan_r9","episcan_r9rt","ba_con","escape_con"],
    help="Optional dataset override. If omitted, uses dataset_to_run below.",
)

parser.add_argument(
    "--ms-matched",
    type=str,
    default=None,
    choices=["true","false"],
    help="Optional MS-matching override: true or false. If omitted, uses ms_matched below.",
)

parser.add_argument(
    "--overwrite",
    action="store_true",
    help="Overwrite an existing final model/results without prompting.",
)

args=parser.parse_args()
gpu_id=args.gpu

if torch.cuda.is_available():
    torch.cuda.set_device(gpu_id)

    print(
        f"Using GPU: {torch.cuda.get_device_name(gpu_id)} "
        f"(ID: {torch.cuda.current_device()})"
    )

device=torch.device(
    f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# USER CONFIGURATION
# ============================================================

# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------

# Options:
#   "yd"
#   "episcan_r9"
#   "episcan_r9rt"
#   "ba_con"
#   "escape_con"

dataset_to_run="yd"
# dataset_to_run="episcan_r9"
# dataset_to_run="episcan_r9rt"
# dataset_to_run="ba_con"
# dataset_to_run="escape_con"


# ------------------------------------------------------------
# Match MS allele x length support to assay
# ------------------------------------------------------------

# False:
#     use the full MS training set
#
# True:
#     match MS training data to the assay allele x length support
#     using the same dataset-specific procedure as the existing update scripts
ms_matched=False
# ms_matched=True


# ------------------------------------------------------------
# Optional command-line overrides
# ------------------------------------------------------------

if args.dataset is not None:
    dataset_to_run=args.dataset

if args.ms_matched is not None:
    ms_matched=args.ms_matched.lower()=="true"

overwrite_existing=args.overwrite


# ------------------------------------------------------------
# Retraining seed
# ------------------------------------------------------------

# Controls blank model initialization + training sampling.
retraining_seed=42


# ------------------------------------------------------------
# Full retraining optimization budget
# ------------------------------------------------------------

# Original MHCPrime was ~131 optimizer steps/epoch x 120 epochs.
total_steps=120*131


# ------------------------------------------------------------
# Optimization
# ------------------------------------------------------------

learning_rate=2e-4
weight_decay=0.01
warmup_ratio=0.1

num_workers=16


# ------------------------------------------------------------
# Batch size / merged-loss mode
# ------------------------------------------------------------

# Overall intended training batch size.
batch_size=3072

# False:
#   MS batch    = 3072
#   assay batch = 3072
#   separate forwards/losses
#
# True:
#   classification only
#   MS batch    = 1536
#   assay batch = 1536
#   merged total = 3072
#   one forward + one joint classification loss
merged_loss=False


# ------------------------------------------------------------
# Intermediate model checkpoints
# ------------------------------------------------------------

# None:
#     no intermediate model checkpoints
#
# Integer:
#     e.g. 1000 -> every 1000 steps
#
# List:
#     e.g. [1000,5000,10000,15720]

def steps_to_max(max_val, interval=50, start=0):
    out = list(range(start, max_val, interval))
    if not out or out[-1] != max_val:
        out.append(max_val)
    return out

save_every_n_steps=steps_to_max(15720, interval=100, start=1)
print(len(save_every_n_steps))

# False:
#     checkpoint models are used for validation scoring after training,
#     then removed from disk
#
# True:
#     checkpoint models are retained in the checkpoints directory
save_validation_checkpoint_models=False


# ------------------------------------------------------------
# Final model name
# ------------------------------------------------------------

if ms_matched:
    final_model_name=(
        f"{dataset_to_run}_retraining_ms_matched_"
        f"steps_{total_steps}_seed_{retraining_seed}.pt"
    )
else:
    final_model_name=(
        f"{dataset_to_run}_retraining_"
        f"steps_{total_steps}_seed_{retraining_seed}.pt"
    )


# ------------------------------------------------------------
# Inference
# ------------------------------------------------------------

inference_batch_size=3072
inference_num_workers=16


# ============================================================
# VALIDATE CONFIGURATION
# ============================================================

valid_datasets={
    "yd",
    "episcan_r9",
    "episcan_r9rt",
    "ba_con",
    "escape_con",
}

classification_datasets={
    "yd",
    "episcan_r9",
    "episcan_r9rt",
}

continuous_datasets={
    "ba_con",
    "escape_con",
}

if dataset_to_run not in valid_datasets:
    raise ValueError(
        f"dataset_to_run must be one of {sorted(valid_datasets)}, "
        f"got '{dataset_to_run}'."
    )

if merged_loss and dataset_to_run in continuous_datasets:
    raise ValueError(
        "merged_loss=True is classification-only. "
        f"'{dataset_to_run}' is a continuous dataset."
    )


# ------------------------------------------------------------
# Set stream batch sizes
# ------------------------------------------------------------

if merged_loss:
    if batch_size%2!=0:
        raise ValueError(
            "batch_size must be even when merged_loss=True."
        )

    ms_batch_size=batch_size//2
    nonms_batch_size=batch_size//2

else:
    ms_batch_size=batch_size
    nonms_batch_size=batch_size


# ------------------------------------------------------------
# Validation checkpoint steps
# ------------------------------------------------------------

if save_every_n_steps is None:
    validation_checkpoint_steps=[]

elif isinstance(save_every_n_steps,int):
    validation_checkpoint_steps=list(
        range(
            save_every_n_steps,
            total_steps+1,
            save_every_n_steps,
        )
    )

else:
    validation_checkpoint_steps=sorted(
        {
            int(step)
            for step in save_every_n_steps
            if int(step)<=total_steps
        }
    )


# ============================================================
# OUTPUT PATHS
# ============================================================

# Data / result output.
joint_retraining_dir_name=(
    "Joint_Retraining_MS_Matched"
    if ms_matched
    else "Joint_Retraining"
)

data_output_dir=(
    DATA_DIR/
    joint_retraining_dir_name/
    dataset_to_run
)

# Model output.
model_output_dir=(
    MODEL_CHECKPOINTS_DIR/
    joint_retraining_dir_name/
    dataset_to_run
)

# Final model.
final_model_path=(
    model_output_dir/
    final_model_name
)

# Optional intermediate checkpoints.
checkpoint_dir=(
    model_output_dir/
    "checkpoints"
)


# ============================================================
# OVERWRITE CHECK
#
# We ONLY stop/ask if the requested FINAL MODEL already exists.
# Existing directories/results alone do not trigger the prompt.
# ============================================================

if final_model_path.exists():

    if overwrite_existing:
        print(
            f"\nWARNING: Final model already exists:\n"
            f"{final_model_path}\n\n"
            f"--overwrite supplied. Continuing without prompt.\n"
        )

    else:
        response=input(
            f"\nWARNING: Final model already exists:\n"
            f"{final_model_path}\n\n"
            f"Overwrite this model and rerun? [y/n]: "
        ).strip().lower()

        if response not in {"y","yes"}:
            raise SystemExit(
                "Stopped by user. Existing final model was not modified."
            )

        print(
            "Continuing. Existing final model may be overwritten.\n"
        )


# Create result/model directories.
data_output_dir.mkdir(
    parents=True,
    exist_ok=True,
)

model_output_dir.mkdir(
    parents=True,
    exist_ok=True,
)

if save_every_n_steps is not None:
    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# PRINT CONFIG
# ============================================================

print()
print("="*80)
print("JOINT RETRAINING CONFIGURATION")
print("="*80)

print("Dataset:",dataset_to_run)
print("MS matched:",ms_matched)
print("Seed:",retraining_seed)

print("Total steps:",total_steps)
print("Learning rate:",learning_rate)
print("Weight decay:",weight_decay)
print("Warmup ratio:",warmup_ratio)

print("Merged loss:",merged_loss)
print("MS stream batch size:",ms_batch_size)
print("Assay stream batch size:",nonms_batch_size)

if merged_loss:
    print(
        "Merged total batch size:",
        ms_batch_size+nonms_batch_size,
    )

print("Workers:",num_workers)
print("Checkpoint schedule:",save_every_n_steps)
print("Save validation checkpoint models:",save_validation_checkpoint_models)
print("Overwrite existing:",overwrite_existing)

print()
print("Data output directory:")
print(data_output_dir)

print()
print("Model output directory:")
print(model_output_dir)

print()
print("Final model:")
print(final_model_path)

if save_every_n_steps is not None:
    print()
    print("Intermediate checkpoint directory:")
    print(checkpoint_dir)

print("="*80)
print()


# ============================================================
# START FULL-RUN TIMER
#
# Includes:
#   data loading
#   processing
#   cache construction
#   training
#   model saving
#   final inference
#   output saving
# ============================================================

full_run_t0=time.perf_counter()


# ============================================================
# MHC PSEUDOSEQUENCES
# ============================================================

mhc_seq_dict={
    k.removeprefix("HLA-")
     .replace("*","")
     .replace(":",""):v
    for k,v in mhc_seq_dict.items()
}


# ============================================================
# FEATURE DATA
# ============================================================

print("Loading feature data...")

aa_features_table_ext,descriptor_groups=load_feature_data()

processed_feature_table,corr_info=process_aa_feature_table(
    aa_features_table_ext
)

feature_names=list(
    processed_feature_table.columns
)

print()


# ============================================================
# LOAD FULL MS TRAIN / VAL / TEST
# ============================================================

print("Loading full MS train/validation/test data...")

ms_train_test_dir=(
    DATA_DIR/
    "Training_Datasets"/
    "Processed"/
    "MS_Train_Test_Sets"
)

ms_train_data=pd.read_csv(
    ms_train_test_dir/
    "ms_train_data_remaining_minimal.csv",
    index_col=0,
)

ms_val_data=pd.read_csv(
    ms_train_test_dir/
    "ms_val_data.csv",
    index_col=0,
)

ms_test_data=pd.read_csv(
    ms_train_test_dir/
    "ms_test_data.csv",
    index_col=0,
)

ms_train_data["domain_id"]=0

print(
    f"Full MS train rows: {len(ms_train_data):,}"
)
print(
    f"Full MS val rows:   {len(ms_val_data):,}"
)
print(
    f"Full MS test rows:  {len(ms_test_data):,}"
)

print()


# ============================================================
# YD / HOLEC
# ============================================================

if dataset_to_run=="yd":

    dataset_prefix="holec"
    task_type="classification"

    print("Loading YD data...")

    data_dir=(
        DATA_DIR/
        "Training_Datasets"/
        "Processed"/
        "Holec_et_al_2025"
    )

    ft_train=pd.read_csv(
        data_dir/
        "holec_train_set_remaining.csv",
        index_col=0,
    )

    ft_val=pd.read_csv(
        data_dir/
        "holec_val_set.csv",
        index_col=0,
    )

    ft_test=pd.read_csv(
        data_dir/
        "holec_test_set.csv",
        index_col=0,
    )

    ft_train["domain_id"]=1

    # --------------------------------------------------------
    # FULL MS + YD
    #
    # RETRAINING DIFFERENCE:
    # no allele matching
    # no length matching
    # --------------------------------------------------------

    train_set=pd.concat(
        [
            ms_train_data,
            ft_train,
        ],
        ignore_index=True,
    )

    if ms_matched:
        train_set["seq_len"]=train_set.seq.str.replace("$","").apply(len)
        train_set=train_set[train_set.allele.isin(ft_train.allele.unique())].query("seq_len == 9").reset_index(drop=True)

    # Full MS validation/test.
    ms_val_processed=ms_val_data.copy()
    ms_test_processed=ms_test_data.copy()

    for name,df in [
        ("ms_test",ms_test_processed),
        ("ms_val",ms_val_processed),
        ("ft_test",ft_test),
        ("ft_val",ft_val),
    ]:

        df=full_process_data(
            df,
            mhc_seq_dict,
            remove_flank=True,
            add_pseudo_label=False,
        )

        df=create_seq_allele_column(df)

        if name=="ms_test":
            ms_test_processed=df

        elif name=="ms_val":
            ms_val_processed=df

        elif name=="ft_test":
            ft_test=df

        else:
            ft_val=df


    filtered_columns=[
        "seq",
        "allele",
        "label",
        "seq_allele",
        "n_flank",
        "c_flank",
        "mhc_a_1",
        "mhc_b_1",
        "mhc_c_1",
        "mhc_a_2",
        "mhc_b_2",
        "mhc_c_2",
        "sa_ma",
        "mhcprime",
        "group",
    ]

    ms_test_processed=filter_single_df_to_cols(
        ms_test_processed,
        filtered_columns,
    )

    ms_val_processed=filter_single_df_to_cols(
        ms_val_processed,
        filtered_columns,
    )

    ft_test=filter_single_df_to_cols(
        ft_test,
        filtered_columns,
    )

    ft_val=filter_single_df_to_cols(
        ft_val,
        filtered_columns,
    )


    test_sets={
        "holec_test":ft_test,
        "ms_test":ms_test_processed,
    }

    val_sets={
        "holec_val":ft_val,
        "ms_val":ms_val_processed,
    }


# ============================================================
# EPISCAN R9 / R9RT
# ============================================================

elif dataset_to_run in {
    "episcan_r9",
    "episcan_r9rt",
}:

    dataset_prefix=dataset_to_run
    task_type="classification"

    print("Loading EpiScan data...")

    data_dir=(
        DATA_DIR/
        "Training_Datasets"/
        "Processed"/
        "EpiScan"
    )

    r9_train=pd.read_csv(
        data_dir/
        "episcan_r9_train_set_remaining.csv",
        index_col=0,
    )

    rt_train=pd.read_csv(
        data_dir/
        "episcan_rt_train_set_remaining.csv",
        index_col=0,
    )

    r9_val=pd.read_csv(
        data_dir/
        "episcan_r9_val_set.csv",
        index_col=0,
    )

    rt_val=pd.read_csv(
        data_dir/
        "episcan_rt_val_set.csv",
        index_col=0,
    )

    r9_test=pd.read_csv(
        data_dir/
        "episcan_r9_test_set.csv",
        index_col=0,
    )

    rt_test=pd.read_csv(
        data_dir/
        "episcan_rt_test_set.csv",
        index_col=0,
    )

    sd3_formatted=pd.read_csv(
        data_dir/
        "sd3_formatted.csv",
        index_col=0,
    )


    r9_train["domain_id"]=1
    rt_train["domain_id"]=1


    # R9:
    #   R9 +/-
    #   RT positives only
    if dataset_to_run=="episcan_r9":

        ft_train=pd.concat(
            [
                r9_train,
                rt_train.query("label == 1"),
            ],
            ignore_index=True,
        )


    # R9RT:
    #   R9 +/-
    #   RT +/-
    else:

        ft_train=pd.concat(
            [
                r9_train,
                rt_train,
            ],
            ignore_index=True,
        )


    # --------------------------------------------------------
    # FULL MS + EPISCAN
    #
    # RETRAINING DIFFERENCE:
    # NO iso-allele filtering
    # NO 9-mer restriction on MS
    # --------------------------------------------------------

    train_set=pd.concat(
        [
            ms_train_data,
            ft_train,
        ],
        ignore_index=True,
    )

    if ms_matched:
        train_set["seq_len"]=train_set.seq.apply(len)
        iso_alleles=['A0201','A0301','B0801','B5701']
        train_set=train_set[train_set.allele.isin(iso_alleles)].query("seq_len == 9").reset_index(drop=True)


    # Combined R9 / RT validation and test sets.
    r9_val["dataset_source"]="r9"
    rt_val["dataset_source"]="rt"

    ft_val=pd.concat(
        [
            r9_val,
            rt_val,
        ],
        ignore_index=True,
    )


    r9_test["dataset_source"]="r9"
    rt_test["dataset_source"]="rt"

    ft_test=pd.concat(
        [
            r9_test,
            rt_test,
        ],
        ignore_index=True,
    )


    # Full MS validation/test.
    ms_val_processed=ms_val_data.copy()
    ms_test_processed=ms_test_data.copy()


    for name,df in [
        ("ms_test",ms_test_processed),
        ("ms_val",ms_val_processed),
        ("ft_test",ft_test),
        ("ft_val",ft_val),
        ("sd3",sd3_formatted),
    ]:

        df=full_process_data(
            df,
            mhc_seq_dict,
            remove_flank=True,
            add_pseudo_label=False,
        )

        df=create_seq_allele_column(df)

        if name=="ms_test":
            ms_test_processed=df

        elif name=="ms_val":
            ms_val_processed=df

        elif name=="ft_test":
            ft_test=df

        elif name=="ft_val":
            ft_val=df

        else:
            sd3_formatted=df


    filtered_columns=[
        "seq",
        "allele",
        "label",
        "seq_allele",
        "n_flank",
        "c_flank",
        "mhc_a_1",
        "mhc_b_1",
        "mhc_c_1",
        "mhc_a_2",
        "mhc_b_2",
        "mhc_c_2",
        "sa_ma",
        "mhcprime",
        "dataset_source",
    ]


    ms_test_processed=filter_single_df_to_cols(
        ms_test_processed,
        filtered_columns,
    )

    ms_val_processed=filter_single_df_to_cols(
        ms_val_processed,
        filtered_columns,
    )

    ft_test=filter_single_df_to_cols(
        ft_test,
        filtered_columns,
    )

    ft_val=filter_single_df_to_cols(
        ft_val,
        filtered_columns,
    )

    sd3_formatted=filter_single_df_to_cols(
        sd3_formatted,
        filtered_columns,
    )


    test_sets={
        "episcan_r9_rt_test":ft_test,
        "cov_test":sd3_formatted,
        "ms_test":ms_test_processed,
    }

    val_sets={
        "episcan_r9_rt_val":ft_val,
        "ms_val":ms_val_processed,
    }


# ============================================================
# BA CONTINUOUS
# ============================================================

elif dataset_to_run=="ba_con":

    dataset_prefix="ba_con"
    task_type="continuous"

    print("Loading BA data...")

    data_dir=(
        DATA_DIR/
        "Training_Datasets"/
        "Processed"/
        "BA"
    )

    ft_train=pd.read_csv(
        data_dir/
        "train_df_ba_for_updating_remaining.csv",
        index_col=0,
    )

    ft_val=pd.read_csv(
        data_dir/
        "ba_val_set.csv",
        index_col=0,
    )

    ft_test=pd.read_csv(
        data_dir/
        "test_df_ba_for_updating.csv",
        index_col=0,
    )

    ft_train["domain_id"]=1


    # --------------------------------------------------------
    # FULL MS + BA
    #
    # RETRAINING DIFFERENCE:
    # NO filter_domain0_to_domain1_pairs()
    # --------------------------------------------------------

    train_set=pd.concat(
        [
            ms_train_data,
            ft_train,
        ],
        ignore_index=True,
    )

    if ms_matched:
        train_set["seq_len"]=train_set.seq.str.replace("$","").apply(len)
        train_set=filter_domain0_to_domain1_pairs(
            train_set,allele_col="allele",seq_len_col="seq_len",domain_col="domain_id"
        ).reset_index(drop=True)


    # Full MS validation/test.
    ms_val_processed=ms_val_data.copy()
    ms_test_processed=ms_test_data.copy()


    for name,df in [
        ("ms_test",ms_test_processed),
        ("ms_val",ms_val_processed),
        ("ft_test",ft_test),
        ("ft_val",ft_val),
    ]:

        df=full_process_data(
            df,
            mhc_seq_dict,
            remove_flank=True,
            add_pseudo_label=False,
        )

        df=create_seq_allele_column(df)

        if name=="ms_test":
            ms_test_processed=df

        elif name=="ms_val":
            ms_val_processed=df

        elif name=="ft_test":
            ft_test=df

        else:
            ft_val=df


    filtered_columns=[
        "seq",
        "allele",
        "label",
        "seq_allele",
        "n_flank",
        "c_flank",
        "mhc_a_1",
        "mhc_b_1",
        "mhc_c_1",
        "mhc_a_2",
        "mhc_b_2",
        "mhc_c_2",
        "sa_ma",
        "mhcprime",
        "ba_value",
    ]


    ms_test_processed=filter_single_df_to_cols(
        ms_test_processed,
        filtered_columns,
    )

    ms_val_processed=filter_single_df_to_cols(
        ms_val_processed,
        filtered_columns,
    )

    ft_test=filter_single_df_to_cols(
        ft_test,
        filtered_columns,
    )

    ft_val=filter_single_df_to_cols(
        ft_val,
        filtered_columns,
    )


    test_sets={
        "ba_test":ft_test,
        "ms_test":ms_test_processed,
    }

    val_sets={
        "ba_val":ft_val,
        "ms_val":ms_val_processed,
    }


# ============================================================
# ESCAPE-SEQ CONTINUOUS
# ============================================================

elif dataset_to_run=="escape_con":

    dataset_prefix="escape_con"
    task_type="continuous"

    print("Loading ESCAPE-seq data...")

    data_dir=(
        DATA_DIR/
        "Training_Datasets"/
        "Processed"/
        "ESCAPE_seq"
    )


    # --------------------------------------------------------
    # CLEANED ESCAPE-seq files.
    # Keep these exactly.
    # --------------------------------------------------------

    ft_train=pd.read_csv(
        data_dir/
        "escape_con_train_set_remaining_cleaned.csv",
        index_col=0,
    )

    ft_val=pd.read_csv(
        data_dir/
        "escape_con_val_set_cleaned.csv",
        index_col=0,
    )

    ft_test=pd.read_csv(
        data_dir/
        "escape_con_test_set_cleaned.csv",
        index_col=0,
    )


    ft_train["domain_id"]=1


    # --------------------------------------------------------
    # FULL MS + ESCAPE
    #
    # RETRAINING DIFFERENCE:
    # NO filter_domain0_to_domain1_pairs()
    # --------------------------------------------------------

    train_set=pd.concat(
        [
            ms_train_data,
            ft_train,
        ],
        ignore_index=True,
    )

    if ms_matched:
        train_set["seq_len"]=train_set.seq.str.replace("$","").apply(len)
        train_set=filter_domain0_to_domain1_pairs(
            train_set,allele_col="allele",seq_len_col="seq_len",domain_col="domain_id"
        ).reset_index(drop=True)


    # Full MS validation/test.
    ms_val_processed=ms_val_data.copy()
    ms_test_processed=ms_test_data.copy()


    for name,df in [
        ("ms_test",ms_test_processed),
        ("ms_val",ms_val_processed),
        ("ft_test",ft_test),
        ("ft_val",ft_val),
    ]:

        df=full_process_data(
            df,
            mhc_seq_dict,
            remove_flank=True,
            add_pseudo_label=False,
        )

        df=create_seq_allele_column(df)

        if name=="ms_test":
            ms_test_processed=df

        elif name=="ms_val":
            ms_val_processed=df

        elif name=="ft_test":
            ft_test=df

        else:
            ft_val=df


    filtered_columns=[
        "seq",
        "allele",
        "label",
        "seq_allele",
        "n_flank",
        "c_flank",
        "mhc_a_1",
        "mhc_b_1",
        "mhc_c_1",
        "mhc_a_2",
        "mhc_b_2",
        "mhc_c_2",
        "sa_ma",
        "mhcprime",
        "escore",
        "ba_value",
        "name2",
        "name",
        "gene",
        "mutation",
        "mut_type",
    ]


    ms_test_processed=filter_single_df_to_cols(
        ms_test_processed,
        filtered_columns,
    )

    ms_val_processed=filter_single_df_to_cols(
        ms_val_processed,
        filtered_columns,
    )

    ft_test=filter_single_df_to_cols(
        ft_test,
        filtered_columns,
    )

    ft_val=filter_single_df_to_cols(
        ft_val,
        filtered_columns,
    )


    test_sets={
        "escape_test":ft_test,
        "ms_test":ms_test_processed,
    }

    val_sets={
        "escape_val":ft_val,
        "ms_val":ms_val_processed,
    }


# ============================================================
# DATA SUMMARY
# ============================================================

print()
print("="*80)
print("RETRAINING DATA")
print("="*80)

print("Dataset:",dataset_to_run)
print("Task type:",task_type)
print("Dataset prefix:",dataset_prefix)

print(
    "MS train rows:",
    f"{(train_set['domain_id']==0).sum():,}",
)

print(
    "Assay train rows:",
    f"{(train_set['domain_id']==1).sum():,}",
)

print(
    "Total joint train rows:",
    f"{len(train_set):,}",
)

print("Test sets:",list(test_sets))
print("Validation sets:",list(val_sets))

print_data_stats(train_set)

print()


# ============================================================
# TOKENIZER / BLANK MHCPrime
# ============================================================

print("="*80)
print("INITIALIZING BLANK MHCPrime")
print("="*80)

tokenizer=AminoAcidTokenizer()


# Same MHCPrime architecture used in the current lambda script.
base_model_params={
    "vocab_size":tokenizer.vocab_size,

    "embed_dim":480,
    "fc_hidden_dims":[240,120],
    "fc_dropout":0.1,
    "n_heads":16,

    "max_peptide_len":34,
    "max_mhc_len":34,

    "n_peptide_layers":2,
    "n_mhc_layers":1,
    "n_concat_layers":3,

    "add_pos_enc":True,
    "use_segment_embedder":True,
    "dim_feedforward_multiplier":1,

    "debugging":False,

    "fc_output_dim":1,
    "pooling":"cls",
    "ffnn_use_layernorm":False,

    "use_feature_table":True,
    "feature_table":processed_feature_table,
    "sequence_embedder_feature_names":feature_names,
    "sequence_embedder_feature_mode":"concat",
    "sequence_embedder_tokenizer":tokenizer,

    "peptide_start_pos":1,
    "mhc_start_pos":35,

    "pos_enc_type":"sincos",

    # Blank-model initialization seed.
    "seed":retraining_seed,


    # --------------------------------------------------------
    # Peptide conformer
    # --------------------------------------------------------

    "add_peptide_conformer":True,
    "peptide_conformer_num_layers":2,
    "peptide_conformer_n_heads":16,
    "peptide_conformer_conv_kernel":5,
    "peptide_conformer_dropout":0.0,


    # --------------------------------------------------------
    # MHC conformer
    # --------------------------------------------------------

    "add_mhc_conformer":False,
    "mhc_conformer_num_layers":2,
    "mhc_conformer_n_heads":16,
    "mhc_conformer_conv_kernel":5,
    "mhc_conformer_dropout":0.0,


    # --------------------------------------------------------
    # Fusion conformer
    # --------------------------------------------------------

    "add_fusion_conformer":True,
    "fusion_conformer_num_layers":1,
    "fusion_conformer_n_heads":16,
    "fusion_conformer_conv_kernel":7,
    "fusion_conformer_dropout":0.0,
    "fusion_conformer_mask_bos":True,

    "ff_pre_post_swiglu":False,


    # --------------------------------------------------------
    # Legacy initialization compatibility
    # --------------------------------------------------------

    "legacy_init_compat":True,
}


clear_all_gpu_memory()


# ------------------------------------------------------------
# IMPORTANT:
#
# Blank/random initialization.
# There is deliberately NO checkpoint load here.
# ------------------------------------------------------------

retraining_model=init_model(
    base_model_params,
    print_check=False,
    print_params=True,
)

retraining_model.to(device)


print()
print("Blank MHCPrime initialized.")
print(
    "Model device:",
    next(retraining_model.parameters()).device,
)
print()


# ============================================================
# CREATE TEST / VALIDATION CACHES
#
# Done BEFORE training.
#
# MS validation/test sets remain FULL MS val/test:
# ms_matched only changes the MS TRAINING support.
# ============================================================

print("="*80)
print("CREATING TEST / VALIDATION CACHES")
print("="*80)


test_set_cache_paths={}

for test_set_name,test_data_set in test_sets.items():

    print(
        "Caching test set:",
        test_set_name,
    )

    data_to_run_path=(
        DATA_DIR/
        f"{test_set_name}_{gpu_id}_retraining_cache.pt"
    )

    build_token_cache(
        test_data_set,
        data_to_run_path,
        tokenizer,
        use_domain_id=False,
    )

    test_set_cache_paths[
        test_set_name
    ]=data_to_run_path


val_set_cache_paths={}

for val_set_name,val_data_set in val_sets.items():

    print(
        "Caching validation set:",
        val_set_name,
    )

    data_to_run_path=(
        DATA_DIR/
        f"{val_set_name}_{gpu_id}_retraining_cache.pt"
    )

    build_token_cache(
        val_data_set,
        data_to_run_path,
        tokenizer,
        use_domain_id=False,
    )

    val_set_cache_paths[
        val_set_name
    ]=data_to_run_path


print()


# ============================================================
# RETRAINING PARAMETERS
# ============================================================

print("="*80)
print("SETTING RETRAINING PARAMETERS")
print("="*80)


# ------------------------------------------------------------
# Classification
# ------------------------------------------------------------

retraining_params_cls=dict(

    tokenizer=tokenizer,
    device=device,


    # --------------------------------------------------------
    # Steps
    # --------------------------------------------------------

    steps=total_steps,


    # --------------------------------------------------------
    # Intermediate MODEL checkpoints
    #
    # These go to Model_Checkpoints/Joint_Retraining/...
    # --------------------------------------------------------

    output_dir=(
        str(checkpoint_dir)
        if save_every_n_steps is not None
        else None
    ),

    save_every_n_steps=save_every_n_steps,


    # --------------------------------------------------------
    # General
    # --------------------------------------------------------

    seed=retraining_seed,

    num_workers=num_workers,

    collate_fn=collate_fn,


    # --------------------------------------------------------
    # Sampling
    # --------------------------------------------------------

    num_ms_pos=2000,

    num_nonms_pos=2000,

    ms_neg_ratio=1.0,

    nonms_neg_ratio=1.0,

    auto_calibrate=True,

    calibration_steps=1,

    ms_batch_size=ms_batch_size,

    nonms_batch_size=nonms_batch_size,


    # --------------------------------------------------------
    # Optimization
    #
    # SAME PepCL scheduler machinery.
    # --------------------------------------------------------

    encoder_lr=learning_rate,

    decoder_lr=learning_rate,

    weight_decay=weight_decay,

    warmup_ratio=warmup_ratio,

    update_domain_embeddings=False,


    # --------------------------------------------------------
    # Joint training
    # --------------------------------------------------------

    use_replay=True,

    primary_task_id=1,

    sequential_backprop=True,


    # --------------------------------------------------------
    # MS classification
    # --------------------------------------------------------

    ms_loss_type="logsmoothap",

    ms_loss_alpha=10.0,

    ms_lambda=1.0,


    # --------------------------------------------------------
    # Assay classification
    # --------------------------------------------------------

    nonms_loss_type="logsmoothap",

    nonms_loss_alpha=10.0,

    ft_lambda=1.0,


    # --------------------------------------------------------
    # NO teacher / NO DR
    # --------------------------------------------------------

    dr_lambda=0.0,

    dr_loss_type="mse",


    # --------------------------------------------------------
    # New retraining mode
    # --------------------------------------------------------

    merged_loss=merged_loss,


    # --------------------------------------------------------
    # Logging
    #
    # Explicitly enabled.
    # --------------------------------------------------------

    enable_logging=True,
)


# ------------------------------------------------------------
# Continuous
# ------------------------------------------------------------

retraining_params_con=dict(

    tokenizer=tokenizer,
    device=device,


    # --------------------------------------------------------
    # Steps
    # --------------------------------------------------------

    steps=total_steps,


    # --------------------------------------------------------
    # Intermediate model checkpoints
    # --------------------------------------------------------

    output_dir=(
        str(checkpoint_dir)
        if save_every_n_steps is not None
        else None
    ),

    save_every_n_steps=save_every_n_steps,


    # --------------------------------------------------------
    # General
    # --------------------------------------------------------

    seed=retraining_seed,

    num_workers=num_workers,

    collate_fn=collate_fn,


    # --------------------------------------------------------
    # Sampling
    # --------------------------------------------------------

    num_ms_pos=2000,

    num_nonms_pos=2000,

    ms_neg_ratio=1.0,

    nonms_neg_ratio=1.0,

    auto_calibrate=True,

    calibration_steps=1,

    ms_batch_size=ms_batch_size,

    nonms_batch_size=nonms_batch_size,


    # --------------------------------------------------------
    # Optimization
    # --------------------------------------------------------

    encoder_lr=learning_rate,

    decoder_lr=learning_rate,

    weight_decay=weight_decay,

    warmup_ratio=warmup_ratio,

    update_domain_embeddings=False,


    # --------------------------------------------------------
    # Joint training
    # --------------------------------------------------------

    use_replay=True,

    primary_task_id=1,

    sequential_backprop=True,


    # --------------------------------------------------------
    # MS classification loss
    # --------------------------------------------------------

    ms_loss_type="logsmoothap",

    ms_loss_alpha=10.0,

    ms_lambda=1.0,


    # --------------------------------------------------------
    # New assay continuous loss
    # --------------------------------------------------------

    nonms_loss_type="spearman",

    nonms_loss_alpha=0.01,

    ft_lambda=1.0,


    # --------------------------------------------------------
    # NO teacher / NO DR
    # --------------------------------------------------------

    dr_lambda=0.0,

    dr_loss_type="mse",


    # --------------------------------------------------------
    # Continuous-data settings
    #
    # Preserved from current PepCL script.
    # --------------------------------------------------------

    continuous_target_col="ba_value",

    continuous_ft_sampling_mode="allele_aware",


    # --------------------------------------------------------
    # Merged loss is classification-only.
    # --------------------------------------------------------

    merged_loss=False,


    # --------------------------------------------------------
    # Logging
    # --------------------------------------------------------

    enable_logging=True,
)


# Select parameter set.
if task_type=="continuous":

    retraining_params=(
        retraining_params_con.copy()
    )

    print(
        "Using CONTINUOUS retraining parameters."
    )

else:

    retraining_params=(
        retraining_params_cls.copy()
    )

    print(
        "Using CLASSIFICATION retraining parameters."
    )


print()
print(
    "Steps:",
    total_steps,
)

print(
    "LR:",
    learning_rate,
)

print(
    "MS batch:",
    ms_batch_size,
)

print(
    "Assay batch:",
    nonms_batch_size,
)

print(
    "Merged loss:",
    retraining_params["merged_loss"],
)

print(
    "Logging enabled:",
    retraining_params["enable_logging"],
)

print()


# ============================================================
# JOINT RETRAINING
# ============================================================

print("="*80)
print("STARTING JOINT RETRAINING")
print("="*80)


training_t0=time.perf_counter()


trainer=PepCL_Retraining(

    model=retraining_model,

    train_df=train_set,

    mhc_seq_dict=mhc_seq_dict,

    **retraining_params,
)


training_stats=trainer.train()


training_seconds=(
    time.perf_counter()-training_t0
)


print()
print("="*80)
print("RETRAINING COMPLETE")
print("="*80)

print(
    f"Training seconds: "
    f"{training_seconds:,.2f}"
)

print(
    f"Training minutes: "
    f"{training_seconds/60:,.2f}"
)

print(
    f"Training hours: "
    f"{training_seconds/3600:,.3f}"
)

print()


# ============================================================
# SAVE FINAL MODEL
#
# MODEL_CHECKPOINTS_DIR, not DATA_DIR.
# ============================================================

print("="*80)
print("SAVING FINAL MODEL")
print("="*80)

print(final_model_path)


torch.save(
    retraining_model.state_dict(),
    final_model_path,
)


print("Final model saved.")
print()


# ============================================================
# SAVE TRAINING STATS + FULL LOGS
#
# These go to DATA_DIR alongside test/val results.
# ============================================================

print("="*80)
print("SAVING TRAINING STATS / LOGS")
print("="*80)


training_stats_path=(
    data_output_dir/
    "training_stats.pkl"
)

with open(
    training_stats_path,
    "wb",
) as f:

    pickle.dump(
        training_stats,
        f,
    )


print(
    "Training stats:",
    training_stats_path,
)


# Full PepCL_Retraining logging output.
training_logs=trainer.get_logs()


training_logs_path=(
    data_output_dir/
    "training_logs.pkl"
)

with open(
    training_logs_path,
    "wb",
) as f:

    pickle.dump(
        training_logs,
        f,
    )


print(
    "Training logs:",
    training_logs_path,
)

print()


# ============================================================
# FINAL MODEL SCORE COLUMN NAME
# ============================================================

model_score_name=(
    Path(final_model_name).stem
)


# ============================================================
# FINAL TEST-SET INFERENCE
# ============================================================

print("="*80)
print("RUNNING FINAL TEST-SET INFERENCE")
print("="*80)


retraining_model=(
    retraining_model.to(device)
)

retraining_model.eval()


for test_set_name,test_data_set in test_sets.items():

    data_to_run_path=(
        test_set_cache_paths[
            test_set_name
        ]
    )

    print()
    print(
        "Test set:",
        test_set_name,
    )

    print(
        "Cache:",
        data_to_run_path,
    )


    with torch.no_grad():

        output_scores=run_mhcprime_fast(

            retraining_model,

            data_to_run_path,

            batch_size=(
                inference_batch_size
            ),

            num_workers=(
                inference_num_workers
            ),

            compile_model=False,

            use_domain_id=False,
        )


    test_data_set[
        model_score_name
    ]=(
        output_scores
        .detach()
        .cpu()
        .tolist()
    )


    test_sets[
        test_set_name
    ]=test_data_set


# ============================================================
# FINAL VALIDATION-SET INFERENCE
# ============================================================

print()
print("="*80)
print("RUNNING FINAL VALIDATION-SET INFERENCE")
print("="*80)


for val_set_name,val_data_set in val_sets.items():

    data_to_run_path=(
        val_set_cache_paths[
            val_set_name
        ]
    )

    print()
    print(
        "Validation set:",
        val_set_name,
    )

    print(
        "Cache:",
        data_to_run_path,
    )


    with torch.no_grad():

        output_scores=run_mhcprime_fast(

            retraining_model,

            data_to_run_path,

            batch_size=(
                inference_batch_size
            ),

            num_workers=(
                inference_num_workers
            ),

            compile_model=False,

            use_domain_id=False,
        )


    val_data_set[
        model_score_name
    ]=(
        output_scores
        .detach()
        .cpu()
        .tolist()
    )


    val_sets[
        val_set_name
    ]=val_data_set


# ============================================================
# VALIDATION-CHECKPOINT INFERENCE
#
# Checkpoint models are scored on VALIDATION SETS ONLY.
#
# This always uses the full validation sets already stored in
# val_sets. ms_matched only changes the MS TRAINING support and
# does not partition or restrict validation data.
#
# Score-column naming:
#     <full final model name>_step_N
#
# Example:
#     ..._seed_42_step_50
# ============================================================

if validation_checkpoint_steps:

    print()
    print("="*80)
    print("RUNNING VALIDATION-CHECKPOINT INFERENCE")
    print("="*80)

    print(
        "Validation checkpoint steps:",
        validation_checkpoint_steps,
    )

    for checkpoint_step in validation_checkpoint_steps:

        checkpoint_path=(
            checkpoint_dir/
            f"pepcl_step_{checkpoint_step}.pt"
        )

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Expected validation checkpoint not found: "
                f"{checkpoint_path}"
            )

        checkpoint_score_name=(
            f"{model_score_name}_step_{checkpoint_step}"
        )

        print()
        print(
            "Checkpoint step:",
            checkpoint_step,
        )
        print(
            "Checkpoint:",
            checkpoint_path,
        )
        print(
            "Score column:",
            checkpoint_score_name,
        )

        clear_all_gpu_memory()

        checkpoint_model=init_model(
            base_model_params,
            print_check=False,
            print_params=False,
        )

        checkpoint_state_dict=torch.load(
            checkpoint_path,
            map_location="cpu",
        )

        checkpoint_model.load_state_dict(
            checkpoint_state_dict,
            strict=True,
        )

        checkpoint_model.to(device)
        checkpoint_model.eval()

        for val_set_name,val_data_set in val_sets.items():

            data_to_run_path=(
                val_set_cache_paths[
                    val_set_name
                ]
            )

            print(
                "Validation set:",
                val_set_name,
            )

            with torch.no_grad():

                output_scores=run_mhcprime_fast(

                    checkpoint_model,

                    data_to_run_path,

                    batch_size=(
                        inference_batch_size
                    ),

                    num_workers=(
                        inference_num_workers
                    ),

                    compile_model=False,

                    use_domain_id=False,
                )

            val_data_set[
                checkpoint_score_name
            ]=(
                output_scores
                .detach()
                .cpu()
                .tolist()
            )

            val_sets[
                val_set_name
            ]=val_data_set

        checkpoint_model.to("cpu")
        del checkpoint_model
        del checkpoint_state_dict
        clear_all_gpu_memory()


# ============================================================
# REMOVE VALIDATION CHECKPOINT MODELS IF NOT RETAINED
# ============================================================

if (
    validation_checkpoint_steps
    and not save_validation_checkpoint_models
):

    print()
    print(
        "Removing validation checkpoint model files..."
    )

    for checkpoint_step in validation_checkpoint_steps:

        checkpoint_path=(
            checkpoint_dir/
            f"pepcl_step_{checkpoint_step}.pt"
        )

        if checkpoint_path.exists():
            os.remove(checkpoint_path)


# ============================================================
# SAVE FINAL TEST DATASETS
#
# DATA_DIR/Joint_Retraining/<dataset>/
# ============================================================

print()
print("="*80)
print("SAVING TEST SETS")
print("="*80)


for test_set_name,test_data_set in test_sets.items():

    save_path=(
        data_output_dir/
        f"{test_set_name}.csv"
    )

    print(
        test_set_name,
        "->",
        save_path,
    )

    test_data_set.to_csv(
        save_path
    )


# ============================================================
# SAVE FINAL VALIDATION DATASETS
# ============================================================

print()
print("="*80)
print("SAVING VALIDATION SETS")
print("="*80)


for val_set_name,val_data_set in val_sets.items():

    save_path=(
        data_output_dir/
        f"{val_set_name}.csv"
    )

    print(
        val_set_name,
        "->",
        save_path,
    )

    val_data_set.to_csv(
        save_path
    )


# ============================================================
# REMOVE TEMPORARY CACHES
# ============================================================

print()
print(
    "Removing temporary inference caches..."
)


for test_name,test_path in (
    test_set_cache_paths.items()
):

    if os.path.exists(test_path):
        os.remove(test_path)


for val_name,val_path in (
    val_set_cache_paths.items()
):

    if os.path.exists(val_path):
        os.remove(val_path)


# ============================================================
# FULL-RUN TIMING
# ============================================================

full_run_seconds=(
    time.perf_counter()-full_run_t0
)


runtime_summary={

    # --------------------------------------------------------
    # Dataset / task
    # --------------------------------------------------------

    "dataset":dataset_to_run,

    "dataset_prefix":dataset_prefix,

    "task_type":task_type,

    "ms_matched":bool(
        ms_matched
    ),


    # --------------------------------------------------------
    # Paths
    # --------------------------------------------------------

    "data_output_dir":str(
        data_output_dir
    ),

    "model_output_dir":str(
        model_output_dir
    ),

    "model_name":final_model_name,

    "model_path":str(
        final_model_path
    ),

    "training_stats_path":str(
        training_stats_path
    ),

    "training_logs_path":str(
        training_logs_path
    ),


    # --------------------------------------------------------
    # Training config
    # --------------------------------------------------------

    "seed":int(
        retraining_seed
    ),

    "steps":int(
        total_steps
    ),

    "learning_rate":float(
        learning_rate
    ),

    "weight_decay":float(
        weight_decay
    ),

    "warmup_ratio":float(
        warmup_ratio
    ),

    "merged_loss":bool(
        merged_loss
    ),

    "configured_batch_size":int(
        batch_size
    ),

    "ms_batch_size":int(
        ms_batch_size
    ),

    "nonms_batch_size":int(
        nonms_batch_size
    ),

    "num_workers":int(
        num_workers
    ),

    "save_every_n_steps":(
        save_every_n_steps
    ),

    "save_validation_checkpoint_models":bool(
        save_validation_checkpoint_models
    ),

    "validation_checkpoint_steps":[
        int(step)
        for step in validation_checkpoint_steps
    ],

    "overwrite_existing":bool(
        overwrite_existing
    ),


    # --------------------------------------------------------
    # Dataset counts
    # --------------------------------------------------------

    "ms_train_rows":int(
        (
            train_set[
                "domain_id"
            ]==0
        ).sum()
    ),

    "assay_train_rows":int(
        (
            train_set[
                "domain_id"
            ]==1
        ).sum()
    ),

    "total_train_rows":int(
        len(train_set)
    ),


    # --------------------------------------------------------
    # Timing
    # --------------------------------------------------------

    "training_seconds":float(
        training_seconds
    ),

    "training_minutes":float(
        training_seconds/60
    ),

    "training_hours":float(
        training_seconds/3600
    ),

    "full_run_seconds":float(
        full_run_seconds
    ),

    "full_run_minutes":float(
        full_run_seconds/60
    ),

    "full_run_hours":float(
        full_run_seconds/3600
    ),
}


runtime_path=(
    data_output_dir/
    "runtime_summary.json"
)


with open(
    runtime_path,
    "w",
) as f:

    json.dump(
        runtime_summary,
        f,
        indent=2,
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("="*80)
print("JOINT RETRAINING RUN COMPLETE")
print("="*80)

print(
    "Dataset:",
    dataset_to_run,
)

print()
print(
    "Final model:"
)
print(
    final_model_path
)

print()
print(
    "Data/results directory:"
)
print(
    data_output_dir
)

print()
print(
    "Model directory:"
)
print(
    model_output_dir
)

if save_every_n_steps is not None:

    print()

    if save_validation_checkpoint_models:
        print(
            "Intermediate checkpoints:"
        )

        print(
            checkpoint_dir
        )

    else:
        print(
            "Intermediate validation checkpoints were scored and removed."
        )


print()

print(
    f"Training time: "
    f"{training_seconds/3600:.3f} h"
)

print(
    f"Full run time: "
    f"{full_run_seconds/3600:.3f} h"
)


print()

print(
    "Training stats:",
    training_stats_path,
)

print(
    "Training logs:",
    training_logs_path,
)

print(
    "Runtime summary:",
    runtime_path,
)

print(
    "Test CSVs:",
    list(test_sets),
)

print(
    "Validation CSVs:",
    list(val_sets),
)

print(
    "Validation checkpoint steps:",
    validation_checkpoint_steps,
)

print("="*80)