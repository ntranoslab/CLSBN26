"""
Run cmd: python Multi_Dataset_Updating_Long_Horizon.py -g gpu_id
"""
import sys
from pathlib import Path
import shutil
import time

PROJECT_ROOT=Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0,str(PROJECT_ROOT))

DATA_DIR=PROJECT_ROOT / "Data"
SRC_DIR=PROJECT_ROOT / "src"
MODEL_CHECKPOINTS_DIR=PROJECT_ROOT / "Model_Checkpoints"
ANALYSIS_DIR=PROJECT_ROOT / "Analysis"

from Data.mhc_pseudo_seqs import mhc_seq_dict
from src.Model.mhcprime import *
from src.model_utils import *
from src.general_utils import *
from src.get_feature_data import *
from src.mhcprime_trainer import *
from src.mhcprime_inference import *
from src.pepcl import *
from src.pepcl_multi import *

import warnings
warnings.filterwarnings("ignore")

import argparse
parser=argparse.ArgumentParser(description="Select specific GPU to run model on.")
parser.add_argument("-g","--gpu",type=int,default=0,help="Which GPU ID to use (e.g. 0, 1, 2...)")
args=parser.parse_args(); gpu_id=args.gpu

GPU_ID=gpu_id
torch.cuda.set_device(GPU_ID)
print(f"Using GPU: {torch.cuda.get_device_name(GPU_ID)} (ID: {torch.cuda.current_device()})")
device='cuda' if torch.cuda.is_available() else 'cpu'

# ============================================================
# CUSTOMIZE RUN
# ============================================================
seeds=[0,1,2,3,4]


# All seed-specific dataframe outputs are written inside this directory as seed_0/, seed_1/, ...
full_output_root=DATA_DIR / "Multi_Dataset_Updating_Results" / "T1_Allele_Partition_Sets"

# num_workers=16
num_workers=4

# ============================================================
# COMMON SETUP
# ============================================================
mhc_seq_dict={k.removeprefix("HLA-").replace("*","").replace(":",""):v for k,v in mhc_seq_dict.items()}

print("Loading feature data...")
aa_features_table_ext,descriptor_groups=load_feature_data()
processed_feature_table,corr_info=process_aa_feature_table(aa_features_table_ext)
feature_names=list(processed_feature_table.columns)
print(end="\n")

# ============================================================
# LOAD ALL TRAIN / VALIDATION / TEST SETS
# ============================================================
print("Loading all train, validation, and test sets...")

# MS: use the ablation training set + standard validation/test sets.
ms_datasets_dir=DATA_DIR / "Training_Datasets" / "Processed" / "MS_Train_Test_Sets"
ms_train_data=pd.read_csv(f"{ms_datasets_dir}/ms_train_data_remaining_minimal.csv",index_col=0)
ms_val_data=pd.read_csv(f"{ms_datasets_dir}/ms_val_data.csv",index_col=0)
ms_test_data=pd.read_csv(f"{ms_datasets_dir}/ms_test_data.csv",index_col=0)

# YD / Holec: use the ablation train-remaining set + validation/test sets.
holec_datasets_dir=DATA_DIR / "Training_Datasets" / "Processed" / "Holec_et_al_2025"
holec_train_set=pd.read_csv(f"{holec_datasets_dir}/holec_train_set_remaining.csv",index_col=0)
holec_val_set=pd.read_csv(f"{holec_datasets_dir}/holec_val_set.csv",index_col=0)
holec_test_set=pd.read_csv(f"{holec_datasets_dir}/holec_test_set.csv",index_col=0)

# EpiScan R9/RT: use both train-remaining sets and both validation/test sets.
episcan_train_test_dir=DATA_DIR / "Training_Datasets" / "Processed" / "EpiScan"
episcan_r9_train_set=pd.read_csv(f"{episcan_train_test_dir}/episcan_r9_train_set_remaining.csv",index_col=0)
episcan_rt_train_set=pd.read_csv(f"{episcan_train_test_dir}/episcan_rt_train_set_remaining.csv",index_col=0)
episcan_r9_val_set=pd.read_csv(f"{episcan_train_test_dir}/episcan_r9_val_set.csv",index_col=0)
episcan_rt_val_set=pd.read_csv(f"{episcan_train_test_dir}/episcan_rt_val_set.csv",index_col=0)
episcan_r9_test_set=pd.read_csv(f"{episcan_train_test_dir}/episcan_r9_test_set.csv",index_col=0)
episcan_rt_test_set=pd.read_csv(f"{episcan_train_test_dir}/episcan_rt_test_set.csv",index_col=0)
sd3_formatted=pd.read_csv(f"{episcan_train_test_dir}/sd3_formatted.csv",index_col=0)

# Preserve source before merging EpiScan R9/RT validation/test sets.
episcan_r9_val_set["dataset_source"]="r9"
episcan_rt_val_set["dataset_source"]="rt"
episcan_r9_test_set["dataset_source"]="r9"
episcan_rt_test_set["dataset_source"]="rt"

# BA: pre-created multi-dataset classification train/test split.
ba_train_test_dir=DATA_DIR / "Training_Datasets" / "Processed" / "BA"
ba_train_set=pd.read_csv(f"{ba_train_test_dir}/ba_train_set_for_md_updating.csv",index_col=0)
ba_test_set=pd.read_csv(f"{ba_train_test_dir}/ba_test_set_for_md_updating.csv",index_col=0)

# ESCAPE-seq: pre-created multi-dataset classification train/test split.
escape_train_test_dir=DATA_DIR / "Training_Datasets" / "Processed" / "ESCAPE_seq"
escape_train_set=pd.read_csv(f"{escape_train_test_dir}/escape_train_set_for_md_updating.csv",index_col=0)
escape_test_set=pd.read_csv(f"{escape_train_test_dir}/escape_test_set_for_md_updating.csv",index_col=0)
print(end="\n")

# ============================================================
# PROCESS SETS -- SAME WRAPPER / PROCESSING AS ORIGINAL SCRIPT
# ============================================================
print("Processing train, validation, and test sets...")

def processing_wrapper(df):
    df=full_process_data(df,mhc_seq_dict,remove_flank=True,add_pseudo_label=False)
    df=create_seq_allele_column(df)
    df["peptide"]=df.seq.str.replace("$","")
    df["seq_len"]=df.seq.str.replace("$","").apply(len)
    return df

ms_train_data=processing_wrapper(ms_train_data)
ms_val_data=processing_wrapper(ms_val_data)
ms_test_data=processing_wrapper(ms_test_data)

holec_train_set=processing_wrapper(holec_train_set)
holec_val_set=processing_wrapper(holec_val_set)
holec_test_set=processing_wrapper(holec_test_set)

episcan_r9_train_set=processing_wrapper(episcan_r9_train_set)
episcan_rt_train_set=processing_wrapper(episcan_rt_train_set)
episcan_r9_val_set=processing_wrapper(episcan_r9_val_set)
episcan_rt_val_set=processing_wrapper(episcan_rt_val_set)
episcan_r9_test_set=processing_wrapper(episcan_r9_test_set)
episcan_rt_test_set=processing_wrapper(episcan_rt_test_set)
sd3_formatted=processing_wrapper(sd3_formatted)

ba_train_set=processing_wrapper(ba_train_set)
ba_test_set=processing_wrapper(ba_test_set)

escape_train_set=processing_wrapper(escape_train_set)
escape_test_set=processing_wrapper(escape_test_set)
print(end="\n")

# ============================================================
# MERGE EPISCAN R9 / RT SETS
# ============================================================
print("Merging EpiScan R9/RT sets...")
episcan_train_set=pd.concat([episcan_r9_train_set,episcan_rt_train_set])
episcan_val_set=pd.concat([episcan_r9_val_set,episcan_rt_val_set])
episcan_test_set=pd.concat([episcan_r9_test_set,episcan_rt_test_set])
print_data_stats(episcan_train_set)
print_data_stats(episcan_val_set)
print_data_stats(episcan_test_set)
print(end="\n")

# ============================================================
# AGGREGATE COMMON DATAFRAMES
# ============================================================
# Replay set is held separately to create the adaptive buffer.
train_sets={
    "ms":ms_train_data,
    "yd":holec_train_set,
    "episcan":episcan_train_set,
    "ba":ba_train_set,
    "escape":escape_train_set,
}

# BA and ESCAPE-seq intentionally have no validation sets for this classification-updating experiment.
test_sets={
    "ms_test":ms_test_data,
    "yd_test":holec_test_set,
    "episcan_test":episcan_test_set,
    "ba_test":ba_test_set,
    "escape_test":escape_test_set,
    "cov_test":sd3_formatted,
}

val_sets={
    "ms_val":ms_val_data,
    "yd_val":holec_val_set,
    "episcan_val":episcan_val_set,
}

# ============================================================
# FILTER EVALUATION COLUMNS
# ============================================================
# Same minimal evaluation columns as the original script; dataset_source is retained
# so merged EpiScan R9/RT validation/test rows remain identifiable.
filtered_columns=['seq','allele','label','seq_allele','n_flank','c_flank','mhc_a_1','mhc_b_1','mhc_c_1','mhc_a_2','mhc_b_2','mhc_c_2','sa_ma','mhcprime','dataset_source','escore','ba_value']
test_sets=filter_dict_columns(test_sets,filtered_columns)
val_sets=filter_dict_columns(val_sets,filtered_columns)

# ============================================================
# REMOVE TRAIN / TEST OVERLAPS -- SAME OPERATION AS ORIGINAL SCRIPT
# ============================================================
print("Cleaning train/test cross-dataset peptide-allele overlaps...")
seq_allele_overlap_matrix,train_sets=seq_allele_overlap_table(train_sets,test_sets,seq_col="seq",allele_col="allele",remove_test_from_train=True)
print(end="\n")

# MS is used as replay buffer separately, not as an update-domain train set.
ms_train_data=train_sets["ms"].copy()
if "ms" in train_sets:
    print("Deleting MS key; used as buffer separately.")
    del train_sets["ms"]

# Keep clean processed templates so every seed starts with independent dataframes.
test_sets_template={k:v.copy() for k,v in test_sets.items()}
val_sets_template={k:v.copy() for k,v in val_sets.items()}

# ============================================================
# LOAD MHCPrime BASE FOR ABLATIONS
# ============================================================
print("Loading MHCPrime base checkpoint...")
tokenizer=AminoAcidTokenizer()

base_model_params={
    'vocab_size':tokenizer.vocab_size,
    'embed_dim':480,
    'fc_hidden_dims':[240,120],
    'fc_dropout':0.1,
    'n_heads':16,
    'max_peptide_len':34,
    'max_mhc_len':34,
    "n_peptide_layers":2,
    "n_mhc_layers":1,
    'n_concat_layers':3,
    'add_pos_enc':True,
    'use_segment_embedder':True,
    'dim_feedforward_multiplier':1,
    'debugging':False,
    'fc_output_dim':1,
    'pooling':'cls',
    "ffnn_use_layernorm":False,
    'use_feature_table':True,
    "feature_table":processed_feature_table,
    "sequence_embedder_feature_names":feature_names,
    'sequence_embedder_feature_mode':"concat",
    'sequence_embedder_tokenizer':tokenizer,
    "peptide_start_pos":1,
    "mhc_start_pos":35,
    "pos_enc_type":"sincos",
    "seed":42,
    "add_peptide_conformer":True,
    "peptide_conformer_num_layers":2,
    "peptide_conformer_n_heads":16,
    "peptide_conformer_conv_kernel":5,
    "peptide_conformer_dropout":0.0,
    "add_mhc_conformer":False,
    "mhc_conformer_num_layers":2,
    "mhc_conformer_n_heads":16,
    "mhc_conformer_conv_kernel":5,
    "mhc_conformer_dropout":0.0,
    "add_fusion_conformer":True,
    "fusion_conformer_num_layers":1,
    "fusion_conformer_n_heads":16,
    "fusion_conformer_conv_kernel":7,
    "fusion_conformer_dropout":0.0,
    "fusion_conformer_mask_bos":True,
    "ff_pre_post_swiglu":False,
    "legacy_init_compat":True,
}

base_model_dir=MODEL_CHECKPOINTS_DIR / "MHCPrime_Base_for_Ablations" / "Checkpoints"
ckpt_path=base_model_dir / "model_final.pt"
device=torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")

clear_all_gpu_memory()
mhcprime=init_model(base_model_params,print_check=False,print_params=True)
state_dict=torch.load(ckpt_path,map_location="cpu")
mhcprime.load_state_dict(state_dict,strict=True)
mhcprime.to(device); mhcprime.eval()
print("Model loaded from checkpoint...")
print(f"Checkpoint: {ckpt_path}")
print(f"Model device: {next(mhcprime.parameters()).device}\n")

pepcl_multi_base_params=dict(
    tokenizer=tokenizer,
    device="cuda",
    steps=300,
    output_dir=None,
    save_every_n_steps=2000,
    seed=42,
    num_workers=num_workers,
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

# ============================================================
# SEED LOOP
# ============================================================
full_output_root.mkdir(parents=True,exist_ok=True)


# merge all update sets
merged_train_set_with_domain_ids = pd.concat(train_sets, names=["dataset_source"]).reset_index(level="dataset_source").reset_index(drop=True)

print_data_stats(merged_train_set_with_domain_ids)

# assign domains
domain_id_mapping={
    "yd": 1,
    "episcan": 2,
    "ba": 3,
    "escape": 4,
}

merged_train_set_with_domain_ids["domain_id"]=merged_train_set_with_domain_ids.dataset_source.map(domain_id_mapping)

print_data_stats(merged_train_set_with_domain_ids)

for pepcl_seed in seeds:
    print("Seed: ", pepcl_seed)

    model_save_root=MODEL_CHECKPOINTS_DIR / "T1_Allele_Partition_Sets" / f"Seed_{pepcl_seed}"
    model_save_root.mkdir(parents=True,exist_ok=True)

    # get allele-specific partitions for each dataset
    partition_counts={
        "yd": 4,
        "episcan": 4,
        "ba": 4,
        "escape": 4,
    }
    train_sets_allele_partitions, train_sets_allele_partitions_summary=partition_datasets_by_allele(train_sets, partition_counts, allele_col="allele", label_col="label", seed=pepcl_seed)

    save_dict_pickle(train_sets_allele_partitions, f"{full_output_root}/train_sets_allele_partitions_seed_{pepcl_seed}.pkl")
    save_dict_pickle(train_sets_allele_partitions_summary, f"{full_output_root}/train_sets_allele_partitions_summary_seed_{pepcl_seed}.pkl")

    """
    PepCL adaptive buffer

    NOTE: make sure the correct train set is used.
    """
    model_pepcl_multi_r_dr=copy.deepcopy(mhcprime)

    pepcl_multi_r_dr_models={}

    current_base_params=pepcl_multi_base_params.copy()
    current_base_params["seed"]=pepcl_seed

    task_order_name="t1_allele_partition"

    task_order=list(train_sets_allele_partitions.keys())
    print("Task order: ", task_order)

    temp_ms_train_data=ms_train_data.copy()
    temp_ms_train_data["domain_id"]=0
    temp_train_data=temp_ms_train_data.copy()

    # task_order=['yd_1']
    for domain_id,ft_train_set_name in enumerate(task_order,start=1):
        print(f"\n--- Stage s{domain_id}: {ft_train_set_name} ---")

        temp_ft_train_set_df=train_sets_allele_partitions[ft_train_set_name].copy()
        temp_ft_train_set_df["domain_id"]=domain_id
        temp_train_data=pd.concat([temp_train_data,temp_ft_train_set_df],ignore_index=True)
        temp_train_data.reset_index(inplace=True,drop=True)
        print_data_stats(temp_train_data)

        replay_domain_ids=list(range(domain_id))
        update_domain_id=domain_id
        print(f"Replay domains: {replay_domain_ids}")
        print(f"Update domain: {update_domain_id}")

        # =====================================================
        # PepCL Multi: equal replay representation, replay + DR
        # =====================================================
        print("PepCL Multi w/ Replay and DR")
        replay_dr_params=current_base_params.copy(); replay_dr_params["dr_lambda"]=1e-5
        pepcl=PepCL_Multi(
            model=model_pepcl_multi_r_dr,train_df=temp_train_data,use_replay=True,
            replay_domain_ids=replay_domain_ids,update_domain_id=update_domain_id,
            ms_replay_fraction="equal",**replay_dr_params,
        )
        training_stats=pepcl.train()


        pepcl_multi_r_dr_model_name=f"{task_order_name}_s{domain_id}_pepcl_multi_replay_dr_full_buffer"

        pepcl_multi_r_dr_models[pepcl_multi_r_dr_model_name]=copy.deepcopy(model_pepcl_multi_r_dr)

        print(pepcl_multi_r_dr_model_name,id(pepcl_multi_r_dr_models[pepcl_multi_r_dr_model_name]))

    for model_name, model in pepcl_multi_r_dr_models.items():
        print(model_name)
        torch.save(model.state_dict(), f"{model_save_root}/{model_name}.pt")

    """
    PepCL adaptive buffer; fixed 50% MS in replay

    NOTE: make sure the correct train set is used.
    """
    model_pepcl_multi_r_dr_50pct_ms=copy.deepcopy(mhcprime)

    pepcl_multi_r_dr_50pct_ms_models={}

    current_base_params=pepcl_multi_base_params.copy()
    current_base_params["seed"]=pepcl_seed

    task_order_name="t1_allele_partition"

    task_order=list(train_sets_allele_partitions.keys())
    print("Task order: ", task_order)

    temp_ms_train_data=ms_train_data.copy()
    temp_ms_train_data["domain_id"]=0
    temp_train_data=temp_ms_train_data.copy()

    # task_order=['yd_1']
    for domain_id, ft_train_set_name in enumerate(tqdm(task_order, desc="Tasks"), start=1):
        print(f"\n--- Stage s{domain_id}: {ft_train_set_name} ---")

        temp_ft_train_set_df=train_sets_allele_partitions[ft_train_set_name].copy()
        temp_ft_train_set_df["domain_id"]=domain_id
        temp_train_data=pd.concat([temp_train_data,temp_ft_train_set_df],ignore_index=True)
        temp_train_data.reset_index(inplace=True,drop=True)
        print_data_stats(temp_train_data)

        replay_domain_ids=list(range(domain_id))
        update_domain_id=domain_id
        print(f"Replay domains: {replay_domain_ids}")
        print(f"Update domain: {update_domain_id}")

        # =====================================================
        # PepCL Multi: fixed 50% MS replay, replay + DR
        # =====================================================
        print("PepCL Multi w/ Replay and DR, 50% MS replay")
        replay_dr_50pct_ms_params=current_base_params.copy(); replay_dr_50pct_ms_params["dr_lambda"]=1e-5
        pepcl=PepCL_Multi(
            model=model_pepcl_multi_r_dr_50pct_ms,train_df=temp_train_data,use_replay=True,
            replay_domain_ids=replay_domain_ids,update_domain_id=update_domain_id,
            ms_replay_fraction=0.5,**replay_dr_50pct_ms_params,
        )
        training_stats=pepcl.train()

        pepcl_multi_r_dr_50pct_ms_model_name=f"{task_order_name}_s{domain_id}_pepcl_multi_replay_dr_full_buffer_50pct_ms"

        pepcl_multi_r_dr_50pct_ms_models[pepcl_multi_r_dr_50pct_ms_model_name]=copy.deepcopy(model_pepcl_multi_r_dr_50pct_ms)

        print(pepcl_multi_r_dr_50pct_ms_model_name,id(pepcl_multi_r_dr_50pct_ms_models[pepcl_multi_r_dr_50pct_ms_model_name]))

    for model_name, model in pepcl_multi_r_dr_50pct_ms_models.items():
        print(model_name)
        torch.save(model.state_dict(), f"{model_save_root}/{model_name}.pt")