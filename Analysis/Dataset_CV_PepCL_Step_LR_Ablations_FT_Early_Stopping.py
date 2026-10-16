"""
FT early-stopping CV rerun script using PepCL_Logging_Ext_Early_Stopping.

Select DATASET_TO_RUN at the top. Loads the saved early-stopping config dictionary
selected from the 5-seed analysis, then reruns FT only on each requested CV split.
For each split, the script uses the exact split-specific train/validation construction
and active validation sets from the corresponding CV step/LR ablation script.

Model naming:
    <prefix>_ft_steps_<full_budget>_lr_<lr_string>_seed_<seed>_step_<stop>

Each CV split is saved in its own DATA_DIR output directory.

Run:
    python Dataset_CV_PepCL_Step_LR_Ablations_FT_Early_Stopping.py -g GPU_ID
"""

import sys, os, copy, warnings, argparse
from pathlib import Path
warnings.filterwarnings("ignore")

PROJECT_ROOT=Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0,str(PROJECT_ROOT))
DATA_DIR=PROJECT_ROOT/"Data"
SRC_DIR=PROJECT_ROOT/"src"
ANALYSIS_DIR=PROJECT_ROOT/"Analysis"

from Data.mhc_pseudo_seqs import mhc_seq_dict
from src.Model.mhcprime import *
from src.model_utils import *
from src.general_utils import *
from src.get_feature_data import *
from src.mhcprime_trainer import *
from src.mhcprime_inference import *
from src.pepcl import *
from src.pepcl_logging import *

# ============================================================================
# USER SETTINGS
# ============================================================================

# holec, episcan_r9, episcan_r9rt, ba_con, escape_con

# DATASET_TO_RUN="holec" # yd
# DATASET_TO_RUN="episcan_r9"
# DATASET_TO_RUN="episcan_r9rt"
# DATASET_TO_RUN="ba_con"
DATASET_TO_RUN="escape_con"


DATASET_SPLITS=[1,2,3,4]
CV_SEEDS=[0]  # CV ablation scripts use seed 0 per split; change only if intentionally rerunning more seeds.
SAVE_LOG_DATA=False
NUM_WORKERS=4

# Same directory used as ft_es_config_dir when the 5-seed early-stop configs were saved.
FT_ES_CONFIG_DIR=DATA_DIR/"FT_LR_Step_Ablations_Early_Stopping_Configs"

EARLY_STOP_CONFIG_FILES={
    "holec":"holec_early_stop_configs_final.pkl",
    "episcan_r9":"episcan_r9_early_stop_configs_final.pkl",
    "episcan_r9rt":"episcan_r9rt_early_stop_configs_final.pkl",
    "ba_con":"ba_con_early_stop_configs_final.pkl",
    "escape_con":"escape_con_early_stop_configs_final.pkl",
    "escape_con":"escape_con_cleaned_early_stop_configs_final.pkl",
}


OUTPUT_DIR_PATTERNS={
    "holec": "YD_FT_Early_Stopping_w_Alt_MHCPrime_CV_Datasplit_Final_{split}",
    "episcan_r9": "EpiScan_R9_FT_Early_Stopping_w_Alt_MHCPrime_CV_Datasplit_Final_{split}",
    "episcan_r9rt": "EpiScan_R9_RT_FT_Early_Stopping_w_Alt_MHCPrime_CV_Datasplit_Final_{split}",
    "ba_con": "BA_CON_FT_Early_Stopping_w_Alt_MHCPrime_CV_Datasplit_Final_{split}",
    "escape_con": "ESCAPE_seq_CON_FT_Early_Stopping_w_Alt_MHCPrime_CV_Datasplit_Cleaned_Final_{split}",
}


MODEL_PREFIXES={
    "holec":"holec",
    "episcan_r9":"episcan_r9",
    "episcan_r9rt":"episcan_r9rt",
    "ba_con":"ba_con",
    "escape_con":"escape_con",
}

if DATASET_TO_RUN not in EARLY_STOP_CONFIG_FILES:
    raise ValueError(f"DATASET_TO_RUN must be one of {list(EARLY_STOP_CONFIG_FILES)}, got '{DATASET_TO_RUN}'.")

# ============================================================================
# GPU
# ============================================================================
parser=argparse.ArgumentParser(description="Select specific GPU to run model on.")
parser.add_argument("-g","--gpu",type=int,default=0,help="Which GPU ID to use (e.g. 0, 1, 2...)")
args=parser.parse_args(); gpu_id=args.gpu
GPU_ID=gpu_id
torch.cuda.set_device(GPU_ID)
print(f"Using GPU: {torch.cuda.get_device_name(GPU_ID)} (ID: {torch.cuda.current_device()})")
device=torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

# ============================================================================
# HELPERS
# ============================================================================
def validate_early_stop_configs(configs):
    if not isinstance(configs,dict) or len(configs)==0: raise ValueError("Early-stopping config file is empty or not a dict.")
    seen=set()
    for i,cfg in configs.items():
        if not isinstance(cfg,dict): raise TypeError(f"Config {i} is not a dict: {cfg}")
        missing=[k for k in ("lr","steps","stop") if k not in cfg]
        if missing: raise ValueError(f"Config {i} missing keys {missing}: {cfg}")
        lr=float(cfg["lr"]); steps=int(cfg["steps"]); stop=int(cfg["stop"])
        if lr<=0 or steps<=0 or stop<=0 or stop>steps: raise ValueError(f"Invalid config {i}: {cfg}")
        key=(lr,steps,stop)
        if key in seen: raise ValueError(f"Duplicate early-stopping configuration found: {key}")
        seen.add(key)
    return configs

def run_inference_into_sets(model,sets_dict,cache_paths,device,num_workers):
    model=model.to(device); model.eval()
    for set_name,data_set in sets_dict.items():
        path=cache_paths[set_name]; print(set_name,path)
        with torch.no_grad():
            scores=run_mhcprime_fast(model,path,batch_size=3072,num_workers=num_workers,compile_model=False,use_domain_id=False)
        yield set_name,scores.detach().cpu().tolist()

def save_sets(sets_dict,output_dir,label):
    print(f"Saving {label} sets...")
    for set_name,data_set in sets_dict.items(): print(set_name); data_set.to_csv(f"{output_dir}/{set_name}.csv")
    print(end="\n")

def build_base_model(tokenizer,processed_feature_table,feature_names,device):
    base_model_params={
        'vocab_size':tokenizer.vocab_size,'embed_dim':480,'fc_hidden_dims':[240,120],'fc_dropout':0.1,'n_heads':16,
        'max_peptide_len':34,'max_mhc_len':34,'n_peptide_layers':2,'n_mhc_layers':1,'n_concat_layers':3,
        'add_pos_enc':True,'use_segment_embedder':True,'dim_feedforward_multiplier':1,'debugging':False,'fc_output_dim':1,
        'pooling':'cls','ffnn_use_layernorm':False,'use_feature_table':True,'feature_table':processed_feature_table,
        'sequence_embedder_feature_names':feature_names,'sequence_embedder_feature_mode':'concat',
        'sequence_embedder_tokenizer':tokenizer,'peptide_start_pos':1,'mhc_start_pos':35,'pos_enc_type':'sincos','seed':42,
        'add_peptide_conformer':True,'peptide_conformer_num_layers':2,'peptide_conformer_n_heads':16,
        'peptide_conformer_conv_kernel':5,'peptide_conformer_dropout':0.0,'add_mhc_conformer':False,
        'mhc_conformer_num_layers':2,'mhc_conformer_n_heads':16,'mhc_conformer_conv_kernel':5,'mhc_conformer_dropout':0.0,
        'add_fusion_conformer':True,'fusion_conformer_num_layers':1,'fusion_conformer_n_heads':16,
        'fusion_conformer_conv_kernel':7,'fusion_conformer_dropout':0.0,'fusion_conformer_mask_bos':True,
        'ff_pre_post_swiglu':False,'legacy_init_compat':True,
    }
    ckpt_path=PROJECT_ROOT/"Model_Checkpoints"/"MHCPrime_Base_for_Ablations"/"Checkpoints"/"model_final.pt"
    clear_all_gpu_memory(); model=init_model(base_model_params,print_check=False,print_params=True)
    state_dict=torch.load(ckpt_path,map_location="cpu"); model.load_state_dict(state_dict,strict=True); model.to(device); model.eval()
    print("Model loaded from checkpoint..."); print(f"Checkpoint: {ckpt_path}"); print(f"Model device: {next(model.parameters()).device}\n")
    return model

# ============================================================================
# FEATURES / PSEUDOSEQUENCES -- same setup as CV step/LR scripts
# ============================================================================
mhc_seq_dict={k.removeprefix("HLA-").replace("*","").replace(":",""):v for k,v in mhc_seq_dict.items()}
print("Loading feature data...")
aa_features_table_ext,descriptor_groups=load_feature_data()
processed_feature_table,corr_info=process_aa_feature_table(aa_features_table_ext)
feature_names=list(processed_feature_table.columns)
print(end="\n")

tokenizer=AminoAcidTokenizer()

# ============================================================================
# LOAD EARLY-STOPPING CONFIGS SELECTED FROM THE 5-SEED RUNS
# ============================================================================
config_path=FT_ES_CONFIG_DIR/EARLY_STOP_CONFIG_FILES[DATASET_TO_RUN]
print(f"Loading early-stopping configs: {config_path}")
early_stop_configs=validate_early_stop_configs(load_dict_pickle(config_path))
print(f"Loaded {len(early_stop_configs)} true early-stopping configurations:")
for i,cfg in early_stop_configs.items(): print(f"  {i}: lr={cfg['lr']}, steps={cfg['steps']}, stop={cfg['stop']}")
print()

# ============================================================================
# LOAD DATASET-SPECIFIC CV SPLIT DICTIONARIES ONCE
# ============================================================================
if DATASET_TO_RUN=="holec":
    train_val_sets=load_dict_pickle(f"{DATA_DIR}/Training_Datasets/Processed/Holec_et_al_2025/holec_train_val_sets.pkl")
elif DATASET_TO_RUN in {"episcan_r9","episcan_r9rt"}:
    episcan_r9_train_val_sets=load_dict_pickle(f"{DATA_DIR}/Training_Datasets/Processed/EpiScan/episcan_r9_train_val_sets.pkl")
    episcan_rt_train_val_sets=load_dict_pickle(f"{DATA_DIR}/Training_Datasets/Processed/EpiScan/episcan_rt_train_val_sets.pkl")
elif DATASET_TO_RUN=="ba_con":
    train_val_sets=load_dict_pickle(f"{DATA_DIR}/Training_Datasets/Processed/BA/ba_con_train_val_sets.pkl")
elif DATASET_TO_RUN=="escape_con":
    # train_val_sets=load_dict_pickle(f"{DATA_DIR}/Training_Datasets/Processed/ESCAPE_seq/escape_con_train_val_sets.pkl")
    train_val_sets=load_dict_pickle(f"{DATA_DIR}/Training_Datasets/Processed/ESCAPE_seq/escape_con_train_val_sets_cleaned.pkl")

# ============================================================================
# CV SPLITS
# ============================================================================
for ds_split in DATASET_SPLITS:
    print("\n"+"#"*100)
    print(f"Dataset: {DATASET_TO_RUN} | CV Dataset Split: {ds_split}")
    print("#"*100+"\n")

    # ------------------------------------------------------------------------
    # Common MS data -- reloaded per split exactly as in CV source scripts.
    # ------------------------------------------------------------------------
    print("Loading MS data...")
    ms_train_test_dir=f"{DATA_DIR}/Training_Datasets/Processed/MS_Train_Test_Sets"
    ms_train_data=pd.read_csv(f"{ms_train_test_dir}/ms_train_data_remaining_minimal.csv",index_col=0)
    ms_val_data=pd.read_csv(f"{ms_train_test_dir}/ms_val_data.csv",index_col=0)

    # ========================================================================
    # DATASET-SPECIFIC CV DATA PREPARATION
    # Mirrors each corresponding CV step/LR script.
    # ========================================================================
    if DATASET_TO_RUN=="holec":
        print("Loading YD data...")
        d=f"{DATA_DIR}/Training_Datasets/Processed/Holec_et_al_2025"
        train_val_dict=train_val_sets[ds_split]
        holec_train_set_remaining=train_val_dict["train"]
        holec_val_set=train_val_dict["val"]

        print("Merging MS/YD data...")
        ms_train_data["domain_id"]=0; holec_train_set_remaining["domain_id"]=1
        ms_holec_train_set=pd.concat([ms_train_data,holec_train_set_remaining]); ms_holec_train_set.reset_index(inplace=True,drop=True)
        print_data_stats(ms_holec_train_set)
        ms_holec_train_set["seq_len"]=ms_holec_train_set.seq.str.replace("$","").apply(len)
        ms_holec_train_set_matched_allele_length=ms_holec_train_set[ms_holec_train_set.allele.isin(holec_train_set_remaining.allele.unique().tolist())].query("seq_len == 9")
        ms_holec_train_set_matched_allele_length.reset_index(inplace=True,drop=True); print_data_stats(ms_holec_train_set_matched_allele_length)

        print("Matching allele- and length distributions for MS/YD data...")
        ms_val_data_matched_allele_length=ms_val_data[ms_val_data.allele.isin(holec_val_set.allele.unique().tolist())].query("seq_len == 9")

        ms_val_data_matched_allele_length=create_seq_allele_column(full_process_data(ms_val_data_matched_allele_length,mhc_seq_dict,remove_flank=True,add_pseudo_label=False))
        ms_val_data=create_seq_allele_column(full_process_data(ms_val_data,mhc_seq_dict,remove_flank=True,add_pseudo_label=False))
        holec_val_set=create_seq_allele_column(full_process_data(holec_val_set,mhc_seq_dict,remove_flank=True,add_pseudo_label=False))

        filtered_columns=['seq','allele','label','seq_allele','n_flank','c_flank','mhc_a_1','mhc_b_1','mhc_c_1','mhc_a_2','mhc_b_2','mhc_c_2','sa_ma','mhcprime','group']
        ms_val_data_matched_allele_length=filter_single_df_to_cols(ms_val_data_matched_allele_length,filtered_columns)

        val_sets_run_once={"ms_val":ms_val_data_matched_allele_length}
        val_sets={"holec_val":holec_val_set}
        ft_train_df=ms_holec_train_set_matched_allele_length.query("domain_id == 1")
        ft_loss_type="logsmoothap"; ft_loss_alpha=10.0; continuous_target_col="ba_value"; continuous_sampling_mode="allele_aware"

    elif DATASET_TO_RUN in {"episcan_r9","episcan_r9rt"}:
        print("Loading EpiScan data...")
        d=f"{DATA_DIR}/Training_Datasets/Processed/EpiScan"
        episcan_r9_train_set_remaining=episcan_r9_train_val_sets[ds_split]["train"]
        episcan_rt_train_set_remaining=episcan_rt_train_val_sets[ds_split]["train"]
        episcan_r9_val_set=episcan_r9_train_val_sets[ds_split]["val"]
        episcan_rt_val_set=episcan_rt_train_val_sets[ds_split]["val"]

        print("Merging MS/FT data...")
        ms_train_data["domain_id"]=0; episcan_r9_train_set_remaining["domain_id"]=1; episcan_rt_train_set_remaining["domain_id"]=1
        ms_r9rt_pos_r9_neg_train_set=pd.concat([ms_train_data,episcan_r9_train_set_remaining,episcan_rt_train_set_remaining.query("label == 1")])
        ms_r9rt_pos_r9rt_neg_train_set=pd.concat([ms_train_data,episcan_r9_train_set_remaining,episcan_rt_train_set_remaining])
        ms_r9rt_pos_r9_neg_train_set.reset_index(inplace=True,drop=True); ms_r9rt_pos_r9rt_neg_train_set.reset_index(inplace=True,drop=True)
        print_data_stats(ms_r9rt_pos_r9_neg_train_set); print_data_stats(ms_r9rt_pos_r9rt_neg_train_set)

        ms_r9rt_pos_r9_neg_train_set["seq_len"]=ms_r9rt_pos_r9_neg_train_set.seq.apply(len)
        ms_r9rt_pos_r9rt_neg_train_set["seq_len"]=ms_r9rt_pos_r9rt_neg_train_set.seq.apply(len)
        iso_alleles=['A0201','A0301','B0801','B5701']
        ms_r9rt_pos_r9_neg_train_set_matched_allele_length=ms_r9rt_pos_r9_neg_train_set[ms_r9rt_pos_r9_neg_train_set.allele.isin(iso_alleles)].query("seq_len == 9")
        ms_r9rt_pos_r9rt_neg_train_set_matched_allele_length=ms_r9rt_pos_r9rt_neg_train_set[ms_r9rt_pos_r9rt_neg_train_set.allele.isin(iso_alleles)].query("seq_len == 9")
        ms_r9rt_pos_r9_neg_train_set_matched_allele_length.reset_index(inplace=True,drop=True)
        ms_r9rt_pos_r9rt_neg_train_set_matched_allele_length.reset_index(inplace=True,drop=True)
        print_data_stats(ms_r9rt_pos_r9_neg_train_set_matched_allele_length); print_data_stats(ms_r9rt_pos_r9rt_neg_train_set_matched_allele_length)

        print("Merging R9/RT val sets...")
        episcan_r9_val_set["dataset_source"]="r9"; episcan_rt_val_set["dataset_source"]="rt"
        episcan_r9_rt_val_sets=pd.concat([episcan_r9_val_set,episcan_rt_val_set])

        print("Matching allele- and length distributions for MS/FT data...")
        ms_val_data_matched_allele_length=ms_val_data[ms_val_data.allele.isin(iso_alleles)].query("seq_len == 9")

        ms_val_data_matched_allele_length=create_seq_allele_column(full_process_data(ms_val_data_matched_allele_length,mhc_seq_dict,remove_flank=True,add_pseudo_label=False))
        ms_val_data=create_seq_allele_column(full_process_data(ms_val_data,mhc_seq_dict,remove_flank=True,add_pseudo_label=False))
        episcan_r9_rt_val_sets=create_seq_allele_column(full_process_data(episcan_r9_rt_val_sets,mhc_seq_dict,remove_flank=True,add_pseudo_label=False))

        filtered_columns=['seq','allele','label','seq_allele','n_flank','c_flank','mhc_a_1','mhc_b_1','mhc_c_1','mhc_a_2','mhc_b_2','mhc_c_2','sa_ma','mhcprime','dataset_source']
        ms_val_data_matched_allele_length=filter_single_df_to_cols(ms_val_data_matched_allele_length,filtered_columns)
        episcan_r9_rt_val_sets=filter_single_df_to_cols(episcan_r9_rt_val_sets,filtered_columns)

        val_sets_run_once={"ms_val":ms_val_data_matched_allele_length}
        val_sets={"episcan_r9_rt_val":episcan_r9_rt_val_sets}
        if DATASET_TO_RUN=="episcan_r9":
            ft_train_df=ms_r9rt_pos_r9_neg_train_set_matched_allele_length.query("domain_id == 1")
        else:
            ft_train_df=ms_r9rt_pos_r9rt_neg_train_set_matched_allele_length.query("domain_id == 1")
        ft_loss_type="logsmoothap"; ft_loss_alpha=10.0; continuous_target_col="ba_value"; continuous_sampling_mode="allele_aware"

    elif DATASET_TO_RUN=="ba_con":
        print("Loading BA data...")
        train_val_dict=train_val_sets[ds_split]
        d=DATA_DIR/"Training_Datasets"/"Processed"/"BA"
        train_df_ba_for_updating_remaining=train_val_dict["train"]
        ba_val_set=train_val_dict["val"]

        print("Merging MS/FT data...")
        ms_train_data["domain_id"]=0; train_df_ba_for_updating_remaining["domain_id"]=1
        ms_ba_train_set=pd.concat([ms_train_data,train_df_ba_for_updating_remaining]); ms_ba_train_set.reset_index(inplace=True,drop=True)
        print_data_stats(ms_ba_train_set)
        ms_ba_train_set["seq_len"]=ms_ba_train_set.seq.str.replace("$","").apply(len)
        ms_ba_train_set_matched_allele_length=filter_domain0_to_domain1_pairs(ms_ba_train_set,allele_col='allele',seq_len_col='seq_len',domain_col='domain_id')
        ms_ba_train_set_matched_allele_length.reset_index(inplace=True,drop=True); print_data_stats(ms_ba_train_set_matched_allele_length)

        print("Matching allele- and length distributions for MS/FT data...")
        ms_val_data_matched_allele_length=filter_to_shared_pairs(ms_val_data,ba_val_set,allele_col='allele',seq_len_col='seq_len')

        ms_val_data_matched_allele_length=create_seq_allele_column(full_process_data(ms_val_data_matched_allele_length,mhc_seq_dict,remove_flank=True,add_pseudo_label=False))
        ba_val_set=create_seq_allele_column(full_process_data(ba_val_set,mhc_seq_dict,remove_flank=True,add_pseudo_label=False))
        ms_val_data=create_seq_allele_column(full_process_data(ms_val_data,mhc_seq_dict,remove_flank=True,add_pseudo_label=False))

        filtered_columns=['seq','allele','label','seq_allele','n_flank','c_flank','mhc_a_1','mhc_b_1','mhc_c_1','mhc_a_2','mhc_b_2','mhc_c_2','sa_ma','mhcprime','ba_value']
        ms_val_data_matched_allele_length=filter_single_df_to_cols(ms_val_data_matched_allele_length,filtered_columns)

        val_sets_run_once={"ms_val":ms_val_data_matched_allele_length}
        val_sets={"ba_val":ba_val_set}
        ft_train_df=ms_ba_train_set_matched_allele_length.query("domain_id == 1")
        ft_loss_type="spearman"; ft_loss_alpha=0.01; continuous_target_col="ba_value"; continuous_sampling_mode="allele_aware"

    elif DATASET_TO_RUN=="escape_con":
        print("Loading ESCAPE-seq data...")
        train_val_dict=train_val_sets[ds_split]
        d=f"{DATA_DIR}/Training_Datasets/Processed/ESCAPE_seq"
        escape_train_remaining=train_val_dict["train"]
        escape_val=train_val_dict["val"]

        print("Merging MS/FT data...")
        ms_train_data["domain_id"]=0; escape_train_remaining["domain_id"]=1
        ms_escape_train_set=pd.concat([ms_train_data,escape_train_remaining]); ms_escape_train_set.reset_index(inplace=True,drop=True)
        print_data_stats(ms_escape_train_set)
        ms_escape_train_set["seq_len"]=ms_escape_train_set.seq.str.replace("$","").apply(len)
        ms_escape_train_set_matched_allele_length=filter_domain0_to_domain1_pairs(ms_escape_train_set,allele_col='allele',seq_len_col='seq_len',domain_col='domain_id')
        ms_escape_train_set_matched_allele_length.reset_index(inplace=True,drop=True); print_data_stats(ms_escape_train_set_matched_allele_length)

        print("Matching allele- and length distributions for MS/FT data...")
        ms_val_data_matched_allele_length=filter_to_shared_pairs(ms_val_data,escape_val,allele_col='allele',seq_len_col='seq_len')

        ms_val_data_matched_allele_length=create_seq_allele_column(full_process_data(ms_val_data_matched_allele_length,mhc_seq_dict,remove_flank=True,add_pseudo_label=False))
        escape_val=create_seq_allele_column(full_process_data(escape_val,mhc_seq_dict,remove_flank=True,add_pseudo_label=False))
        ms_val_data=create_seq_allele_column(full_process_data(ms_val_data,mhc_seq_dict,remove_flank=True,add_pseudo_label=False))

        filtered_columns=['seq','allele','label','seq_allele','n_flank','c_flank','mhc_a_1','mhc_b_1','mhc_c_1','mhc_a_2','mhc_b_2','mhc_c_2','sa_ma','mhcprime','escore','ba_value','name2','name','gene','mutation','mut_type']
        ms_val_data_matched_allele_length=filter_single_df_to_cols(ms_val_data_matched_allele_length,filtered_columns)

        val_sets_run_once={"ms_val":ms_val_data_matched_allele_length}
        val_sets={"escape_val":escape_val}
        ft_train_df=ms_escape_train_set_matched_allele_length.query("domain_id == 1")
        ft_loss_type="spearman"; ft_loss_alpha=0.01; continuous_target_col="ba_value"; continuous_sampling_mode="allele_aware"

    # ========================================================================
    # BASE MHCPrime -- fresh model per split, same checkpoint as CV scripts
    # ========================================================================
    print("Load MHCPrime for this CV split...")
    mhcprime=build_base_model(tokenizer,processed_feature_table,feature_names,device)

    # ========================================================================
    # ========================================================================
    print("Creating validation set caches for inference...")
    val_sets_run_once_cache_paths={}
    for val_set_name_run_once,val_data_set_run_once in val_sets_run_once.items():
        print(val_set_name_run_once); p=f"{DATA_DIR}/{val_set_name_run_once}_{gpu_id}_cache.pt"
        build_token_cache(val_data_set_run_once,p,tokenizer,use_domain_id=False); val_sets_run_once_cache_paths[val_set_name_run_once]=p
    val_set_cache_paths={}
    for val_set_name,val_data_set in val_sets.items():
        print(val_set_name); p=f"{DATA_DIR}/{val_set_name}_{gpu_id}_cache.pt"
        build_token_cache(val_data_set,p,tokenizer,use_domain_id=False); val_set_cache_paths[val_set_name]=p
    print()

    # ========================================================================
    # OUTPUT DIRECTORY -- separate directory for each CV split, under DATA_DIR
    # ========================================================================
    full_output_dir=DATA_DIR/OUTPUT_DIR_PATTERNS[DATASET_TO_RUN].format(split=ds_split)
    full_log_output_dir=full_output_dir/"logs"
    if os.path.exists(full_output_dir):
        response=input(f"\nWARNING: Output directory already exists:\n{full_output_dir}\nContinuing may overwrite existing outputs.\nContinue? [y/n]: ").strip().lower()
        if response not in {"y","yes"}: raise SystemExit("Stopped by user. Existing output directory was not modified.")
        print("Continuing with existing output directory. Existing files may be overwritten.\n")
    print("Output directory for everything:",full_output_dir); os.makedirs(full_output_dir,exist_ok=True)
    if SAVE_LOG_DATA: print("Output directory for logs:",full_log_output_dir); os.makedirs(full_log_output_dir,exist_ok=True)

    # ========================================================================
    # BASE MHCPrime INFERENCE ONCE FOR THIS SPLIT
    # ========================================================================
    print("Run base MHCPrime inference once on all validation sets...")
    base_model=copy.deepcopy(mhcprime)
    for set_name,scores in run_inference_into_sets(base_model,val_sets_run_once,val_sets_run_once_cache_paths,device,NUM_WORKERS): val_sets_run_once[set_name]["mhcprime"]=scores
    for set_name,scores in run_inference_into_sets(base_model,val_sets,val_set_cache_paths,device,NUM_WORKERS): val_sets[set_name]["mhcprime"]=scores
    print(); save_sets(val_sets_run_once,full_output_dir,"initial val run once"); save_sets(val_sets,full_output_dir,"initial val")
    del base_model; clear_all_gpu_memory()

    # ========================================================================
    # BASE FT PARAMETERS -- same loss/sampling settings as corresponding CV script
    # ========================================================================
    pepcl_base_params=dict(
        tokenizer=tokenizer,
        device=("cuda" if DATASET_TO_RUN in {"ba_con","escape_con"} else device),
        steps=200,
        output_dir=None,
        save_every_n_steps=None,
        seed=42,
        num_workers=NUM_WORKERS,
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
        nonms_loss_type=ft_loss_type,
        nonms_loss_alpha=ft_loss_alpha,
        ft_lambda=1.0,
        dr_lambda=0.0,
        dr_loss_type="mse",
    )
    if DATASET_TO_RUN in {"ba_con","escape_con"}:
        pepcl_base_params["continuous_target_col"]=continuous_target_col
        pepcl_base_params["continuous_ft_sampling_mode"]=continuous_sampling_mode

    # ========================================================================
    # RUN EARLY-STOPPING FT CONFIGS ON THIS CV SPLIT
    # ========================================================================
    dataset_prefix=MODEL_PREFIXES[DATASET_TO_RUN]
    print(f"Running FT early stopping for dataset={DATASET_TO_RUN}, split={ds_split}, seeds={CV_SEEDS}\n")

    for config_idx in sorted(early_stop_configs):
        cfg=early_stop_configs[config_idx]
        learning_rate=float(cfg["lr"]); steps=int(cfg["steps"]); stop=int(cfg["stop"])
        learning_rate_string=lr_to_string(learning_rate)

        for pepcl_seed in CV_SEEDS:
            print("="*90)
            print(f"Dataset: {DATASET_TO_RUN} | Split: {ds_split} | Config: {config_idx} | Seed: {pepcl_seed}")
            print(f"Full step budget: {steps} | Early stop: {stop} | LR: {learning_rate} ({learning_rate_string})")

            params=copy.deepcopy(pepcl_base_params)
            params["seed"]=pepcl_seed; params["steps"]=steps
            params["encoder_lr"]=learning_rate; params["decoder_lr"]=learning_rate
            params["save_every_n_steps"]=None; params["dr_lambda"]=0.0

            model_ft=copy.deepcopy(mhcprime)
            print("FT with fixed-step early stopping")
            pepcl=PepCL_Logging_Ext_Early_Stopping(
                model=model_ft,
                train_df=ft_train_df,
                use_replay=False,
                mhc_seq_dict=mhc_seq_dict,
                enable_logging=True,
                early_stopping_step=stop,
                **params,
            )
            training_stats_ft=pepcl.train()

            model_name=f"{dataset_prefix}_ft_steps_{steps}_lr_{learning_rate_string}_seed_{pepcl_seed}_step_{stop}"
            print(f"Model name: {model_name}")

            if SAVE_LOG_DATA:
                ft_logs=pepcl.get_logs(as_dataframes=True); ft_logs["training_stats"]=training_stats_ft
                print("Saving logs"); save_log_dict(ft_logs,f"{full_log_output_dir}/{model_name}.pkl")

            for set_name,scores in run_inference_into_sets(model_ft,val_sets_run_once,val_sets_run_once_cache_paths,device,NUM_WORKERS): val_sets_run_once[set_name][model_name]=scores
            for set_name,scores in run_inference_into_sets(model_ft,val_sets,val_set_cache_paths,device,NUM_WORKERS): val_sets[set_name][model_name]=scores

            save_sets(val_sets_run_once,full_output_dir,"val run once"); save_sets(val_sets,full_output_dir,"val")
            del pepcl,model_ft,training_stats_ft
            if SAVE_LOG_DATA: del ft_logs
            clear_all_gpu_memory()

    # ========================================================================
    # SPLIT CLEANUP
    # ========================================================================
    print("Removing inference caches for this split...")
    for _,p in val_sets_run_once_cache_paths.items():
        if os.path.exists(p): os.remove(p)
    for _,p in val_set_cache_paths.items():
        if os.path.exists(p): os.remove(p)

    del mhcprime,ft_train_df,val_sets_run_once,val_sets,val_sets_run_once_cache_paths,val_set_cache_paths
    clear_all_gpu_memory()
    print(f"Finished dataset split {ds_split}.\n")

print("All requested CV splits complete.")
