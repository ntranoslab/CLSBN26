"""
Updates without allele-aware sampling.

Run cmd: python Dataset_CV_PepCL_Allele_Aware_Ablations_Script.py -g gpu_id
"""
import sys
from pathlib import Path
import shutil

PROJECT_ROOT=Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0,str(PROJECT_ROOT))
DATA_DIR=PROJECT_ROOT / "Data"
SRC_DIR=PROJECT_ROOT / "src"
ANALYSIS_DIR=PROJECT_ROOT / "Analysis"

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
parser=argparse.ArgumentParser(description="Select specific GPU to run model on.")
parser.add_argument("-g","--gpu",type=int,default=0,help="Which GPU ID to use (e.g. 0, 1, 2...)")
args=parser.parse_args(); gpu_id=args.gpu
GPU_ID=gpu_id
torch.cuda.set_device(GPU_ID)
print(f"Using GPU: {torch.cuda.get_device_name(GPU_ID)} (ID: {torch.cuda.current_device()})")
device=f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"

mhc_seq_dict={k.removeprefix("HLA-").replace("*","").replace(":",""):v for k,v in mhc_seq_dict.items()}
print("Loading feature data...")
aa_features_table_ext,descriptor_groups=load_feature_data()
processed_feature_table,corr_info=process_aa_feature_table(aa_features_table_ext)
feature_names=list(processed_feature_table.columns)
print(end="\n")

# ============================================================
# CUSTOMIZE RUN
# ============================================================
valid_datasets_to_path_prefix={"yd":"YD","episcan_r9":"EpiScan_R9","episcan_r9rt":"EpiScan_R9_RT","ba_con":"BA_CON","escape_con":"ESCAPE_seq_CON"}
valid_datasets=set(valid_datasets_to_path_prefix)

default_lr_steps={"lr":2e-4,"steps":300}
default_lr=default_lr_steps["lr"]
default_steps=default_lr_steps["steps"]


# pepcl and dr
run_pepcl_r_dr=True
# run_pepcl_r_dr=False

 # dr only
run_pepcl_dr=True
# run_pepcl_dr=False

# "yd", "episcan_r9", "episcan_r9rt", "ba_con", "escape_con"
# dataset_to_run="yd" 
# dataset_to_run="episcan_r9"
# dataset_to_run="episcan_r9rt"
# dataset_to_run="ba_con"
dataset_to_run="escape_con"

# Final path is: {path_prefix}_{path_suffix}_CV_Datasplit_{ds_split}/
# path_suffix="Allele_Aware_Ablations_w_Alt_MHCPrime"
path_suffix="Allele_Aware_Ablations_w_Alt_MHCPrime_Cleaned"


# CV split IDs and fixed training seed(s). CV scripts use one model seed while varying the data split.
dataset_splits=[1,2,3,4]
seeds=[0]
dr_lambda_list=[0, 1e-5]
num_workers=4

if dataset_to_run not in valid_datasets: raise ValueError(f"dataset_to_run must be one of {sorted(valid_datasets)}, got '{dataset_to_run}'.")
path_prefix=valid_datasets_to_path_prefix[dataset_to_run]
print(f"Dataset: {dataset_to_run}")
print(f"Path suffix: {path_suffix}")
print(f"CV splits: {dataset_splits}")
print(f"Seeds: {seeds}")
print(f"Lambda values: {dr_lambda_list}")

# ============================================================
# LOAD CV SPLITS FOR SELECTED DATASET
# ============================================================
if dataset_to_run=="yd":
    split_data=load_dict_pickle(f"{DATA_DIR}/Training_Datasets/Processed/Holec_et_al_2025/holec_train_val_sets.pkl")
elif dataset_to_run in {"episcan_r9","episcan_r9rt"}:
    r9_split_data=load_dict_pickle(f"{DATA_DIR}/Training_Datasets/Processed/EpiScan/episcan_r9_train_val_sets.pkl")
    rt_split_data=load_dict_pickle(f"{DATA_DIR}/Training_Datasets/Processed/EpiScan/episcan_rt_train_val_sets.pkl")
elif dataset_to_run=="ba_con":
    split_data=load_dict_pickle(f"{DATA_DIR}/Training_Datasets/Processed/BA/ba_con_train_val_sets.pkl")
else:
    # split_data=load_dict_pickle(f"{DATA_DIR}/Training_Datasets/Processed/ESCAPE_seq/escape_con_train_val_sets.pkl")
    split_data=load_dict_pickle(f"{DATA_DIR}/Training_Datasets/Processed/ESCAPE_seq/escape_con_train_val_sets_cleaned.pkl")
    

# ============================================================
# LOAD BASE MODEL ONCE
# ============================================================
print("Load MHCPrime for ablations...")
tokenizer=AminoAcidTokenizer()
base_model_params={
    'vocab_size':tokenizer.vocab_size,'embed_dim':480,'fc_hidden_dims':[240,120],'fc_dropout':0.1,'n_heads':16,
    'max_peptide_len':34,'max_mhc_len':34,"n_peptide_layers":2,"n_mhc_layers":1,'n_concat_layers':3,'add_pos_enc':True,
    'use_segment_embedder':True,'dim_feedforward_multiplier':1,'debugging':False,'fc_output_dim':1,'pooling':'cls',
    "ffnn_use_layernorm":False,'use_feature_table':True,"feature_table":processed_feature_table,
    "sequence_embedder_feature_names":feature_names,'sequence_embedder_feature_mode':'concat','sequence_embedder_tokenizer':tokenizer,
    "peptide_start_pos":1,"mhc_start_pos":35,"pos_enc_type":"sincos","seed":42,
    "add_peptide_conformer":True,"peptide_conformer_num_layers":2,"peptide_conformer_n_heads":16,
    "peptide_conformer_conv_kernel":5,"peptide_conformer_dropout":0.0,
    "add_mhc_conformer":False,"mhc_conformer_num_layers":2,"mhc_conformer_n_heads":16,"mhc_conformer_conv_kernel":5,
    "mhc_conformer_dropout":0.0,"add_fusion_conformer":True,"fusion_conformer_num_layers":1,"fusion_conformer_n_heads":16,
    "fusion_conformer_conv_kernel":7,"fusion_conformer_dropout":0.0,"fusion_conformer_mask_bos":True,"ff_pre_post_swiglu":False,
    "legacy_init_compat":True,
}
MODEL_CHECKPOINTS_DIR=PROJECT_ROOT / "Model_Checkpoints"
base_model_dir=MODEL_CHECKPOINTS_DIR / "MHCPrime_Base_for_Ablations" / "Checkpoints"
ckpt_path=base_model_dir / "model_final.pt"
device=torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")
clear_all_gpu_memory()
mhcprime=init_model(base_model_params,print_check=False,print_params=True)
state_dict=torch.load(ckpt_path,map_location="cpu")
mhcprime.load_state_dict(state_dict,strict=True); mhcprime.to(device); mhcprime.eval()
print("Model loaded from checkpoint...")
print(f"Checkpoint: {ckpt_path}")
print(f"Model device: {next(mhcprime.parameters()).device}\n")

# ============================================================
# BASE PEPCL PARAMS
# ============================================================
pepcl_base_params_cls=dict(
    tokenizer=tokenizer,device=device,steps=200,output_dir=None,seed=42,num_workers=16,collate_fn=collate_fn,
    num_ms_pos=2000,num_nonms_pos=2000,ms_neg_ratio=1.0,nonms_neg_ratio=1.0,auto_calibrate=True,calibration_steps=1,
    ms_batch_size=3072,nonms_batch_size=3072,encoder_lr=2e-4,decoder_lr=2e-4,weight_decay=0.01,warmup_ratio=0.1,
    update_domain_embeddings=False,primary_task_id=1,sequential_backprop=True,ms_loss_type="logsmoothap",ms_loss_alpha=10.0,
    ms_lambda=1.0,nonms_loss_type="logsmoothap",nonms_loss_alpha=10.0,ft_lambda=1.0,dr_lambda=1e-5,dr_loss_type="mse", allele_aware_sampling=False
)
pepcl_base_params_con=dict(
    tokenizer=tokenizer,device=device,steps=200,output_dir=None,save_every_n_steps=2000,seed=42,num_workers=16,collate_fn=collate_fn,
    num_ms_pos=2000,num_nonms_pos=2000,ms_neg_ratio=1.0,nonms_neg_ratio=1.0,auto_calibrate=True,calibration_steps=1,
    ms_batch_size=3072,nonms_batch_size=3072,encoder_lr=2e-4,decoder_lr=2e-4,weight_decay=0.01,warmup_ratio=0.1,
    update_domain_embeddings=False,primary_task_id=1,sequential_backprop=True,ms_loss_type="logsmoothap",ms_loss_alpha=10.0,
    ms_lambda=1.0,nonms_loss_type="spearman",nonms_loss_alpha=0.01,ft_lambda=1.0,dr_lambda=1e-5,dr_loss_type="mse",
    continuous_target_col="ba_value",continuous_ft_sampling_mode="pooled",allele_aware_sampling=False,
)
pepcl_base_params=(pepcl_base_params_con if dataset_to_run in {"ba_con","escape_con"} else pepcl_base_params_cls).copy()
pepcl_base_params["steps"]=default_steps
pepcl_base_params["encoder_lr"]=default_lr
pepcl_base_params["decoder_lr"]=default_lr
pepcl_base_params["num_workers"]=num_workers
print("Using", "CON" if dataset_to_run in {"ba_con","escape_con"} else "CLS", "PepCL parameters")
print("LR and steps being used:",pepcl_base_params["encoder_lr"],pepcl_base_params["decoder_lr"],pepcl_base_params["steps"])

# ============================================================
# CV SPLIT LOOP
# ============================================================
for ds_split in dataset_splits:
    print(f"\n{'='*90}\nDATASET: {dataset_to_run} | CV DATA SPLIT: {ds_split}\n{'='*90}\n")

    # --------------------------------------------------------
    # COMMON MS DATA -- reloaded each split to avoid mutation carry-over
    # --------------------------------------------------------
    print("Loading MS data...")
    ms_train_test_dir=f"{DATA_DIR}/Training_Datasets/Processed/MS_Train_Test_Sets"
    ms_train_data=pd.read_csv(f"{ms_train_test_dir}/ms_train_data_remaining_minimal.csv",index_col=0)
    ms_val_data=pd.read_csv(f"{ms_train_test_dir}/ms_val_data.csv",index_col=0)
    ms_train_data["domain_id"]=0

    # --------------------------------------------------------
    # YD / HOLEC
    # --------------------------------------------------------
    if dataset_to_run=="yd":
        dataset_prefix="holec"; task_type="classification"
        data_dir=f"{DATA_DIR}/Training_Datasets/Processed/Holec_et_al_2025"
        if ds_split not in split_data: raise KeyError(f"YD split {ds_split} not found. Available: {sorted(split_data)}")
        ft_train=split_data[ds_split]["train"].copy(); ft_val=split_data[ds_split]["val"].copy()

        train_set=pd.concat([ms_train_data,ft_train],ignore_index=True)
        train_set["seq_len"]=train_set.seq.str.replace("$","").apply(len)
        train_set=train_set[train_set.allele.isin(ft_train.allele.unique())].query("seq_len == 9").reset_index(drop=True)
        ms_val_data["seq_len"]=ms_val_data.seq.str.replace("$","").apply(len)
        ms_val_matched=ms_val_data[ms_val_data.allele.isin(ft_val.allele.unique())].query("seq_len == 9").copy()

        for name,df in [("ms_val",ms_val_matched),("ft_val",ft_val)]:
            df=full_process_data(df,mhc_seq_dict,remove_flank=True,add_pseudo_label=False); df=create_seq_allele_column(df)
            if name=="ms_val": ms_val_matched=df
            else: ft_val=df
        filtered_columns=['seq','allele','label','seq_allele','n_flank','c_flank','mhc_a_1','mhc_b_1','mhc_c_1','mhc_a_2','mhc_b_2','mhc_c_2','sa_ma','mhcprime','group']
        ft_val=filter_single_df_to_cols(ft_val,filtered_columns)
        val_sets_run_once={"ms_val":ms_val_matched}; val_sets={"holec_val":ft_val}

    # --------------------------------------------------------
    # EPISCAN R9 / R9RT
    # --------------------------------------------------------
    elif dataset_to_run in {"episcan_r9","episcan_r9rt"}:
        dataset_prefix=dataset_to_run; task_type="classification"
        data_dir=f"{DATA_DIR}/Training_Datasets/Processed/EpiScan"
        if ds_split not in r9_split_data or ds_split not in rt_split_data:
            raise KeyError(f"EpiScan split {ds_split} not found. R9: {sorted(r9_split_data)}, RT: {sorted(rt_split_data)}")
        r9_train=r9_split_data[ds_split]["train"].copy(); rt_train=rt_split_data[ds_split]["train"].copy()
        r9_val=r9_split_data[ds_split]["val"].copy(); rt_val=rt_split_data[ds_split]["val"].copy()
        r9_train["domain_id"]=1; rt_train["domain_id"]=1
        ft_train=pd.concat([r9_train,rt_train.query("label == 1")],ignore_index=True) if dataset_to_run=="episcan_r9" else pd.concat([r9_train,rt_train],ignore_index=True)

        train_set=pd.concat([ms_train_data,ft_train],ignore_index=True); train_set["seq_len"]=train_set.seq.apply(len)
        iso_alleles=['A0201','A0301','B0801','B5701']
        train_set=train_set[train_set.allele.isin(iso_alleles)].query("seq_len == 9").reset_index(drop=True)
        r9_val["dataset_source"]="r9"; rt_val["dataset_source"]="rt"; ft_val=pd.concat([r9_val,rt_val],ignore_index=True)
        ms_val_data["seq_len"]=ms_val_data.seq.str.replace("$","").apply(len)
        ms_val_matched=ms_val_data[ms_val_data.allele.isin(iso_alleles)].query("seq_len == 9").copy()

        for name,df in [("ms_val",ms_val_matched),("ft_val",ft_val)]:
            df=full_process_data(df,mhc_seq_dict,remove_flank=True,add_pseudo_label=False); df=create_seq_allele_column(df)
            if name=="ms_val": ms_val_matched=df
            else: ft_val=df
        filtered_columns=['seq','allele','label','seq_allele','n_flank','c_flank','mhc_a_1','mhc_b_1','mhc_c_1','mhc_a_2','mhc_b_2','mhc_c_2','sa_ma','mhcprime','dataset_source']
        ft_val=filter_single_df_to_cols(ft_val,filtered_columns)
        val_sets_run_once={"ms_val":ms_val_matched}; val_sets={"episcan_r9_rt_val":ft_val}

    # --------------------------------------------------------
    # BA CONTINUOUS
    # --------------------------------------------------------
    elif dataset_to_run=="ba_con":
        dataset_prefix="ba_con"; task_type="continuous"
        data_dir=f"{DATA_DIR}/Training_Datasets/Processed/BA"
        if ds_split not in split_data: raise KeyError(f"BA split {ds_split} not found. Available: {sorted(split_data)}")
        ft_train=split_data[ds_split]["train"].copy(); ft_val=split_data[ds_split]["val"].copy()

        train_set=pd.concat([ms_train_data,ft_train],ignore_index=True); train_set["seq_len"]=train_set.seq.str.replace("$","").apply(len)
        train_set=filter_domain0_to_domain1_pairs(train_set,allele_col="allele",seq_len_col="seq_len",domain_col="domain_id").reset_index(drop=True)
        ms_val_data["seq_len"]=ms_val_data.seq.str.replace("$","").apply(len)
        ft_val["seq_len"]=ft_val.seq.str.replace("$","").apply(len)
        ms_val_matched=filter_to_shared_pairs(ms_val_data,ft_val,allele_col="allele",seq_len_col="seq_len")

        for name,df in [("ms_val",ms_val_matched),("ft_val",ft_val)]:
            df=full_process_data(df,mhc_seq_dict,remove_flank=True,add_pseudo_label=False); df=create_seq_allele_column(df)
            if name=="ms_val": ms_val_matched=df
            else: ft_val=df
        filtered_columns=['seq','allele','label','seq_allele','n_flank','c_flank','mhc_a_1','mhc_b_1','mhc_c_1','mhc_a_2','mhc_b_2','mhc_c_2','sa_ma','mhcprime','ba_value']
        ft_val=filter_single_df_to_cols(ft_val,filtered_columns)
        val_sets_run_once={"ms_val":ms_val_matched}; val_sets={"ba_val":ft_val}

    # --------------------------------------------------------
    # ESCAPE-SEQ CONTINUOUS
    # --------------------------------------------------------
    else:
        dataset_prefix="escape_con"; task_type="continuous"
        data_dir=f"{DATA_DIR}/Training_Datasets/Processed/ESCAPE_seq"
        if ds_split not in split_data: raise KeyError(f"ESCAPE split {ds_split} not found. Available: {sorted(split_data)}")
        ft_train=split_data[ds_split]["train"].copy(); ft_val=split_data[ds_split]["val"].copy()
        
        ft_train["domain_id"]=1

        train_set=pd.concat([ms_train_data,ft_train],ignore_index=True); train_set["seq_len"]=train_set.seq.str.replace("$","").apply(len)
        train_set=filter_domain0_to_domain1_pairs(train_set,allele_col="allele",seq_len_col="seq_len",domain_col="domain_id").reset_index(drop=True)
        ms_val_data["seq_len"]=ms_val_data.seq.str.replace("$","").apply(len)
        ft_val["seq_len"]=ft_val.seq.str.replace("$","").apply(len)
        ms_val_matched=filter_to_shared_pairs(ms_val_data,ft_val,allele_col="allele",seq_len_col="seq_len")

        for name,df in [("ms_val",ms_val_matched),("ft_val",ft_val)]:
            df=full_process_data(df,mhc_seq_dict,remove_flank=True,add_pseudo_label=False); df=create_seq_allele_column(df)
            if name=="ms_val": ms_val_matched=df
            else: ft_val=df
        filtered_columns=['seq','allele','label','seq_allele','n_flank','c_flank','mhc_a_1','mhc_b_1','mhc_c_1','mhc_a_2','mhc_b_2','mhc_c_2','sa_ma','mhcprime','escore','ba_value','name2','name','gene','mutation','mut_type']
        ft_val=filter_single_df_to_cols(ft_val,filtered_columns)
        val_sets_run_once={"ms_val":ms_val_matched}; val_sets={"escape_val":ft_val}

    # --------------------------------------------------------
    # COMMON OUTPUTS / OUTPUT DIRECTORY
    # --------------------------------------------------------
    ft_train_set=train_set.query("domain_id == 1").copy(); pepcl_train_set=train_set
    print(f"Dataset: {dataset_to_run} | split: {ds_split} | task: {task_type} | prefix: {dataset_prefix}")
    print(f"FT train rows: {len(ft_train_set):,} | PepCL train rows: {len(pepcl_train_set):,}")
    print(f"Validation sets run once: {list(val_sets_run_once)} | Validation sets: {list(val_sets)}")
    print_data_stats(pepcl_train_set)

    full_output_dir=f"{DATA_DIR}/{path_prefix}_{path_suffix}_CV_Datasplit_{ds_split}/"
    full_log_output_dir=f"{full_output_dir}/logs"
    if os.path.exists(full_output_dir):
        response=input(f"\nWARNING: Output directory already exists:\n{full_output_dir}\nContinuing may overwrite existing outputs.\nContinue? [y/n]: ").strip().lower()
        if response not in {"y","yes"}: raise SystemExit("Stopped by user. Existing output directory was not modified.")
        print("Continuing with existing output directory. Existing files may be overwritten.\n")
    print("Output directory for everything:",full_output_dir); os.makedirs(full_output_dir,exist_ok=True)
    print("Output directory for logs:",full_log_output_dir); os.makedirs(full_log_output_dir,exist_ok=True)

    # --------------------------------------------------------
    # CACHE + BASE MHCPrime INFERENCE
    # --------------------------------------------------------
    print("Creating validation set caches for inference...")
    val_sets_run_once_cache_paths={}
    for val_set_name_run_once,val_data_set_run_once in val_sets_run_once.items():
        print(val_set_name_run_once); data_to_run_path=f"{DATA_DIR}/{val_set_name_run_once}_{str(gpu_id)}_cache.pt"
        build_token_cache(val_data_set_run_once,data_to_run_path,tokenizer,use_domain_id=False); val_sets_run_once_cache_paths[val_set_name_run_once]=data_to_run_path
    val_set_cache_paths={}
    for val_set_name,val_data_set in val_sets.items():
        print(val_set_name); data_to_run_path=f"{DATA_DIR}/{val_set_name}_{str(gpu_id)}_cache.pt"
        build_token_cache(val_data_set,data_to_run_path,tokenizer,use_domain_id=False); val_set_cache_paths[val_set_name]=data_to_run_path

    print("Run base MHCPrime inference once on all validation sets...")
    base_and_ft_models={"mhcprime":copy.deepcopy(mhcprime)}
    for set_name_run_once,data_set_run_once in val_sets_run_once.items():
        path=val_sets_run_once_cache_paths[set_name_run_once]; print(set_name_run_once,path)
        for model_name,model in base_and_ft_models.items():
            model=model.to(device); model.eval()
            with torch.no_grad(): output_scores=run_mhcprime_fast(model,path,batch_size=3072,num_workers=num_workers,compile_model=False,use_domain_id=False)
            data_set_run_once[model_name]=output_scores.detach().cpu().tolist()
        val_sets_run_once[set_name_run_once]=data_set_run_once
    for set_name,data_set in val_sets.items():
        path=val_set_cache_paths[set_name]; print(set_name,path)
        for model_name,model in base_and_ft_models.items():
            model=model.to(device); model.eval()
            with torch.no_grad(): output_scores=run_mhcprime_fast(model,path,batch_size=3072,num_workers=num_workers,compile_model=False,use_domain_id=False)
            data_set_run_once[model_name]=output_scores.detach().cpu().tolist()
        val_sets[set_name]=data_set

    print("Saving initial val sets run once...")
    for name,df in val_sets_run_once.items(): print(name); df.to_csv(f"{full_output_dir}/{name}.csv")
    print("Saving initial val sets...")
    for name,df in val_sets.items(): print(name); df.to_csv(f"{full_output_dir}/{name}.csv")

    # --------------------------------------------------------
    # LAMBDA SWEEP
    # --------------------------------------------------------
    learning_rate_string=lr_to_string(default_lr)
    for pepcl_seed in seeds:
        for dr_lambda in dr_lambda_list:
            dr_lambda_string="0" if dr_lambda==0 else lr_to_string(dr_lambda)
            pepcl_base_params["seed"]=pepcl_seed
            pepcl_base_params["save_every_n_steps"]=None
            pepcl_base_params["dr_lambda"]=dr_lambda
            print(f"\nDataset split: {ds_split} | Seed: {pepcl_seed} | Steps: {default_steps} | LR: {default_lr} | DR Lambda: {dr_lambda}")

            final_models_to_run={}
            pepcl_r_dr_model_name=f"{dataset_prefix}_pepcl_r_dr_steps_{default_steps}_lr_{learning_rate_string}_dr_lambda_{dr_lambda_string}_seed_{pepcl_seed}"
            pepcl_dr_model_name=f"{dataset_prefix}_pepcl_dr_steps_{default_steps}_lr_{learning_rate_string}_dr_lambda_{dr_lambda_string}_seed_{pepcl_seed}"

            if run_pepcl_r_dr:
                print("PepCL w/ replay and drift regularization")
                model_pepcl_r_dr=copy.deepcopy(mhcprime)
                pepcl=PepCL_Logging_Ext(model=model_pepcl_r_dr,train_df=pepcl_train_set,use_replay=True,mhc_seq_dict=mhc_seq_dict,enable_logging=False,**pepcl_base_params)
                training_stats=pepcl.train(); final_models_to_run[pepcl_r_dr_model_name]=model_pepcl_r_dr

            if run_pepcl_dr:
                print("PepCL w/ DR only")
                model_pepcl_dr=copy.deepcopy(mhcprime)
                pepcl=PepCL_Logging_Ext(model=model_pepcl_dr,train_df=pepcl_train_set,use_replay=False,mhc_seq_dict=mhc_seq_dict,enable_logging=False,**pepcl_base_params)
                training_stats=pepcl.train(); final_models_to_run[pepcl_dr_model_name]=model_pepcl_dr

            clear_all_gpu_memory()
            for set_name_run_once,data_set_run_once in val_sets_run_once.items():
                path=val_sets_run_once_cache_paths[set_name_run_once]; print(set_name_run_once,path)
                for model_name,model in final_models_to_run.items():
                    model=model.to(device); model.eval()
                    with torch.no_grad(): output_scores=run_mhcprime_fast(model,path,batch_size=3072,num_workers=num_workers,compile_model=False,use_domain_id=False)
                    data_set_run_once[model_name]=output_scores.detach().cpu().tolist()
                val_sets_run_once[set_name_run_once]=data_set_run_once
            for set_name,data_set in val_sets.items():
                path=val_set_cache_paths[set_name]; print(set_name,path)
                for model_name,model in final_models_to_run.items():
                    model=model.to(device); model.eval()
                    with torch.no_grad(): output_scores=run_mhcprime_fast(model,path,batch_size=3072,num_workers=num_workers,compile_model=False,use_domain_id=False)
                    data_set_run_once[model_name]=output_scores.detach().cpu().tolist()
                val_sets[set_name]=data_set

            print("Saving val sets run once...")
            for name,df in val_sets_run_once.items(): print(name); df.to_csv(f"{full_output_dir}/{name}.csv")
            print("Saving val sets...")
            for name,df in val_sets.items(): print(name); df.to_csv(f"{full_output_dir}/{name}.csv")

    # Remove this split's temporary caches before moving to the next split.
    for path in val_sets_run_once_cache_paths.values():
        if os.path.exists(path): os.remove(path)
    for path in val_set_cache_paths.values():
        if os.path.exists(path): os.remove(path)

print("\nAll requested CV lambda sweeps complete.")
