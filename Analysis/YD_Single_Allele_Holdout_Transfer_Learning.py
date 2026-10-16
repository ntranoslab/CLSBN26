"""
Run cmd: python YD_Single_Allele_Holdout_Transfer_Learning.py -g gpu_id
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
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# mhc pseudosequences (loaded above)
mhc_seq_dict={k.removeprefix("HLA-").replace("*","").replace(":",""):v for k,v in mhc_seq_dict.items()}

# get processed feature data
print("Loading feature data...")
aa_features_table_ext, descriptor_groups=load_feature_data()
processed_feature_table, corr_info = process_aa_feature_table(aa_features_table_ext)
feature_names = list(processed_feature_table.columns) # pass to FeatureProjectionEncoder
print(end="\n")

# set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# import ms train data
print("Loading training data...")
# train_data_path = DATA_DIR / "Training_Datasets" / "Processed" / "MS_Train_Test_Sets" / "ms_train_data.csv"

# remaining after validation removal
train_data_path = DATA_DIR / "Training_Datasets" / "Processed" / "MS_Train_Test_Sets" / "ms_train_data_remaining_minimal.csv"

ms_train_data = pd.read_csv(train_data_path, index_col=0)
print_data_stats(ms_train_data)
print(end="\n")

# process train data (no flanks)
print("Processing training data...")
ms_train_data=full_process_data(ms_train_data, mhc_seq_dict, remove_flank=True, add_pseudo_label=False)
ms_train_data=create_seq_allele_column(ms_train_data)
ms_train_data["method_group"] = "MS"
method_group_map={"MS":0, "nonMS":1}
ms_train_data["domain_id"]=ms_train_data.method_group.map(method_group_map)
ms_train_data.reset_index(inplace=True, drop=True)
ms_train_data["example_id"]=ms_train_data.index.tolist()
print_data_stats(ms_train_data)
print(end="\n")

# get ms test data
test_data_path = DATA_DIR / "Training_Datasets" / "Processed" / "MS_Train_Test_Sets" / "ms_test_data.csv"
ms_test_data=pd.read_csv(test_data_path, index_col=0)
ms_test_data=full_process_data(ms_test_data, mhc_seq_dict, remove_flank=True, add_pseudo_label=False)
ms_test_data=create_seq_allele_column(ms_test_data)
ms_test_data["method_group"] = "MS"
ms_test_data["domain_id"]=0
print_data_stats(ms_test_data)

# get holec data
holec_data_path = DATA_DIR / "Training_Datasets" / "Processed" / "Holec_et_al_2025" / "holec_et_al_pathogen_data.csv"
holec_et_al_pathogen_data=pd.read_csv(holec_data_path)
holec_et_al_pathogen_data=full_process_data(holec_et_al_pathogen_data, mhc_seq_dict, remove_flank=True, add_pseudo_label=False)
holec_et_al_pathogen_data=create_seq_allele_column(holec_et_al_pathogen_data)
holec_et_al_pathogen_data["method_group"] = "nonMS"
holec_et_al_pathogen_data["domain_id"]=1
print_data_stats(holec_et_al_pathogen_data)

# merge to create final train set
ms_holec_train_data=ms_train_data.copy()
ms_holec_train_data=pd.concat([ms_holec_train_data, holec_et_al_pathogen_data], axis=0) 
ms_holec_train_data.reset_index(inplace=True, drop=True)

ms_holec_train_data["example_id"]=ms_holec_train_data.index

print_data_stats(ms_holec_train_data)

# match allele and len
ms_holec_train_data["seq_len"]=ms_holec_train_data.seq.str.replace("$", "").apply(len)
ms_holec_train_data_matched_allele=ms_holec_train_data[ms_holec_train_data.allele.isin(holec_et_al_pathogen_data.allele.unique().tolist())].query("seq_len == 9")
ms_holec_train_data_matched_allele.reset_index(inplace=True, drop=True)

ms_holec_train_data_matched_allele["example_id"]=ms_holec_train_data_matched_allele.index

print_data_stats(ms_holec_train_data_matched_allele)

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

print("Setting up directories...")
MODEL_CHECKPOINTS_DIR = PROJECT_ROOT / "Model_Checkpoints"

dataset_name_dir="YD"
# allele_transfer_learning_dir = MODEL_CHECKPOINTS_DIR / "Allele_Transfer_Learning_Models" / dataset_name_dir

allele_transfer_learning_dir = MODEL_CHECKPOINTS_DIR / "Allele_Transfer_Learning_Models_Optimized" / dataset_name_dir

allele_transfer_learning_dir_ft = allele_transfer_learning_dir / "FT"
allele_transfer_learning_dir_pepcl = allele_transfer_learning_dir / "PepCL"

allele_transfer_learning_dir_ft.mkdir(parents=True, exist_ok=True)
allele_transfer_learning_dir_pepcl.mkdir(parents=True, exist_ok=True)

print("Output dirs set:", allele_transfer_learning_dir_ft)
print("Output dirs set:", allele_transfer_learning_dir_pepcl)

# transfer_learning_outputs_dir = DATA_DIR / "Holec_Single_Allele_Heldout_Tranfer_Learning_Outputs" / "output_dfs"

transfer_learning_outputs_dir = DATA_DIR / "Holec_Single_Allele_Heldout_Tranfer_Learning_Outputs_Optimized" / "output_dfs"

transfer_learning_outputs_dir.mkdir(parents=True, exist_ok=True)

print(end="\n")

# init model
print("Initializing model...")
# base_model_dir = MODEL_CHECKPOINTS_DIR / "MHCPrime_Base" / "Checkpoints"

base_model_dir = MODEL_CHECKPOINTS_DIR / "MHCPrime_Base_for_Ablations" / "Checkpoints"

mhcprime=init_model(base_model_params, print_check=False, print_params=True)
mhcprime.load_state_dict(torch.load(f"{base_model_dir}/model_final.pt"))
mhcprime.eval()
print("Model loaded from checkpoint...")
print(end="\n")
clear_all_gpu_memory()

alleles_to_run_per_gpu={
    # 0: ['A0201', 'C0401', 'A0101', 'B3901'], 
    # 1: ['A2402', 'B5502', 'A0301', 'B5801'], 
    0: ['B3502', 'C0102', 'A3001', 'C0303'], 
    3: ['B3503', 'C1202', 'A3002', 'A1101']
}

base_pepcl_params=dict(
            device="cuda",
            # steps = 200,
            steps = 300, # NOTE: updated steps for all (300 is optimal for FT here as well)
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
            nonms_loss_type = "logsmoothap",
            nonms_loss_alpha = 10.0,
            ft_lambda = 1.0,
            dr_lambda = 1e-5,
            dr_loss_type = "mse",
)

alleles_to_run = alleles_to_run_per_gpu[GPU_ID]
print(alleles_to_run)

for allele in tqdm(alleles_to_run):

    base_and_ft_models={} # initialized inside since we run inf after each run

    print(allele)

    # NOTE: holdout strategy depends on what we are training
    train_data_holec_aho=ms_holec_train_data_matched_allele.copy()

    train_data_holec_aho=ms_holec_train_data_matched_allele[ms_holec_train_data_matched_allele["allele"] != allele].copy()

    # get test set, which is for that specific allele basically, the holec seq data
    holec_allele_test_set=holec_et_al_pathogen_data[holec_et_al_pathogen_data.allele == allele]

    # get ms test set (full allele, we can filter by length later)
    ms_allele_test_set=ms_test_data[ms_test_data.allele == allele]

    # compute number of steps needed for domain id 0 and 1, depending on whether we are replay or no replay
    base_model = copy.deepcopy(mhcprime) 
    model_pepcl = copy.deepcopy(mhcprime) 
    model_ft = copy.deepcopy(mhcprime) 

    model_name_pepcl=f"model_pepcl_{allele}" 
    model_name_ft=f"model_ft_{allele}" # no replay

    print("PepCL")
    # run updated pepcl function
    pepcl=PepCL(
            model=model_pepcl,
            train_df=train_data_holec_aho,
            tokenizer=tokenizer,
            use_replay=True,
            **base_pepcl_params,
    )
    training_stats=pepcl.train()


    print("FT")
    # run updated pepcl function
    pepcl=PepCL(
            model=model_ft,
            train_df=train_data_holec_aho,
            tokenizer=tokenizer,
            use_replay=False,
            **base_pepcl_params,
    )
    training_stats=pepcl.train()


    print("Finished training...")
    print(end="\n")

    # add to model registry after ft
    base_and_ft_models["mhcprime"]=base_model
    base_and_ft_models[model_name_pepcl]=model_pepcl
    base_and_ft_models[model_name_ft]=model_ft

    # run inference and test

    build_caches=True
    # build_caches=False

    if build_caches:

        # holec_allele_test_set
        data_to_run_path=f"{DATA_DIR}/holec_allele_test_set_{GPU_ID}.pt"
        build_token_cache(holec_allele_test_set, data_to_run_path, tokenizer, use_domain_id=False) # or .parquet -> provide DF directly now!

        # ms_allele_test_set
        data_to_run_path=f"{DATA_DIR}/ms_allele_test_set_{GPU_ID}.pt"
        build_token_cache(ms_allele_test_set, data_to_run_path, tokenizer, use_domain_id=False) # or .parquet -> provide DF directly now!

        
    data_to_run_path=f"{DATA_DIR}/holec_allele_test_set_{GPU_ID}.pt"
    for model_name, model in base_and_ft_models.items():

        use_domain_id=False

        score_col=model_name
        print(score_col)

        model.eval()
        with torch.no_grad():
            num_workers=16
            output_scores=run_mhcprime_fast(model, data_to_run_path, batch_size=3072, num_workers=num_workers, compile_model=False, use_domain_id=use_domain_id)

        holec_allele_test_set[score_col]=output_scores.detach().cpu().tolist()


    data_to_run_path=f"{DATA_DIR}/ms_allele_test_set_{GPU_ID}.pt"
    for model_name, model in base_and_ft_models.items():

        use_domain_id=False

        score_col=model_name
        print(score_col)

        model.eval()
        with torch.no_grad():
            num_workers=16
            output_scores=run_mhcprime_fast(model, data_to_run_path, batch_size=3072, num_workers=num_workers, compile_model=False, use_domain_id=use_domain_id)

        ms_allele_test_set[score_col]=output_scores.detach().cpu().tolist()

    # holec_allele_test_set
    data_to_run_path=f"{DATA_DIR}/holec_allele_test_set_{GPU_ID}.pt"
    os.remove(data_to_run_path)

    # ms_allele_test_set
    data_to_run_path=f"{DATA_DIR}/ms_allele_test_set_{GPU_ID}.pt"
    os.remove(data_to_run_path)

    print("Saving", model_name_pepcl)
    torch.save(model_pepcl.state_dict(), f"{allele_transfer_learning_dir_pepcl}/{model_name_pepcl}.pt")

    print("Saving", model_name_ft)
    torch.save(model_ft.state_dict(), f"{allele_transfer_learning_dir_ft}/{model_name_ft}.pt")

    print("Saving dataframes")
    # save holec seq test set
    holec_test_set_name=f"holec_allele_test_set_{allele}"
    ms_test_set_name=f"ms_allele_test_set_{allele}"

    # save
    holec_allele_test_set.to_csv(f"{transfer_learning_outputs_dir}/{holec_test_set_name}.csv")
    ms_allele_test_set.to_csv(f"{transfer_learning_outputs_dir}/{ms_test_set_name}.csv")
