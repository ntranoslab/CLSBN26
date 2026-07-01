"""
Call to train mhcprime on MS data. Creates the base checkpoint model.

Run cmd: python train_mhcprime_script.py -g gpu_id
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

# import ms train data
print("Loading training data...")
train_data_path = DATA_DIR / "Training_Datasets" / "Processed" / "MS_Train_Test_Sets" / "ms_train_data.csv"
ms_train_data = pd.read_csv(train_data_path)
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

# get tokenizer
tokenizer = AminoAcidTokenizer()

# set base model params
model_params={
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



# init model
print("Initializing model...")
model=init_model(model_params, print_check=False)
print(end="\n")

print("Setting up output directories...")
MODEL_CHECKPOINTS_DIR = PROJECT_ROOT / "Model_Checkpoints"

run_name = "MHCPrime_Base"
run_dir = MODEL_CHECKPOINTS_DIR / run_name
output_dir = run_dir / "Checkpoints"

print("Run dir:", run_dir)
print("Checkpoint dir:", output_dir)

if run_dir.exists():
    print(run_dir)
    choice = input(
        f"This run directory already exists. Do you want to continue and overwrite/add to it? [y/N]: "
    ).strip().lower()

    if choice not in ("y", "yes"):
        print("Exiting.")
        sys.exit()

output_dir.mkdir(parents=True, exist_ok=True)
print("Output dir set:", output_dir)
print(end="\n")

# training params
training_params = {
    "n_epochs": 120,
    "batch_size": 3072,
    "num_pos_per_epoch": 200_000,
    "neg_pos_ratio": 1,
    "num_workers": 16,
    "persistent_workers": True,
    "device": "cuda",

    "save_every_n_epochs": 60,
    "output_dir": output_dir,

    "loss_type": "logsmoothap",
    "loss_hp": 10.0,

    "encoder_lr": 2e-4,
    "decoder_lr": 2e-4,

    "seed": 42,
    "optimizer_type": "AdamW",

    "lr_warmup_epochs": None,
    "percentage_of_total_steps_for_warmup": None,
}

# train
print("Starting training...")
training_stats=train_mhcprime(
    model=model,
    train_df=ms_train_data,
    tokenizer=tokenizer,
    **training_params
)
print("Finished training...")
print(end="\n")

print("Saving everything...")

if output_dir is not None:
    torch.save(model.state_dict(), output_dir / "model_final.pt")
    save_dict_pickle(training_stats, output_dir / "training_stats.pkl")

    # save train and model params
    save_dict_pickle(training_params, run_dir / "training_params.pkl")

    model_params_to_save = model_params.copy()
    model_params_to_save.pop("feature_table", None)
    model_params_to_save.pop("vocab_size", None)
    model_params_to_save.pop("sequence_embedder_tokenizer", None)

    save_dict_pickle(model_params_to_save, run_dir / "model_params.pkl")

clear_all_gpu_memory()