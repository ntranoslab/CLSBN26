"""
Run cmd: python update_mhcprime_with_heldout_alleles.py -g gpu_id
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


# get splits
allele_holdout_splits_path = DATA_DIR / "Training_Datasets" / "Processed" / "MS_Train_Test_Sets" / "allele_holdout_splits.pkl"
allele_holdout_splits=load_dict_pickle(allele_holdout_splits_path)

# load base models
models={}
MODEL_CHECKPOINTS_DIR = PROJECT_ROOT / "Model_Checkpoints"
aho_dir = MODEL_CHECKPOINTS_DIR / "Alleles_Heldout_Models"
for allele_split, allele_list in allele_holdout_splits.items():
    model_dir = f"{aho_dir}/MHCPrime_Base_AHO_Split_{str(allele_split)}/Checkpoints"
    model_name=f"mhcprime_base_aho_split_{str(allele_split)}"

    model=init_model(base_model_params, print_check=False, print_params=True)
    model.load_state_dict(torch.load(f"{model_dir}/model_final.pt"))
    model.eval()
    models[model_name]=model
    print("Model loaded from checkpoint...")

clear_all_gpu_memory()

output_dir=MODEL_CHECKPOINTS_DIR / "Alleles_Heldout_Updated_Models"
output_dir.mkdir(parents=True, exist_ok=True)

pepcl_base_params=dict(
            tokenizer=tokenizer,
            device="cuda",
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
            nonms_loss_type = "logsmoothap",
            nonms_loss_alpha = 10.0,
            ft_lambda = 1.0,
            dr_lambda = 1e-5,
            dr_loss_type = "mse"
)

for allele_split, allele_list in allele_holdout_splits.items():
    print(allele_split, allele_list, len(allele_list))
    ms_train_data_aho=ms_train_data.copy()

    ms_train_data_aho=add_domain_id_with_alleles(ms_train_data_aho, allele_list, allele_col='allele')

    # init model
    print("Initializing models...")
    base_model_name=f"mhcprime_base_aho_split_{str(allele_split)}"
    model_ft_name=f"mhcprime_ft_aho_split_{str(allele_split)}"
    model_pepcl_name=f"mhcprime_pepcl_aho_split_{str(allele_split)}"

    model_ft=copy.deepcopy(models[base_model_name])
    model_pepcl=copy.deepcopy(models[base_model_name])
    print(end="\n")

    print("PepCL")
    pepcl=PepCL(
        model=model_pepcl,
        train_df=ms_train_data_aho,
        use_replay = True,
        **pepcl_base_params,
    )
    training_stats=pepcl.train()    

    print("FT")
    pepcl=PepCL(
        model=model_ft,
        train_df=ms_train_data_aho,
        use_replay = False,
        **pepcl_base_params,
    )
    training_stats=pepcl.train()    

    print("Saving models ...")
    torch.save(model_pepcl.state_dict(), output_dir / f"{model_pepcl_name}.pt")
    torch.save(model_ft.state_dict(), output_dir / f"{model_ft_name}.pt")

    clear_all_gpu_memory()