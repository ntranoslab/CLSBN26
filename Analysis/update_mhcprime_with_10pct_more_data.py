"""
Call to train mhcprime on MS data. Creates the base checkpoint model.

Run cmd: python update_mhcprime_with_10pct_more_data.py -g gpu_id
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
MODEL_CHECKPOINTS_DIR = PROJECT_ROOT / "Model_Checkpoints"

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

# process
ms_train_data=full_process_data(ms_train_data, mhc_seq_dict, remove_flank=True, add_pseudo_label=False)
ms_train_data=create_seq_allele_column(ms_train_data)

# remove padding from seq allele
ms_train_data.seq_allele=ms_train_data.seq_allele.str.replace("$", "")
print_data_stats(ms_train_data)
print(end="\n")

# seq allele splits
data_splits_path = DATA_DIR / "Training_Datasets" / "Processed" / "MS_Train_Test_Sets" / "ms_train_data_seq_allele_dict_10_pct_splits.pkl"
ms_train_data_seq_allele_dict_10_pct_splits = load_dict_pickle(data_splits_path)

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
        'sequence_embedder_feature_mode': "concat", # replace or concat
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
run_name = "MHCPrime_Base"
run_dir = MODEL_CHECKPOINTS_DIR / run_name
model_dir = run_dir / "Checkpoints"

mhcprime=init_model(base_model_params, print_check=False, print_params=True)
mhcprime.load_state_dict(torch.load(f"{model_dir}/model_final.pt"))
mhcprime.eval()
print("Model loaded from checkpoint...")

# add split models
split_percentages_dict={
    10: [0],
    20: [0, 1],
    30: [0, 1, 2],
    40: [0, 1, 2, 3],
    50: [0, 1, 2, 3, 4],
    60: [0, 1, 2, 3, 4, 5],
    70: [0, 1, 2, 3, 4, 5, 6],
    80: [0, 1, 2, 3, 4, 5, 6, 7],
    90: [0, 1, 2, 3, 4, 5, 6, 7, 8],
}

models={"mhcprime": mhcprime}
for pt_pct in split_percentages_dict.keys():
    model_dir=f"{MODEL_CHECKPOINTS_DIR}/MHCPrime_Base_{str(pt_pct)}pct/Checkpoints"
    model_name=f"mhcprime_{str(pt_pct)}pct"
    print(model_name)
    model=init_model(base_model_params, print_check=False, print_params=True)
    model.load_state_dict(torch.load(f"{model_dir}/model_final.pt"))
    models[model_name]=model


# create split dir based on steps
steps = 1000
in_distribution_updating_dir = MODEL_CHECKPOINTS_DIR / f"MS_10pct_more_updating_models_{steps}steps"
in_distribution_updating_dir.mkdir(parents=True, exist_ok=True)

pepcl_params=dict(
        tokenizer=tokenizer,
        device="cuda",
        steps = steps,
        output_dir = None,
        save_every_n_steps = 2000,
        seed = 42,
        num_workers = 16,
        collate_fn = collate_fn,
        num_ms_pos = 2000,
        ms_neg_ratio = 1.0,
        num_nonms_pos = 2000,
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
        dr_lambda = 0,
        dr_loss_type = "mse"

)

# run model updating with 10% more data
for pt_pct in split_percentages_dict.keys():
    print("Split percentage: ", pt_pct)

    # create update and replay set
    train_ft_data=ms_train_data.copy()
    train_ft_data=assign_domain_split(train_ft_data, ms_train_data_seq_allele_dict_10_pct_splits, pt_pct, split_percentages_dict, seq_allele_col='seq_allele')
    train_ft_data.reset_index(inplace=True, drop=True)
    print_data_stats(train_ft_data)

    # init models
    model_name=f"mhcprime_{str(pt_pct)}pct"
    model_ft = copy.deepcopy(models[model_name])
    model_pepcl = copy.deepcopy(models[model_name])

    print("FT")
    pepcl=PepCL(
        model=model_ft,
        train_df=train_ft_data,
        use_replay=False,
        **pepcl_params,
    )
    training_stats=pepcl.train()

    print("PepCL")
    pepcl=PepCL(
            model=model_pepcl,
            train_df=train_ft_data,
            use_replay=True,
            **pepcl_params,
    )
    training_stats=pepcl.train()

    # save models
    print("Saving models")
    torch.save(model_ft.state_dict(), in_distribution_updating_dir / f"mhcprime_{str(pt_pct)}pct_updated_on_10pct_ft.pt")
    torch.save(model_pepcl.state_dict(), in_distribution_updating_dir / f"mhcprime_{str(pt_pct)}pct_updated_on_10pct_pepcl.pt")

    clear_all_gpu_memory()