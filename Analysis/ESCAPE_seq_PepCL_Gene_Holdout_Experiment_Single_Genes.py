"""
Run cmd: python ESCAPE_seq_PepCL_Gene_Holdout_Experiment_Single_Genes.py -g gpu_id
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
train_data_path = DATA_DIR / "Training_Datasets" / "Processed" / "MS_Train_Test_Sets" / "ms_train_data.csv"
ms_train_data = pd.read_csv(train_data_path, index_col=0)
print_data_stats(ms_train_data)
print(end="\n")

# get ms test data
test_data_path = DATA_DIR / "Training_Datasets" / "Processed" / "MS_Train_Test_Sets" / "ms_test_data.csv"
ms_test_data=pd.read_csv(test_data_path, index_col=0)
print_data_stats(ms_test_data)
print(end="\n")

# escape seq sets
escape_seq_datasets_dir=f"{DATA_DIR}/Training_Datasets/Processed/ESCAPE_seq"

oncogene_mut_wt_df=pd.read_csv(f"{escape_seq_datasets_dir}/oncogene_mut_wt_df.csv", index_col=0)
oncogene_wt_df_flat=pd.read_csv(f"{escape_seq_datasets_dir}/oncogene_wt_df_flat.csv", index_col=0)
oncogene_mut_df_flat=pd.read_csv(f"{escape_seq_datasets_dir}/oncogene_mut_df_flat.csv", index_col=0)

ms_train_data=full_process_data(ms_train_data, mhc_seq_dict, remove_flank=True, add_pseudo_label=False)
ms_train_data=create_seq_allele_column(ms_train_data)

ms_test_data=full_process_data(ms_test_data, mhc_seq_dict, remove_flank=True, add_pseudo_label=False)
ms_test_data=create_seq_allele_column(ms_test_data)

oncogene_mut_wt_df=full_process_data(oncogene_mut_wt_df, mhc_seq_dict, remove_flank=True, add_pseudo_label=False)
oncogene_mut_wt_df=create_seq_allele_column(oncogene_mut_wt_df)

oncogene_wt_df_flat=full_process_data(oncogene_wt_df_flat, mhc_seq_dict, remove_flank=True, add_pseudo_label=False)
oncogene_wt_df_flat=create_seq_allele_column(oncogene_wt_df_flat)

oncogene_mut_df_flat=full_process_data(oncogene_mut_df_flat, mhc_seq_dict, remove_flank=True, add_pseudo_label=False)
oncogene_mut_df_flat=create_seq_allele_column(oncogene_mut_df_flat)

# set mut type
oncogene_wt_df_flat["mut_type"]="wt"
oncogene_mut_df_flat["mut_type"]="mut"

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


pepcl_base_params_con=dict(
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
            nonms_loss_type = "spearman",
            nonms_loss_alpha = 0.01,
            ft_lambda = 1.0,
            dr_lambda = 1e-5,
            dr_loss_type = "mse",
            continuous_target_col="ba_value",
            continuous_ft_sampling_mode="allele_aware",
)

print("Making output dir")
central_ft_dir=f"{DATA_DIR}/ESCAPE_seq_gene_holdout_experiment"
os.makedirs(central_ft_dir, exist_ok=True)

output_df_dir="inference_dfs"
os.makedirs(f"{central_ft_dir}/{output_df_dir}", exist_ok=True)

MODEL_CHECKPOINTS_DIR = PROJECT_ROOT / "Model_Checkpoints"

pepcl_models_dir="ESCAPE_seq_gene_holdout_experiment/PepCL"
os.makedirs(f"{MODEL_CHECKPOINTS_DIR}/{pepcl_models_dir}", exist_ok=True)

ft_models_dir="ESCAPE_seq_gene_holdout_experiment/FT"
os.makedirs(f"{MODEL_CHECKPOINTS_DIR}/{ft_models_dir}", exist_ok=True)

# init model
print("Initializing model...")
# base_model_dir = MODEL_CHECKPOINTS_DIR / "MHCPrime_Base" / "Checkpoints"
# mhcprime=init_model(base_model_params, print_check=False, print_params=True)
# mhcprime.load_state_dict(torch.load(f"{base_model_dir}/model_final.pt"))
# mhcprime.eval()
# print("Model loaded from checkpoint...")
# print(end="\n")
# clear_all_gpu_memory()

base_model_dir = MODEL_CHECKPOINTS_DIR / "MHCPrime_Base" / "Checkpoints"
ckpt_path = base_model_dir / "model_final.pt"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

print("Starting training loop")
gene_groups={
    "KRAS": ["KRAS", "KRAS-NRAS", "KRAS-KRAS-NRAS-NRAS-HRAS-HRAS", "KRAS-HRAS", "KRAS-NRAS-HRAS", "HRAS", "NRAS-HRAS"], 
    "FLT3": ["FLT3", "KIT", "PDGFRA", "FLT3-KIT-PDGFRA", "FLT3-PDGFRA"],
    "IDH2": ["IDH2", "IDH2-IDH2"],
    "PIK3CA": ["PIK3CA", "PIK3CA-PIK3CA"],
    "CTNNB1": ["CTNNB1", "CTNNB1-CTNNB1"],
}

genes_to_run=['JAK2', 'MAX', 'MPL', 'MED12', 'TP53', 'SRSF2', 'NOTCH2', 'EZH2', 'AKT1', 'IDH1', 'ATM', 'MYD88', 'EGFR', 'FGFR3', 'PTEN', 'SF3B1', 'CREBBP', 'MYC', 'FBXW7', 'BRAF', 'RET', 'SMAD4', 'ERBB2', 'GNAS', 'DNMT3A', 'ALK', 'KIAA1549', 'EWSR1']


def flatten_dict_values(d):
    return [item for values in d.values() for item in values]

exclude_genes=flatten_dict_values(gene_groups)

# for gene in tqdm(oncogene_mut_df_flat_processed.gene.unique()):
for gene in tqdm(genes_to_run):

    if gene in exclude_genes:
        print("Excluding", gene)
        continue
    
    print(gene)
    print("Getting data")
    genes=[gene]
    mut_wt_df_no_gene_resolved, mut_wt_df_gene=create_wt_mut_gene_train_test_sets(oncogene_mut_df_flat, oncogene_wt_df_flat, ms_train_data, ms_test_data, genes)

    mut_wt_df_no_gene_resolved["domain_id"]=1

    mut_wt_df_no_gene_resolved.rename(columns={"escore":"ba_value"}, inplace=True)

    # merge to create final train set
    new_train_data_ms_full_pt_and_ft_no_gene=ms_train_data.copy()

    new_train_data_ms_full_pt_and_ft_no_gene["domain_id"]=0

    new_train_data_ms_full_pt_and_ft_no_gene=pd.concat([new_train_data_ms_full_pt_and_ft_no_gene, mut_wt_df_no_gene_resolved], axis=0) # no A6801

    new_train_data_ms_full_pt_and_ft_no_gene.reset_index(inplace=True, drop=True)

    print_data_stats(new_train_data_ms_full_pt_and_ft_no_gene)

    new_train_data_ms_full_pt_and_ft_no_gene["seq_len"]=new_train_data_ms_full_pt_and_ft_no_gene.seq.str.replace("$", "").apply(len)

    new_train_data_ms_full_pt_and_ft_no_gene_matched_allele=new_train_data_ms_full_pt_and_ft_no_gene[new_train_data_ms_full_pt_and_ft_no_gene.allele.isin(mut_wt_df_no_gene_resolved.allele.unique().tolist())].query("seq_len == 9") # w/ train alleles included

    new_train_data_ms_full_pt_and_ft_no_gene_matched_allele.reset_index(inplace=True, drop=True)

    def filter_sequences_with_x(df, seq_col='seq'):
        return df[df[seq_col].str.contains('X', na=False)].reset_index(drop=True)

    def filter_sequences_without_x(df, seq_col='seq'):
        return df[~df[seq_col].str.contains('X', na=False)].reset_index(drop=True)

    new_train_data_ms_full_pt_and_ft_no_gene_matched_allele_noX=filter_sequences_without_x(new_train_data_ms_full_pt_and_ft_no_gene_matched_allele)

    new_train_data_ms_full_pt_and_ft_no_gene_matched_allele_noX_noA68=new_train_data_ms_full_pt_and_ft_no_gene_matched_allele_noX.query("allele != 'A6801'")

    print(new_train_data_ms_full_pt_and_ft_no_gene_matched_allele.shape)
    print(new_train_data_ms_full_pt_and_ft_no_gene_matched_allele_noX.shape)
    print(new_train_data_ms_full_pt_and_ft_no_gene_matched_allele_noX_noA68.shape)

    print_data_stats(new_train_data_ms_full_pt_and_ft_no_gene_matched_allele_noX_noA68)
    
    # init model
    model_base = copy.deepcopy(mhcprime)
    model_ft = copy.deepcopy(mhcprime)
    model_pepcl = copy.deepcopy(mhcprime)

    model_name_ft=f"model_ft_{gene}"
    model_name_pepcl=f"model_pepcl_{gene}"    

    print("FT")
    # run updated pepcl function
    pepcl=PepCL(
            model=model_ft,
            train_df=new_train_data_ms_full_pt_and_ft_no_gene_matched_allele_noX_noA68,
            use_replay = False,
            **pepcl_base_params_con,
    )
    training_stats=pepcl.train()

    print("PepCL")
    # run updated pepcl function
    pepcl=PepCL(
            model=model_pepcl,
            train_df=new_train_data_ms_full_pt_and_ft_no_gene_matched_allele_noX_noA68,
            use_replay = True,
            **pepcl_base_params_con,
    )
    training_stats=pepcl.train()

    # score models
    base_and_ft_models={"model_base": model_base, "model_ft": model_ft, "model_pepcl": model_pepcl}

    for model_name, model in base_and_ft_models.items():
        model.eval()
        with torch.no_grad():
            mut_wt_df_gene=run_mhcprime(model, mut_wt_df_gene, tokenizer, batch_size=3072, num_workers=0, score_col=model_name, device="cuda", return_embeddings=False)


    mut_wt_df_gene["name_allele"]=[f"{n}_{a}" for n, a in zip(mut_wt_df_gene.name2, mut_wt_df_gene.allele)]

    score_cols=["model_base", "model_ft", "model_pepcl"]

    wt_mut_gene_diff_noX=pair_wt_mut_and_diff_with_quadrants_v2(
        df=filter_sequences_without_x(mut_wt_df_gene),
        score_cols=score_cols + ["escore"],
        # id_col="name2",
        id_col="name_allele",
        mut_type_col = "mut_type",   # values like "wt" / "mut"
        wt_value = "wt",
        mut_value = "mut",
        allele_col = "allele",
        mutation_col = "mutation",
        seq_col = "seq",
        label_col = "label",
        quadrant_col = "wt_mut_quadrant",
        seed = 42,
    )

    wt_mut_gene_diff_noX=add_quadrant_binary_label(
        df=wt_mut_gene_diff_noX,
        quadrant_col="wt_mut_quadrant",
        quadrant_name="WT-/MUT+", # shows the best in kd
        # quadrant_name="WT+/MUT-", # shows the best in kd
        out_label_col = "quadrant_label",
    )
    wt_mut_gene_diff_noX_negated=wt_mut_gene_diff_noX.copy()

    wt_mut_gene_diff_noX_negated=add_quadrant_binary_label(
        df=wt_mut_gene_diff_noX_negated,
        quadrant_col="wt_mut_quadrant",
        # quadrant_name="WT-/MUT+", # shows the best in kd
        quadrant_name="WT+/MUT-", # shows the best in kd
        out_label_col = "quadrant_label",
    )

    for score_col in score_cols + ["escore"]:
        wt_mut_gene_diff_noX_negated[score_col]=-wt_mut_gene_diff_noX_negated[score_col]

    mut_wt_df_gene["all"]=1
    allele_ap_matrix=compute_ap_matrix_optimized(filter_sequences_without_x(mut_wt_df_gene).query("allele != 'A6801'"), score_cols + ["escore"], label_col='label', allele_col='all') # use reverse_rank_score_cols instead of rank_score_cols
    print(allele_ap_matrix.mean())
    print(end="\n")

    wt_mut_gene_diff_noX["all"]=1
    allele_ap_matrix=compute_ap_matrix_optimized(wt_mut_gene_diff_noX.query("allele != 'A6801'"), score_cols + ["escore"], label_col='quadrant_label', allele_col='all') # use reverse_rank_score_cols instead of rank_score_cols

    print(allele_ap_matrix.mean())
    # allele_ap_matrix.median()

    wt_mut_gene_diff_noX_negated["all"]=1
    allele_ap_matrix=compute_ap_matrix_optimized(wt_mut_gene_diff_noX_negated.query("allele != 'A6801'"), score_cols + ["escore"], label_col='quadrant_label', allele_col='all') # use reverse_rank_score_cols instead of rank_score_cols

    print(allele_ap_matrix.mean())
    # allele_ap_matrix.median()

    """
    Save
    """

    print("Saving DF")
    mut_wt_df_gene.to_csv(f"{central_ft_dir}/{output_df_dir}/mut_wt_df_gene_{gene}.csv")

    print("Saving", model_name_ft)
    torch.save(model_ft.state_dict(), f"{MODEL_CHECKPOINTS_DIR}/{ft_models_dir}/{model_name_ft}.pt")

    print("Saving", model_name_pepcl)
    torch.save(model_pepcl.state_dict(), f"{MODEL_CHECKPOINTS_DIR}/{pepcl_models_dir}/{model_name_pepcl}.pt")

