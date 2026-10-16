"""
ESCAPE-seq continuous-value updating ablation script.

Run cmd: python ESCAPE_seq_CON_CV_PepCL_Step_LR_Ablations_Script.py -g gpu_id
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
from src.pepcl_logging import *

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
device = f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu'

# mhc pseudosequences (loaded above)
mhc_seq_dict={k.removeprefix("HLA-").replace("*","").replace(":",""):v for k,v in mhc_seq_dict.items()}

# get processed feature data
print("Loading feature data...")
aa_features_table_ext, descriptor_groups=load_feature_data()
processed_feature_table, corr_info = process_aa_feature_table(aa_features_table_ext)
feature_names = list(processed_feature_table.columns) # pass to FeatureProjectionEncoder
print(end="\n")

# format pseudosequences
from Data.mhc_pseudo_seqs import mhc_seq_dict
mhc_seq_dict={k.removeprefix("HLA-").replace("*","").replace(":",""):v for k,v in mhc_seq_dict.items()}

# get feature table and processed features
aa_features_table_ext, descriptor_groups=load_feature_data()
processed_feature_table, corr_info = process_aa_feature_table(aa_features_table_ext)
feature_names = list(processed_feature_table.columns)  # pass to FeatureProjectionEncoder

# cleaned version
escape_con_train_val_sets=load_dict_pickle(f"{DATA_DIR}/Training_Datasets/Processed/ESCAPE_seq/escape_con_train_val_sets_cleaned.pkl")


for ds_split, train_val_dict in escape_con_train_val_sets.items():
    print("Dataset Split", ds_split)


    # load ms train and validation sets
    print("Loading MS data...")
    ms_train_test_dir=f"{DATA_DIR}/Training_Datasets/Processed/MS_Train_Test_Sets"
    # ms_train_data_remaining_minimal=pd.read_csv(f"{ms_train_test_dir}/ms_train_data_remaining_minimal.csv", index_col=0)

    ms_train_data=pd.read_csv(f"{ms_train_test_dir}/ms_train_data_remaining_minimal.csv", index_col=0)

    # NOTE: LOAD IN BASE MS DATA
    # ms_train_data=pd.read_csv(f"{ms_train_test_dir}/ms_train_data.csv", index_col=0)

    ms_val_data=pd.read_csv(f"{ms_train_test_dir}/ms_val_data.csv", index_col=0)

    # load ft train and validation sets
    print("Loading ESCAPE-seq data...")

    escape_train_remaining=train_val_dict["train"]
    escape_val=train_val_dict["val"]


    # merge the train sets and assign domain id
    print("Merging MS/FT data...")
    # ms_train_data_remaining_minimal["domain_id"]=0
    ms_train_data["domain_id"]=0 # NOTE: USE BASE MS TRAIN DATA
    escape_train_remaining["domain_id"]=1

    ms_escape_train_set=pd.concat([ms_train_data, escape_train_remaining])
    ms_escape_train_set.reset_index(inplace=True, drop=True)
    print_data_stats(ms_escape_train_set)

    # match allele and length distribution between domains in train
    ms_escape_train_set["seq_len"]=ms_escape_train_set.seq.str.replace("$", "").apply(len)

    ms_escape_train_set_matched_allele_length=filter_domain0_to_domain1_pairs(ms_escape_train_set, allele_col='allele', seq_len_col='seq_len', domain_col='domain_id')
    ms_escape_train_set_matched_allele_length.reset_index(inplace=True, drop=True)
    print_data_stats(ms_escape_train_set_matched_allele_length)

    print("Matching allele- and length distributions for MS/FT data...")
    # filter the val set down to matching allele and len
    ms_val_data["seq_len"]=ms_val_data.seq.str.replace("$", "").apply(len) # seq.str.replace("$", "") check for padding

    # match allele length distribuction for ms data
    escape_val["seq_len"]=escape_val.seq.str.replace("$", "").apply(len)

    # NOTE: we can match to the val set as well for BA. Do that later perhaps.
    ms_val_data_matched_allele_length=filter_to_shared_pairs(ms_val_data, escape_val, allele_col='allele', seq_len_col='seq_len')


    # process the val sets for ms and ft

    ms_val_data_matched_allele_length=full_process_data(ms_val_data_matched_allele_length, mhc_seq_dict, remove_flank=True, add_pseudo_label=False)
    ms_val_data_matched_allele_length=create_seq_allele_column(ms_val_data_matched_allele_length)


    escape_val=full_process_data(escape_val, mhc_seq_dict, remove_flank=True, add_pseudo_label=False)
    escape_val=create_seq_allele_column(escape_val)


    ms_val_data=full_process_data(ms_val_data, mhc_seq_dict, remove_flank=True, add_pseudo_label=False)
    ms_val_data=create_seq_allele_column(ms_val_data)

    # filter cols for validation sets
    filtered_columns=['seq', 'allele', 'label', 'seq_allele', 'n_flank', 'c_flank', 'mhc_a_1', 'mhc_b_1', 'mhc_c_1', 'mhc_a_2', 'mhc_b_2', 'mhc_c_2', 'sa_ma', 'mhcprime', 'escore', 'ba_value', 'name2', 'name', 'gene', 'mutation', 'mut_type']

    ms_val_data_matched_allele_length=filter_single_df_to_cols(ms_val_data_matched_allele_length, filtered_columns)
    escape_val=filter_single_df_to_cols(escape_val, filtered_columns)

    ms_val_data=filter_single_df_to_cols(ms_val_data, filtered_columns)


    val_sets_run_once={
        "ms_val": ms_val_data_matched_allele_length,
    }

    val_sets={
        "escape_val": escape_val, 
    }

    """
    Load model
    """
    print("Load MHCPrime for ablations...")

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

    # load base model (optionally, can load model_params directly)
    MODEL_CHECKPOINTS_DIR = PROJECT_ROOT / "Model_Checkpoints"
    base_model_dir = MODEL_CHECKPOINTS_DIR / "MHCPrime_Base_for_Ablations" / "Checkpoints"
    # base_model_dir = MODEL_CHECKPOINTS_DIR / "MHCPrime_Base" / "Checkpoints"
    ckpt_path = base_model_dir / "model_final.pt"

    device = torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")

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

    """
    Create validation set caches for inference
    """
    print("Creating validation set caches for inference...")
    val_sets_run_once_cache_paths={}

    for val_set_name, val_data_set in val_sets_run_once.items():
        print(val_set_name)
        data_to_run_path=f"{DATA_DIR}/{val_set_name}_{str(gpu_id)}_cache.pt"
        build_token_cache(val_data_set, data_to_run_path, tokenizer, use_domain_id=False)
        val_sets_run_once_cache_paths[val_set_name]=data_to_run_path

    val_set_cache_paths={}

    for val_set_name, val_data_set in val_sets.items():
        print(val_set_name)
        data_to_run_path=f"{DATA_DIR}/{val_set_name}_{str(gpu_id)}_cache.pt"
        build_token_cache(val_data_set, data_to_run_path, tokenizer, use_domain_id=False)
        val_set_cache_paths[val_set_name]=data_to_run_path


    print(end="\n")

    """
    Create output directories
    """
    # optional model saving
    print("Creating output directories...")

    full_output_dir=f"{DATA_DIR}/ESCAPE_seq_CON_Step_and_Learning_Rate_Ablations_w_Alt_MHCPrime_CV_Datasplit_Cleaned_{ds_split}/"


    full_log_output_dir=f"{full_output_dir}/logs"

    # check before creating directories
    if os.path.exists(full_output_dir):
        response = input(
            f"\nWARNING: Output directory already exists:\n{full_output_dir}\n"
            "Continuing may overwrite existing outputs.\n"
            "Continue? [y/n]: "
        ).strip().lower()

        if response not in {"y", "yes"}:
            raise SystemExit("Stopped by user. Existing output directory was not modified.")

        print("Continuing with existing output directory. Existing files may be overwritten.\n")

    print("Output directory for everything:", full_output_dir)
    os.makedirs(full_output_dir, exist_ok=True)
    print("Output directory for logs:", full_log_output_dir)
    os.makedirs(full_log_output_dir, exist_ok=True)


    """
    Run model inference for the base MHCPrime model
    """

    print("Run base MHCPrime inference once on validation sets...")

    # get models
    base_model=copy.deepcopy(mhcprime)

    base_and_ft_models={
        # base model
        "mhcprime": base_model,
    }
    print(base_and_ft_models.keys())

    for val_set_name, val_data_set in val_sets_run_once.items():
        
        data_to_run_path=val_sets_run_once_cache_paths[val_set_name]

        for model_name, model in base_and_ft_models.items():
            model = model.to(device)
            model.eval()
            with torch.no_grad():
                output_scores=run_mhcprime_fast(model, data_to_run_path, batch_size=3072, num_workers=16, compile_model=False, use_domain_id=False)

            val_data_set[model_name]=output_scores.detach().cpu().tolist()

        # add df back to dictionary
        val_sets_run_once[val_set_name]=val_data_set

    for val_set_name, val_data_set in val_sets.items():
        
        data_to_run_path=val_set_cache_paths[val_set_name]
        print(val_set_name, data_to_run_path)

        for model_name, model in base_and_ft_models.items():
            model = model.to(device)
            model.eval()
            with torch.no_grad():
                output_scores=run_mhcprime_fast(model, data_to_run_path, batch_size=3072, num_workers=16, compile_model=False, use_domain_id=False)

            val_data_set[model_name]=output_scores.detach().cpu().tolist()

        # add df back to dictionary
        val_sets[val_set_name]=val_data_set

    print(end="\n")

    print("Saving initial val sets run once...")
    for val_set_name, val_data_set in val_sets_run_once.items():
        print(val_set_name)
        val_data_set.to_csv(f"{full_output_dir}/{val_set_name}.csv")
    os.listdir(full_output_dir)
    print(end="\n")

    print("Saving initial val sets...")
    for val_set_name, val_data_set in val_sets.items():
        print(val_set_name)
        val_data_set.to_csv(f"{full_output_dir}/{val_set_name}.csv")
    os.listdir(full_output_dir)
    print(end="\n")


    """
    Set base pepcl params
    """

    print("Set base parameters for PepCL...")
    # for continuous value training
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


    # get hyperparameters for search

    step_budgets = [25, 50, 100, 200, 300, 400, 600, 800]
    learning_rates = [2e-5, 5e-5, 1e-4, 2e-4]
    seeds = [0]

    ckpt_to_save_per_budget = {
        25: [5, 10, 15, 20, 25],
        50: [5, 10, 20, 30, 40, 50],
        100: [10, 20, 40, 60, 80, 100],
        200: [10, 20, 25, 50, 75, 100, 150, 200],
        300: [10, 25, 50, 75, 100, 150, 200, 250, 300],
        400: [10, 25, 50, 100, 150, 200, 250, 300, 350, 400],
        600: [10, 25, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600],
        800: [10, 25, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800],
    }


    learning_rate_strings=lr_to_strings(learning_rates)

    # num_workers=2
    num_workers=4
    # num_workers=8
    pepcl_base_params_con["num_workers"] = num_workers

    for pepcl_seed in seeds: # make this the first seed actually

        pepcl_base_params_con["seed"] = pepcl_seed

        for steps in step_budgets:

            pepcl_base_params_con["steps"] = steps
            pepcl_base_params_con["save_every_n_steps"] = ckpt_to_save_per_budget[steps] # to ensure logging

            for learning_rate, learning_rate_string in zip(learning_rates, learning_rate_strings):

                pepcl_base_params_con["encoder_lr"] = learning_rate
                pepcl_base_params_con["decoder_lr"] = learning_rate

                print(f"Seed: {str(pepcl_seed)}")
                print(f"Steps: {str(steps)}")
                print(f"LR: {str(learning_rate)}")

                model_ft = copy.deepcopy(mhcprime)
                pepcl_base_params_con["dr_lambda"]=0 # must set to 0 now to get no dr in general

                print("FT")
                # run updated pepcl function
                pepcl=PepCL_Logging_Ext(
                            model=model_ft,
                            train_df=ms_escape_train_set_matched_allele_length.query("domain_id == 1"), # contains domain id 1 only
                            use_replay = False,
                            mhc_seq_dict=mhc_seq_dict, # must now be provided
                            enable_logging=True,
                            **pepcl_base_params_con,
                )
                training_stats_ft, model_states_ft=pepcl.train() # store ft model states
                ft_logs=pepcl.get_logs(as_dataframes=True) # store logging
                ft_logs["training_stats"]=training_stats_ft


                print("PepCL w/ replay and drift regularization")
                model_pepcl_r_dr = copy.deepcopy(mhcprime)
                pepcl_base_params_con["dr_lambda"]=1e-5 # must reset here since it was run as 0 above

                # NOTE: SET NONE FOR THE PEPCL FOR NOW, ONLY DO TRACKING FOR FT
                pepcl_base_params_con["save_every_n_steps"] = [steps]

                # run updated pepcl function
                pepcl=PepCL_Logging_Ext(
                            model=model_pepcl_r_dr,
                            train_df=ms_escape_train_set_matched_allele_length, # contains both domain ids
                            use_replay = True,
                            mhc_seq_dict=mhc_seq_dict, # must now be provided
                            enable_logging=True,
                            **pepcl_base_params_con,
                )
                training_stats_pepcl_r_dr, model_states_pepcl_r_dr=pepcl.train() # store pepcl model states
                pepcl_r_dr_logs=pepcl.get_logs(as_dataframes=True) # store logging
                pepcl_r_dr_logs["training_stats"]=training_stats_pepcl_r_dr

                clear_all_gpu_memory()

                # adjust model state keys
                def add_prefix_to_keys(d, prefix):
                    return {f"{prefix}{k}": v for k, v in d.items()}

                ft_model_prefix = f"escape_con_ft_steps_{str(steps)}_lr_{learning_rate_string}_seed_{str(pepcl_seed)}_step_"
                # pepcl_r_model_prefix = f"escape_con_pepcl_r_steps_{str(steps)}_lr_{learning_rate_string}_seed_{str(pepcl_seed)}_step_"
                pepcl_r_dr_model_prefix = f"escape_con_pepcl_r_dr_steps_{str(steps)}_lr_{learning_rate_string}_seed_{str(pepcl_seed)}_step_"

                model_states_ft=add_prefix_to_keys(model_states_ft, ft_model_prefix)
                # model_states_pepcl_r=add_prefix_to_keys(model_states_pepcl_r, pepcl_r_model_prefix)
                model_states_pepcl_r_dr=add_prefix_to_keys(model_states_pepcl_r_dr, pepcl_r_dr_model_prefix)

                print(model_states_ft.keys())
                # print(model_states_pepcl_r.keys())
                print(model_states_pepcl_r_dr.keys())

                ft_model_name=ft_model_prefix.removesuffix("_step_")
                # pepcl_r_model_name=pepcl_r_model_prefix.removesuffix("_step_")
                pepcl_r_dr_model_name=pepcl_r_dr_model_prefix.removesuffix("_step_")

                
                print("Saving logs")
                save_log_dict(ft_logs, f"{full_log_output_dir}/{ft_model_name}.pkl")
                save_log_dict(pepcl_r_dr_logs, f"{full_log_output_dir}/{pepcl_r_dr_model_name}.pkl")

                # get all models to run
                model_checkpoints_for_trajectories = model_states_ft | model_states_pepcl_r_dr # removed for now: model_states_pepcl_r

                final_models_to_run = {
                    ft_model_name: model_ft,
                    # pepcl_r_model_name: model_pepcl_r,
                    pepcl_r_dr_model_name: model_pepcl_r_dr,
                }

                all_models_to_run_for_val_set=model_checkpoints_for_trajectories | final_models_to_run

                
                for val_set_name, val_data_set in val_sets_run_once.items():
                    
                    data_to_run_path=val_sets_run_once_cache_paths[val_set_name]

                    for model_name, model in final_models_to_run.items(): # NOTE: CHANGED DICT HERE
                        model = model.to(device)
                        model.eval()
                        with torch.no_grad():
                            output_scores=run_mhcprime_fast(model, data_to_run_path, batch_size=3072, num_workers=num_workers, compile_model=False, use_domain_id=False)

                        val_data_set[model_name]=output_scores.detach().cpu().tolist()

                    # add df back to dictionary
                    val_sets_run_once[val_set_name]=val_data_set

                for val_set_name, val_data_set in val_sets.items():
                    
                    data_to_run_path=val_set_cache_paths[val_set_name]
                    print(val_set_name, data_to_run_path)

                    for model_name, model in all_models_to_run_for_val_set.items(): # NOTE: CHANGED DICT HERE
                        model = model.to(device)
                        model.eval()
                        with torch.no_grad():
                            output_scores=run_mhcprime_fast(model, data_to_run_path, batch_size=3072, num_workers=num_workers, compile_model=False, use_domain_id=False)

                        val_data_set[model_name]=output_scores.detach().cpu().tolist()

                    # add df back to dictionary
                    val_sets[val_set_name]=val_data_set

                print(end="\n")

                print("Saving val sets run once...")
                for val_set_name, val_data_set in val_sets_run_once.items():
                    print(val_set_name)
                    val_data_set.to_csv(f"{full_output_dir}/{val_set_name}.csv")
                os.listdir(full_output_dir)
                print(end="\n")

                print("Saving val sets...")
                for val_set_name, val_data_set in val_sets.items():
                    print(val_set_name)
                    val_data_set.to_csv(f"{full_output_dir}/{val_set_name}.csv")
                os.listdir(full_output_dir)
                print(end="\n")

    for val_name, val_path in val_sets_run_once_cache_paths.items():
        os.remove(val_path)

    for val_name, val_path in val_set_cache_paths.items():
        os.remove(val_path)
