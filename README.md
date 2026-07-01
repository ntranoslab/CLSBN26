# PepCL Reproduction Repository

This repository contains the manuscript-specific code, notebooks, analysis scripts, figure-generation notebooks, and reproduction materials associated with the PepCL peptide–MHC class I continual learning study.

This repository is intended primarily as a **manuscript reproduction repository**. It is not structured as a pip-installable Python package. Packaged, reusable versions of the main software components are available separately:

* **MHCPrime**: https://github.com/ntranoslab/MHCPrime
* **PepCL**: https://github.com/ntranoslab/PepCL

Users who want to run the packaged base model or PepCL update workflow directly should use the MHCPrime and PepCL repositories linked above. This manuscript repository contains the broader analysis and experiment code used to generate the results reported in the manuscript.

## Repository structure

```text
CLSBN26/
├── Analysis/
├── src/
├── Figures/
├── LICENSE
└── README.md
```

### `Analysis/`

The `Analysis/` directory contains scripts and notebooks used for manuscript analyses, including data collection, data processing, model training, PepCL updating experiments, benchmarking, interpretability analyses, and other experiments discussed in the manuscript.

### `src/`

The `src/` directory contains project code used by the manuscript analyses, including model definitions, training utilities, PepCL-related code, metric functions, preprocessing utilities, and analysis helpers.

For general use, the packaged versions of the model and PepCL framework should be accessed through the standalone MHCPrime and PepCL repositories:

* https://github.com/ntranoslab/MHCPrime
* https://github.com/ntranoslab/PepCL

### `Figures/`

The `Figures/` directory contains notebooks and scripts used to generate manuscript figures and supplementary figures. It also contains raw figure outputs and associated figure-level data where applicable.

## Data and model checkpoints

Two large directories are required for full manuscript reproduction but are not included directly in this GitHub repository:

```text
Data/
Model_Checkpoints/
```

These folders can be downloaded from the following Google Drive folder:

```text
https://drive.google.com/drive/folders/1Onb7cGkSlsv4r32lKCHy-moYwGKu9Lim?usp=sharing
```

The `Data/` directory is required for figure generation and manuscript analyses. The `Model_Checkpoints/` directory is required for analyses that load trained or updated model checkpoints.

After downloading, place both folders at the top level of the repository. The expected top-level layout for full reproduction is therefore:

```text
CLSBN26/
├── Analysis/
├── src/
├── Figures/
├── Data/                 # must be downloaded
├── Model_Checkpoints/    # must be downloaded
├── LICENSE
└── README.md
```

## Recommended usage

This repository is intended for reproducing manuscript analyses rather than as the primary user-facing software interface.

For most users:

1. Use **MHCPrime** to load and run the packaged base peptide-MHC model.
2. Use **PepCL** to run packaged continual learning updates and quickstart workflows.
3. Use this manuscript repository to inspect or reproduce the analyses, training runs, update experiments, and figure-generation workflows described in the manuscript.

## Environment

The original analyses were developed in a Python/Conda environment with GPU-enabled PyTorch. All training and updating experiments were performed and designed for a single NVIDIA H100.

## Citation

--
