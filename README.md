# Temporal-Constrained Counterfactuals

This is an improved fork of the original repository for generating counterfactual explanations under temporal constraints. In this fork, we attempted to replicate the experiments in the paper using the provided code and implemented an enhanced pipeline structure.

## Overview

This project generates temporal counterfactual explanations for predictive process monitoring using Linear Temporal Logic over Finite traces (LTLf) constraints. We've implemented improvements to the pipeline structure and conducted experiments on multiple datasets.

## Table of Contents

- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Installing UV Package Manager](#installing-uv-package-manager)
  - [Installing Dependencies](#installing-dependencies)
  - [Installing MONA](#installing-mona)
  - [Datasets](#datasets)
- [Running Experiments](#running-experiments)
  - [Quick Start - Full Original Pipeline](#quick-start---(mostly)-original-full-pipeline)
- [New Pipeline Implementation](new-(unfinished)-pipeline-structure)
  - [New Individual Pipeline Components](#individual-pipeline-components)
- [Documentation](#documentation)

## Installation

### Prerequisites

- Python 3.8 or higher
- WSL (Windows Subsystem for Linux) - Required for MONA installation on Windows
- Git

### Installing UV Package Manager

We recommend using [Astral's UV](https://github.com/astral-sh/uv) package manager for fast and reliable dependency management.

#### Install UV

**On Windows (PowerShell):**
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**On macOS/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

After installation, restart your terminal or source your shell configuration.

### Installing Dependencies

Once UV is installed, you can install all project dependencies using our improved `requirements.txt` file:

```bash
# Create a virtual environment (optional but recommended)
uv venv

# Activate the virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install all dependencies
uv pip install -r requirements.txt

# Install the project in development mode
uv pip install -e .
```

**Alternative (using pip):**
```bash
pip install -r requirements.txt
pip install -e .
```

### Installing MONA

MONA (Monadic second-order logic of One or two successors) is required for LTLf formula processing. The automatic installation during `pip install -e .` may fail on some systems.

#### Manual MONA Installation (Recommended for Windows)

The original MONA installation guide did not work reliably. In this guide, the authors proposed downloading it through the requirements file or downloading it manually from the [MONA Website](https://www.brics.dk/mona/index.html). Here's the updated procedure that worked for us:

**On Windows (using WSL):**

[Windows WSL Guide](https://learn.microsoft.com/en-us/windows/wsl/install)
1. **Install WSL** (if not already installed):
   ```powershell
   wsl --install
   ```
   We used the Ubuntu distribution.
   Restart your computer if this is your first time installing WSL.

2. **Open WSL terminal** and install MONA:
   ```bash
   # Update package lists
   sudo apt-get update
   
   # Install dependencies
   sudo apt-get install build-essential flex bison
   
   # Download MONA (version 1.4)
   wget http://www.brics.dk/mona/download/mona-1.4-18.tar.gz
   
   # Extract
   tar -xzf mona-1.4-18.tar.gz
   cd mona-1.4
   
   # Configure, compile, and install
   ./configure
   make
   sudo make install
   
   # Verify installation
   mona --version
   ```

#### Alternative: Install without MONA

If you encounter issues with MONA, you can install the package without it:

```bash
pip install -e . --no-build-isolation --config-settings setup-file=setup_no_mona.py
```

Note: Some LTLf-related features will not be available without MONA.

### Datasets
- To download the 3 datasets used in this paper, due to the 50MB limit on the supplementary material, please download them from the following link:
   - [Drive datasets link](https://drive.google.com/file/d/1pFZVNgPZibwGPwqLoNC-M-8KAA-FZff2/view?usp=drive_link)
- Make sure to save the datasets inside the repository folder in separate folders for each dataset; BPIC17_O_ACCEPTED/full.xes, bpic2012_O_ACCEPTED-COMPLETE/full.xes, synthetic_data/full.xes

The experiments use one synthetic dataset and 2 real-world event log datasets:

1. **BPIC 2017** - Dutch loan application process
   - Original dataset: `BPIC17_O_ACCEPTED/full.xes`
   - Sampled version: `sampled_logs/` (created by `create_subsample_bpi2017.py`)
   - Prefix lengths tested: 15, 20, 25, 30

2. **BPIC 2012** - Dutch financial institution loan application
   - Dataset: `bpic2012_O_ACCEPTED-COMPLETE/full.xes`
   - Prefix lengths tested: 20, 25, 30, 35

3. **Synthetic Data** - Insurance claim management process
   - Dataset: `synthetic_data/full.xes`
   - Prefix lengths tested: 7, 9, 11, 13
  
### Dataset Configuration

Datasets are configured in `dataset_confs.py` with specifications for:
- Timestamp columns
- Activity columns
- Case ID columns
- Labeling strategies

## Running Experiments

### Quick Start - Full (mostly) Original Pipeline

To reproduce the experiments from our paper on the BPIC17 dataset:

1. **Create the subsampled dataset:**
   ```bash
   python create_subsample_bpi2017.py
   ```
   
   This script creates a stratified subsample of the BPIC17 dataset with 5,000 cases, split into train/validation/test sets. The sampled data will be saved in the `sampled_logs/` directory.

2. **Run the complete experiment pipeline:**
   ```bash
   python run_ltlf_cf_pipeline.py
   ```
   
   This runs the full pipeline, including:
   - Model training for different prefix lengths
   - LTLf conformance checking at multiple complexity levels (10%, 25%, 50%)
   - Counterfactual generation with various configurations
   - Results saved to `results/` directory


## New (unfinished) Pipeline Structure
### Individual Pipeline Components

Our improved pipeline structure allows you to run individual components separately for more flexibility:

#### 1. Train Predictive Model

Train a predictive model for outcome prediction:

```bash
python train_model.py --dataset <dataset_name> --prefix <prefix_length> [OPTIONS]
```

**Required Arguments:**
- `--dataset`: Dataset name (choices: `sampled_logs`, `BPIC17_O_ACCEPTED`, `bpic2012_O_ACCEPTED-COMPLETE`, `synthetic_data`)
- `--prefix`: Prefix length (integer, e.g., 15, 20, 25, 30)

**Optional Arguments:**
- `--seed`: Random seed for reproducibility (default: 42)
- `--no-save`: Don't save the trained model

**Example:**
```bash
# Train model for sampled_logs dataset with prefix length 20
python train_model.py --dataset sampled_logs --prefix 20

# Train without saving the model
python train_model.py --dataset BPIC17_O_ACCEPTED --prefix 25 --no-save
```

**Output:** Trained model saved to `models/{dataset}_prefix{prefix}_model.pkl`

#### 2. Generate Counterfactuals

Generate counterfactual explanations using a pre-trained model:

```bash
python generate_counterfactuals.py --dataset <dataset_name> --prefix <prefix_length> [OPTIONS]
```

**Required Arguments:**
- `--dataset`: Dataset name
- `--prefix`: Prefix length

**Optional Arguments:**
- `--ltl-percentage`: LTLf model complexity (choices: `10%`, `25%`, `50%`; default: `50%`)
- `--model-file`: Path to pre-trained model file (default: `models/{dataset}_prefix{prefix}_model.pkl`)
- `--seed`: Random seed (default: 42)

**Example:**
```bash
# Generate CFs with 50% LTL complexity
python generate_counterfactuals.py --dataset sampled_logs --prefix 20

# Generate CFs with custom model and 25% LTL complexity
python generate_counterfactuals.py --dataset sampled_logs --prefix 20 --ltl-percentage 25% --model-file models/my_custom_model.pkl
```

**Output:** Results saved to `results/{dataset}/{ltl_percentage}/`

#### 3. Run Full Pipeline for Multiple Configurations

Run the complete pipeline across multiple datasets, prefix lengths, and LTL percentages:

```bash
python run_full_pipeline.py [OPTIONS]
```

**Optional Arguments:**
- `--datasets`: Space-separated list of datasets to run (default: all)
- `--ltl-percentages`: Space-separated list of LTL percentages (choices: `10%`, `25%`, `50%`; default: `50%`)
- `--seed`: Random seed (default: 42)

**Examples:**
```bash
# Run all datasets with all LTL percentages
python run_full_pipeline.py --ltl-percentages 10% 25% 50%

# Run specific datasets with 50% LTL complexity
python run_full_pipeline.py --datasets sampled_logs BPIC17_O_ACCEPTED

# Run with custom seed
python run_full_pipeline.py --seed 123 --ltl-percentages 25%
```

**Output:** Results for all configurations in `results/` directory

## Documentation

For detailed information about the project structure, pipeline implementation, and future improvements, please visit our **[Wiki](https://github.com/Npb01/AAAI2025-temporal-constrained-counterfactuals/wiki)**.

The wiki includes:
- **Pipeline Architecture** - Detailed explanation of the new modular pipeline structure
- **Original Setup** - Documentation of the baseline implementation
- **Future Improvements** - Planned enhancements and known limitations

## Project Structure

```
.
├── create_subsample_bpi2017.py    # Create stratified subsample of BPIC17 (new)
├── run_ltlf_cf_pipeline.py        # Run complete experiment (original)
├── run_full_pipeline.py           # Run pipeline for all configs (new)
├── train_model.py                 # Train predictive models (new)
├── generate_counterfactuals.py    # Generate CFs standalone (new)
├── requirements.txt               # Python dependencies
├── dataset_confs.py               # Dataset configurations
├── setup.py                       # Package installation
├── setup_no_mona.py              # Installation without MONA
│
├── src/                          # Source code
│   ├── pipeline/                 # New modular pipeline
│   │   ├── config.py
│   │   ├── data_pipeline.py
│   │   ├── model_pipeline.py
│   │   ├── ltl_pipeline.py
│   │   └── cf_pipeline.py
│   ├── encoding/                 # Feature encoding
│   ├── predictive_model/         # ML models
│   ├── explanation/              # CF generation
│   └── evaluation/               # Metrics
│
├── local_packages/               # Local dependencies
│   ├── Declare4Py/              # LTLf processing
│   └── DiCE/                    # Counterfactual engine
│
├── models/                       # Saved models
├── results/                      # Experimental results
└── sampled_logs/                # Subsampled datasets
```

## Key Features

- **Improved Pipeline Architecture**: Modular design with separate components for data processing, model training, LTL parsing, and counterfactual generation
- **Enhanced Predictive Models**: XGBoost-based models with hyperparameter optimization
- **LTLf Conformance Checking**: Integration with Declare4Py for temporal constraint verification
- **Multiple CF Strategies**: Baseline genetic algorithm and adaptive heuristics (APRIORI, ONLINE, Mutate-and-Retry)
- **Comprehensive Evaluation**: Metrics for validity, proximity, sparsity, and temporal consistency

## Results Analysis

Analysis of experimental results can be found in:
- 
- `results_analysis.ipynb` - Jupyter notebook with visualizations and statistical analysis
- `results/` directory - Raw experimental outputs organized by dataset and configuration

## LTLp Formulas for Each Dataset

For each dataset, we used different Linear Temporal Logic over Process Traces (LTLp) formulas to check coverage at 10%, 25%, and 50%. Below are the specific formulas used for each dataset:

### BPIC2012 Dataset

- **10%:** F(osentcomplete) ∧ G(osentcomplete → ¬(aacceptedcomplete) U (wcompleterenaanvraagcomplete)) ∧ F(osentbackcomplete)


- **25%:** F(osentcomplete) ∧ G(osentcomplete → ¬(aacceptedcomplete) U (wcompleterenaanvraagcomplete)) ∧ F(osentbackcomplete) ∧ G(wcompleterenaanvraagstart → F(aacceptedcomplete)) ∧ (F(wnabellenoffertesstart) ∧ F(wnabellenoffertescomplete)) ∧ (F(oselectedcomplete) ∨ F(wvaliderenaanvraagstart))


- **50%:** F(osentcomplete) ∧ G(osentcomplete → ¬(aacceptedcomplete) U (wcompleterenaanvraagcomplete)) ∧ G(wcompleterenaanvraagschedule → F(wcompleterenaanvraagstart)) ∧ (F(wnabellenoffertesstart) ∨ F(wnabellenoffertescomplete)) ∧ (F(oselectedcomplete) ∨ F(wvaliderenaanvraagstart)) ∧ asubmittedcomplete ∧ F(oselectedcomplete ∨ apartlysubmittedcomplete) ∧ G(ocreatedcomplete → F(osentbackcomplete)) ∧ F(afinalizedcomplete) ∨ F(apreacceptedcomplete) ∨ F(wafhandelenleadscomplete)


### BPIC17 Dataset

- **10%:** acreateapplication ∧ ¬(aconcept) U (wcompleteapplication)


- **25%:** acreateapplication ∧ ¬(aconcept) U (wcompleteapplication) ∧ (F(ocreateoffer) → F(wcallafteroffers)) ∧ F(wcompleteapplication)


- **50%:** acreateapplication ∧ ¬(aconcept) U (wcompleteapplication) ∧ G(ocreateoffer → (F(wcallafteroffers) ∨ F(wvalidateapplication))) ∧ (F(ocreated) → X(osentmailandonline ∨ osentonlineonly)) ∧ G((aincomplete ∨ apending) → (X(wcallincompletefiles) ∧ F(wvalidateapplication)))


### Claim Management Dataset

- **10%:** G(contacthospital → X(acceptclaim ∨ rejectclaim))

- **25%:** G(contacthospital → X(acceptclaim ∨ rejectclaim)) ∧ F(createquestionnaire)

- **50%:** (F(contacthospital) → F(highinsurancecheck)) ∧ G(preparenotificationcontent → X(sendnotificationbyphone ∨ sendnotificationbypost)) ∧ G(createquestionnaire → F(preparenotificationcontent)) ∧ register

## Requirements

Key dependencies include:
- Python 3.8+
- pm4py - Process mining library
- xgboost - Gradient boosting framework
- scikit-learn - Machine learning utilities
- pandas, numpy - Data manipulation
- declare4py - LTLf formula processing
- dice-ml - Counterfactual generation
- MONA - Automata toolkit (external)

See `requirements.txt` for a complete list with versions.

## Troubleshooting

### MONA Installation Issues
- **Windows**: Ensure WSL is properly installed and updated
- **macOS**: May need to install Xcode Command Line Tools: `xcode-select --install`
- **Linux**: Ensure build-essential, flex, and bison are installed

### Import Errors
- Verify virtual environment is activated
- Reinstall with: `uv pip install -r requirements.txt --force-reinstall`
- Check Python version: `python --version` (should be 3.8+)

### Memory Issues
- Large datasets may require 16GB+ RAM
- Consider reducing sample size in `create_subsample_bpi2017.py`
- Use smaller prefix lengths for initial testing

## Contributing

This is a research project for the Seminar Process Analytics (2IMI00) by two MSc Data Science and Artificial AI students at the TUe. For questions or contributions, please open an issue on GitHub.


**For more detailed documentation, visit the [Wiki](https://github.com/Npb01/AAAI2025-temporal-constrained-counterfactuals/wiki).**
