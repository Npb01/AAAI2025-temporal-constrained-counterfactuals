
## Project Installation Guide

Welcome to the installation guide for the paper `Generating counterfactual explanations under temporal constraints`. 


We recommend the use of a virtual environment to avoid possible clashes between your local Python version and the required libraries. A virtual environment can be created with [Conda](https://www.anaconda.com/) or with the [venv](https://docs.python.org/3/library/venv.html) Python utility. 
Please follow the steps below to set up the project, including the installation of dependencies and the MONA tool.

### Step 1: Install the Package

To install the package and its dependencies, including the MONA tool, use the following commands:

```bash
pip install -r requirements.txt
pip install -e .
```

### Step 2: Handling MONA Installation Issues
To install the required python dependencies, run ```pip install -r requirements.txt``` to install the required python libraries, including the DiCE and Declare4Py packages in the local_packages directory.
The `pip install -e .` command should then automatically download, compile, and install the MONA tool and all the required packages. However, if the MONA installation fails during this process, you can manually install MONA by following these instructions:

We also provide a second setup python file, not included the automatic installation of MONA, named ```setup_no_mona.py```. This file can be used to install the package without the MONA tool.
Simply run ```pip install -e . --no-build-isolation --config-settings setup-file=setup_no_mona.py``` to install the required dependencies without MONA.```
#### Option 1: Manual Installation of MONA

1. **Download MONA**:
   - Visit the following page: [MONA Download](https://www.brics.dk/mona/download.html)
   - Download the latest MONA tarball from the provided links.

2. **Extract and Install MONA**:
   - Extract the tarball using the command:
     ```bash
     tar -xvzf mona-<version>.tar.gz
     ```
   - Navigate to the extracted directory:
     ```bash
     cd mona-<version>
     ```
   - Configure, compile, and install MONA:
     ```bash
     ./configure
     make
     sudo make install
     ```

   Replace `<version>` with the appropriate version number downloaded.

3. **Verify Installation**:
   - Ensure MONA is installed correctly by running the following command:
     ```bash
     mona --help
     ```
   - If the help message appears, MONA is installed successfully.

#### Option 2: Use an Alternative Download Source

If the primary download link for MONA is not working, you can try downloading it from an alternative link provided on the [MONA Download Page](https://www.brics.dk/mona/download.html). Follow the same steps as above to install and verify MONA.


### Step 3: Dataset download and experiment execution
 - To download the 3 datasets used in this paper, due to the 50MB limit on the supplementery material, please download them from the following link:
   - [Drive datasets link](https://drive.google.com/file/d/1pFZVNgPZibwGPwqLoNC-M-8KAA-FZff2/view?usp=drive_link)
 - To run the experiments, simply run ```python run_ltlf_cf_pipeline.py``` in the root directory of the project. This will run the experiments for the 3 datasets and the 4 methods, generating the results in the `results` folder.


### Step 4: Analysis of results
The jupyter notebook included in the folder details all of the analysis performed for the paper