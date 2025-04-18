# ST311_group_project

## Instructions 

### 1. Register for a HuggingFace account

If you haven't already, create a HuggingFace account [here](https://huggingface.co/join). We will need it later. Go to step 2 if you already have an account. 

### 2. Pull the latest changes (Obviously) 

Use `git pull` or Github Desktop to pull the latest changes to this repository. 

### 3. Set up Python environment

Create a new conda environment for this project: 
```bash
conda create -n st311 python=3.11
```

Activate environment: 
```bash
conda activate st311 
```

Install required dependencies (make sure you're in the root directory of the project): 
```bash
pip install -r requirements.txt 
```

### 4. Create `.env` file

Create a file `.env` at the root level of the project. 

Copy the contents in `.env.sample` exactly into `.env` (You do not need to make any changes to `.env` for now as we are not using any private tokens yet) 

### 5. Preprocess LongBench dataset

Run `notebooks/NB0-benchmarks_preprocessing.ipynb` to generate a filtered `longbench_filtered.csv` in the `data/` folder. The data folder is gitignored, so any data will not be tracked by git. 
