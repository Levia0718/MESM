# Python Environment

- Base environment: Python 3.10.1, torch 2.2.1+cu121, torch_scatter 2.1.2+pt22cu121, torch-geometric 2.6.1, biopython 1.80
- Others can see [requirements.txt](requirements.txt)
```
conda create -n MESM python=3.10
conda activate MESM
pip install -r requirements.txt
```

# Dataset

You can get the pretrained data and four datasets(SHS27k, SHS148k, SYS30k, and SYS60k) from the [Google Drive](https://drive.google.com/drive/folders/1j7Y0IjkFmfavnPsTEvc8ehuFdsqX2kF5?usp=sharing).  
The raw_data folder contains the original data and the processed_data folder contains the filtered data with PDB.
- Yeast_AF2DB: PDB files of Yeast
- STRING_AF2DB: PDB files of STRING
- protein.actions.SHS27k.txt: PPI network of SHS27k
- protein.SHS27k.sequences.dictionary.tsv: Sequence of Proteins in SHS27k
## Multimodal Protein Pre-training

### Preparation before pre-training

If you want to perform pre-training, you need to place the following folders in the designated location:
- '/STRING_AF2DB' and '/Yeast_AF2DB' should be put in 'MESM/datasets/raw_data'.

### Run the following py files in order

1. `python utils/data/sequence/__init__.py`
2. `python utils/data/structure/__init__.py`
3. `python utils/data/pointcloud/__init__.py`
4. `python scripts/pretrain/train_sequence.py`
5. `python scripts/pretrain/train_structure.py`
6. `python scripts/pretrain/train_pointcloud.py`
7. `python utils/data/fusion/__init__.py`
8. `python scripts/pretrain/train_fusion.py`
9. `python utils/data/multimodal_representation/__init__.py`
10. `python datasets/protein_data/multimodal_protein_representations/save_pickle.py`

## Preparation before training

After placing the following two files, you can proceed to train or test the model:
- 'protein.actions.SHS27k.txt' should be put in 'MESM/datasets/processed_data_SHS27k'.
- 'all_protein_STRING.pickle' and 'all_protein_Yeast.pickle' should be put in 'MESM/datasets/protein_data/multimodal_protein_representations'.(These are pre-trained files.)

# Usage instructions

## Training

```
python train.py --dataset SHS27k --split_mode bfs
```
- `--dataset` can be SHS27k, SHS148k, SYS30k or SYS60k
- `--split_mode` can be bfs, dfs, or random

## Testing

```
python test.py --dataset SHS27k --split_mode bfs --model_save_path xxx
```
- `--dataset` can be SHS27k, SHS148k, SYS30k or SYS60k
- `--split_mode` can be bfs, dfs, or random
- `--model_save_path` xxx is the name of the folder where your trained model parameter files and training logs are saved, such as a folder xxx under the path results/SHS27k/bfs/xxx