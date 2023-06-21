# Performance comparison between multi-center histopathology datasets of a weakly-supervised deep learning model for pancreatic ductal adenocarcinoma detection

Studying the performance differences between different deep learning models trained on multiple histopathology databases for pancreatic ductal carcinoma vs control classification.

## Reference files

Reference files must be created for each datasets with the following columns:

| Path | WSI_name | Patient_ID | Label | n_patches |
|------|----------|------------|-------|-----------|

In the case of the MHMC dataset, the n_patches column can be ommited:


| Path | WSI_name | Patient_ID | Label |
|------|----------|------------|-------|

See the [data folder](data/) for concrete examples.

## Model

The model presented is based on a Resnet-50 pre-trained on Imagenet. The features from 100 tiles are extracted and then fused inside the model. Please refer to [wsi_model.py](src/wsi_model.py) for implementation details.

## Usage

### Creating the tile databases

```python

python3 patch_gen_grid.py --wsi_path ../TCGA/ --patch_path ../TCGA/TCGA_256x256/ --mask_path ../TCGA/TCGA_Masks/ --patch_size 256 --max_patches_per_slide 4000
python3 patch_gen_grid.py --wsi_path ../TCIA/ --patch_path ../TCIA/TCIA_256x256/ --mask_path ../TCIA/TCIA_Masks/ --patch_size 256 --max_patches_per_slide 4000
python3 patch_gen_grid.py --wsi_path ../GTEX/ --patch_path ../GTEX/GTEX_256x256/ --mask_path ../GTEX/GTEX_Masks/ --patch_size 256 --max_patches_per_slide 4000

```

### Creating ref files

```python

python3 create_refs.py

```

### Performing the independent patient-wise stratified 10-Fold CV on TCIA and TCGA+GTEx:

```python

python3 main.py --path_csv ../data/tcia_ref.csv --save_dir runs/tcia_kfold --train --batch_size 4 --lr 1e-3 --bag_size 100 --max_patch_per_wsi 200 --log 0 --flag tcia_kfold
python3 main.py --path_csv ../data/tcga_gtex_data.csv --save_dir runs/tcgagtex_kfold --train --batch_size 4 --lr 1e-3 --bag_size 100 --max_patch_per_wsi 200 --log 0 --flag tcgagtex_kfold

```

## Performing full-training on TCIA and TCGA+GTEx

```python

python3 main.py --path_csv ../data/tcia_ref.csv --save_dir runs/tcia_fulltrain --fulltrain --batch_size 4 --lr 1e-3 --bag_size 100 --max_patch_per_wsi 200 --log 0 --flag tcia_fulltrain
python3 main.py --path_csv ../data/tcga_gtex_data.csv --save_dir runs/tcgagtex_fulltrain --fulltrain --batch_size 4 --lr 1e-3 --bag_size 100 --max_patch_per_wsi 200 --log 0 --flag tcgagtex_fulltrain

``` 

## Evaluating fully-trained models on the different databases

```python

# tcga-gtex on tcia
python3 main.py --path_csv ../data/tcia_ref.csv --save_dir runs/tcga_gtex_on --evaluate --batch_size 4 --lr 1e-3 --bag_size 100 --max_patch_per_wsi 100 --log 0 --flag tcgagtex_on_tcia --checkpoint runs/tcgagtex_fulltrain/model_dict_best.pt

# tcia on tcga-gtex
python3 main.py --path_csv ../data/tcga_gtex_data.csv --save_dir runs/tcia_on_tcgagtex --evaluate --batch_size 4 --lr 1e-3 --bag_size 100 --max_patch_per_wsi 100 --log 0 --flag tcia_on_tcgagtex --checkpoint runs/tcia_fulltrain/model_dict_best.pt

# tcia on mhmc
python3 main.py --path_csv ../data/mhmc_ref.csv --png --save_dir runs/tcia_on_mhmc --evaluate --batch_size 4 --lr 1e-3 --bag_size 100 --max_patch_per_wsi 100 --log 0 --flag tcia_on_mhmc --checkpoint runs/tcia_fulltrain/model_dict_best.pt

# tcga-gtex on mhmc
python3 main.py --path_csv ../data/mhmc_ref.csv --png --save_dir runs/tcgagtex_on_mhmc --evaluate --batch_size 4 --lr 1e-3 --bag_size 100 --max_patch_per_wsi 100 --log 0 --flag tcia_on_mhmc --checkpoint runs/tcgagtex_fulltrain/model_dict_best.pt

```
## CPTAC k-Fold per country of origin

```python
python3 main.py --path_csv ../data/tcia_ref.csv --save_dir runs/tcia_kfold_country --country --batch_size 4 --lr 1e-3 --bag_size 100 --max_patch_per_wsi 200 --log 0 --flag tcia_kfold_patient_country
```
