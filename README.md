# Performance comparison between multi-center histopathology datasets of a weakly-supervised deep learning model for pancreatic ductal adenocarcinoma detection

*Francisco Carrillo-Perez, Francisco M. Ortuno, Alejandro Börjesson, Ignacio Rojas, and Luis Javier Herrera*

[Journal link](https://cancerimagingjournal.biomedcentral.com/articles/10.1186/s40644-023-00586-3)

#### Abstract
Pancreatic ductal carcinoma patients have a really poor prognosis given its difficult early detection and the lack of early symptoms. Digital pathology is routinely used by pathologists to diagnose the disease. However, visually inspecting the tissue is a time-consuming task, which slows down the diagnostic procedure. With the advances occurred in the area of artificial intelligence, specifically with deep learning models, and the growing availability of public histology data, clinical decision support systems are being created. However, the generalization capabilities of these systems are not always tested, nor the integration of publicly available datasets for pancreatic ductal carcinoma detection (PDAC).

In this work, we explored the performace of two weakly-supervised deep learning models using the two more widely available datasets with pancreatic ductal carcinoma histology images, The Cancer Genome Atlas Project (TCGA) and the Clinical Proteomic Tumor Analysis Consortium (CPTAC). In order to have sufficient training data, the TCGA dataset was integrated with the Genotype-Tissue Expression (GTEx) project dataset, which contains healthy pancreatic samples.

We showed how the model trained on CPTAC generalizes better than the one trained on the integrated dataset, obtaining an inter-dataset accuracy of 90.62% ± 2.32 and an outer-dataset accuracy of 92.17% when evaluated on TCGA + GTEx. Furthermore, we tested the performance on another dataset formed by tissue micro-arrays, obtaining an accuracy of 98.59%. We showed how the features learned in an integrated dataset do not differentiate between the classes, but between the datasets, noticing that a stronger normalization might be needed when creating clinical decision support systems with datasets obtained from different sources. To mitigate this effect, we proposed to train on the three available datasets, improving the detection performance and generalization capabilities of a model trained only on TCGA + GTEx and achieving a similar performance to the model trained only on CPTAC.

The integration of datasets where both classes are present can mitigate the batch effect present when integrating datasets, improving the classification performance, and accurately detecting PDAC across different datasets.

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

## Citation

If you find this work useful please cite it as follows:

```
Carrillo-Perez, F., Ortuno, F.M., Börjesson, A. et al. Performance comparison between multi-center
histopathology datasets of a weakly-supervised deep learning model
for pancreatic ductal adenocarcinoma detection. Cancer Imaging 23, 66 (2023). https://doi.org/10.1186/s40644-023-00586-3
```
