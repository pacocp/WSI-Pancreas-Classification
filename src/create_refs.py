import pandas as pd
import glob

labels_tcga = []
labels_gtex = []
labels_tcia = []
patient_id_tcga = []
patient_id_gtex = []
patient_id_tcia = []
paths_wsi_tcga = []
paths_wsi_gtex = []
paths_wsi_tcia = []
name_wsi_tcga = []
name_wsi_gtex = []
name_wsi_tcia = []
for path in glob.glob('../GDC/**/*.h5', recursive=True):
    patient_id = path.split('/')[-1].split('-')[:3]
    wsi_name = path.split('/')[-1]
    if '-11Z-' in path:
        label = 'Control'
    elif '-01Z-' in path:
        label = 'Tumor'
    patient_id = '-'.join(patient_id)
    labels_tcga.append(label)
    patient_id_tcga.append(patient_id)
    paths_wsi_tcga.append(path)
    name_wsi_tcga.append(wsi_name)

df = pd.DataFrame()
df['Path'] = paths_wsi_tcga
df['WSI_name'] = name_wsi_tcga
df['Patient_ID'] = patient_id_tcga
df['Label'] = labels_tcga
df.to_csv('../data/tcga_ref.csv', index=False, sep=',')

for path in glob.glob('../GTEx/**/*.h5', recursive=True):
    patient_id = path.split('/')[-1].split('-')[:2]
    wsi_name = path.split('/')[-1]
    patient_id = '-'.join(patient_id)
    labels_gtex.append('Control')
    patient_id_gtex.append(patient_id)
    paths_wsi_gtex.append(path)
    name_wsi_gtex.append(wsi_name)

df = pd.DataFrame()
df['Path'] = paths_wsi_gtex
df['WSI_name'] = name_wsi_gtex
df['Patient_ID'] = patient_id_gtex
df['Label'] = labels_gtex
df.to_csv('../data/gtex_ref.csv', index=False, sep=',')

sample_sheet_tcia = pd.read_csv('../TCIA/cohort.csv')
for path in glob.glob('../TCIA/**/*.h5', recursive=True):
    slide_id = path.split('/')[-1].split('.svs')[0]
    wsi_name = path.split('/')[-1]
    row = sample_sheet_tcia[sample_sheet_tcia['Slide_ID'] == slide_id]
    patient_id = row['Case_ID'].values
    patient_id = '-'.join(patient_id)
    label = row['Specimen_Type'].values
    if label == 'tumor_tissue':
        labels_tcia.append('Tumor')
    elif label == 'normal_tissue':
        labels_tcia.append('Control')
    
    patient_id_tcia.append(patient_id)
    paths_wsi_tcia.append(path)
    name_wsi_tcia.append(wsi_name)

df = pd.DataFrame()
df['Path'] = paths_wsi_tcia
df['WSI_name'] = name_wsi_tcia
df['Patient_ID'] = patient_id_tcia
df['Label'] = labels_tcia
df.to_csv('../data/tcia_ref.csv', index=False, sep=',')