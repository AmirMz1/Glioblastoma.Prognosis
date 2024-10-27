from octavecgan import *
# from attentionUnet import AttentionUNet
from octaveunet import OctaveUNet

import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision.utils as vutils

import xml.etree.ElementTree as ET
import csv

# import SimpleITK as sitk
import numpy as np, os, pandas as pd, cv2
# from radiomics import featureextractor
from skimage.measure import regionprops, shannon_entropy
from scipy.stats import kurtosis, entropy
from skimage.feature import hog

# parser = argparse.ArgumentParser()
# parser.add_argument('--t1_TCGA', required=True, type=str)
# parser.add_argument('--falir_TCGA', required=True, type=str)

# args = parser.parse_args()


def load_checkpoint(netG, optimizerG):
    print(f"Loading generator checkpoint")
    netG.load_state_dict(torch.load('weights/best_octave_unet_generator.pth', map_location=torch.device(device)))
    segmet_mdoel.load_state_dict(torch.load('weights/unet_best.pth', map_location=torch.device(device)))


def extract_radiomics_features(image, mask):
    # sitk_image = sitk.GetImageFromArray(image)
    # sitk_mask = sitk.GetImageFromArray(mask)
    features = {}

    mask = mask.astype(int)
    props = regionprops(mask)

    features['Centroid1'], features['Centroid2'], features['Centroid3'] = props[0]['Centroid']
    features['MajorAxisLength'] = props[0]['major_axis_length']
    # features['MinorAxisLength'] = props[0]['minor_axis_length']

    features['Extent'] = props[0]['extent']
    features['Diameter'] = props[0]['equivalent_diameter']
    features['Eigen1'], features['Eigen2'], features['Eigen3'] = props[0]['inertia_tensor_eigvals']
    features['Solidity'] = props[0]['solidity']
    Vectors = props[0]['inertia_tensor']
    FirstAxis = Vectors[0]
    Denominator = np.power((Vectors[0][0]),2)+np.power((Vectors[0][1]),2)+ np.power((Vectors[0][2]),2)
    Denominator = np.sqrt(Denominator)
    features['FirstAxisLength'] = Denominator
    features['FirstAxis1'],features['FirstAxis2'], features['FirstAxis3'] = FirstAxis/Denominator

    SecondAxis = Vectors[1]
    Denominator = np.power((Vectors[1][0]),2)+np.power((Vectors[1][1]),2)+ np.power((Vectors[1][2]),2)
    Denominator = np.sqrt(Denominator)
    features['SecondAxisLength'] = Denominator
    features['SecondAxis1'], features['SecondAxis2'], features['SecondAxis3'] = SecondAxis/Denominator

    ThirdAxis = Vectors[2]
    Denominator = np.power((Vectors[2][0]),2)+np.power((Vectors[2][1]),2)+ np.power((Vectors[2][2]),2)
    Denominator = np.sqrt(Denominator)
    features['ThirdAxisLength'] = Denominator
    features['ThirdAxis1'], features['ThirdAxis2'], features['ThirdAxis3'] = ThirdAxis/Denominator

    # features['kurt'] = (kurtosis(kurtosis(kurtosis(image, fisher=True))))
    histo=0
    i=0
    for i in range(image.shape[0]):
        fd, hog_image = hog(image[i,:,:], orientations=8, pixels_per_cell=(16, 16),cells_per_block=(1, 1), visualize=True)
        #print(np.sum(hog_image))
        histo = histo+np.sum((hog_image))
    features['histo'] = histo/1000
    hemorrhage  = np.sum(np.sum(np.sum(image)))
    features['hemorrhage'] = hemorrhage/135442
    
    def fractal_dimension(Z, threshold=0.9):

        def boxcount(Z, k):
            S = np.add.reduceat(
                np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
                                    np.arange(0, Z.shape[1], k), axis=1)

            return len(np.where((S > 0) & (S < k*k))[0])


        Z = (Z < threshold)


        p = min(Z.shape)

        n = 2**np.floor(np.log(p)/np.log(2))

        n = int(np.log(n)/np.log(2))

        sizes = 2**np.arange(n, 1, -1)

        counts = []
        for size in sizes:
            counts.append(boxcount(Z, size))
        if len(counts) > 1:  # Avoid warnings from np.polyfit
            coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
            return -coeffs[0]
        return None


    features['FractalDim'] = fractal_dimension(image)
    features['Entropy'] = shannon_entropy(image, base=2)

    return features
    
def read_clinical(file_path):

    # Define namespaces
    namespaces = {
        "clin_shared": "http://tcga.nci/bcr/xml/clinical/shared/2.7",
        "shared": "http://tcga.nci/bcr/xml/shared/2.7",
        "admin": "http://tcga.nci/bcr/xml/clinical/admin/2.7",
        "gbm": "http://tcga.nci/bcr/xml/clinical/gbm/2.7"
        # namespaces={"shared": "http://tcga.nci/bcr/xml/shared/2.7"}
    }

    # Parse the XML file
    tree = ET.parse(file_path)
    root = tree.getroot()

    # Define CSV headers
    headers = [
        "patient_id", "bcr_patient_barcode", "tumor_tissue_site", "histological_type", "gender", "vital_status", 
        "days_to_birth", "days_to_death", "race", "age_at_initial_diagnosis", "year_of_initial_diagnosis", 
        "tumor_status", "karnofsky_performance_score"
    ]
    patient_info = {}
    for patient in root.findall(".//gbm:patient", namespaces):
            patient_info["patient_id"] = patient.findtext(".//shared:patient_id", namespaces=namespaces)
            patient_info["bcr_patient_barcode"] = patient.findtext(".//shared:bcr_patient_barcode", namespaces=namespaces)
            patient_info["tumor_tissue_site"] = patient.findtext(".//clin_shared:tumor_tissue_site", namespaces=namespaces)
            patient_info["histological_type"] = patient.find(".//shared:histological_type", namespaces=namespaces).text
            patient_info["gender"] = patient.find(".//shared:gender", namespaces=namespaces).text
            patient_info["vital_status"] = patient.findtext(".//clin_shared:vital_status", namespaces=namespaces)
            patient_info["days_to_birth"] = patient.findtext(".//clin_shared:days_to_birth", namespaces=namespaces)
            patient_info["days_to_death"] = patient.findtext(".//clin_shared:days_to_death", namespaces=namespaces)
            patient_info["race"] = patient.findtext(".//clin_shared:race_list/clin_shared:race", namespaces=namespaces)
            patient_info["age_at_initial_diagnosis"] = patient.findtext(".//clin_shared:age_at_initial_pathologic_diagnosis", namespaces=namespaces)
            patient_info["year_of_initial_diagnosis"] = patient.findtext(".//clin_shared:year_of_initial_pathologic_diagnosis", namespaces=namespaces)
            patient_info["tumor_status"] = patient.findtext(".//clin_shared:person_neoplasm_cancer_status", namespaces=namespaces)
            patient_info["karnofsky_performance_score"] = patient.findtext(".//clin_shared:karnofsky_performance_score", namespaces=namespaces)

    return patient_info


# def read_cnv_data(path):
#     data = pd.read_csv(path, sep="\t")
#     cnv_filtered = []

#     # Loop through each row and create an entry for each corresponding value
#     for start, end, num_probes, segment_mean in zip(data['Start'], data['End'], data['Num_Probes'], data['Segment_Mean']):
#         cnv_filtered.append({
#             'Start': start,
#             'End': end,
#             'Num_Probes': num_probes,
#             'Segment_Mean': segment_mean
#         })

#     return cnv_filtered

def read_cnv_data(path):
	data = pd.read_csv(path, sep="\t")

	cnv_filterd = {}

	cnv_filterd['Start'] = np.mean(data['Start'].values)
	cnv_filterd['End'] = np.mean(data['End'].values)
	cnv_filterd['Num_Probes'] = np.mean(data['Num_Probes'].values)
	cnv_filterd['Segment_Mean'] = np.mean(data['Segment_Mean'].values)

	return cnv_filterd


def read_transcriptome_data(path):
	# data = pd.read_csv(path, sep="\t")
	data = pd.read_csv(path, sep="\t", comment='#')
	data = data[4:]

	transcriptome = {}

	# transcriptome['unstranded'] = np.mean(data['unstranded'].values)
	transcriptome['stranded_first'] = np.mean(data['stranded_first'].values)
	transcriptome['stranded_second'] = np.mean(data['stranded_second'].values)
	transcriptome['tpm_unstranded'] = np.mean(data['tpm_unstranded'].values)
	transcriptome['fpkm_unstranded'] = np.mean(data['fpkm_unstranded'].values)
	transcriptome['fpkm_uq_unstranded'] = np.mean(data['fpkm_uq_unstranded'].values)


	return transcriptome

def normalize_images(images):
    max_value = np.max(images)
    if max_value == 0:
        print("Warning: Maximum value in image set is zero, skipping normalization.")
        return images  # در صورتی که ماکسیمم صفر است، نرمال‌سازی نکنید
    return np.clip(images / max_value, 0, 1)

def GAN_SEGMENT(t1_TCGA, falir_TCGA, segmet_mdoel, netG):
    name = t1_TCGA.split('_')[1]
    # Load data
    t1_images = np.load(t1_TCGA)
    flair_images = np.load(falir_TCGA)

    if len(t1_images) > len(flair_images):
        t1_images = t1_images[:len(flair_images)]
    if len(t1_images) < len(flair_images):
        flair_images = flair_images[:len(t1_images)]

    
    t1, flair = map(lambda x: torch.tensor(x, dtype=torch.float32).unsqueeze(1), [t1_images, flair_images])
    data = torch.cat((t1, flair), dim=1)

    real_cpu = data.to(device).float()
    output = netG(real_cpu)
    output = output.squeeze(1).detach().cpu().numpy()


    dataset = np.stack([flair_images, t1_images, output], axis=-1)
    segmet_mdoel = segmet_mdoel.to(device)
    radiomics_data = []
    x = 1
    for img in dataset:
        image = img.transpose(2, 0, 1)
        image = normalize_images(image)
        

        image = torch.tensor(image, dtype=torch.float32).to(device)
        image = image.unsqueeze(0)
        # pred = torch.sigmoid(segmet_mdoel(image))
        pred = segmet_mdoel(image)

        pred = pred.detach().cpu().numpy().squeeze()
        image = image.detach().cpu().numpy().squeeze()
        
        path = f'results/segmented_tcia/{name}'
        if not os.path.exists(path):
            os.makedirs(path)

        if not len(np.unique(pred)) == 1:
            # Create a color overlay using the mask
            rgb_mask = np.zeros((pred.shape[1], pred.shape[2], 3), dtype=np.float32)
            for i in range(pred.shape[0]):
                rgb_mask[..., i] = pred[i, :, :] * 255

            cv2.imwrite(f'{path}/{x}.jpg', rgb_mask)
            x += 1

            radiomics_features = extract_radiomics_features(image, rgb_mask)
            radiomics_data.append(radiomics_features)
        

    return radiomics_data, name

# [{}]


if __name__ == "__main__": 
    from config import get_args

    args = get_args()

    if args.device == 'TPU':
        import torch_xla
        import torch_xla.core.xla_model as xm
        device = xm.xla_device()
    else:
        device = ('cuda:0' if torch.cuda.is_available() else 'cpu') 

    segmet_mdoel = OctaveUNet(n_classes=3).to(device)
    # OctaveUnet / Generator
    netG = define_G(input_nc=2, output_nc=1, norm='batch', use_dropout=False, gpu_ids=[])
    netG = netG.to(device)

    optimizerG = optim.Adam(netG.parameters(), lr=1e-4, betas=(0.5, 0.999))

    # Load checkpoints if resuming training
    load_checkpoint(netG, optimizerG)

    CLINICAL_FOLDER = "clinical_filtered_files"
    DATA_DIR = "dataset/TCIA"
    CNV_FILTERD_DIR = "cnv_filtered_files"
    # Specify the file name
    CSV_FILE = "features.csv"
    CSV_FILE_cnv = "cnv_features.csv"

    TRANSCRIOTOME_DIR = "transcriptome_filtered_files"


    DATA_FILES = [n for n in os.listdir(DATA_DIR) if "t1" in n]
    CLINICAL_FILES = os.listdir(CLINICAL_FOLDER)
    CNV_FILTERD_FILES = os.listdir(CNV_FILTERD_DIR)
    TRANSCRIOTOME_FILES = os.listdir(TRANSCRIOTOME_DIR)


    dataset = []
    for data in DATA_FILES:
        print(data)
        flair = 'flair' + data[2:]
        flair = f"{DATA_DIR}/{flair}"
        T1 = f"{DATA_DIR}/{data}"

        radiomics, name = GAN_SEGMENT(T1, flair, segmet_mdoel, netG)

        radiomics, name = GAN_SEGMENT(T1, flair, segmet_mdoel, netG)

        for radiomic in radiomics:

            clinical_file = list(filter(lambda value: value.startswith(name), CLINICAL_FILES))
            cnv_file = list(filter(lambda value: value.startswith(name), CNV_FILTERD_FILES))
            transcriotome_file = list(filter(lambda value: value.startswith(name), TRANSCRIOTOME_FILES))

            clinical_data = read_clinical(f"{CLINICAL_FOLDER}/{clinical_file[0]}")
            cnv_data = read_cnv_data(f"{CNV_FILTERD_DIR}/{cnv_file[0]}")
            transcriotome = read_transcriptome_data(f"{TRANSCRIOTOME_DIR}/{transcriotome_file[0]}")
            

            combined_dict = {'name':name, **clinical_data, **transcriotome, **cnv_data, **radiomic}
            # combined_dict = {'name':name, **clinical_data, **transcriotome, **radiomic}

            dataset.append(combined_dict)

    # Get the list of column headers (keys from the first dictionary)
    headers = dataset[0].keys()

    # Writing the data to a CSV file
    with open(CSV_FILE, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=headers)

        # Write the header
        writer.writeheader()

        # Write the rows of data
        writer.writerows(dataset)

    print(f"CSV file '{CSV_FILE}' created successfully.")





