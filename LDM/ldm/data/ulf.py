import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, EnsureTyped,
    CropForegroundd, SpatialPadd, CenterSpatialCropd,
    ScaleIntensityRangePercentilesd, RandFlipd, RandRotate90d
)
from monai.data import Dataset as MonaiDataset


def get_transforms(phase="train"):
    modalities = ["ulf", "hf"]

    if phase == "train":
        train_transforms = Compose(
            [
                RandFlipd(keys=modalities, prob=0.1, spatial_axis=0, allow_missing_keys=True),
                RandFlipd(keys=modalities, prob=0.1, spatial_axis=1, allow_missing_keys=True),
                RandFlipd(keys=modalities, prob=0.1, spatial_axis=2, allow_missing_keys=True),
                # Add rotation for more augmentation
                RandRotate90d(keys=modalities, prob=0.1, max_k=3, allow_missing_keys=True),
            ]
        )
    else:
        train_transforms = Compose([])
    
    return Compose(
        [
            LoadImaged(keys=modalities, allow_missing_keys=True),
            EnsureChannelFirstd(keys=modalities, allow_missing_keys=True),
            Orientationd(keys=modalities, axcodes="RAS", allow_missing_keys=True),
            EnsureTyped(keys=modalities, allow_missing_keys=True),
            # Crop foreground based on HF image
            CropForegroundd(keys=modalities, source_key="hf", margin=0, allow_missing_keys=True),
            # Pad to target size (160, 160, 128) based on your data
            SpatialPadd(keys=modalities, spatial_size=(160, 160, 128), allow_missing_keys=True),
            CenterSpatialCropd(keys=modalities, roi_size=(160, 160, 128), allow_missing_keys=True),
            # Normalize intensity to [-1, 1] range
            ScaleIntensityRangePercentilesd(
                keys=modalities, 
                lower=0.5, 
                upper=99.5, 
                b_min=-1, 
                b_max=1, 
                allow_missing_keys=True
            ),
            train_transforms
        ]
    )


def get_ulf_hf_dataset(ulf_path, hf_path, csv_path=None, phase="train"):
    """
    Load paired ULF and HF MRI dataset
    
    Args:
        ulf_path: Path to ULF data folder (e.g., 'Dataset/ULF-v1/ULF-v1Tr')
        hf_path: Path to HF data folder (e.g., 'Dataset/HF/HFTr')
        csv_path: Optional CSV file with subject IDs and splits
        phase: 'train', 'val', or 'test'
    """
    transform = get_transforms(phase=phase)
    
    # Get list of files
    if csv_path is not None:
        df = pd.read_csv(csv_path)
        datalist = []
        for sub_id in df["id"].tolist():
            split_list = df[df["id"] == sub_id]["split"].tolist()
            if split_list and split_list[0] == phase:
                datalist.append(sub_id)
    else:
        # Get all .nii.gz files
        ulf_files = sorted([f for f in os.listdir(ulf_path) if f.endswith('.nii.gz')])
        hf_files = sorted([f for f in os.listdir(hf_path) if f.endswith('.nii.gz')])
        
        # Extract subject IDs (assuming filenames match)
        datalist = [f.replace('.nii.gz', '') for f in ulf_files]

    data = []
    
    for subject in datalist:
        # Construct file paths
        ulf_file = os.path.join(ulf_path, f"{subject}.nii.gz")
        hf_file = os.path.join(hf_path, f"{subject}.nii.gz")
        
        # Check if both files exist
        if not os.path.exists(ulf_file):
            print(f"Warning: ULF file not found: {ulf_file}")
            continue
        if not os.path.exists(hf_file):
            print(f"Warning: HF file not found: {hf_file}")
            continue
        
        data.append({
            "ulf": ulf_file,
            "hf": hf_file,
            "subject_id": subject,
            "path": ulf_file
        })
                    
    print(f"{phase} - num of subjects: {len(data)}")
    
    return MonaiDataset(data=data, transform=transform)


class Brain3DBase(Dataset):
    def __init__(self, **kwargs):
        super().__init__()
        self.data = None

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        return self.data[i]
    

class ULFtoHFBase(Brain3DBase):
    def __init__(self):
        super().__init__()
        self.modalities = ["ulf", "hf"]
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        item = dict(self.data[i])
        
        # For ULF-to-HF, source is always ULF, target is always HF
        item["source"] = item["ulf"]
        item["target"] = item["hf"]
        item["subject_id"] = item.get("subject_id", f"subject_{i}")
        # Add target_class for consistency with BraTS dataset (single class = 0)
        item["target_class"] = torch.tensor(0)
        
        return item


class ULFtoHFTrain(ULFtoHFBase):
    def __init__(self, ulf_path, hf_path, csv_path=None, **kwargs):
        super().__init__()
        self.data = get_ulf_hf_dataset(ulf_path, hf_path, csv_path, phase="train")


class ULFtoHFVal(ULFtoHFBase):
    def __init__(self, ulf_path, hf_path, csv_path=None, **kwargs):
        super().__init__()
        self.data = get_ulf_hf_dataset(ulf_path, hf_path, csv_path, phase="val")


class ULFtoHFTest(ULFtoHFBase):
    def __init__(self, ulf_path, hf_path, csv_path=None, **kwargs):
        super().__init__()
        self.data = get_ulf_hf_dataset(ulf_path, hf_path, csv_path, phase="test")


# For compatibility with existing code
class ULFtoHF2021Train(ULFtoHFTrain):
    pass


class ULFtoHF2021Val(ULFtoHFVal):
    pass


class ULFtoHF2021Test(ULFtoHFTest):
    pass