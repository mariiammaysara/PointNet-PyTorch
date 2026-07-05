import os
import glob
import urllib.request
import zipfile
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

def download_modelnet40(data_dir: str):
    """
    Downloads and extracts the preprocessed ModelNet40 dataset in HDF5 format
    if not already present.
    
    Expected source: https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip
    """
    url = "https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip"
    zip_path = os.path.join(data_dir, "modelnet40_ply_hdf5_2048.zip")
    extract_path = os.path.join(data_dir, "modelnet40_ply_hdf5_2048")
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        
    if not os.path.exists(extract_path):
        if not os.path.exists(zip_path):
            print(f"Downloading ModelNet40 from {url}...")
            urllib.request.urlretrieve(url, zip_path)
            print("Download complete.")
            
        print(f"Extracting to {data_dir}...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(data_dir)
        print("Extraction complete.")
        
        # Clean up the zip file to save space
        try:
            os.remove(zip_path)
        except OSError:
            pass

def load_data(root_dir: str, split: str):
    """
    Helper function to load the ModelNet40 preprocessed HDF5 files.
    """
    # The dataset contains train_files.txt and test_files.txt pointing to the .h5 files
    split_file = os.path.join(root_dir, "modelnet40_ply_hdf5_2048", f"{split}_files.txt")
    files = []
    
    if os.path.exists(split_file):
        with open(split_file, "r") as f:
            for line in f:
                filename = os.path.basename(line.strip())
                files.append(os.path.join(root_dir, "modelnet40_ply_hdf5_2048", filename))
    else:
        # Fallback to search folder directly
        files = sorted(glob.glob(os.path.join(root_dir, "modelnet40_ply_hdf5_2048", f"ply_data_{split}*.h5")))
        
    if len(files) == 0:
        raise FileNotFoundError(
            f"No HDF5 files found for split '{split}' in {root_dir}. "
            "Please ensure the dataset is downloaded and extracted properly."
        )
        
    all_data = []
    all_labels = []
    for file_path in files:
        with h5py.File(file_path, "r") as f:
            data = f["data"][:]
            label = f["label"][:]
            all_data.append(data)
            all_labels.append(label)
            
    all_data = np.concatenate(all_data, axis=0)       # shape: (N, 2048, 3)
    all_labels = np.concatenate(all_labels, axis=0).squeeze()  # shape: (N,)
    
    return all_data, all_labels

class ModelNet40Dataset(Dataset):
    """
    PyTorch Dataset for ModelNet40 point clouds.
    
    Expected HDF5 file content:
    - 'data': shape (N, 2048, 3)
    - 'label': shape (N,)
    """
    def __init__(self, root_dir: str, split: str = 'train', num_points: int = 1024,
                 augment: bool = False, _mock: bool = False):
        """
        Args:
            root_dir (str): Root directory where data is saved or will be downloaded.
            split (str): Split to load, either 'train' or 'test'.
            num_points (int): Number of points to sample from each point cloud.
            augment (bool): If True, apply data augmentation (rotation + jitter).
            _mock (bool): Internal testing flag to initialize dataset with dummy data.
        """
        self.root_dir = root_dir
        self.split = split
        self.num_points = num_points
        self.augment = augment
        
        if _mock:
            # Generate dummy mock data for testing
            self.data = np.random.randn(10, 2048, 3)
            self.labels = np.random.randint(0, 40, size=(10,))
        else:
            download_modelnet40(self.root_dir)
            self.data, self.labels = load_data(self.root_dir, self.split)
            
        print(f"Loaded {len(self.data)} point clouds for split: {self.split}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Retrieve points and label
        point_set = self.data[index]  # (2048, 3)
        label = self.labels[index]    # scalar
        
        # 1. Random sampling of num_points from the 2048 points without replacement
        choice = np.random.choice(point_set.shape[0], self.num_points, replace=False)
        point_set = point_set[choice, :]
        
        # 2. Data Augmentation (Only if augment=True, e.g. during training)
        if self.augment:
            # Rotation around Y-axis (up axis)
            theta = np.random.uniform(0, 2 * np.pi)
            cos_t = np.cos(theta)
            sin_t = np.sin(theta)
            
            # The rotation matrix around the Y-axis:
            # [ cos(theta)  0  sin(theta)]
            # [     0       1      0     ]
            # [-sin(theta)  0  cos(theta)]
            rotation_matrix = np.array([
                [cos_t, 0, sin_t],
                [0, 1, 0],
                [-sin_t, 0, cos_t]
            ])
            # Apply rotation: shape (N, 3) @ (3, 3) -> (N, 3)
            point_set = np.dot(point_set, rotation_matrix)
            
            # Random jitter (additive Gaussian noise, clipped to avoid outliers)
            # Paper parameters: standard dev = 0.02, noise clipped to 0.05
            jitter = np.clip(np.random.normal(0, 0.02, size=point_set.shape), -0.05, 0.05)
            point_set = point_set + jitter
            
        # 3. Normalize the point cloud to fit in a unit sphere (centered at origin)
        # Shift centroid to origin
        centroid = np.mean(point_set, axis=0)
        point_set = point_set - centroid
        
        # Scale by maximum distance to origin
        max_dist = np.max(np.sqrt(np.sum(point_set ** 2, axis=1)))
        if max_dist > 0:
            point_set = point_set / max_dist
            
        # 4. Return variables as PyTorch tensors
        points_tensor = torch.tensor(point_set, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return points_tensor, label_tensor
