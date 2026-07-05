import torch
import sys
import os

# Add the project root to the python path to resolve imports correctly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import ModelNet40Dataset

def test_dataset_mock():
    # Instantiate with _mock=True to avoid download requirement during testing
    dataset = ModelNet40Dataset(root_dir="mock_dir", split="train", num_points=512, augment=False, _mock=True)
    
    assert len(dataset) == 10, f"Expected mock dataset size 10, got {len(dataset)}"
    
    points, label = dataset[0]
    
    # Check shape
    assert points.shape == (512, 3), f"Expected points shape (512, 3), got {points.shape}"
    assert isinstance(label, torch.Tensor) and label.dim() == 0, f"Expected scalar LongTensor label, got {label}"
    
    print(f"Sample points shape: {points.shape}")
    print(f"Sample label: {label.item()}")
    print(f"Label type: {label.dtype}")
    
    # Verify normalization (max distance from origin is close to 1.0)
    distances = torch.sqrt(torch.sum(points ** 2, dim=1))
    max_dist = torch.max(distances).item()
    assert abs(max_dist - 1.0) < 1e-5, f"Expected point cloud to fit in unit sphere with max dist 1.0, got {max_dist}"
    print(f"Max distance from origin (unit sphere test): {max_dist:.4f}")
    
    print("Basic mock dataset test passed!")

def test_dataset_augmentation():
    dataset_aug = ModelNet40Dataset(root_dir="mock_dir", split="train", num_points=1024, augment=True, _mock=True)
    
    points, label = dataset_aug[0]
    assert points.shape == (1024, 3)
    
    # Verify that augmented point cloud still fits in unit sphere
    distances = torch.sqrt(torch.sum(points ** 2, dim=1))
    max_dist = torch.max(distances).item()
    assert abs(max_dist - 1.0) < 1e-5, f"Expected point cloud to fit in unit sphere with max dist 1.0, got {max_dist}"
    
    # Label range should be within [0, 39]
    for i in range(len(dataset_aug)):
        _, l = dataset_aug[i]
        assert 0 <= l.item() < 40, f"Label {l.item()} out of ModelNet40 range"
        
    print("Augmentation test passed!")

if __name__ == "__main__":
    test_dataset_mock()
    test_dataset_augmentation()
    print("All dataset tests passed successfully!")
