import torch
import sys
import os

# Add the project root to the python path to resolve imports correctly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.pointnet import PointNetClassifier

def test_pointnet_classifier_with_feat_transform():
    # Input shape: (batch=2, num_points=1024, 3)
    x = torch.randn(2, 1024, 3)
    model = PointNetClassifier(num_classes=40, feature_transform=True)
    model.eval()
    
    with torch.no_grad():
        logits, trans, trans_feat = model(x)
        
    # Check shapes
    assert logits.shape == (2, 40), f"Expected logits shape (2, 40), got {logits.shape}"
    assert trans.shape == (2, 3, 3), f"Expected input transform shape (2, 3, 3), got {trans.shape}"
    assert trans_feat.shape == (2, 64, 64), f"Expected feature transform shape (2, 64, 64), got {trans_feat.shape}"
    print("PointNetClassifier with feature transform test passed!")

def test_pointnet_classifier_without_feat_transform():
    # Input shape: (batch=2, num_points=1024, 3)
    x = torch.randn(2, 1024, 3)
    model = PointNetClassifier(num_classes=40, feature_transform=False)
    model.eval()
    
    with torch.no_grad():
        logits, trans, trans_feat = model(x)
        
    # Check shapes
    assert logits.shape == (2, 40), f"Expected logits shape (2, 40), got {logits.shape}"
    assert trans.shape == (2, 3, 3), f"Expected input transform shape (2, 3, 3), got {trans.shape}"
    assert trans_feat is None, f"Expected feature transform to be None, got {trans_feat}"
    print("PointNetClassifier without feature transform test passed!")

def test_pointnet_classifier_ablations():
    x = torch.randn(2, 1024, 3)
    
    # Test use_input_transform=False
    model = PointNetClassifier(num_classes=40, use_input_transform=False)
    model.eval()
    with torch.no_grad():
        logits, trans, trans_feat = model(x)
    assert trans is None, f"Expected input transform (trans) to be None when use_input_transform=False, got {trans}"
    assert logits.shape == (2, 40)
    
    # Test pooling_type="avg"
    model = PointNetClassifier(num_classes=40, pooling_type="avg")
    model.eval()
    with torch.no_grad():
        logits, trans, trans_feat = model(x)
    assert logits.shape == (2, 40)
    print("PointNetClassifier ablation options test passed!")

if __name__ == "__main__":
    test_pointnet_classifier_with_feat_transform()
    test_pointnet_classifier_without_feat_transform()
    test_pointnet_classifier_ablations()
    print("All PointNetClassifier tests passed successfully!")
