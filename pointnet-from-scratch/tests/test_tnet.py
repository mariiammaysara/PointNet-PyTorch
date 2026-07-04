import torch
import sys
import os

# Add the project root to the python path to resolve imports correctly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.pointnet import TNet

def test_tnet_shape():
    # Create a random tensor of shape (batch=2, k=3, num_points=1024)
    x = torch.randn(2, 3, 1024)
    tnet = TNet(k=3)
    out = tnet(x)
    
    assert out.shape == (2, 3, 3), f"Expected shape (2, 3, 3), got {out.shape}"
    print("Shape test passed!")

def test_tnet_identity_init():
    # Verify that the initial predictions are indeed close to identity matrices
    x = torch.randn(2, 3, 1024)
    tnet = TNet(k=3)
    tnet.eval()
    with torch.no_grad():
        out = tnet(x)
        
    expected = torch.eye(3).unsqueeze(0).repeat(2, 1, 1)
    # Check if the output is close to the identity matrix
    assert torch.allclose(out, expected, atol=1e-5), f"Initial output is not the identity matrix: {out}"
    print("Identity initialization test passed!")

if __name__ == "__main__":
    test_tnet_shape()
    test_tnet_identity_init()
    print("All TNet tests passed successfully!")
