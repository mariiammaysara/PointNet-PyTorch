import torch
import sys
import os

# Add the project root to the python path to resolve imports correctly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import feature_transform_regularizer, get_classification_loss

def test_feature_transform_regularizer_identity():
    # Batch size 4, matrix size 64 x 64
    batch_size = 4
    k = 64
    
    # Batch of identity matrices
    identity_batch = torch.eye(k).unsqueeze(0).repeat(batch_size, 1, 1)
    
    loss = feature_transform_regularizer(identity_batch)
    
    # Loss should be 0 or extremely close to 0 due to floating point precision
    assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6), f"Expected loss to be ~0, got {loss.item()}"
    print("Identity matrix regularization test passed!")

def test_classification_loss_no_reg():
    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
    targets = torch.tensor([0, 1])
    
    loss = get_classification_loss(logits, targets, feature_trans=None)
    # This should be pure cross entropy, which is small since we predict correct logits
    assert loss > 0
    print("Classification loss without regularization test passed!")

def test_classification_loss_with_reg():
    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
    targets = torch.tensor([0, 1])
    
    # Identity matrix (should not add any regularization loss)
    identity_feat = torch.eye(64).unsqueeze(0).repeat(2, 1, 1)
    loss_with_identity = get_classification_loss(logits, targets, feature_trans=identity_feat, reg_weight=0.1)
    loss_without_reg = get_classification_loss(logits, targets, feature_trans=None)
    
    assert torch.allclose(loss_with_identity, loss_without_reg, atol=1e-6), "Identity transform should not change total loss"
    
    # Non-orthogonal matrix (should add regularization loss)
    bad_feat = torch.eye(64).unsqueeze(0).repeat(2, 1, 1) * 2.0
    loss_with_bad = get_classification_loss(logits, targets, feature_trans=bad_feat, reg_weight=0.1)
    
    assert loss_with_bad > loss_without_reg, "Non-orthogonal transform should increase total loss"
    print("Classification loss with regularization test passed!")

if __name__ == "__main__":
    test_feature_transform_regularizer_identity()
    test_classification_loss_no_reg()
    test_classification_loss_with_reg()
    print("All utils tests passed successfully!")
