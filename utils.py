import torch
import torch.nn.functional as F

def feature_transform_regularizer(trans: torch.Tensor) -> torch.Tensor:
    """
    Computes the feature transform regularization loss from the PointNet paper.
    
    L_reg = ||I - A @ A^T||_F^2
    
    This constraint encourages the learned feature transformation matrix (A) to be
    close to orthogonal. Without this constraint, the feature transform (a 64x64 matrix,
    which is much higher dimensional than the 3x3 input transform) can become unstable
    during training since it has significantly more degrees of freedom. An orthogonal
    transformation preserves distances, volumes, and angles, preventing the network
    from scaling or distorting features arbitrarily.
    
    Args:
        trans (torch.Tensor): Feature transform matrix of shape (batch_size, k, k).
        
    Returns:
        torch.Tensor: Scalar tensor containing the regularization loss.
    """
    batch_size = trans.size(0)
    k = trans.size(1)
    
    # Create batch of identity matrices on the same device and type as trans
    identity = torch.eye(k, device=trans.device).unsqueeze(0).expand(batch_size, -1, -1)
    
    # A @ A^T
    trans_transpose = trans.transpose(1, 2)
    a_at = torch.bmm(trans, trans_transpose)
    
    # Calculate the squared Frobenius norm: ||I - A @ A^T||_F^2, averaged over the batch
    diff = identity - a_at
    loss = torch.mean(torch.sum(diff ** 2, dim=(1, 2)))
    
    return loss

def get_classification_loss(logits: torch.Tensor, targets: torch.Tensor, feature_trans: torch.Tensor, reg_weight: float = 0.001) -> torch.Tensor:
    """
    Computes the total PointNet classification loss.
    
    Combines standard Cross-Entropy loss with the feature transform regularization loss.
    
    Args:
        logits (torch.Tensor): Model output logits of shape (batch_size, num_classes).
        targets (torch.Tensor): Ground truth labels of shape (batch_size,).
        feature_trans (torch.Tensor or None): Feature transform matrix of shape (batch_size, 64, 64),
                                             or None if feature transform is disabled.
        reg_weight (float): Scalar weight for the feature transform regularization loss.
        
    Returns:
        torch.Tensor: Total scalar loss tensor.
    """
    ce_loss = F.cross_entropy(logits, targets)
    
    if feature_trans is not None:
        reg_loss = feature_transform_regularizer(feature_trans)
        total_loss = ce_loss + reg_loss * reg_weight
    else:
        total_loss = ce_loss
        
    return total_loss

def set_seed(seed: int = 42):
    """
    Sets the random seed for python, numpy, and PyTorch (CPU and CUDA)
    to ensure reproducibility.
    """
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
