import os
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from sklearn.model_selection import train_test_split
import pandas as pd
import segmentation_models_pytorch as smp


# def create_dataset(image_dir, mask_dir, img_size=(256, 256), test_size=0.2, random_state=42):
def create_dataset(dir, img_size=(256, 256), test_size=0.2, random_state=42):
    """
    Reads CSV image slices and their corresponding masks from the given directories,
    performs necessary preprocessing (e.g., normalization), and splits the data into
    training and validation sets.

    Expected naming convention for files:
      - For a slice with a burst:
          e.g. slice_20240608_y155_x270406_SkylineHS.csv
          e.g. slice_20240608_y155_x270406_SkylineHS_mask.csv
      - For a non-burst slice:
          e.g. slice_20240420_y0_x3391_PeachMountain_2020_nonburst.csv
      - The filename structure is:
          slice_{date}_y{top_left_y_coordinate}_x{top_left_x_coordinate}_{location}
          [optional: _{time}][optional: _nonburst]_mask.csv   (for mask files)
          or without the _mask extension for image files.

    Parameters:
        # image_dir (str): Directory containing the CSV image slices.
        # mask_dir (str): Directory containing the corresponding CSV mask files.
        dir (str): Directory containing the CSV image slices and masks.
        img_size (tuple): Expected image size (default: (256, 256)).
        test_size (float): Proportion of the dataset to reserve for validation (default: 0.2).
        random_state (int): Random seed for reproducibility (default: 42).

    Returns:
        tuple: ((train_images, train_masks), (val_images, val_masks))
               where train_images and train_masks are numpy arrays of training data,
               and val_images and val_masks are numpy arrays of validation data.
    """
    # Get a sorted list of CSV file paths for images and masks.
    # image_files = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.csv')])
    # mask_files  = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith('.csv')])
    all_files = [os.path.join(dir, f) for f in os.listdir(dir) if f.endswith('.csv')]
    
    image_files = [f for f in all_files if '_mask' not in f]
    mask_files = [f for f in all_files if '_mask' in f]

    # Optionally, sort them to ensure pairing:
    image_files = sorted(image_files)
    mask_files = sorted(mask_files)
    
    images = []
    masks = []
    
    # Loop through each image and corresponding mask file.
    for img_path, mask_path in zip(image_files, mask_files):
        # Load the CSV data, convert to float32 and normalize to [0,1]
        img = pd.read_csv(img_path).values.astype(np.float32)
        max_val = np.max(img)
        img = img / max_val if max_val != 0 else img
        
        mask = pd.read_csv(mask_path).values.astype(np.float32)
        max_val_mask = np.max(mask)
        mask = mask / max_val_mask if max_val_mask != 0 else mask
        
        # Expand the dimensions to add the channel dimension (making shape: 256x256x1)
        img = np.expand_dims(img, axis=-1)
        mask = np.expand_dims(mask, axis=-1)
        
        images.append(img)
        masks.append(mask)
    
    # Convert list of images and masks to numpy arrays.
    images = np.array(images)
    masks = np.array(masks)
    
    # Split the dataset into training and validation sets.
    train_images, val_images, train_masks, val_masks = train_test_split(
        images, masks, test_size=test_size, random_state=random_state)
    
    return (train_images, train_masks), (val_images, val_masks)

def build_unet(input_shape=(256, 256, 1), num_classes=1, encoder_weights='imagenet'):
    """
    Constructs a UNet model with an option to load pre-trained encoder weights for transfer learning.

    Parameters:
        input_shape (tuple): Specifies the dimensions of the input images.
                             For example, (256, 256, 1) represents 256x256 grayscale images.
        num_classes (int): The number of output classes (typically 1 for binary segmentation or 2 for multi-class segmentation).
        encoder_weights (str or None): If set to 'imagenet', pre-trained weights for the encoder will be loaded.
                                       If None, the encoder's weights will be randomly initialized.

    Returns:
        model: A UNet model instance constructed using the segmentation_models library.
    """
    # Use the Unet model from the segmentation_models library as an example.
    model = smp.Unet(
        encoder_name='resnet34',  # select backbone architecture
        encoder_weights=encoder_weights,  # 'imagenet' to load pre-trained weights
        in_channels=input_shape[-1],  # number of input channels
        classes=num_classes,          # number of output classes
        activation='sigmoid'          # final activation function
    )
    return model

def freeze_encoder_weights(model):
    """
    Freezes the weights of the encoder (or backbone) part of the model.
    It assumes that the model has an attribute 'encoder' or 'backbone'.
    """
    if hasattr(model, 'encoder'):
        for param in model.encoder.parameters():
            param.requires_grad = False
    elif hasattr(model, 'backbone'):
        for param in model.backbone.parameters():
            param.requires_grad = False
    else:
        print("Encoder/backbone not found. Please freeze the relevant layers manually.")

def unfreeze_encoder_weights(model):
    """
    Unfreezes the weights of the encoder (or backbone) part of the model.
    """
    if hasattr(model, 'encoder'):
        for param in model.encoder.parameters():
            param.requires_grad = True
    elif hasattr(model, 'backbone'):
        for param in model.backbone.parameters():
            param.requires_grad = True
            
            
def focal_loss(y_pred, y_true, alpha=1.0, gamma=2.0, reduction='mean'):
    """
    Focal Loss implementation for addressing class imbalance.
    
    Args:
        y_pred (Tensor): Predicted probabilities [N, C, H, W]
        y_true (Tensor): Ground truth binary masks [N, C, H, W]
        alpha (float): Weighting factor for rare class (default: 1.0)
        gamma (float): Focusing parameter (default: 2.0)
        reduction (str): 'mean', 'sum', or 'none'
    
    Returns:
        Tensor: Focal loss
    """
    # Compute binary cross entropy
    bce = F.binary_cross_entropy(y_pred, y_true, reduction='none')
    
    # Compute p_t
    p_t = y_pred * y_true + (1 - y_pred) * (1 - y_true)
    
    # Compute alpha_t
    alpha_t = alpha * y_true + (1 - alpha) * (1 - y_true)
    
    # Compute focal weight
    focal_weight = alpha_t * (1 - p_t) ** gamma
    
    # Apply focal weight
    focal_loss_val = focal_weight * bce
    
    if reduction == 'mean':
        return focal_loss_val.mean()
    elif reduction == 'sum':
        return focal_loss_val.sum()
    else:
        return focal_loss_val


def boundary_loss(y_pred, y_true, kernel_size=3):
    """
    Boundary-aware loss to improve edge prediction accuracy.
    
    Args:
        y_pred (Tensor): Predicted probabilities [N, C, H, W]  
        y_true (Tensor): Ground truth binary masks [N, C, H, W]
        kernel_size (int): Kernel size for edge detection
        
    Returns:
        Tensor: Boundary loss
    """
    # Create Laplacian kernel for edge detection
    device = y_pred.device
    laplacian_kernel = torch.tensor([[[
        [0, -1, 0],
        [-1, 4, -1], 
        [0, -1, 0]
    ]]], dtype=torch.float32, device=device)
    
    # Compute edges in ground truth
    gt_edges = F.conv2d(y_true, laplacian_kernel, padding=1)
    gt_edges = (gt_edges.abs() > 0.1).float()
    
    # Compute edges in predictions
    pred_edges = F.conv2d(y_pred, laplacian_kernel, padding=1)
    pred_edges = torch.sigmoid(pred_edges)
    
    # Compute boundary loss (weighted by edge presence)
    boundary_loss_val = F.binary_cross_entropy(pred_edges, gt_edges, reduction='none')
    edge_weight = gt_edges + 0.1  # Give higher weight to edge pixels
    
    return (boundary_loss_val * edge_weight).mean()


def adaptive_iou_loss(y_pred, y_true, smooth=1e-6, power=1):
    """
    Adaptive IoU loss with optional power weighting.
    
    Args:
        y_pred (Tensor): Predicted probabilities [N, C, H, W]
        y_true (Tensor): Ground truth binary masks [N, C, H, W]
        smooth (float): Smoothing factor to avoid division by zero
        power (int): Power for weighting rare positive samples
        
    Returns:
        Tensor: Adaptive IoU loss
    """
    # Flatten tensors
    y_pred_flat = y_pred.view(-1)
    y_true_flat = y_true.view(-1)
    
    # Compute intersection and union
    intersection = (y_pred_flat * y_true_flat).sum()
    union = y_pred_flat.sum() + y_true_flat.sum() - intersection
    
    # Compute IoU
    iou = (intersection + smooth) / (union + smooth)
    
    # Apply power weighting for imbalanced data
    if power != 1:
        # Weight by inverse class frequency
        pos_ratio = y_true_flat.mean()
        weight = (1 - pos_ratio) ** power if pos_ratio > 0 else 1.0
        iou_loss = (1 - iou) * weight
    else:
        iou_loss = 1 - iou
        
    return iou_loss


def combined_loss(y_true, y_pred, loss_weights=None, focal_params=None, boundary_weight=0.2):
    """
    Enhanced combined loss with multiple components:
    - Focal Loss (for class imbalance)
    - Adaptive IoU Loss (for overlap quality)  
    - Boundary Loss (for edge accuracy)
    
    Parameters:
        y_true (Tensor): Ground truth masks (shape: [N, C, H, W])
        y_pred (Tensor): Predicted masks (shape: [N, C, H, W])
        loss_weights (dict): Weights for different loss components
        focal_params (dict): Parameters for focal loss
        boundary_weight (float): Weight for boundary loss component
        
    Returns:
        Tensor: Combined loss with detailed breakdown.
    """
    # Default parameters
    if loss_weights is None:
        loss_weights = {'focal': 1.0, 'iou': 1.0, 'boundary': 0.2}
    
    if focal_params is None:
        focal_params = {'alpha': 0.75, 'gamma': 2.0}  # Slightly favor positive class
    
    # Compute individual loss components
    focal_loss_val = focal_loss(y_pred, y_true, 
                               alpha=focal_params['alpha'], 
                               gamma=focal_params['gamma'])
    
    iou_loss_val = adaptive_iou_loss(y_pred, y_true, power=1.5)  # Slight power weighting
    
    boundary_loss_val = boundary_loss(y_pred, y_true)
    
    # Combine losses with weights
    total_loss = (loss_weights['focal'] * focal_loss_val + 
                  loss_weights['iou'] * iou_loss_val + 
                  loss_weights['boundary'] * boundary_loss_val)
    
    # Store individual components for monitoring (optional)
    if hasattr(total_loss, '_loss_components'):
        total_loss._loss_components = {
            'focal': focal_loss_val.item(),
            'iou': iou_loss_val.item(), 
            'boundary': boundary_loss_val.item(),
            'total': total_loss.item()
        }
    
    return total_loss


def simple_combined_loss(y_true, y_pred):
    """
    Backward compatible simple combined loss (original version).
    Use this for quick comparison with the original implementation.
    """
    # Binary cross entropy loss (assumes y_pred contains probabilities)
    bce = F.binary_cross_entropy(y_pred, y_true)
    
    # Compute intersection and union along spatial and channel dimensions
    intersection = torch.sum(y_true * y_pred, dim=[1, 2, 3])
    union = torch.sum(y_true + y_pred, dim=[1, 2, 3]) - intersection
    # Compute Jaccard (IOU) loss for each sample and average over batch
    iou_loss = 1 - (intersection + 1e-6) / (union + 1e-6)
    return bce + iou_loss.mean()


def compute_metrics(y_true, y_pred):
    """
    Computes segmentation evaluation metrics: IOU and F1 score.
    The predictions are thresholded at 0.5 to produce binary masks.
    
    Parameters:
        y_true (Tensor): Ground truth masks.
        y_pred (Tensor): Predicted masks.
        
    Returns:
        dict: A dictionary with 'iou' and 'f1' metrics.
    """
    y_true_bin = (y_true > 0.5).float()
    y_pred_bin = (y_pred > 0.5).float()
    
    intersection = torch.sum(y_true_bin * y_pred_bin, dim=[1, 2, 3])
    union = torch.sum(y_true_bin + y_pred_bin, dim=[1, 2, 3]) - intersection
    iou = (intersection / (union + 1e-6)).mean()
    
    precision = intersection / (torch.sum(y_pred_bin, dim=[1, 2, 3]) + 1e-6)
    recall = intersection / (torch.sum(y_true_bin, dim=[1, 2, 3]) + 1e-6)
    f1 = (2 * precision * recall / (precision + recall + 1e-6)).mean()
    
    return {'iou': iou.item(), 'f1': f1.item()}


def train_one_epoch(model, dataloader, optimizer, device, loss_weights=None, focal_params=None):
    """
    Performs one training epoch with configurable enhanced loss function.

    Iterates over the training dataloader, computes predictions, loss,
    backpropagates gradients, and updates model parameters.
    
    Parameters:
        model (nn.Module): The segmentation model to train.
        dataloader (DataLoader): PyTorch DataLoader for the training set.
        optimizer (torch.optim.Optimizer): Optimizer for updating the model parameters.
        device (torch.device): Device on which to run the model (e.g., 'cuda' or 'cpu').
        loss_weights (dict): Weights for different loss components. If None, uses default for radio bursts.
        focal_params (dict): Parameters for focal loss. If None, uses default for radio bursts.
    
    Returns:
        tuple: (average training loss, loss components dict)
    """
    # Set default parameters optimized for radio burst detection
    if loss_weights is None:
        loss_weights = {'focal': 1.2, 'iou': 1.5, 'boundary': 0.2}
    if focal_params is None:
        focal_params = {'alpha': 0.8, 'gamma': 2.5}
    
    model.train()
    total_loss = 0.0
    total_components = {'focal': 0.0, 'iou': 0.0, 'boundary': 0.0}
    num_batches = 0
    
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        preds = model(x)  # Forward pass
        
        # Use enhanced loss function
        loss = combined_loss(y, preds, 
                           loss_weights=loss_weights,
                           focal_params=focal_params)
        
        # Track individual components for monitoring
        with torch.no_grad():
            focal_val = focal_loss(preds, y, **focal_params)
            iou_val = adaptive_iou_loss(preds, y, power=1.5)
            boundary_val = boundary_loss(preds, y)
            
            total_components['focal'] += focal_val.item()
            total_components['iou'] += iou_val.item()
            total_components['boundary'] += boundary_val.item()
        
        loss.backward()   # Compute gradients
        optimizer.step()  # Update weights
        total_loss += loss.item()
        num_batches += 1
    
    avg_loss = total_loss / num_batches
    avg_components = {k: v / num_batches for k, v in total_components.items()}
    
    return avg_loss, avg_components


def validate_one_epoch(model, dataloader, device, loss_weights=None, focal_params=None):
    """
    Evaluates the model on the validation set for one epoch with enhanced loss.

    Computes both the loss and the evaluation metrics (IOU, F1 score) without updating the model.
    
    Parameters:
        model (nn.Module): The segmentation model.
        dataloader (DataLoader): PyTorch DataLoader for the validation set.
        device (torch.device): Device on which to run the model.
        loss_weights (dict): Weights for different loss components. If None, uses default for radio bursts.
        focal_params (dict): Parameters for focal loss. If None, uses default for radio bursts.
    
    Returns:
        tuple: (average validation loss, dictionary of average metrics)
    """
    # Set default parameters optimized for radio burst detection
    if loss_weights is None:
        loss_weights = {'focal': 1.2, 'iou': 1.5, 'boundary': 0.2}
    if focal_params is None:
        focal_params = {'alpha': 0.8, 'gamma': 2.5}
    
    model.eval()
    total_loss = 0.0
    num_batches = 0
    total_metrics = {'iou': 0.0, 'f1': 0.0}
    
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            preds = model(x)
            
            # Use enhanced loss function with same parameters as training
            loss = combined_loss(y, preds, 
                               loss_weights=loss_weights,
                               focal_params=focal_params)
            total_loss += loss.item()
            
            metrics_dict = compute_metrics(y, preds)
            total_metrics['iou'] += metrics_dict['iou']
            total_metrics['f1'] += metrics_dict['f1']
            num_batches += 1
            
    avg_loss = total_loss / num_batches
    avg_metrics = {k: v / num_batches for k, v in total_metrics.items()}
    return avg_loss, avg_metrics


def adjust_learning_rate(optimizer, new_lr):
    """
    Adjusts the learning rate of the given optimizer.
    
    Parameters:
        optimizer (torch.optim.Optimizer): The optimizer whose learning rate is to be adjusted.
        new_lr (float): The new learning rate.
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr
    print("Learning rate adjusted to:", new_lr)


def save_checkpoint(model, optimizer, epoch, best_metric, checkpoint_dir):
    """
    Saves the model checkpoint along with optimizer state.
    Checkpoints are saved to a specified directory with filenames including epoch and best_metric.

    Parameters:
        model (nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer whose state to save.
        epoch (int): The current epoch number.
        best_metric: The best metric value achieved (e.g., best validation loss).
        checkpoint_dir (str): Directory in which to save the checkpoint.
    """
    # Create the checkpoint directory if it does not exist
    os.makedirs(checkpoint_dir, exist_ok=True)
    # Generate a unique filename (e.g., "checkpoint_epoch_{epoch}_metric_{best_metric:.4f}.pth")
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}_metric_{best_metric:.4f}.pth")
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_metric': best_metric
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at epoch {epoch} with best metric {best_metric} -> {checkpoint_path}")


def train_model(model, train_loader, val_loader, initial_lr=1e-3, freeze_epochs=100, total_epochs=150,
                checkpoint_dir='./checkpoints', patience=10, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                loss_config='imbalanced'):
    """
    Manages the entire training process in two phases.
    
    Phase 1: Freeze the encoder and train the remaining parts.
    Phase 2: Unfreeze the encoder, reduce the learning rate, and fine-tune the entire model.
    
    Early stopping is applied if the validation loss does not improve for 'patience' epochs.
    
    Parameters:
        model (nn.Module): The segmentation model.
        train_loader (DataLoader): Training DataLoader.
        val_loader (DataLoader): Validation DataLoader.
        initial_lr (float): Initial learning rate.
        freeze_epochs (int): Epochs with frozen encoder.
        total_epochs (int): Total training epochs.
        checkpoint_dir (str): Directory where checkpoints will be saved.
        patience (int): Epochs with no improvement before early stopping.
        device (torch.device): Training device.
    """
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=initial_lr)
    best_val_loss = float('inf')
    no_improve = 0  # Counter for early stopping
    
    # Phase 1: Freeze encoder and train remaining layers.
    print("Phase 1: Freezing encoder and training decoder only.")
    freeze_encoder_weights(model)
    for epoch in range(1, freeze_epochs + 1):
        train_loss, train_components = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_metrics = validate_one_epoch(model, val_loader, device)
        print(f"Epoch {epoch}/{freeze_epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - IOU: {val_metrics['iou']:.4f} - F1: {val_metrics['f1']:.4f}")
        print(f"  Loss Components - Focal: {train_components['focal']:.4f}, IoU: {train_components['iou']:.4f}, Boundary: {train_components['boundary']:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            save_checkpoint(model, optimizer, epoch, best_val_loss, checkpoint_dir)
        else:
            no_improve += 1
        if no_improve >= patience:
            print("Early stopping in Phase 1.")
            break
            
    # Phase 2: Unfreeze encoder and fine-tune entire model.
    print("Phase 2: Unfreezing encoder and fine-tuning entire model.")
    unfreeze_encoder_weights(model)
    adjust_learning_rate(optimizer, initial_lr / 10)
    no_improve = 0  # Reset early stopping counter for phase 2
    for epoch in range(freeze_epochs + 1, total_epochs + 1):
        train_loss, train_components = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_metrics = validate_one_epoch(model, val_loader, device)
        print(f"Epoch {epoch}/{total_epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - IOU: {val_metrics['iou']:.4f} - F1: {val_metrics['f1']:.4f}")
        print(f"  Loss Components - Focal: {train_components['focal']:.4f}, IoU: {train_components['iou']:.4f}, Boundary: {train_components['boundary']:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            save_checkpoint(model, optimizer, epoch, best_val_loss, checkpoint_dir)
        else:
            no_improve += 1
        if no_improve >= patience:
            print("Early stopping in Phase 2.")
            break

