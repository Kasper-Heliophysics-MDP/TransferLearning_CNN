import os
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from sklearn.model_selection import train_test_split
import pandas as pd

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
    try:
        from segmentation_models import Unet
    except ImportError:
        raise ImportError("Please install the segmentation_models library: pip install segmentation-models")
    
    model = Unet(
        backbone_name='resnet34', # resnet34, resnet50, resnet101, resnet152
        input_shape=input_shape,
        classes=num_classes,
        encoder_weights=encoder_weights
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
            
            
def combined_loss(y_true, y_pred):
    """
    Computes the combined loss: binary cross-entropy (BCE) plus Jaccard (IOU) loss.
    
    Parameters:
        y_true (Tensor): Ground truth masks (shape: [N, C, H, W])
        y_pred (Tensor): Predicted masks (shape: [N, C, H, W])
        
    Returns:
        Tensor: Combined loss.
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


def train_one_epoch(model, dataloader, optimizer, device):
    """
    Performs one training epoch.

    Iterates over the training dataloader, computes predictions, loss,
    backpropagates gradients, and updates model parameters.
    
    Parameters:
        model (nn.Module): The segmentation model to train.
        dataloader (DataLoader): PyTorch DataLoader for the training set.
        optimizer (torch.optim.Optimizer): Optimizer for updating the model parameters.
        device (torch.device): Device on which to run the model (e.g., 'cuda' or 'cpu').
    
    Returns:
        float: Average training loss for this epoch.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        preds = model(x)  # Forward pass
        loss = combined_loss(y, preds)
        loss.backward()   # Compute gradients
        optimizer.step()  # Update weights
        total_loss += loss.item()
        num_batches += 1
    return total_loss / num_batches


def validate_one_epoch(model, dataloader, device):
    """
    Evaluates the model on the validation set for one epoch.

    Computes both the loss and the evaluation metrics (IOU, F1 score) without updating the model.
    
    Parameters:
        model (nn.Module): The segmentation model.
        dataloader (DataLoader): PyTorch DataLoader for the validation set.
        device (torch.device): Device on which to run the model.
    
    Returns:
        tuple: (average validation loss, dictionary of average metrics)
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    total_metrics = {'iou': 0.0, 'f1': 0.0}
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            preds = model(x)
            loss = combined_loss(y, preds)
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


def save_checkpoint(model, optimizer, epoch, best_metric, checkpoint_path):
    """
    Saves the model checkpoint along with optimizer state.
    
    Parameters:
        model (nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer whose state to save.
        epoch (int): The current epoch number.
        best_metric: The best metric value achieved (e.g., best validation loss).
        checkpoint_path (str): Path to save the checkpoint.
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_metric': best_metric
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at epoch {epoch} with best metric {best_metric}")


def train_model(model, train_loader, val_loader, initial_lr=1e-3, freeze_epochs=100, total_epochs=150,
                checkpoint_path='./best_model.pth', patience=10, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    """
    Manages the entire training process with two phases:
    
    Phase 1: The encoder is frozen, and only the remaining part of the model is trained.
             This phase runs for 'freeze_epochs' iterations.
             
    Phase 2: The encoder is unfrozen for fine-tuning the entire model, and the learning rate
             is reduced. This phase runs from freeze_epochs+1 to total_epochs.
             
    Early stopping is applied in both phases if the validation loss does not improve for 'patience' epochs.
    
    Parameters:
        model (nn.Module): The segmentation model to be trained.
        train_loader (DataLoader): DataLoader for the training set.
        val_loader (DataLoader): DataLoader for the validation set.
        initial_lr (float): Initial learning rate.
        freeze_epochs (int): Number of epochs to train with frozen encoder.
        total_epochs (int): Total number of training epochs.
        checkpoint_path (str): File path for saving checkpoints.
        patience (int): Number of epochs to wait before early stopping if no improvement.
        device (torch.device): The device to use for training.
    """
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=initial_lr)
    best_val_loss = float('inf')
    no_improve = 0  # Counter for early stopping
    
    # Phase 1: Freeze encoder and train the remaining layers.
    print("Phase 1: Freezing encoder and training decoder only.")
    freeze_encoder_weights(model)
    for epoch in range(1, freeze_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_metrics = validate_one_epoch(model, val_loader, device)
        print(f"Epoch {epoch}/{freeze_epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - IOU: {val_metrics['iou']:.4f} - F1: {val_metrics['f1']:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            save_checkpoint(model, optimizer, epoch, best_val_loss, checkpoint_path)
        else:
            no_improve += 1
        if no_improve >= patience:
            print("Early stopping in Phase 1.")
            break
    
    # Phase 2: Unfreeze encoder and fine-tune the entire model.
    print("Phase 2: Unfreezing encoder and fine-tuning entire model.")
    unfreeze_encoder_weights(model)
    adjust_learning_rate(optimizer, initial_lr / 10)
    no_improve = 0  # Reset early stopping counter
    for epoch in range(freeze_epochs + 1, total_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_metrics = validate_one_epoch(model, val_loader, device)
        print(f"Epoch {epoch}/{total_epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - IOU: {val_metrics['iou']:.4f} - F1: {val_metrics['f1']:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            save_checkpoint(model, optimizer, epoch, best_val_loss, checkpoint_path)
        else:
            no_improve += 1
        if no_improve >= patience:
            print("Early stopping in Phase 2.")
            break
