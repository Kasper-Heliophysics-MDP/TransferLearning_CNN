import os
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from sklearn.model_selection import train_test_split
import pandas as pd
import segmentation_models_pytorch as smp


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


def build_deeplabv3(input_shape=(256, 256, 1), num_classes=1, encoder_weights=None, encoder_name='resnet34'):
    """
    Constructs a DeepLabV3+ model optimized for radio burst detection.
    
    DeepLabV3+ advantages for radio burst detection:
    - Atrous Spatial Pyramid Pooling (ASPP) captures multi-scale temporal patterns
    - Better handling of objects at different scales (short vs long bursts)
    - Improved boundary detection compared to UNet
    - More robust feature extraction for sparse signals
    
    Parameters:
        input_shape (tuple): Specifies the dimensions of the input images.
                             For example, (256, 256, 1) represents 256x256 grayscale spectrograms.
        num_classes (int): The number of output classes (typically 1 for binary segmentation).
        encoder_weights (str or None): If None, encoder weights will be randomly initialized.
                                       Recommended to use None for spectrogram data.
        encoder_name (str): Backbone architecture. Options: 'resnet34', 'resnet18', 'efficientnet-b0'
    
    Returns:
        model: A DeepLabV3+ model instance optimized for radio burst segmentation.
    """
    print(f"🚀 Building DeepLabV3+ with {encoder_name} backbone")
    print(f"   Input: {input_shape}, Classes: {num_classes}")
    print(f"   Encoder weights: {'None (from scratch)' if encoder_weights is None else encoder_weights}")
    
    model = smp.DeepLabV3Plus(
        encoder_name=encoder_name,           # backbone architecture
        encoder_weights=encoder_weights,      # None for spectrogram-specific training
        in_channels=input_shape[-1],         # number of input channels (1 for grayscale spectrogram)
        classes=num_classes,                 # number of output classes
        activation='sigmoid',                # final activation function for binary segmentation
        encoder_depth=5,                     # depth of feature extraction
        decoder_channels=256,                # decoder feature channels
    )
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✅ Model created successfully!")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    return model


def build_deeplabv3_rgb_to_mono(input_shape=(256, 256, 1), num_classes=1, 
                               conversion_method='average', encoder_name='resnet34'):
    """
    Constructs a DeepLabV3+ model with RGB-to-mono weight conversion for radio burst detection.
    
    This function performs transfer learning by:
    1. Loading ImageNet pretrained RGB DeepLabV3+ model
    2. Converting RGB first layer weights to single channel
    3. Transferring all other weights to single channel model
    
    Conversion methods:
    - 'average': Simple average of RGB channels (W_mono = (W_R + W_G + W_B) / 3)
    - 'luminance': Weighted average based on human perception (0.299*R + 0.587*G + 0.114*B)
    
    Parameters:
        input_shape (tuple): Input dimensions for single channel spectrograms
        num_classes (int): Number of output classes (typically 1 for binary segmentation)
        conversion_method (str): Method for RGB to mono conversion ('average' or 'luminance')
        encoder_name (str): Backbone architecture name
    
    Returns:
        model: DeepLabV3+ model with converted weights from ImageNet RGB pretraining
    """
    import torch
    import torch.nn.functional as F
    
    print(f"🚀 Building DeepLabV3+ with RGB-to-mono conversion")
    print(f"   Conversion method: {conversion_method}")
    print(f"   Encoder: {encoder_name}")
    
    # Step 1: Build RGB model with ImageNet pretraining
    print("📥 Loading ImageNet pretrained RGB model...")
    model_rgb = smp.DeepLabV3Plus(
        encoder_name=encoder_name,
        encoder_weights='imagenet',     # Load ImageNet pretrained weights
        in_channels=3,                  # RGB input channels
        classes=num_classes,
        activation='sigmoid',
        encoder_depth=5,
        decoder_channels=256
    )
    
    # Step 2: Build target single channel model
    print("🎯 Creating target single channel model...")
    model_mono = smp.DeepLabV3Plus(
        encoder_name=encoder_name,
        encoder_weights=None,           # No pretraining, will load converted weights
        in_channels=input_shape[-1],    # Single channel input
        classes=num_classes,
        activation='sigmoid',
        encoder_depth=5,
        decoder_channels=256
    )
    
    # Step 3: Convert RGB weights to single channel
    print(f"🔄 Converting RGB first layer weights using '{conversion_method}' method...")
    
    # Get RGB first layer weights: shape [64, 3, 7, 7]
    rgb_conv1_weight = model_rgb.encoder.conv1.weight.data
    print(f"   RGB weights shape: {rgb_conv1_weight.shape}")
    
    if conversion_method == 'average':
        # Simple average across RGB channels
        mono_conv1_weight = rgb_conv1_weight.mean(dim=1, keepdim=True)
        print("   Applied simple average conversion")
        
    elif conversion_method == 'luminance':
        # Weighted average based on luminance perception
        # Y = 0.299*R + 0.587*G + 0.114*B (ITU-R BT.601 standard)
        rgb_weights = torch.tensor([0.299, 0.587, 0.114], 
                                 dtype=rgb_conv1_weight.dtype,
                                 device=rgb_conv1_weight.device)
        
        # Apply weighted combination: [64, 3, 7, 7] -> [64, 1, 7, 7]
        mono_conv1_weight = (rgb_conv1_weight * rgb_weights.view(1, 3, 1, 1)).sum(dim=1, keepdim=True)
        print("   Applied luminance-weighted conversion")
        
    else:
        raise ValueError(f"Unsupported conversion method: {conversion_method}. Use 'average' or 'luminance'")
    
    print(f"   Converted weights shape: {mono_conv1_weight.shape}")
    
    # Step 4: Transfer weights from RGB model to mono model
    print("🔗 Transferring weights...")
    
    rgb_state_dict = model_rgb.state_dict()
    mono_state_dict = model_mono.state_dict()
    
    transferred_count = 0
    converted_count = 0
    skipped_count = 0
    
    for key in mono_state_dict.keys():
        if key == 'encoder.conv1.weight':
            # Special handling for first convolution layer
            mono_state_dict[key] = mono_conv1_weight
            converted_count += 1
            print(f"   ✅ Converted: {key}")
            
        elif key in rgb_state_dict and rgb_state_dict[key].shape == mono_state_dict[key].shape:
            # Direct transfer for matching shapes
            mono_state_dict[key] = rgb_state_dict[key]
            transferred_count += 1
            
        else:
            # Skip incompatible weights
            skipped_count += 1
            if skipped_count <= 3:  # Only print first few skipped items
                print(f"   ⚠️ Skipped: {key} (shape mismatch)")
    
    # Load converted state dict
    model_mono.load_state_dict(mono_state_dict)
    
    print(f"📊 Weight transfer summary:")
    print(f"   ✅ Converted layers: {converted_count}")
    print(f"   ✅ Transferred layers: {transferred_count}")
    print(f"   ⚠️ Skipped layers: {skipped_count}")
    
    # Step 5: Validation
    print("🔍 Validating converted model...")
    
    # Test forward pass
    test_input = torch.randn(1, input_shape[-1], *input_shape[:2])
    model_mono.eval()
    
    with torch.no_grad():
        test_output = model_mono(test_input)
    
    total_params = sum(p.numel() for p in model_mono.parameters())
    
    print(f"✅ Validation successful:")
    print(f"   Input shape: {test_input.shape}")
    print(f"   Output shape: {test_output.shape}")
    print(f"   Output range: [{test_output.min():.4f}, {test_output.max():.4f}]")
    print(f"   Total parameters: {total_params:,}")
    
    print("🎉 RGB-to-mono conversion completed successfully!")
    print("   Model ready for training on radio burst data")
    
    return model_mono


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


def simple_combined_loss(y_true, y_pred):
    """
    Original simple combined loss function: BCE + IoU Loss
    
    This is the ORIGINAL loss function without focal loss, boundary loss, etc.
    Use this for simple model comparison and when you want to avoid loss function complexity.
    
    Parameters:
        y_true (Tensor): Ground truth masks (shape: [N, C, H, W])
        y_pred (Tensor): Predicted masks (shape: [N, C, H, W])
        
    Returns:
        Tensor: Combined BCE + IoU loss
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


def train_one_epoch_simple(model, dataloader, optimizer, device):
    """
    Performs one training epoch with SIMPLE loss function (original version).

    Iterates over the training dataloader, computes predictions, loss,
    backpropagates gradients, and updates model parameters.
    
    Parameters:
        model (nn.Module): The segmentation model to train.
        dataloader (DataLoader): PyTorch DataLoader for the training set.
        optimizer (torch.optim.Optimizer): Optimizer for updating the model parameters.
        device (torch.device): Device on which to run the model (e.g., 'cuda' or 'cpu').
    
    Returns:
        float: Average training loss for the epoch
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        preds = model(x)  # Forward pass
        
        # Use SIMPLE loss function (original BCE + IoU)
        loss = simple_combined_loss(y, preds)
        
        loss.backward()   # Compute gradients
        optimizer.step()  # Update weights
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


def validate_one_epoch_simple(model, dataloader, device):
    """
    Evaluates the model on the validation set for one epoch with SIMPLE loss.

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
            
            # Use SIMPLE loss function (original BCE + IoU)
            loss = simple_combined_loss(y, preds)
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


def train_model_simple(model, train_loader, val_loader, initial_lr=1e-3, freeze_epochs=100, total_epochs=150,
                      checkpoint_dir='./checkpoints', patience=10, 
                      device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    """
    Manages the entire training process in two phases with SIMPLE loss function.
    
    This is the ORIGINAL training function without enhanced loss components.
    Use this for straightforward model architecture comparison.
    
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
        
    Returns:
        tuple: (trained_model, training_history)
    """
    print("🔧 Using SIMPLE loss function (BCE + IoU only)")
    print("   No focal loss, boundary loss, or other enhancements")
    print("   Good for clean model architecture comparison")
    
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=initial_lr)
    best_val_loss = float('inf')
    no_improve = 0  # Counter for early stopping
    history = []
    
    # Phase 1: Freeze encoder and train remaining layers (if freeze_epochs > 0)
    if freeze_epochs > 0:
        print(f"Phase 1: Freezing encoder and training decoder for {freeze_epochs} epochs.")
        freeze_encoder_weights(model)
        
        for epoch in range(1, freeze_epochs + 1):
            train_loss = train_one_epoch_simple(model, train_loader, optimizer, device)
            val_loss, val_metrics = validate_one_epoch_simple(model, val_loader, device)
            
            # Record history
            epoch_stats = {
                'epoch': epoch,
                'phase': 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_iou': val_metrics['iou'],
                'val_f1': val_metrics['f1']
            }
            history.append(epoch_stats)
            
            print(f"Epoch {epoch}/{freeze_epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - IOU: {val_metrics['iou']:.4f} - F1: {val_metrics['f1']:.4f}")
            
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
    if freeze_epochs < total_epochs:
        print(f"Phase 2: Unfreezing encoder and fine-tuning entire model.")
        unfreeze_encoder_weights(model)
        adjust_learning_rate(optimizer, initial_lr / 10)
        no_improve = 0  # Reset early stopping counter for phase 2
        
        start_epoch = max(freeze_epochs + 1, 1)
        for epoch in range(start_epoch, total_epochs + 1):
            train_loss = train_one_epoch_simple(model, train_loader, optimizer, device)
            val_loss, val_metrics = validate_one_epoch_simple(model, val_loader, device)
            
            # Record history
            epoch_stats = {
                'epoch': epoch,
                'phase': 2,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_iou': val_metrics['iou'],
                'val_f1': val_metrics['f1']
            }
            history.append(epoch_stats)
            
            print(f"Epoch {epoch}/{total_epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - IOU: {val_metrics['iou']:.4f} - F1: {val_metrics['f1']:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improve = 0
                save_checkpoint(model, optimizer, epoch, best_val_loss, checkpoint_dir)
            else:
                no_improve += 1
                
            if no_improve >= patience:
                print("Early stopping in Phase 2.")
                break
    
    print(f"✅ Training completed!")
    print(f"   Best validation loss: {best_val_loss:.4f}")
    if history:
        best_f1 = max([h['val_f1'] for h in history])
        print(f"   Best F1 score: {best_f1:.4f}")
    
    return model, history


# Model factory function for easy switching between architectures
def build_model_by_name(model_name='unet', input_shape=(256, 256, 1), num_classes=1, 
                       encoder_weights=None, encoder_name='resnet34', conversion_method='average'):
    """
    Factory function to build different model architectures by name.
    
    Parameters:
        model_name (str): Model architecture name. Options:
                         - 'unet': Standard UNet model
                         - 'deeplabv3': DeepLabV3+ model from scratch
                         - 'deeplabv3_rgb_convert': DeepLabV3+ with RGB-to-mono conversion
        input_shape (tuple): Input dimensions for single channel spectrograms
        num_classes (int): Number of output classes
        encoder_weights (str or None): Pretrained weights ('imagenet' or None)
        encoder_name (str): Backbone architecture name
        conversion_method (str): RGB conversion method for 'deeplabv3_rgb_convert'
    
    Returns:
        model: The requested model architecture
    """
    print(f"🏗️ Building {model_name} model...")
    
    if model_name == 'unet':
        return build_unet(input_shape, num_classes, encoder_weights)
        
    elif model_name == 'deeplabv3':
        return build_deeplabv3(input_shape, num_classes, encoder_weights, encoder_name)
        
    elif model_name == 'deeplabv3_rgb_convert':
        # Always use RGB conversion for this option (ignore encoder_weights parameter)
        return build_deeplabv3_rgb_to_mono(input_shape, num_classes, conversion_method, encoder_name)
        
    else:
        raise ValueError(f"Unknown model name: {model_name}. "
                        f"Options: 'unet', 'deeplabv3', 'deeplabv3_rgb_convert'")


# Backward compatibility aliases
combined_loss = simple_combined_loss  # For drop-in replacement
train_one_epoch = train_one_epoch_simple
validate_one_epoch = validate_one_epoch_simple
train_model = train_model_simple
