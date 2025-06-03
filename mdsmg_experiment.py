#!/usr/bin/env python3

import os
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
from huggingface_hub import hf_hub_download

class MaterialDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None, image_size=(256, 256)):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.image_size = image_size
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        
        if image_path.endswith('.tif') or image_path.endswith('.tiff'):
            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            elif len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image = np.array(Image.open(image_path).convert('RGB'))
            
        if mask_path.endswith('.tif') or mask_path.endswith('.tiff'):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            mask = np.array(Image.open(mask_path).convert('L'))
        
        image = cv2.resize(image, self.image_size)
        mask = cv2.resize(mask, self.image_size, interpolation=cv2.INTER_NEAREST)
        mask = (mask > 127).astype(np.uint8)
        
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            mask = torch.from_numpy(mask).long()
            
        return image, mask

def load_dataset_paths(base_path, dataset_name):
    dataset_path = os.path.join(base_path, dataset_name)
    image_dir = os.path.join(dataset_path, 'images')
    mask_dir = os.path.join(dataset_path, 'masks')
    
    if not os.path.exists(image_dir) or not os.path.exists(mask_dir):
        print(f"Warning: Directory does not exist for {dataset_name}")
        return [], []
    
    image_extensions = ['*.png', '*.tif', '*.tiff', '*.jpg', '*.jpeg']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(image_dir, ext)))
    
    mask_paths = []
    for img_path in image_paths:
        img_name = os.path.basename(img_path)
        img_name_no_ext = os.path.splitext(img_name)[0]
        
        for mask_ext in ['png', 'tif', 'tiff']:
            mask_path = os.path.join(mask_dir, f"{img_name_no_ext}.{mask_ext}")
            if os.path.exists(mask_path):
                mask_paths.append(mask_path)
                break
    
    valid_pairs = []
    valid_mask_paths = []
    for img_path, mask_path in zip(image_paths, mask_paths):
        if os.path.exists(img_path) and os.path.exists(mask_path):
            valid_pairs.append(img_path)
            valid_mask_paths.append(mask_path)
    
    return valid_pairs, valid_mask_paths

def load_micronet_weights(model_name='micronet-resnet50'):
    try:
        if model_name == 'micronet-resnet50':
            filenames_to_try = [
                "resnet50_micronet_weights.pth",
                "resnet50_imagenet-micronet_weights.pth", 
                "pytorch_model.bin"
            ]
            repo_id = "jstuckner/microscopy-resnet50-micronet"
            
        elif model_name == 'micronet-se_resnext101':
            filenames_to_try = [
                "resnext101_micronet_weights.pth",
                "resnext101_imagenet-micronet_weights.pth",
                "pytorch_model.bin"
            ]
            repo_id = "jstuckner/microscopy-resnext101-micronet"
        else:
            raise ValueError(f"Unknown MicroNet model: {model_name}")
        
        weights = None
        for filename in filenames_to_try:
            try:
                model_path = hf_hub_download(repo_id=repo_id, filename=filename)
                weights = torch.load(model_path, map_location='cpu')
                print(f"Successfully loaded MicroNet weights for {model_name} from {filename}")
                break
            except Exception:
                continue
        
        return weights
    
    except Exception as e:
        print(f"Failed to load MicroNet weights for {model_name}: {str(e)}")
        return None

def create_encoder(encoder_name='resnet50', pretrained='imagenet'):
    if encoder_name == 'resnet50':
        if pretrained == 'imagenet':
            encoder = smp.encoders.get_encoder('resnet50', in_channels=3, weights='imagenet')
        elif pretrained == 'micronet':
            encoder = smp.encoders.get_encoder('resnet50', in_channels=3, weights=None)
            micronet_weights = load_micronet_weights('micronet-resnet50')
            if micronet_weights:
                try:
                    encoder.load_state_dict(micronet_weights, strict=False)
                except Exception:
                    model_dict = encoder.state_dict()
                    pretrained_dict = {k: v for k, v in micronet_weights.items() 
                                     if k in model_dict and v.shape == model_dict[k].shape}
                    model_dict.update(pretrained_dict)
                    encoder.load_state_dict(model_dict)
        else:
            encoder = smp.encoders.get_encoder('resnet50', in_channels=3, weights=None)
            
        encoder_channels = [3, 64, 256, 512, 1024, 2048]
    
    elif encoder_name == 'se_resnext101':
        if pretrained == 'imagenet':
            encoder = smp.encoders.get_encoder('se_resnext101_32x4d', in_channels=3, weights='imagenet')
        elif pretrained == 'micronet':
            encoder = smp.encoders.get_encoder('se_resnext101_32x4d', in_channels=3, weights='imagenet')
            micronet_weights = load_micronet_weights('micronet-se_resnext101')
            if micronet_weights:
                try:
                    model_dict = encoder.state_dict()
                    pretrained_dict = {}
                    for k, v in micronet_weights.items():
                        if k in model_dict and v.shape == model_dict[k].shape:
                            pretrained_dict[k] = v
                    
                    if len(pretrained_dict) > 0:
                        model_dict.update(pretrained_dict)
                        encoder.load_state_dict(model_dict)
                except Exception:
                    pass
        else:
            encoder = smp.encoders.get_encoder('se_resnext101_32x4d', in_channels=3, weights=None)
            
        encoder_channels = [3, 64, 256, 512, 1024, 2048]
    
    return encoder, encoder_channels

class UNetDecoder(nn.Module):
    def __init__(self, encoder_channels, decoder_channels, n_classes=1):
        super(UNetDecoder, self).__init__()
        
        self.decoder_blocks = nn.ModuleList()
        
        for i in range(len(decoder_channels)):
            if i == 0:
                in_ch = encoder_channels[-1]
            else:
                in_ch = decoder_channels[i-1]
            
            out_ch = decoder_channels[i]
            skip_ch = encoder_channels[-(i+2)] if i < len(encoder_channels)-1 else 0
            
            block = nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
                nn.Conv2d(out_ch + skip_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )
            
            self.decoder_blocks.append(block)
        
        self.final_conv = nn.Conv2d(decoder_channels[-1], n_classes, kernel_size=1)
        
    def forward(self, encoder_features):
        x = encoder_features[-1]
        
        for i, decoder_block in enumerate(self.decoder_blocks):
            x = decoder_block[0](x)
            
            if i < len(encoder_features) - 1:
                skip = encoder_features[-(i+2)]
                if x.shape[2:] != skip.shape[2:]:
                    x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
                x = torch.cat([x, skip], dim=1)
            
            for layer in decoder_block[1:]:
                x = layer(x)
        
        return self.final_conv(x)

class ASPPModule(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rates):
        super(ASPPModule, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.atrous_convs = nn.ModuleList()
        for rate in atrous_rates:
            self.atrous_convs.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=rate, dilation=rate, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))
        
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * (2 + len(atrous_rates)), out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
        
    def forward(self, x):
        h, w = x.shape[2:]
        
        feat1 = self.conv1(x)
        atrous_feats = [conv(x) for conv in self.atrous_convs]
        global_feat = self.global_pool(x)
        global_feat = F.interpolate(global_feat, size=(h, w), mode='bilinear', align_corners=False)
        
        concat_feat = torch.cat([feat1] + atrous_feats + [global_feat], dim=1)
        return self.project(concat_feat)

class DeepLabV3Decoder(nn.Module):
    def __init__(self, encoder_channels, n_classes=1):
        super(DeepLabV3Decoder, self).__init__()
        self.aspp = ASPPModule(encoder_channels[-1], 256, [6, 12, 18])
        self.classifier = nn.Conv2d(256, n_classes, 1)
        
    def forward(self, encoder_features):
        x = encoder_features[-1]
        x = self.aspp(x)
        x = self.classifier(x)
        x = F.interpolate(x, scale_factor=32, mode='bilinear', align_corners=False)
        return x

class DeepLabV3PlusDecoder(nn.Module):
    def __init__(self, encoder_channels, n_classes=1):
        super(DeepLabV3PlusDecoder, self).__init__()
        
        self.aspp = ASPPModule(encoder_channels[-1], 256, [6, 12, 18])
        low_level_channels = encoder_channels[1]
        
        self.low_level_proj = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        
        self.decoder = nn.Sequential(
            nn.Conv2d(256 + 48, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.classifier = nn.Conv2d(256, n_classes, 1)
        
    def forward(self, encoder_features):
        low_level = encoder_features[1]
        high_level = encoder_features[-1]
        
        high_level = self.aspp(high_level)
        high_level = F.interpolate(high_level, size=low_level.shape[2:], mode='bilinear', align_corners=False)
        
        low_level = self.low_level_proj(low_level)
        
        concat_feat = torch.cat([high_level, low_level], dim=1)
        decoded = self.decoder(concat_feat)
        output = self.classifier(decoded)
        
        output = F.interpolate(output, scale_factor=2, mode='bilinear', align_corners=False)
        
        return output

class MaterialSegmentationModel(nn.Module):
    def __init__(self, encoder_name='resnet50', decoder_name='unet', pretrained='imagenet', n_classes=1):
        super(MaterialSegmentationModel, self).__init__()
        
        self.encoder, encoder_channels = create_encoder(encoder_name, pretrained)
        
        if decoder_name == 'unet':
            self.decoder = UNetDecoder(encoder_channels[1:], [256, 128, 64, 32], n_classes)
        elif decoder_name == 'deeplabv3':
            self.decoder = DeepLabV3Decoder(encoder_channels[1:], n_classes)
        elif decoder_name == 'deeplabv3plus':
            self.decoder = DeepLabV3PlusDecoder(encoder_channels, n_classes)
        
        self.config = f"{encoder_name}_{decoder_name}_{pretrained}"
        
    def forward(self, x):
        encoder_features = self.encoder(x)
        
        if hasattr(self.decoder, '__class__') and self.decoder.__class__.__name__ == 'DeepLabV3PlusDecoder':
            output = self.decoder(encoder_features)
        else:
            output = self.decoder(encoder_features[1:])
        
        return output

def dice_coefficient(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    
    if target.dim() == 3 and pred.dim() == 4:
        target = target.unsqueeze(1)
    
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice.mean()

def iou_score(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    
    if target.dim() == 3 and pred.dim() == 4:
        target = target.unsqueeze(1)
    
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    return iou.mean()

def train_model(model, train_loader, val_loader, num_epochs=200, device='cuda'):
    model = model.to(device)
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    best_val_iou = 0
    history = {
        'train_losses': [],
        'val_losses': [],
        'val_ious': [],
        'val_dices': []
    }
    
    print(f"Training {model.config} for {num_epochs} epochs on {device}")
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_batches = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for images, masks in pbar:
            images, masks = images.to(device), masks.to(device).float()
            
            optimizer.zero_grad()
            outputs = model(images)
            
            if outputs.shape[2:] != masks.shape[1:]:
                outputs = F.interpolate(outputs, size=masks.shape[1:], mode='bilinear', align_corners=False)
            
            outputs = outputs.squeeze(1)
            loss = criterion(outputs, masks)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            if train_batches % 10 == 0:
                torch.cuda.empty_cache()
        
        model.eval()
        val_loss = 0
        val_iou = 0
        val_dice = 0
        val_batches = 0
        
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device).float()
                
                outputs = model(images)
                if outputs.shape[2:] != masks.shape[1:]:
                    outputs = F.interpolate(outputs, size=masks.shape[1:], mode='bilinear', align_corners=False)
                
                outputs = outputs.squeeze(1)
                loss = criterion(outputs, masks)
                
                val_loss += loss.item()
                
                masks_reshaped = masks.unsqueeze(1) if masks.dim() == 3 else masks
                outputs_reshaped = outputs.unsqueeze(1)
                
                val_iou += iou_score(outputs_reshaped, masks_reshaped).item()
                val_dice += dice_coefficient(outputs_reshaped, masks_reshaped).item()
                val_batches += 1
        
        avg_train_loss = train_loss / train_batches
        avg_val_loss = val_loss / val_batches
        avg_val_iou = val_iou / val_batches
        avg_val_dice = val_dice / val_batches
        
        history['train_losses'].append(avg_train_loss)
        history['val_losses'].append(avg_val_loss)
        history['val_ious'].append(avg_val_iou)
        history['val_dices'].append(avg_val_dice)
        
        scheduler.step(avg_val_loss)
        
        if epoch % 10 == 0 or epoch >= num_epochs - 5:
            print(f'Epoch {epoch+1:3d}: Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val IoU: {avg_val_iou:.4f} | Val Dice: {avg_val_dice:.4f}')
        
        if avg_val_iou > best_val_iou:
            best_val_iou = avg_val_iou
            torch.save(model.state_dict(), f'best_{model.config}.pth')
            if epoch % 10 == 0 or epoch >= num_epochs - 5:
                print(f'  New best IoU: {best_val_iou:.4f}')
        
        torch.cuda.empty_cache()
    
    history['best_val_iou'] = best_val_iou
    print(f"Training completed! Best validation IoU: {best_val_iou:.4f}")
    return history

def run_experiment(dataset_name, encoder_name, decoder_name, pretrained, epochs, batch_size, image_size, device='cuda'):
    print(f"Starting experiment: {dataset_name} | {encoder_name} | {decoder_name} | {pretrained}")
    
    image_paths, mask_paths = load_dataset_paths(".", dataset_name)
    
    if len(image_paths) < 4:
        print(f"Insufficient data for {dataset_name}: {len(image_paths)} samples")
        return None, None
    
    train_images, val_images, train_masks, val_masks = train_test_split(
        image_paths, mask_paths, test_size=0.2, random_state=42
    )
    
    print(f"Dataset split: {len(train_images)} train, {len(val_images)} validation")
    
    train_dataset = MaterialDataset(train_images, train_masks, image_size=(image_size, image_size))
    val_dataset = MaterialDataset(val_images, val_masks, image_size=(image_size, image_size))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    model = MaterialSegmentationModel(encoder_name, decoder_name, pretrained, n_classes=1)
    
    history = train_model(model, train_loader, val_loader, num_epochs=epochs, device=device)
    
    result = {
        'dataset': dataset_name,
        'encoder': encoder_name,
        'decoder': decoder_name,
        'pretrained': pretrained,
        'config': f"{encoder_name}_{decoder_name}_{pretrained}",
        'train_samples': len(train_images),
        'val_samples': len(val_images),
        'best_val_iou': history['best_val_iou'],
        'final_val_dice': history['val_dices'][-1],
        'final_train_loss': history['train_losses'][-1],
        'final_val_loss': history['val_losses'][-1]
    }
    
    print(f"Results: IoU={result['best_val_iou']:.4f} | Dice={result['final_val_dice']:.4f}")
    
    return result, history

def create_visualization(results_df, save_path='experiment_results.png'):
    if len(results_df) == 0:
        print("No results to visualize")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    sorted_results = results_df.sort_values('best_val_iou', ascending=True)
    bars = axes[0,0].barh(range(len(sorted_results)), sorted_results['best_val_iou'])
    axes[0,0].set_yticks(range(len(sorted_results)))
    axes[0,0].set_yticklabels(sorted_results['config'], fontsize=8)
    axes[0,0].set_xlabel('IoU Score')
    axes[0,0].set_title('Performance Ranking by Configuration')
    axes[0,0].grid(True, alpha=0.3)
    
    for i, bar in enumerate(bars):
        if sorted_results.iloc[i]['best_val_iou'] > 0.7:
            bar.set_color('green')
        elif sorted_results.iloc[i]['best_val_iou'] > 0.5:
            bar.set_color('orange')
        else:
            bar.set_color('red')
    
    if 'encoder' in results_df.columns:
        encoder_performance = results_df.groupby('encoder')['best_val_iou'].agg(['mean', 'std'])
        encoder_performance['mean'].plot(kind='bar', ax=axes[0,1], yerr=encoder_performance['std'], capsize=4)
        axes[0,1].set_title('IoU by Encoder')
        axes[0,1].set_ylabel('Mean IoU')
        axes[0,1].tick_params(axis='x', rotation=45)
        axes[0,1].grid(True, alpha=0.3)
    
    if 'pretrained' in results_df.columns:
        pretrain_performance = results_df.groupby('pretrained')['best_val_iou'].agg(['mean', 'std'])
        pretrain_performance['mean'].plot(kind='bar', ax=axes[1,0], yerr=pretrain_performance['std'], capsize=4)
        axes[1,0].set_title('IoU by Pretraining Strategy')
        axes[1,0].set_ylabel('Mean IoU')
        axes[1,0].tick_params(axis='x', rotation=45)
        axes[1,0].grid(True, alpha=0.3)
    
    axes[1,1].hist(results_df['best_val_iou'], bins=10, alpha=0.7, edgecolor='black')
    axes[1,1].axvline(results_df['best_val_iou'].mean(), color='red', linestyle='--', label='Mean')
    axes[1,1].set_xlabel('IoU Score')
    axes[1,1].set_ylabel('Frequency')
    axes[1,1].set_title('IoU Score Distribution')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Material Segmentation Model Garden Experiment')
    
    parser.add_argument('--datasets', nargs='+', default=['AFM'], 
                       help='Dataset names (default: AFM)')
    parser.add_argument('--encoders', nargs='+', default=['resnet50'], 
                       choices=['resnet50', 'se_resnext101'],
                       help='Encoder architectures (default: resnet50)')
    parser.add_argument('--decoders', nargs='+', default=['unet'], 
                       choices=['unet', 'deeplabv3', 'deeplabv3plus'],
                       help='Decoder architectures (default: unet)')
    parser.add_argument('--pretrained', nargs='+', default=['imagenet'], 
                       choices=['imagenet', 'micronet'],
                       help='Pretraining strategies (default: imagenet)')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs (default: 50)')
    parser.add_argument('--batch-size', type=int, default=2,
                       help='Batch size (default: 2)')
    parser.add_argument('--image-size', type=int, default=256,
                       help='Image size (default: 256)')
    parser.add_argument('--device', default='cuda',
                       help='Device to use (default: cuda)')
    parser.add_argument('--output', default='experiment_results.csv',
                       help='Output CSV file (default: experiment_results.csv)')
    parser.add_argument('--visualize', action='store_true',
                       help='Create visualizations')
    
    args = parser.parse_args()
    
    print("Material Data Science Model Garden (MDSMG)")
    print("=" * 80)
    print(f"Datasets: {args.datasets}")
    print(f"Encoders: {args.encoders}")
    print(f"Decoders: {args.decoders}")
    print(f"Pretraining: {args.pretrained}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Image size: {args.image_size}")
    print(f"Device: {args.device}")
    print("=" * 80)
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"GPU Memory available: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    results = []
    experiment_count = 0
    total_experiments = len(args.datasets) * len(args.encoders) * len(args.decoders) * len(args.pretrained)
    
    print(f"Total experiments to run: {total_experiments}")
    
    for dataset_name in args.datasets:
        for encoder in args.encoders:
            for decoder in args.decoders:
                for pretrained in args.pretrained:
                    experiment_count += 1
                    
                    print(f"\nRunning experiment {experiment_count}/{total_experiments}")
                    
                    result, history = run_experiment(
                        dataset_name, encoder, decoder, pretrained, 
                        args.epochs, args.batch_size, args.image_size, args.device
                    )