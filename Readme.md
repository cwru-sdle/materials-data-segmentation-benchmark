# Material Data Science Model Garden (MDSMG)

A comprehensive deep learning framework for material image segmentation using various encoder-decoder architectures.

## Features

- Multiple encoder architectures: ResNet50, SE-ResNeXt101
- Multiple decoder architectures: UNet, DeepLabV3, DeepLabV3+
- Multiple pretraining strategies: ImageNet, MicroNet
- Support for various material datasets
- Command-line interface for easy experimentation
- Automatic result visualization and analysis

## Requirements

```bash
pip install torch torchvision
pip install segmentation-models-pytorch
pip install opencv-python
pip install scikit-learn
pip install pandas matplotlib
pip install tqdm pillow
pip install huggingface-hub
```

## Dataset Structure

Organize your datasets in the following structure:

```
project_root/
├── mdsmg_experiment.py
├── AFM/
│   ├── images/
│   │   ├── image1.png
│   │   ├── image2.tif
│   │   └── ...
│   └── masks/
│       ├── image1.png
│       ├── image2.tif
│       └── ...
├── SEM/
│   ├── images/
│   └── masks/
└── ...
```

Supported image formats: PNG, TIF/TIFF, JPG/JPEG

## Basic Usage

### 1. Single Dataset, Single Configuration

```bash
python mdsmg_experiment.py --datasets AFM --encoders resnet50 --decoders unet --pretrained imagenet
```

### 2. Multiple Configurations

```bash
python mdsmg_experiment.py --datasets AFM SEM --encoders resnet50 se_resnext101 --decoders unet deeplabv3 --pretrained imagenet micronet
```

### 3. Quick Test Run

```bash
python mdsmg_experiment.py --datasets AFM --epochs 10 --batch-size 1 --image-size 128
```

### 4. Full Experiment Suite

```bash
python mdsmg_experiment.py --datasets AFM SEM Carbon --encoders resnet50 se_resnext101 --decoders unet deeplabv3 deeplabv3plus --pretrained imagenet micronet --epochs 200 --visualize
```

## Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--datasets` | list | `['AFM']` | Dataset names to use |
| `--encoders` | list | `['resnet50']` | Encoder architectures |
| `--decoders` | list | `['unet']` | Decoder architectures |
| `--pretrained` | list | `['imagenet']` | Pretraining strategies |
| `--epochs` | int | `50` | Number of training epochs |
| `--batch-size` | int | `2` | Batch size for training |
| `--image-size` | int | `256` | Input image size (square) |
| `--device` | str | `'cuda'` | Device to use (cuda/cpu) |
| `--output` | str | `'experiment_results.csv'` | Output CSV filename |
| `--visualize` | flag | `False` | Create visualization plots |

## Available Options

### Encoders
- `resnet50`: ResNet-50 backbone
- `se_resnext101`: SE-ResNeXt-101 backbone

### Decoders
- `unet`: U-Net decoder
- `deeplabv3`: DeepLabV3 decoder
- `deeplabv3plus`: DeepLabV3+ decoder

### Pretraining
- `imagenet`: ImageNet pretrained weights
- `micronet`: MicroNet pretrained weights (specialized for microscopy)

## Example Commands

### Research Experiment
```bash
# Full comparison study
python mdsmg_experiment.py \
    --datasets AFM SEM Carbon \
    --encoders resnet50 se_resnext101 \
    --decoders unet deeplabv3 deeplabv3plus \
    --pretrained imagenet micronet \
    --epochs 200 \
    --batch-size 2 \
    --image-size 256 \
    --visualize \
    --output full_study_results.csv
```

### Quick Prototyping
```bash
# Fast testing with small configuration
python mdsmg_experiment.py \
    --datasets AFM \
    --encoders resnet50 \
    --decoders unet \
    --epochs 20 \
    --batch-size 4 \
    --image-size 128
```

### CPU Training
```bash
# For systems without GPU
python mdsmg_experiment.py \
    --datasets AFM \
    --device cpu \
    --batch-size 1 \
    --epochs 10
```

### Memory-Constrained Training
```bash
# For limited GPU memory
python mdsmg_experiment.py \
    --datasets AFM \
    --batch-size 1 \
    --image-size 128 \
    --epochs 50
```

## Output Files

The script generates several output files:

1. **CSV Results** (`experiment_results.csv`): Detailed metrics for each configuration
2. **Model Weights** (`best_*.pth`): Best model weights for each configuration
3. **Visualizations** (`experiment_results.png`): Performance comparison plots (if `--visualize` is used)

## Performance Monitoring

The script automatically displays:
- Training progress with loss values
- Validation metrics (IoU, Dice coefficient)
- Best model performance
- GPU memory usage
- Experiment completion status

## Memory Management

For systems with limited GPU memory:

1. Reduce batch size: `--batch-size 1`
2. Reduce image size: `--image-size 128`
3. Use CPU: `--device cpu`
4. Close other GPU applications

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Solution: Reduce batch size and image size
   python mdsmg_experiment.py --batch-size 1 --image-size 128
   ```

2. **Dataset Not Found**
   ```
   # Ensure dataset folder structure is correct
   # Check that images/ and masks/ folders exist
   ```

3. **MicroNet Weights Loading Failed**
   ```
   # This is normal - will fall back to ImageNet weights
   # MicroNet weights are optional
   ```

### Performance Tips

1. **Faster Training**: Use larger batch sizes if memory allows
2. **Better Results**: Use larger image sizes (512x512) if memory allows
3. **Quick Testing**: Start with 10-20 epochs for initial experiments
4. **Production**: Use 100-200 epochs for final results

## Results Interpretation

The CSV output contains:
- `best_val_iou`: Best validation IoU score (higher is better)
- `final_val_dice`: Final validation Dice coefficient
- `config`: Model configuration string
- `train_samples`: Number of training samples
- `val_samples`: Number of validation samples

Typical IoU scores:
- `> 0.8`: Excellent
- `0.6-0.8`: Good
- `0.4-0.6`: Fair
- `< 0.4`: Poor

## Citation


```bibtex
@software{,
  title={},
  author={},
  year={},
  url={}
}
```