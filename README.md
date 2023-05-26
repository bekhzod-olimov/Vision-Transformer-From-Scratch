# Vision Transformer (ViT) from scratch
This repository contains from scratch implemetation, sanity check (with [a timm ViT model](https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py)), and inference of [ViT](https://arxiv.org/pdf/2010.11929.pdf) using PyTorch.

### Run the model script
```python
python model.py
```

### Verify the model script
Double check that the VIT from scratch is implemented correctly by comparing with VIT implementation from timm libarary.
```python
python verify.py
```

### Inference with VIT
Conduct inference with random images from Internet by running this script.
```python
python inference.py
```
