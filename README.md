# DRL4AVSR
Official Implementation of "Decoupled Representation Learning for Robust Audio-visual Speech Recognition"

### Requirements
- python >= 3.9
- torch == 2.5.1+cu124
- torchaudio == 2.5.1+cu124
- torchvision == 0.20.1+cu124
- transformers == 4.52.4
- av == 15.1.0
- librosa == 0.11.0
- opencv-python == 4.11.0
- jiwer == 4.0.0
- umap-learn == 0.5.8

### Training / Inference
`python avtrain.py <cuda_id> [grid|cmlr|lrs3]`

### Datasets
- [GRID](https://spandh.dcs.shef.ac.uk/gridcorpus/)
- [CMLR](https://www.vipazoo.cn/CMLR.html)
- [LRS3-TED](https://www.robots.ox.ac.uk/~vgg/data/lip_reading/) ([link2](https://mmai.io/datasets/lip_reading/))
  
  
