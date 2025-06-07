# Additive Manufacturing Fingerprinting

![Fig1](https://github.com/user-attachments/assets/05cf7bb8-be13-4539-a582-6886d3cf997c)

Code for the paper: Additive Manufacturing Source Identification from Photographs using Deep Learning

Disclaimer: This repo is not plug-and-play and will take some modifications of file paths to run.

The data for the 21 machine model can be found on [Kaggle](https://www.kaggle.com/datasets/milesbimrose/additive-manufacturing-source-identification-uiuc)

[Dataset citation](https://doi.org/10.34740/kaggle/dsv/10072431)

## Running locally

### Setup

- Ensure Python is installed - `python --version`
- (Optional) Create virtual environment - `python -m venv venv`
  - (Windows) active virtual environment - `venv\Scripts\activate`
- Install dependencies - `pip install torch torchvision pandas numpy timm torch_optimizer joblib wandb`

### Running inference

- Run the inference script to get a prediction for a given image (replace "PATH_TO_IMAGE" with an actual path to an image) - `python .\inference.py PATH_TO_IMAGE`
  - i.e. `python .\inference.py .\examples\F3B-1_096_loc05_FaceScan.png`
- Example images can be downloaded from the [Kaggle dataset](https://www.kaggle.com/datasets/milesbimrose/additive-manufacturing-source-identification-uiuc/data)

```bash
(venv) PS C:\dev\fingerprinting> python .\inference.py .\examples\F3B-1_096_loc05_FaceScan.png
Using device: cpu
Model loaded successfully from ./data/Models/fingerprinting_21_printer_best_model_efficientnetv2_m.pth
Image loaded: .\examples\F3B-1_096_loc05_FaceScan.png (size: (5375, 3905))

Results for .\examples\F3B-1_096_loc05_FaceScan.png:
Predicted printer: Form3B-2
Confidence score: 1.0000 (100.00%)

Top 5 predictions:
  1. Form3B-2 - 100.00%
  2. Form3B-5 - 0.00%
  3. M2-2 - 0.00%
  4. Form3B-3 - 0.00%
  5. L1-2 - 0.00%
```
