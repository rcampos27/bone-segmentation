# Application of convolutional neural networks for jaw bone tissue segmentation in tomographies

## Overview

This project is designed for the identification and segmentation of the multiple bone tissues within dental CBCT images. It leverages Convolutional Neural Networks (CNNs) to perform image segmentation tasks, distinguishing the bone tissue between four types according to Rebaudi's Classification:
1. Soft (Low Density Bone Tissue)
2. Normal (Intermediate Density Bone Tissue)
3. Hard (High Density Bone Tissue)
4. Enamel


The project is organized into four main components:

1. **Pre-processing:** This module processes DICOM images, applies intensity transformations, creates segmentation masks, and performs various image enhancements to prepare the data for training.

2. **Dataset:** The custom dataset class manages loading image and mask pairs, applying optional data augmentation transformations, and facilitating data feeding to the model during training.

3. **Model:** The neural network architecture is a U-Net variant designed for image segmentation tasks. It comprises encoding and decoding components for efficient feature extraction and upsampling.

4. **Train:** The training script initializes the model, defines loss functions, optimizers, and carries out the training loop. It also includes functionalities for loading checkpoints, checking accuracy, and saving model predictions.

## Requirements

- Python 3.x
- PyTorch
- NumPy
- OpenCV
- scikit-image
- tqdm

## Usage

1. Run the preprocessing script:

    ```bash
    python preprocessing.py
    ```

2. Train the model:

    ```bash
    python train.py
    ```

3. Evaluate the model (optional):

   With the trained model, load it in a Jupyter Notebook (`.ipynb`) or a Python script (`.py`) file and run
    ```python
    prediction = model.evaluate(sample)
    ```

## Acknowledgments

- [Professor Dr. Adair Santa Catarina](http://lattes.cnpq.br/7041836941307184)
- [Dr. Sreenivas Bhattiprolu](https://github.com/bnsreenu/python_for_microscopists)
