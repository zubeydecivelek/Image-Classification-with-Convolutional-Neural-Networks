# Micro Organism Classification

This project focuses on classifying microorganisms using deep learning techniques. It involves two parts: training custom CNN models from scratch and fine-tuning a pre-trained ResNet-18 network. The dataset consists of images of different microorganisms, divided into various classes.

## Part-1: Training Custom CNN Models

In this part, custom CNN models are built and trained from scratch without using pre-trained weights. The models are designed to learn features specific to the Micro_Organism dataset. The steps involved are as follows:

1. Dataset Preparation:
   - The dataset is organized into folders, with each folder representing a specific microorganism class.
   - Images are resized, converted to tensors, and normalized.
   - The dataset is split into train, validation, and test sets.

2. Model Architecture:
   - Two models are defined: ModelWithoutResidual and ModelWithResidual.
   - ModelWithoutResidual uses standard convolutional layers, max pooling, and fully connected layers.
   - ModelWithResidual incorporates residual blocks in addition to the standard layers.

3. Training and Evaluation:
   - Models are trained using the training set and evaluated on the validation set.
   - Hyperparameters such as learning rate and batch size are varied to find the best model.
   - Loss and accuracy curves are plotted to visualize the training progress.
   - The model with the highest validation accuracy is selected and evaluated on the test set.

## Part-2: Fine-tuning a Pre-trained ResNet-18 Network

In this part, a pre-trained ResNet-18 network is used and fine-tuned to classify the microorganism images. The steps involved are as follows:

1. Pre-trained Model:
   - The ResNet-18 network, pre-trained on the ImageNet dataset, is loaded.
   - All layers except the fully connected layer are frozen.

2. Model Modification:
   - The last fully connected layer is modified to match the number of classes in the Micro_Organism dataset.
   - The weights of the last layer are randomly initialized.

3. Training and Evaluation:
   - Two cases are explored: training only the fully connected layer and freezing the rest, and training the last two convolutional layers and the fully connected layer while freezing the rest.
   - Models are trained using the training set and evaluated on the validation and test sets.
   - Accuracy on the validation and test sets is recorded for comparison.

## Results and Analysis:

1. Part-1 vs. Part-2:
   - Fine-tuning the pre-trained network outperformed training from scratch in terms of accuracy and training efficiency.
   - Fine-tuning utilized the learned features of the pre-trained network, leading to improved generalization and faster convergence.

2. Conclusion:
   - Fine-tuning a pre-trained network is a powerful technique, especially when working with limited training data or when a pre-trained model is available on a related task.
   - The best model achieved higher accuracy on the validation and test sets compared to the models trained from scratch.

## Dependencies:

- Python (>=3.6)
- PyTorch (>=1.7.0)
- Torchvision (>=0.8.0)
- Matplotlib (>=3.3.0)