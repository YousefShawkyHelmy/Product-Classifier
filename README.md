# Product Image Classifier
This repository contains the code for a Product Image Classifier using Machine learning modeling. It is used to enhance an e-commerce app's intelligence, through  categorizing a given image into the app groups categories like fashion, nutrition, or accessories ... etc.

And here are the main  approach and functionalities of the Model:

## 1. Dataset Download
I started by taking screenshots to the products images from the mobile application (you can download it from: [Slash](slash-eg.com)), and grouping them in local directory, each category together in a file, creating by then a folder of labeled image dataset.

## 2. Data Preparation
* First of all, the dataset images needed to be reasized all of the same pixil size. So, I resized all the dataset.
* After that, I split the dataset locally into training, validation, and test sets, creating a separated directory for each set.
* As a last step in the data preprocessing phase: I rescaled all of the images, so the pixel values of each image to be in the range [0,1], which is typically done to facilitate the training process, as neural networks tend to perform better with input values in this range.

## 3. Model Building
The main model used for this classifier is the ResNet50 convolutional neural network. I **fine-tuned** the pre-trained ResNet50 base model by adding  custom classification layers on top of it. 
Meanwhile, all layers in the pre-trained ResNet50 model are set to non-trainable, meaning their weights will not be updated during training. This step is done to keep the pre-trained weights intact and prevent them from being modified, i.e., training only the weights of the custom layers while keeping the pre-trained weights frozen.

After Compiling and Training the model, the Evaluatino came to be: Test Loss: 1.59 and Test Accuracy: 0.69 .

And by testing the model on random photos from the internet, it seems to be working just fine.

Here is a short video demonstrating my code : https://drive.google.com/file/d/1BwKfDvp1rokw7Q94qIh98WCoJzCAq19m/view?usp=sharing