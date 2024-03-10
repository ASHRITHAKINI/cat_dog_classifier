# cat_dog_classifier
Cat and Dog Image Classifier This project implements an image classification model to distinguish between images of cats and dogs using data science techniques in Python. The model is built using the Fastai library, which provides high-level abstractions for training deep learning models.
Project Overview:
Data Collection: The project uses the "Dogs vs. Cats" dataset available in the Fastai library. This dataset contains images of cats and dogs, which are used for training and evaluation.

Data Preprocessing: The images are preprocessed by resizing them to a uniform size of 224x224 pixels. The dataset is split into training and validation sets, with 20% of the data reserved for validation.

Model Selection: The model architecture chosen for this project is a Convolutional Neural Network (CNN) based on the ResNet34 architecture. Transfer learning is used, where the pre-trained ResNet34 model is fine-tuned on the "Dogs vs. Cats" dataset.

Model Training: The model is trained on the training dataset using the Fastai library. The fine-tuning process involves updating the parameters of the pre-trained model to better fit the "Dogs vs. Cats" dataset.

Model Evaluation: The trained model is evaluated on the validation dataset to assess its performance. The model's performance is evaluated using metrics such as error rate and confusion matrix.

Model Deployment: An example of model deployment is provided, where users can upload an image of a cat or dog, and the model predicts whether the image contains a cat or a dog.

Usage:
To use the image classifier:

Clone this repository to your local machine.
Install the required dependencies by running pip install -r requirements.txt.
Run the provided Python script to train the model and make predictions.

Dependencies:
Python 3.x
Fastai
PyTorch
Jupyter Notebook (optional)
