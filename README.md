# Binary Image Classification Neural Network

This repository contains code and documentation for a binary image classification neural network. The primary goal of this project is to experiment with Explainable AI (XAI) techniques, with a particular focus on using Shapley Values from game theory, to interpret and understand the decisions made by the neural network.

## Project Overview

### Data Preparation

The dataset used for this project consists of binary images. Here are the key steps in data preprocessing:

1. **Image Loading**: Images were loaded from the file system using the `os` library. The path to each image was obtained, and the images were read into memory.

2. **Image Processing with OpenCV**: OpenCV was used for various image processing tasks:
   - **Resizing**: Images were resized to a common size of 128x128 pixels to ensure uniformity in the dataset.
   - **Grayscale Conversion**: Images were converted to grayscale to reduce data dimensionality and simplify processing.

3. **Data Representation**: Each image was transformed into a single row in a DataFrame. In this process, a 128x128 grayscale image with pixel values ranging from 0 to 255 was flattened into a row vector with 16,384 columns. Each column represented the grayscale value of a pixel.

4. **Label Encoding**: Labels or categories associated with each image were encoded using a label encoder. This assigned a unique numerical value to each category, making it suitable for machine learning models.

### Neural Network Implementation

The neural network was created from scratch, incorporating the following components:

1. **Forward Propagation**: Forward propagation involved the mathematical functions for linear transformations and activation functions.
   
   - **Mathematics of Forward Propagation**: Given an input \(x\) and the weights \(W\) and biases \(b\) of a neuron, the forward propagation can be expressed as:
   
     \[z = Wx + b\]
     \[a = \text{ReLU}(z)\]

2. **Backward Propagation**: Backpropagation was used for updating the network's weights and biases.

3. **Activation Functions**: ReLU (Rectified Linear Unit) and Softmax activation functions were used.
   
   - **Mathematics of ReLU (Rectified Linear Unit)**:
     \[f(x) = \begin{cases}
     x & \text{if } x > 0 \\
     0 & \text{otherwise}
     \end{cases}\]

   - **Mathematics of Softmax**:
     \[S_i(z) = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}\]

4. **Gradient Descent**: Gradient descent was employed for optimizing the network's parameters.

### Model Evaluation

The performance of the binary image classification model was assessed using several evaluation metrics:

- **Accuracy**: Accuracy measures the proportion of correctly classified samples to the total number of samples. It is calculated as follows:

  \[Accuracy = \frac{TP + TN}{TP + TN + FP + FN}\]

  Where:
  - \(TP\) (True Positives) is the number of correctly predicted positive samples.
  - \(TN\) (True Negatives) is the number of correctly predicted negative samples.
  - \(FP\) (False Positives) is the number of incorrectly predicted positive samples.
  - \(FN\) (False Negatives) is the number of incorrectly predicted negative samples.

- **Precision**: Precision measures the proportion of true positive predictions out of all positive predictions made by the model. It is calculated as follows:

  \[Precision = \frac{TP}{TP + FP}\]

- **Recall (Sensitivity)**: Recall, also known as sensitivity or true positive rate, measures the proportion of true positive predictions out of all actual positive samples. It is calculated as follows:

  \[Recall = \frac{TP}{TP + FN}\]

- **F1 Score**: The F1 score is the harmonic mean of precision and recall. It provides a balance between precision and recall. It is calculated as follows:

  \[F1 \text{ Score} = \frac{2 \cdot Precision \cdot Recall}{Precision + Recall}\]

## Usage

To run this code, follow these steps:

1. Clone this repository to your local machine.
2. Ensure you have the required dependencies installed.
3. Execute the main script to train and test the neural network on your dataset.

## Explainable AI (XAI)

The main goal of this project is to experiment with Explainable AI (XAI) techniques, particularly focusing on using Shapley Values from game theory, to interpret and understand the decisions made by the neural network.

### Shapley Values for Explaining Neural Networks

Shapley Values are a concept derived from cooperative game theory that offers a principled way to explain the predictions of complex machine learning models like neural networks. Here's how Shapley Values can be applied to interpret neural network predictions:

1. **Attributing Predictions**: Shapley Values help attribute the prediction of a model to individual input features (pixels in the case of images). These values quantify how much each pixel contributed to a particular prediction. By analyzing Shapley Values, you can identify which parts of an image influenced the decision, providing insights into feature importance.

2. **Interpreting Black-Box Models**: Neural networks are often considered black-box models because they can be challenging to understand. Shapley Values provide a way to open this black box and gain insights into why a model made a specific prediction. By examining Shapley Values, you can understand which pixels had the most impact on a prediction.

3. **Feature Importance**: Shapley Values can be used to rank the importance of pixels in an image. This ranking is valuable for understanding which regions of an image are critical for the model's decision. Identifying important features can be particularly useful in applications where pinpointing model biases or identifying erroneous classifications is essential.

To calculate Shapley Values for a neural network, various techniques and algorithms can be employed. These methods often involve permutations and combinations of input features to measure their individual and collective impact on predictions. Shapley Values provide a fair and interpretable way to explain complex neural network decisions, making them a valuable tool in the field of Explainable AI (XAI).

## Future Improvements

- Incorporate other XAI techniques like LIME or Integrated Gradients for more comprehensive model interpretability.
- Enhance the model's performance through architectural improvements and data augmentation.
- Provide a user-friendly interface or visualization tool for exploring the model's predictions and interpretations.

## Contact

For questions or feedback, please contact [jarifbegg@gmail.com].

