# Image Classification - Eiffel Tower Prediction
Aim: To predict whether a given image is an Eiffel Tower or not using Convolutional Neural Network

Convolutional Neural Network: 
The process of building a Convolutional Neural Network 

Step 1: 
      Convolution - It is the first layer to extract features from an input image. Convolution preserves the relationship between pixels by learning image features using small squares of input data
      Step 1(B): ReLU - ReLU stands for Rectified Linear Unit for a non-linear operation. The output is ƒ(x) = max(0,x). ReLU’s purpose is to introduce non-linearity in our ConvNet. Since, the real world data would want our ConvNet to learn would be non-negative linear values.

Step 2:
      Pooling - Pooling layers section would reduce the number of parameters when the images are too large. It reduces the dimensionality of each map but retains the important information

Step 3:
      Flattening - Once the pooled featured map is obtained, the next step is to flatten it. Flattening involves transforming the entire pooled feature map matrix into a single column which is then fed to the neural network for processing.

Step 4:
      Full connection - Feature map matrix will be converted as vector (x1, x2, x3, …). With the fully connected layers, we combined these features together to create a model. 

Finally, we have an activation function such as softmax or sigmoid to classify the output.

Implementation:
     This problem statement is implemented using Keras Library (Python 3.6 version) in less than 30 lines of code.
     Training, Test data are available in the Dataset folder.
     Source code is available in Program.py file.
     
     
Installation:
     If Keras/Tensorflow is not installed,

Install Tensorflow using pip install tensorflow

Install keras using conda install keras 

Keras runs on top of Tensorflow. So Tensorflow installation is mandatory.
