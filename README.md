# Traffic

## Experimentation Process

During the experimentation process, I aimed to create a convolutional neural network (CNN) model using the Keras framework. The goal was to build a model that could classify images into different categories. Here is an overview of the steps I followed:

1. **Defining the Model Architecture**: I started by defining the architecture of the CNN model. The model consists of multiple layers including convolutional layers, max-pooling layers, flatten layers, and dense layers. The number of units and other parameters for each layer were determined based on the requirements of the task.

2. **Compiling the Model**: After defining the architecture, I compiled the model using the Adam optimizer. For the loss function, I chose categorical cross-entropy since it is commonly used for multi-class classification problems. I also specified accuracy as a metric to evaluate the performance of the model during training.

3. **Training and Evaluation**: With the model compiled, I used a training dataset to fit the model. The dataset contained images of various categories. During the training process, the model learned to recognize patterns and features in the images and make predictions accordingly. After training, I evaluated the model's performance using an evaluation dataset, measuring both the loss and accuracy.

4. **Observations**: While experimenting, I made several observations:
   - The use of convolutional layers with increasing numbers of filters allowed the model to extract higher-level features from the images.
   - Max-pooling layers helped reduce the spatial dimensions of the feature maps, enabling the model to focus on important features.
   - Adding a dense hidden layer with ReLU activation helped capture complex relationships between features.
   - The use of dropout regularization in the hidden layer prevented overfitting by randomly disabling some neurons during training.
   - The output layer utilized the softmax activation function to generate class probabilities for each category.

## Results

The experimentation process led to the creation of a CNN model with the defined architecture. The model showed promising results in classifying images into different categories. Some key outcomes are as follows:

- The model architecture successfully learned and extracted relevant features from the images, allowing it to make accurate predictions.
- The use of convolutional and max-pooling layers helped capture local patterns and reduce spatial dimensions, respectively.
- The dense hidden layer with ReLU activation and dropout regularization enhanced the model's ability to learn complex relationships and avoid overfitting.
- The output layer with softmax activation provided probability distributions over the categories, enabling effective classification.

## Further Improvements

Although the model achieved satisfactory results, there is always room for improvement. And so:


After conducting further experimentation and researching the top-ranked model in the German traffic sign benchmark, I discovered that their research paper suggested an improved model architecture for the traffic sign classification problem. Taking inspiration from their findings, I made adjustments to my model to incorporate their recommendations. Here is an updated description of the model architecture:

- **Convolutional Layer 1**: The first convolutional layer consists of 200 filters with a 7x7 kernel size, employing the ReLU activation function. This layer captures lower-level features from the input images.

- **Max-Pooling Layer 1**: Following the first convolutional layer, a 2x2 MaxPooling operation is applied to reduce the spatial dimensions of the feature maps, focusing on the most important features while maintaining local spatial relationships.

- **Convolutional Layer 2**: The second convolutional layer comprises 250 filters with a 4x4 kernel size, also utilizing the ReLU activation function. This layer further extracts higher-level features from the feature maps generated by the previous layer.

- **Max-Pooling Layer 2**: Similar to the previous pooling layer, a 2x2 MaxPooling operation is performed to downsample the feature maps and preserve the most salient information.

- **Hidden Layer**: After the pooling layers, a dense hidden layer with 400 units and ReLU activation is introduced. This layer allows the model to capture complex relationships between the extracted features.

- **Dropout Regularization**: To mitigate overfitting, a dropout regularization with a rate of 0.5 is applied to the hidden layer. Dropout randomly deactivates a portion of the neurons during training, encouraging the model to learn more robust and generalized representations.

- **Output Layer**: The final layer of the model is a dense output layer with softmax activation. This layer produces class probabilities for all the categories in the problem, enabling the model to make predictions.

By incorporating the insights from the top-ranked model in the German traffic sign benchmark, I have adjusted my model to include 2 Conv2D layers. These modifications aim to enhance the model's ability to capture relevant features and improve classification performance in traffic sign recognition tasks.

By exploring these avenues and fine-tuning the model, it was possible to further enhance its performance and accuracy in image classification tasks.

**You can watch the training process of the model** [here](https://youtu.be/WdbnVu4TzjA).