# Age Prediction using Convolutional Neural Networks (CNNs) 🧠🎂

## About 🧑‍🔬
This project focuses on building a Convolutional Neural Network (CNN) to predict a person's age from images using deep learning techniques. The model is trained on a dataset consisting of facial images, with each image labeled by the subject's actual age. The model leverages multiple layers of convolutions, pooling, and dropout to generalize well over unseen data. It achieves age prediction by minimizing the Mean Squared Error (MSE) during training, which is essential for regression tasks like this.

## Impact 🌍
Age prediction has a broad range of real-world applications, including:
- **Healthcare**: Assisting in estimating biological age, which can be used in age-related disease diagnosis.
- **Security**: Automated age estimation for identity verification in applications like facial recognition.
- **Social Media**: Ensuring age-appropriate content delivery.
  
The model's architecture can be extended to other age-related tasks or used in conjunction with other systems for more refined predictions, making it valuable in research and commercial applications.

## Methodology ⚙️
The workflow followed for this research project consists of the following key steps:

1. **Data Loading and Preprocessing**: 
    - The images are loaded from a directory, resized to a fixed shape (200x200 pixels), and normalized by scaling pixel values between 0 and 1.
    - Each image is labeled based on the age extracted from the file name.

2. **Train-Test Split**:
    - We use an 80-20 split for training and testing.
  
3. **Model Architecture**: 
    - A deep CNN consisting of convolutional layers, pooling layers, and fully connected layers is designed.
    - Image augmentation (random rotation, zoom) is applied to enhance generalization.
  
4. **Training**:
    - The model is trained for 250 epochs with a batch size of 32, optimizing the loss using the Adam optimizer and tracking Mean Absolute Error (MAE).
  
5. **Evaluation**:
    - Training and validation losses are monitored to evaluate the model’s performance.
    - Random samples from the test set are visualized along with the predicted ages to assess qualitative performance.

## CNN Model Layers and Parameters 🧠

Each layer in the CNN is carefully designed to learn different levels of abstraction in the image data.

### Input Layer
- **Layer**: `InputLayer`
- **Shape**: (200, 200, 3) for RGB images
- **Purpose**: Specifies the input image shape, i.e., 200x200 pixels with 3 color channels.

### Data Augmentation Layers
- **RandomRotation(0.2)**: Randomly rotates images by up to 20%. Helps in making the model invariant to rotational changes.
- **RandomZoom(0.2)**: Randomly zooms images by up to 20%. Aids in making the model robust to different face sizes and positioning.

### Convolutional Layers
1. **Conv2D(32, 3, activation='relu')**: 
   - **Purpose**: Detects basic features such as edges.
   - **Parameters**: 32 filters, 3x3 kernel, ReLU activation.
   - **Reason for ReLU**: ReLU introduces non-linearity, allowing the network to capture complex patterns.
   
2. **Conv2D(32, 3, activation='relu')**: 
   - **Purpose**: Refines the detection of patterns from the first convolutional layer.

3. **MaxPooling2D()**: 
   - **Purpose**: Reduces spatial dimensions (downsampling) while preserving important features, thereby reducing computational complexity.

### Deeper Convolutional Layers
4. **Conv2D(64, 3, activation='relu')** and **Conv2D(64, 3, activation='relu')**:
   - **Purpose**: Captures more detailed patterns from the images, like textures.

5. **MaxPooling2D()**: 
   - **Purpose**: Further downsampling to focus on the most prominent features.

6. **Conv2D(128, 3, activation='relu')** and **Conv2D(128, 3, activation='relu')**:
   - **Purpose**: Extracts even higher-level features, including complex shapes and facial structures.

7. **Conv2D(265, 3, activation='relu')** and **Conv2D(265, 3, activation='relu')**:
   - **Purpose**: Deepest layer to capture the most abstract representations of the data.

8. **MaxPooling2D()**: 
   - **Purpose**: Final downsampling, reducing spatial dimensions before flattening.

### Fully Connected Layers
- **Flatten()**: Converts the 3D output from the convolutional layers into a 1D vector, suitable for the dense layers.
- **Dense(64, activation='relu')**: A fully connected layer to learn complex combinations of features.
- **Dropout(0.5)**: Reduces overfitting by randomly "dropping out" 50% of neurons during each training step.
- **Dense(1)**: Single output neuron for regression to predict the age.

### Model Compilation
- **Loss Function**: Mean Squared Error (MSE) is used, suitable for continuous data predictions.
- **Optimizer**: Adam optimizer for efficient learning.
- **Metrics**: Mean Absolute Error (MAE) to monitor the accuracy of predictions.

## Code Snippets 💻

```python
# Define the CNN model architecture
model = Sequential([
    InputLayer(input_shape=input_shape),
    
    RandomRotation(0.2),
    RandomZoom(0.2),

    Conv2D(32, 3, activation='relu'),
    Conv2D(32, 3, activation='relu'),
    MaxPooling2D(),

    Conv2D(64, 3, activation='relu'),
    Conv2D(64, 3, activation='relu'),
    MaxPooling2D(),

    Conv2D(128, 3, activation='relu'),
    Conv2D(128, 3, activation='relu'),
    MaxPooling2D(),

    Conv2D(265, 3, activation='relu'),
    Conv2D(265, 3, activation='relu'),
    MaxPooling2D(),

    Flatten(),

    Dense(64, activation='relu'),
    Dropout(0.5),

    Dense(1),
])

# Model compilation and training
model.compile(
    optimizer="adam",
    loss="mean_squared_error",
    metrics=["mae"]
)

history = model.fit(train_images, train_labels, epochs=EPOCHS, batch_size=BATCH_SIZE, 
validation_data=(test_images, test_labels), callbacks=[cp_callback])
```

## Findings Section

### Plot the Loss and the MAE

![1a188526-3eb6-4b01-9225-efce14bf563a](https://github.com/user-attachments/assets/cbbf63bd-12e9-403b-8408-8e049df54f7c)

**Visualization:** These two graphs represent the change in loss and Mean Absolute Error (MAE) during the training process. 
- **Left Graph:** Training Loss vs. Validation Loss.
- **Right Graph:** Training MAE vs. Validation MAE.

**Finding:**
- Both the training and validation loss steadily decrease as training progresses, indicating that the model is learning effectively.
- Early in training, there is a significant difference between training and validation loss/MAE, but this difference becomes smaller over time. This suggests that the model is reducing overfitting as the number of epochs increases, and the validation performance improves in tandem.
- The final loss and MAE values indicate that the model has reached a reasonable level of generalization.

**Conclusion:** The CNN model shows solid progress across 250 epochs. The consistent drop in loss and MAE, both in the training and validation data, implies that the model generalizes well and can effectively predict ages within an acceptable margin. Although there are some fluctuations toward the end of training, the overall performance stabilizes. The final MAE is approximately 6.52 on the training set and 6.73 on the validation set, which indicates a reasonably accurate model.

---

### Loss, MAE, Validation Loss, and Validation MAE Over the Course of Training

- **Training Loss:** Started at **745.21** and steadily decreased to around **78.88** after 250 epochs. This sharp decrease demonstrates effective learning and optimization.
- **Validation Loss:** Initial value was **606.09**, which decreased to **92.18** by the end of training. Although slightly higher than training loss, the validation loss demonstrates the model's ability to generalize.
- **MAE:** The Mean Absolute Error (MAE) on the training set started at **21.82** and dropped to **6.52** at the end of training. This is a significant improvement in the model's predictions.
- **Validation MAE:** Similarly, the validation MAE started at **19.90** and ended at **6.73**. This shows that the model's predictions on unseen data become more precise over time.

**Conclusion:** Overall, the model shows strong learning behavior. The final validation results suggest that the model achieves satisfactory generalization on the test set, though there's room for improvement, especially in reducing the validation loss and validation MAE even further.

---

### Evaluate with the testing data

![bed23ae7-a59f-4081-bc83-8eecfa5dbda5](https://github.com/user-attachments/assets/4e6d4e98-144b-442c-95a0-a56892ab3499)

**Visualization:** The images show actual vs. predicted ages for various test samples. These predictions are derived from the trained convolutional neural network (CNN) model, which was evaluated on unseen testing data. The actual age is mentioned at the top of each image, with the predicted age displayed below it.

**Finding:**
- The model performs reasonably well on a wide range of ages. For some samples, the model predicts the age very close to the actual one, showing strong generalization. However, for certain images, particularly at the extreme ends of the age spectrum, there are notable discrepancies (e.g., predicting 61 years for a person aged 78).
- The predictions for younger children show greater deviation from their actual ages compared to adults. This could indicate that the model struggles more with identifying specific features of young faces.

**Conclusion:** The model demonstrates good accuracy in generalizing across most age ranges but shows some inconsistency in extreme cases (very young and older adults). Fine-tuning the model with more age-diverse data or increasing model complexity may help address this.



## Challenges and How I Overcame Them 🧗‍♂️

1. **GPU Memory Issues**:
    - Challenge: TensorFlow would raise memory issues when training on larger datasets.
    - Solution: Enabled memory growth on GPU using the following code:
      ```python
      for gpu in gpus:
          tf.config.experimental.set_memory_growth(gpu, True)
      ```

2. **Overfitting**:
    - Challenge: The model was prone to overfitting with a small dataset.
    - Solution: Implemented dropout layers to mitigate overfitting and employed data augmentation (rotation and zoom).

## Conclusion 🧾
This project demonstrated the power of Convolutional Neural Networks in predicting a continuous variable, i.e., age, from facial images. Despite challenges like overfitting and GPU limitations, careful use of techniques like data augmentation and dropout improved the model's generalization. Future work could include incorporating a larger and more diverse dataset to improve performance further.

The developed model is effective for practical applications and can serve as a basis for further enhancement in age-related machine learning tasks.