# HOMEWORK-_2
NAME: NITHIN AARON KOMMIREDDY
STUDENT ID: 700752900

# Cloud Computing & CNN Tasks

## Question 1: Cloud Computing for Deep Learning

### Elasticity and Scalability in Cloud Computing

- **Elasticity** allows computing resources to be automatically adjusted based on workload demands, optimizing cost and resource utilization during deep learning tasks.
- **Scalability** enables a system to handle increasing workloads by adding resources, either by upgrading existing ones (vertical scaling) or adding more instances (horizontal scaling). This ensures that deep learning models can efficiently scale with growing datasets.

### Comparison of AWS SageMaker, Google Vertex AI, and Microsoft Azure Machine Learning Studio

- **AWS SageMaker:** Supports TensorFlow, PyTorch, and MXNet. Offers SageMaker Autopilot for AutoML and distributed training with managed spot instances. Provides one-click deployment, pay-as-you-go pricing, and integration with AWS services like S3 and Lambda.
- **Google Vertex AI:** Supports TensorFlow, PyTorch, and JAX. Includes Vertex AI AutoML and distributed training with GPUs/TPUs. Features an end-to-end pipeline, flexible pricing, and integration with Google Cloud services like BigQuery.
- **Microsoft Azure Machine Learning Studio:** Supports TensorFlow, PyTorch, and Scikit-learn. Offers Azure AutoML, distributed training with GPU/CPU clusters, and real-time and batch deployment. Pricing includes reserved instance options, with integration into Azure services like Azure Storage and Synapse.

---

## Question 2: Convolution Operations with Different Parameters

- Convolution can be implemented using NumPy and TensorFlow/Keras.
- A **5×5 input matrix** is processed using a **3×3 kernel** with different configurations:
  - Stride = 1, Padding = 'VALID'
  - Stride = 1, Padding = 'SAME'
  - Stride = 2, Padding = 'VALID'
  - Stride = 2, Padding = 'SAME'
- The output feature maps for each case are analyzed.

---

## Question 3: CNN Feature Extraction

### Edge Detection using Sobel Filter
- Load a grayscale image.
- Apply Sobel filters to detect edges in both **x-direction** and **y-direction**.
- Display the original image, along with the results of Sobel-X and Sobel-Y operations.

### Max Pooling and Average Pooling
- Create a **random 4x4 matrix** as an input image.
- Apply a **2x2 Max Pooling operation** to extract the most prominent features.
- Apply a **2x2 Average Pooling operation** to smooth the feature map.
- Compare the original matrix and the results.

---

## Question 4: Implementing and Comparing CNN Architectures

### Implementing AlexNet
- Build AlexNet using **TensorFlow/Keras**, incorporating:
  - Conv2D layers with varying kernel sizes and activation functions.
  - MaxPooling layers for reducing spatial dimensions.
  - Fully connected layers using ReLU activations.
  - Dropout layers to prevent overfitting.
- Print the model summary to review its structure.

### Comparing AlexNet, VGG, and ResNet

- **AlexNet:** Contains 8 layers, using large filters (11x11, 5x5). Follows a simple sequential architecture and provides good performance.
- **VGG:** Has 16-19 layers with small (3x3) filters. Uses a deep but uniform filter approach, offering better performance.
- **ResNet:** Features 50+ layers with residual blocks and skip connections, making it the best performer for deep models.

---


