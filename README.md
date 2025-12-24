
# 🛒 Grocery Object Detection with MobileNetV2



## 📌 Overview
This project develops a grocery object detection and classification system using MobileNetV2 as the backbone model. By applying transfer learning, fine-tuning, and data augmentation, the model is able to classify grocery items into multiple categories. The final trained model achieves a test accuracy of approximately 74.5 percent, demonstrating strong potential for retail automation and shelf analytics.
## ❓ Problem Statement
Retail stores face challenges in managing inventory and ensuring shelves are stocked correctly. Manual classification of grocery items is slow and prone to human error. This project addresses the need for an automated, scalable, and accurate grocery classification system that can support tasks such as product recognition, empty shelf detection, and low-stock alerts.
## 📂 Dataset
I downloaded a collection of grocery images and uploaded them to Roboflow. Using Roboflow, I created a structured dataset with separate splits for training, validation, and testing. This organized dataset was then used to train and evaluate the model effectively.

Preprocessed with ImageDataGenerator for augmentation:

Rescaling

Rotation

Width/Height shift

Zoom

Horizontal flip
## 🛠️ Tools & Technologies
Python

TensorFlow / Keras

Roboflow API

Google Colab

MobileNetV2 pretrained on ImageNet

Google Drive for model and artifact storage
## 🔄 Workflow
Images collected and uploaded into Roboflow to create dataset splits for training, validation, and testing.

Data augmentation applied using ImageDataGenerator to improve robustness.

MobileNetV2 used as the base model with pretrained ImageNet weights.

Custom layers added: Global Average Pooling, Dense layer with ReLU, Dropout, and final Dense softmax layer.

Training performed in two phases:

Phase 1: Train custom layers with the base model frozen.

Phase 2: Fine-tune the last 50 layers of MobileNetV2 with a low learning rate.

Model evaluated on the test set and saved to Google Drive along with class names.

Callbacks such as EarlyStopping and ReduceLROnPlateau were used to optimize training and prevent overfitting.
## 🔑 Key Insights
Transfer learning with MobileNetV2 provides strong baseline performance for grocery classification.

Fine-tuning improves accuracy compared to training only custom layers.

Data augmentation reduces overfitting and improves generalization.

Saving class names separately ensures reproducibility and easier deployment.

The final model achieves a test accuracy of 74.5 percent, which is promising for real-world retail applications.

The workflow is modular, making it easy to extend to detection models like YOLOv8 or integrate with dashboards for analytics.
## 📊 Results
Final Test Accuracy: 82.02% (Test Accuracy: 0.8202019929885864)

Best Validation Accuracy: 82.11% (epoch 19 of fine-tuning, learning rate 1e-5)

Validation Loss at Best Epoch: 0.6239

Final Training Accuracy: 90.56% (epoch 20 of fine-tuning)

Final Training Loss: 0.3005

Steps per Epoch: 648

Learning Rate Schedule:

Initial training at 0.001

Auto-reduced to 0.0002 by ReduceLROnPlateau

Fine-tuning at 0.00001
## Conclusion
This project demonstrates the effectiveness of MobileNetV2 for grocery object detection and classification. With proper dataset preparation, augmentation, and fine-tuning, the model achieves reliable accuracy. Future improvements could include experimenting with YOLOv8 for combined detection and classification, adding Grad-CAM visualizations for interpretability, and deploying the solution as a Streamlit app for real-time grocery recognition. The modular design and artifact management (model file and class names JSON) make this project reproducible, portable, and ready for integration into larger retail analytics systems.
## Contributing
Contributions to this project are welcome! If you have ideas for improvements or additional insights, please open an issue or a pull request. Your contributions will be greatly appreciated.
## 📬 Contact Information
LinkedIn: kaushik-raghani

Email: kaushikraghani23@gmail.com