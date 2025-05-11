# 🧠 Neural Network for Diabetes Prediction
Healthcare-focused neural network project that applies deep learning to predict diabetes with enhanced accuracy using TensorFlow, PyTorch, and modern regularization methods

## Overview  
This project presents a deep learning-based approach to predict diabetes using a forward neural network tailored for image classification. Built using both PyTorch and TensorFlow, the model is trained on a diabetes prediction dataset, demonstrating the practical application of neural networks in the healthcare domain.

---

## Methodology  

The project follows a structured workflow:

- **Library Imports**  
  Loaded essential libraries for data manipulation, visualization, model building, and performance evaluation.

- **Exploratory Data Analysis**  
  Analyzed the distribution of target classes and visualized sample data to identify patterns and inform preprocessing strategies.

- **Data Preprocessing**  
  Cleaned the dataset by handling missing values, applied one-hot encoding for categorical variables, and standardized numerical features.

- **Data Splitting**  
  Divided the dataset into training, validation, and test sets to enable effective model training and evaluation.

- **Model Architecture**  
  Constructed a deep feedforward neural network using TensorFlow and Keras. Configured input, hidden, and output layers with appropriate activation functions.

- **Training**  
  Compiled the model with a suitable loss function, optimizer, and evaluation metrics. Trained the model on the prepared data while tracking performance.

- **Model Evaluation**  
  Assessed model performance using loss and accuracy metrics on validation and test datasets.

---

## Regularization Techniques  

To address overfitting and enhance generalization, the following techniques were implemented:

- **Batch Normalization**  
  Introduced batch normalization after each hidden layer to stabilize training and improve performance.

- **Dropout**  
  Applied dropout layers to randomly deactivate neurons during training, reducing reliance on specific connections.

---

## TensorBoard Analysis  

Monitored training progress using TensorBoard. Observed signs of overfitting around epochs 6–7, where the validation performance began to diverge from training metrics. This insight guided the application of regularization techniques.

---

## Summary & Insights  

- Batch normalization and dropout improved generalization and reduced overfitting.  
- Deeper network layers captured complex data patterns, aiding class separation in the embedding space.  
- Observed a decline in final layer embedding quality, highlighting the need for further tuning in intermediate representations.

---

## Conclusion  

This project showcases the effective application of deep learning models in medical data analysis. Using a forward neural network architecture, diabetes prediction was achieved with promising results. By leveraging both PyTorch and TensorFlow, the project benefits from diverse tooling and deployment options, offering scalability for future enhancements.

---

## Dependencies  

- Python 3.x  
- PyTorch  
- TensorFlow  
- NumPy  
- Pandas  
- Scikit-learn  
- Matplotlib  
- UMAP-learn  
- TensorBoard  

---

## Usage  

To replicate or build upon this analysis:

1. Clone the repository.  
2. Install the listed dependencies.  
3. Execute the Python scripts for model training, regularization, and evaluation.  
4. View outputs and insights in the accompanying Jupyter Notebook or markdown report.

---
