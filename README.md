# Hybrid Quantum–Classical CNN for MNIST Dataset

This project is a capstone project for the **Quantum Computing**

It demonstrates a Hybrid Quantum-Classical Neural Network to perform image classification on a subset of the MNIST dataset.

## Project Description

This model uses a classical Convolutional Neural Network (CNN) for feature extraction and a Parameterized Quantum Circuit (PQC) as a quantum-enhanced classifier.

1.  **Classical CNN**: A small classical CNN processes the input MNIST image and extracts a set of features.
2.  **Quantum Layer**: These classical features are then used as rotation parameters (gates) for a quantum circuit.
3.  **Hybrid Model**: The quantum circuit is measured, and the resulting expectation value is passed to a final classical output layer to produce a classification (e.g., "3" or "6").

This implementation uses **TensorFlow Quantum (TFQ)** and **Cirq**. Due to the complexity of classifying 10 digits, this project is simplified to a binary classification task: distinguishing between digits **3** and **6**.

## How to Run

### 1. Clone the Repository

### 2. Set up a Virtual Environment
It is highly recommended to use a virtual environment.

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Training

```bash
python src/train.py
```
The script will:
Download and preprocess the MNIST dataset.
Build the hybrid quantum-classical model.
Train the model for a few epochs.
Print the final training and validation accuracy.