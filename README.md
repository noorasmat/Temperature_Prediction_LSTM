
# Temperature Prediction Using LSTM

This repository contains a model that predicts the next 10-minute temperature values using IoT sensor data. The model is built with a 2-layer Long Short-Term Memory (LSTM) architecture.

# Use Guide
Install Python, Jupyter Lab using Anaconda if you are using local system, and set colab ='local'. If you are using Google Colab use mode='colab.
- **Temp_Predictor**: LSTM model with a single LSTM layer.
- **Temp_Predictor_Low_LR**: Single LSTM layer model with a lower learning rate (0.001).
- **Temp_Predictor_2LSTMs**: LSTM model with 2 LSTM layers.
- **Temp_Predictor_2LSTMs_Attention**: LSTM model with 2 LSTM layers, incorporating a self-attention mechanism.

## Overview

This project aims to predict future temperature values by leveraging historical data collected from IoT temperature sensors. The model utilizes an LSTM neural network, which excels at capturing temporal dependencies in time series data.

## Data Preparation

1. **Notebook Mode Selection:**
   - Two modes are available: one for Colab (loading data from Google Drive) and another for loading data from the local system.
   
2. **Loading Data:**
   - The dataset consists of temperature readings stored in a CSV file.

3. **Duplicate Removal:**
   - Duplicate entries are eliminated to ensure the uniqueness of timestamps, which is vital for time series integrity.

4. **Handling Missing Values:**
   - Missing values are linearly interpolated to maintain continuity in the data.

5. **Normalization:**
   - Data is normalized to the [0, 1] range using `MinMaxScaler` for faster convergence and equal contribution of features during training.

6. **Train-Test Split:**
   - For this experiment, the dataset was not split to retain more data for training. However, a typical best practice is to perform a split.

## Model Architecture

1. **LSTM Layers:**
   - Two LSTM layers are employed to capture long-term dependencies in the data. LSTMs are chosen for their ability to mitigate issues like the vanishing gradient problem.

2. **Dense Layer:**
   - A fully connected dense layer is used to output the next 10 temperature values.

3. **Reshape Layer:**
   - The output is reshaped to match the required format for evaluation.

## Training

1. **Optimizer:**
   - Adam optimizer is used for its adaptive learning rate, making it suitable for this time series forecasting task.

2. **Loss Function:**
   - Mean Squared Error (MSE) is employed to measure the difference between actual and predicted temperature values.

3. **ModelCheckpoint:**
   - The best model during training is saved based on the training loss. 

## Evaluation

- The model's performance is evaluated on training data using MSE. Though validation is ideal, the focus here is on maximizing training data.

## Experiments

Three experiments were conducted to refine the model's performance:
| Experiment   | Model Details     | Learning Rate | MSE   | Findings                         |
|--------------|-------------------|---------------|-------|----------------------------------|
| Experiment 1 | 1 LSTM Layer       | 0.01          | 0.0185| Lr=0.01 gives better accuracy, thus we picked it for future results.                    |
| Experiment 2 | 1 LSTM with Low Learning Rate| 0.001          | 0.0188| -- |

Two Different Learning Rate Uses

| Experiment   | Model Details     | Learning Rate | MSE   | Findings                         |
|--------------|-------------------|---------------|-------|----------------------------------|
| Experiment 1 | 1 LSTM Layer       | 0.01          | 0.0185| Random results                   |
| Experiment 2 | 2 LSTM Layers      | 0.01          | 0.0181| Better learning with more epochs |
| Experiment 3 | Transformer Model  | 0.01          | 0.0185| 2 LSTM layers perform better     |

## Conclusion

The use of a 2-layer LSTM with a learning rate of 0.01 resulted in the best performance. The Adam optimizer and MSE loss function provided efficient training, and the preprocessing steps ensured quality input data for the model.

--- 
