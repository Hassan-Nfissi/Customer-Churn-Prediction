# Customer Churn Prediction

## Problem Statement

In today's competitive business landscape, customer retention is crucial for sustainable growth and success. This project aims to develop a predictive model that identifies customers at risk of churning – discontinuing their use of our service. Customer churn can lead to significant revenue loss and a decline in market share. By leveraging a Feedforward Neural Network (FNN) with PyTorch, we aim to build a model that accurately predicts customer churn based on historical usage behavior, demographic information, and subscription details. This predictive model will enable us to proactively target high-risk customers with personalized retention strategies, ultimately enhancing customer satisfaction, reducing churn rates, and optimizing business strategies. The goal is to create an effective solution that contributes to the long-term success of our company by fostering customer loyalty and engagement.

## Data Description

The dataset consists of customer information relevant to churn prediction. It includes the following columns:

- **CustomerID**: Unique identifier for each customer.
- **Age**: Age of the customer.
- **Gender**: Gender of the customer (Male or Female).
- **Tenure**: The duration (in months) that the customer has been with the service.
- **Usage Frequency**: How frequently the customer uses the service.
- **Support Calls**: The number of support calls made by the customer.
- **Payment Delay**: Instances of delayed payments by the customer.
- **Subscription Type**: Type of subscription the customer has (e.g., Basic, Premium).
- **Contract Length**: The length of the customer's contract (e.g., month-to-month, annual).
- **Total Spend**: Total amount spent by the customer.
- **Last Interaction**: The last time the customer interacted with the service.
- **Churn**: A binary indicator (1 for churned, 0 for not churned) representing whether the customer has churned.

## Technology Stack

### Python Programming Language
Python serves as the primary programming language for data analysis, modeling, and implementation of machine learning algorithms due to its rich ecosystem of libraries and packages.

### PyTorch
PyTorch is an open-source machine learning framework used for building and training deep learning models. It provides flexibility and control, making it a popular choice for implementing custom neural networks.

### Feedforward Neural Network (FNN)
A Feedforward Neural Network (FNN) is used for predicting customer churn. The network is designed to take both categorical and continuous features as input, process them through multiple hidden layers, and output a probability of churn.

### Batch Normalization
Batch normalization is applied to both categorical and continuous features to normalize the inputs, helping to stabilize and accelerate the training process.

### Dropout
Dropout is used as a regularization technique to prevent overfitting. It randomly drops a fraction of neurons during training, forcing the network to generalize better.

### Embeddings
Embeddings are used to convert categorical variables into dense vectors, which are then passed through the network.

### ReLU Activation
The Rectified Linear Unit (ReLU) activation function is used in the hidden layers to introduce non-linearity into the model.

### Sigmoid Activation
The Sigmoid activation function is used in the output layer to produce a probability score between 0 and 1, indicating the likelihood of a customer churning.

## Model Architecture

```python
import torch
import torch.nn as nn

class FNN(nn.Module):
    def __init__(self, embed_dim, n_cont, layers, p):
        super().__init__()
        self.embeds = nn.ModuleList([nn.Embedding(inp, out) for inp, out in embed_dim])
        self.emb_drop = nn.Dropout(p)
        self.bn_cont = nn.BatchNorm1d(n_cont)

        layers_list = []
        n_embed = sum((out for inp, out in embed_dim))
        n_in = n_embed + n_cont

        for i in layers:
            layers_list.append(nn.Linear(n_in, i))
            layers_list.append(nn.ReLU(inplace=True))
            layers_list.append(nn.BatchNorm1d(i))
            layers_list.append(nn.Dropout(p))
            n_in = i

        layers_list.append(nn.Linear(layers[-1], 1))
        layers_list.append(nn.Sigmoid())

        self.layers = nn.Sequential(*layers_list)

    def forward(self, x_cat, x_cont):
        embeddings = []
        for i, e in enumerate(self.embeds):
            embeddings.append(e(x_cat[:, i]))

        x = torch.cat(embeddings, 1)
        x = self.emb_drop(x)
        x_cont = self.bn_cont(x_cont)
        x = torch.cat([x, x_cont], 1)

        x = self.layers(x)
        return x
"# Customer-Churn-Prediction" 
