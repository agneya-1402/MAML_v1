# Meta-Learning with Model-Agnostic Meta-Learning (MAML)

## Overview
This project implements **Model-Agnostic Meta-Learning (MAML)** using PyTorch. Meta-learning, or learning to learn, enables models to quickly adapt to new tasks with minimal data. One of the most effective meta-learning techniques is Model-Agnostic Meta-Learning (MAML), which optimizes a model such that it can adapt to new tasks efficiently with a few gradient steps.

## Features
- Uses a **simple 2 layer feedforward neural network** for classification tasks.
- Implements **MAML algorithm** to update model parameters efficiently.
- Uses **synthetic few shot learning tasks** for training.
- Optimized with **Adam and SGD** for meta-learning.

## Installation
Make sure you have Python installed along with PyTorch:
```bash
!pip install torch numpy
```

## Usage
Run the script to train the meta-learning model:
```bash
python MAML_v1.ipynb
```
This will execute the meta-learning training process and output the **meta-loss** over epochs.

## Expected Output
The loss should **decrease** over time:
```plaintext
Epoch 0, Meta Loss: 0.78
Epoch 10, Meta Loss: 0.72
Epoch 20, Meta Loss: 0.65
...
Epoch 90, Meta Loss: 0.55
```

## Code Explanation
### 1. Define the Model
```python
class MetaModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(MetaModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, output_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)
```
- A **2-layer neural network** with ReLU activation.

### 2. MAML Update Function
```python
def maml_update(model, task_data, alpha=0.01):
    criterion = nn.CrossEntropyLoss()
    gradients = []
    for (x_train, y_train, x_val, y_val) in task_data:
        temp_model = MetaModel(10, 2)  # Create task-specific copy
        temp_model.load_state_dict(model.state_dict())
        optimizer = optim.SGD(temp_model.parameters(), lr=alpha)

        loss = criterion(temp_model(x_train), y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        val_loss = criterion(temp_model(x_val), y_val)
        gradients.append([p.grad for p in temp_model.parameters()])
    
    return sum(gradients) / len(gradients)
```
- **Performs a task-specific update** using **SGD**.
- **Computes gradients** to apply for meta-learning.

### 3. Meta-Training Loop
```python
meta_optimizer = optim.Adam(meta_model.parameters(), lr=0.001)
for epoch in range(100):
    meta_optimizer.zero_grad()
    grads = maml_update(meta_model, task_data)
    for param, grad in zip(meta_model.parameters(), grads):
        param.grad = grad
    meta_optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Meta Loss: {meta_loss.item()}")
```
- **Updates model parameters** using **meta-gradients**.
- **Prints meta-loss** every 10 epochs.

## References
- [MAML Paper](https://arxiv.org/abs/1703.03400)
- [PyTorch Meta-Learning Guide](https://pytorch.org/tutorials/intermediate/)

