# Neural Network Design Guide

Welcome to the Professional Neural Network Visualizer. This guide explains the core concepts behind the parameters you can control in the sidebar.

## 1. Network Architecture
The architecture defines the "capacity" of your model to learn information.

- **Depth (Hidden Layers)**: Adding more layers allows the network to learn higher-level hierarchical features. *Too many layers can lead to vanishing gradients without proper activation.*
- **Width (Neurons per Layer)**: More neurons allow for more complex decision boundaries. *Too many neurons can lead to overfitting, where the model memorizes the data instead of generalizing.*

## 2. Activation Functions
Activations introduce non-linearity, which is essential for learning non-linear patterns.

- **ReLU (Rectified Linear Unit)**: The industry standard. It's fast and helps prevent vanishing gradients.
- **Sigmoid**: Historically significant, it maps inputs to a (0,1) range. *Prone to saturation at high/low values.*
- **Tanh**: Similar to Sigmoid but outputs (-1,1), providing zero-centered data which helps convergence.

## 3. The Adam Optimizer
Unlike standard Stochastic Gradient Descent (SGD), **Adam** (Adaptive Moment Estimation) computes individual adaptive learning rates for different parameters.

- **Learning Rate**: Controls the "step size" during optimization. Adam is robust, but a rate that's too high can still cause divergence.
- **Beta 1 & 2**: Control the exponential decay rates for the first and second moment estimates.

## 4. Overfitting vs. Underfitting
As you experiment with complex datasets (Circles, Moons), observe the **Decision Boundary**:

- **Underfitting**: The boundary is too simple (e.g., a straight line) and fails to separate classes. *Solution: Add more layers or neurons.*
- **Overfitting**: The boundary is extremely wiggly and tries to catch every noisy data point. *Solution: Reduce architecture complexity or add regularization.*

---
*Built for Advanced Agentic Coding - Neural Network Lab*
