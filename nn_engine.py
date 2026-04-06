import numpy as np

class Activation:
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def sigmoid_prime(x):
        s = Activation.sigmoid(x)
        return s * (1 - s)
    
    @staticmethod
    def relu(x):
        return np.maximum(0, x)
    
    @staticmethod
    def relu_prime(x):
        return (x > 0).astype(float)
    
    @staticmethod
    def tanh(x):
        return np.tanh(x)
    
    @staticmethod
    def tanh_prime(x):
        return 1 - np.tanh(x)**2

    @staticmethod
    def get(name):
        activations = {
            "Sigmoid": (Activation.sigmoid, Activation.sigmoid_prime),
            "ReLU": (Activation.relu, Activation.relu_prime),
            "Tanh": (Activation.tanh, Activation.tanh_prime)
        }
        return activations.get(name, (Activation.sigmoid, Activation.sigmoid_prime))

class Layer:
    def __init__(self, input_size, output_size, activation_name="Sigmoid"):
        # He initialization for ReLU, Xavier for others
        if activation_name == "ReLU":
            self.weights = np.random.randn(input_size, output_size) * np.sqrt(2.0/input_size)
        else:
            self.weights = np.random.randn(input_size, output_size) * np.sqrt(1.0/input_size)
            
        self.biases = np.zeros((1, output_size))
        self.activation_name = activation_name
        self.activation, self.activation_prime = Activation.get(activation_name)
        
        # Internal state for visualization
        self.last_input = None
        self.last_z = None
        self.last_a = None
        self.dw = None
        self.db = None

    def forward(self, input_data):
        self.last_input = input_data
        self.last_z = np.dot(input_data, self.weights) + self.biases
        self.last_a = self.activation(self.last_z)
        return self.last_a

    def backward(self, output_gradient, learning_rate):
        # Grad w.r.t z (weighted sum)
        # delta = output_gradient * activation'(z)
        delta = output_gradient * self.activation_prime(self.last_z)
        
        # Grad w.r.t weights (input^T * delta)
        self.dw = np.dot(self.last_input.T, delta)
        # Grad w.r.t biases (sum(delta))
        self.db = np.sum(delta, axis=0, keepdims=True)
        
        # Grad w.r.t input (delta * weight^T)
        input_gradient = np.dot(delta, self.weights.T)
        
        # Update parameters
        self.weights -= learning_rate * self.dw
        self.biases -= learning_rate * self.db
        
        return input_gradient

class NeuralNetwork:
    def __init__(self, architecture, learning_rate=0.1, activation="Sigmoid"):
        """
        architecture: list of layer sizes, e.g., [2, 4, 1]
        """
        self.layers = []
        for i in range(len(architecture) - 1):
            # All layers use same activation for simplicity in this visualizer
            # but usually output layer might be different (e.g. sigmoid for binary)
            self.layers.append(Layer(architecture[i], architecture[i+1], activation))
        self.learning_rate = learning_rate

    def predict(self, input_data):
        output = input_data
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def train_step(self, x, y):
        # Forward pass
        prediction = self.predict(x)
        
        # Loss (MSE gradient for simplicity)
        # Error = prediction - labels
        error_gradient = 2 * (prediction - y) / y.size
        
        # Backward pass
        grad = error_gradient
        for layer in reversed(self.layers):
            grad = layer.backward(grad, self.learning_rate)
        
        # Return loss for visualization
        return np.mean((prediction - y)**2)

def generate_data(type="circles", n_samples=300, noise=0.1):
    from sklearn.datasets import make_circles, make_moons, make_blobs
    if type == "circles":
        X, y = make_circles(n_samples=n_samples, factor=0.5, noise=noise)
    elif type == "moons":
        X, y = make_moons(n_samples=n_samples, noise=noise)
    else: # Linearly separable
        X, y = make_blobs(n_samples=n_samples, centers=2, cluster_std=noise*5, random_state=42)
    
    # Scale labels for binary class
    y = y.reshape(-1, 1)
    return X, y
