import numpy as np

class Activation:
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
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
        
        # Adam Optimizer state
        self.mw, self.vw = np.zeros_like(self.weights), np.zeros_like(self.weights)
        self.mb, self.vb = np.zeros_like(self.biases), np.zeros_like(self.biases)
        
        # State for visualization
        self.last_input = None
        self.last_z = None
        self.last_a = None
        self.dw, self.db = None, None

    def forward(self, input_data):
        self.last_input = input_data
        self.last_z = np.dot(input_data, self.weights) + self.biases
        self.last_a = self.activation(self.last_z)
        return self.last_a

    def backward(self, output_gradient):
        delta = output_gradient * self.activation_prime(self.last_z)
        self.dw = np.dot(self.last_input.T, delta)
        self.db = np.sum(delta, axis=0, keepdims=True)
        input_gradient = np.dot(delta, self.weights.T)
        return input_gradient

class NeuralNetwork:
    def __init__(self, architecture, learning_rate=0.001, activation="Sigmoid", beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.layers = [Layer(architecture[i], architecture[i+1], activation) for i in range(len(architecture)-1)]
        self.lr = learning_rate
        self.beta1, self.beta2 = beta1, beta2
        self.epsilon = epsilon
        self.t = 0 # timestep for Adam

    def predict(self, x):
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def train_step(self, x, y):
        self.t += 1
        # Forward
        prediction = self.predict(x)
        
        # Loss (MSE derivative)
        grad = 2 * (prediction - y) / y.size
        
        # Backward
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
            
            # Adam Updates
            # Weights
            layer.mw = self.beta1 * layer.mw + (1 - self.beta1) * layer.dw
            layer.vw = self.beta2 * layer.vw + (1 - self.beta2) * (layer.dw**2)
            mw_corr = layer.mw / (1 - self.beta1**self.t)
            vw_corr = layer.vw / (1 - self.beta2**self.t)
            layer.weights -= self.lr * mw_corr / (np.sqrt(vw_corr) + self.epsilon)
            
            # Biases
            layer.mb = self.beta1 * layer.mb + (1 - self.beta1) * layer.db
            layer.vb = self.beta2 * layer.vb + (1 - self.beta2) * (layer.db**2)
            mb_corr = layer.mb / (1 - self.beta1**self.t)
            vb_corr = layer.vb / (1 - self.beta2**self.t)
            layer.biases -= self.lr * mb_corr / (np.sqrt(vb_corr) + self.epsilon)
            
        return np.mean((prediction - y)**2)

class StandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None
        
    def fit_transform(self, x):
        self.mean = np.mean(x, axis=0)
        self.std = np.std(x, axis=0)
        self.std[self.std == 0] = 1 # Avoid division by zero
        return (x - self.mean) / self.std
        
    def transform(self, x):
        return (x - self.mean) / self.std

def generate_data(type="circles", n_samples=300, noise=0.1):
    from sklearn.datasets import make_circles, make_moons, make_blobs
    if type == "circles":
        X, y = make_circles(n_samples=n_samples, factor=0.5, noise=noise)
    elif type == "moons":
        X, y = make_moons(n_samples=n_samples, noise=noise)
    else:
        X, y = make_blobs(n_samples=n_samples, centers=2, cluster_std=noise*5, random_state=42)
    return X, y.reshape(-1, 1)
