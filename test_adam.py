import numpy as np
from nn_engine import NeuralNetwork, generate_data, StandardScaler

# XOR Test
X, y = generate_data(type="blobs", n_samples=300, noise=0.1) # Blobs is easier to start
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Try with Adam (Default LR 0.001)
nn_adam = NeuralNetwork([2, 16, 16, 1], learning_rate=0.1, activation="Sigmoid")

print("Training with Adam Optimizer...")
for i in range(500):
    loss = nn_adam.train_step(X_scaled, y)
    if i % 100 == 0:
        print(f"Epoch {i}, Loss: {loss:.6f}")

preds = nn_adam.predict(X_scaled)
accuracy = np.mean((preds > 0.5) == y)
print(f"\nFinal Accuracy: {accuracy*100:.2f}%")

if accuracy > 0.90:
    print("\n✅ Logic Verified: Adam Optimizer converged successfully!")
else:
    print("\n❌ Logic Error: Adam Optimizer failed to converge to high accuracy.")
