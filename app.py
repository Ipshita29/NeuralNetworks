import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd
import time
from nn_engine import NeuralNetwork, Layer, Activation, generate_data
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(page_title="Neural Network Odyssey", layout="wide", page_icon="🧠")

# Load CSS
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# --- Sidebar Configuration ---
st.sidebar.title("🛠️ Model Configuration")

dataset_name = st.sidebar.selectbox("Select Dataset", ["Circles", "Moons", "Linear Clusters"])
dataset_type = dataset_name.lower().split()[0] # circles, moons, linear

noise_val = st.sidebar.slider("Dataset Noise", 0.0, 0.5, 0.1)
learning_rate = st.sidebar.slider("Learning Rate", 0.001, 1.0, 0.1)
num_hidden_layers = st.sidebar.slider("Hidden Layers", 1, 5, 1)
neurons_per_layer = st.sidebar.slider("Neurons per Hidden Layer", 1, 20, 4)
activation_fn = st.sidebar.selectbox("Activation Function", ["Sigmoid", "ReLU", "Tanh"])
epochs = st.sidebar.number_input("Number of Epochs", min_value=1, max_value=2000, value=200)

if st.sidebar.button("🚀 Re-train Model"):
    st.session_state.trained = False

# --- Initialization ---
# Track sidebar config to detect changes
current_config = {
    "dataset": dataset_type,
    "noise": noise_val,
    "layers": num_hidden_layers,
    "neurons": neurons_per_layer,
    "activation": activation_fn
}

if 'trained' not in st.session_state:
    st.session_state.trained = False
    st.session_state.history = []
    st.session_state.last_config = current_config
    st.session_state.trained_nn = None

# Auto-reset training if architecture changes
if st.session_state.last_config != current_config:
    st.session_state.trained = False
    st.session_state.last_config = current_config
    st.session_state.trained_nn = None

def get_architecture():
    return [2] + [neurons_per_layer] * num_hidden_layers + [1]

# Local instance for "untrained" visualizations, but use session model if trained
nn = NeuralNetwork(get_architecture(), learning_rate, activation_fn)
X, y = generate_data(type=dataset_type, noise=noise_val)

# Global model used for visualizations
viz_nn = st.session_state.trained_nn if st.session_state.trained else nn

# --- Main Tabs ---
tabs = st.tabs([
    "🖥️ Neuron Computation", 
    "📈 Activation Functions", 
    "🏗️ Architecture", 
    "➡️ Forward Prop", 
    "📉 Loss Tracking", 
    "⬅️ Backpropagation", 
    "🎯 Model Complexity"
])

# --- Tab 1: Neuron Computation ---
with tabs[0]:
    st.markdown('<div class="concept-box"><div class="concept-title">Step 1: The Basic Unit (The Neuron)</div><div class="concept-desc">Neurons compute a weighted sum of inputs plus a bias, then apply a non-linear activation function.</div></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Manual Neuron Control")
        w1 = st.slider("Weight 1 (w1)", -5.0, 5.0, 1.0)
        w2 = st.slider("Weight 2 (w2)", -5.0, 5.0, -1.0)
        b = st.slider("Bias (b)", -5.0, 5.0, 0.0)
        x1 = st.number_input("Input x1", value=0.5)
        x2 = st.number_input("Input x2", value=0.2)
        
        z = w1*x1 + w2*x2 + b
        act_fn, _ = Activation.get(activation_fn)
        a = act_fn(z)
        
        st.latex(f"z = ({w1} \\times {x1}) + ({w2} \\times {x2}) + ({b}) = {z:.4f}")
        st.latex(f"a = \\text{{{activation_fn}}}({z:.4f}) = {a:.4f}")

    with col2:
        st.subheader("Neuron Visualization")
        fig = go.Figure()
        # Input nodes
        fig.add_trace(go.Scatter(x=[0, 0], y=[1, -1], mode='markers+text', 
                               text=['x1', 'x2'], marker=dict(size=40, color='#6366f1')))
        # Neuron node
        fig.add_trace(go.Scatter(x=[1], y=[0], mode='markers+text', 
                               text=['Neuron (Σ + f)'], marker=dict(size=80, color='#a855f7')))
        # Weights (lines)
        fig.add_trace(go.Scatter(x=[0, 1], y=[1, 0], mode='lines', line=dict(width=abs(w1)*2, color='white')))
        fig.add_trace(go.Scatter(x=[0, 1], y=[-1, 0], mode='lines', line=dict(width=abs(w2)*2, color='white')))
        
        fig.update_layout(showlegend=False, xaxis=dict(visible=False), yaxis=dict(visible=False), 
                          margin=dict(l=0, r=0, t=10, b=10), height=300, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

# --- Tab 2: Activation Functions ---
with tabs[1]:
    st.markdown('<div class="concept-box"><div class="concept-title">Step 2: Adding Non-Linearity</div><div class="concept-desc">Without activation functions, a neural network is just a linear regressor, no matter how deep. Non-linearity allows it to learn curved boundaries.</div></div>', unsafe_allow_html=True)
    
    x_range = np.linspace(-5, 5, 100)
    
    fig = go.Figure()
    for name in ["Sigmoid", "ReLU", "Tanh"]:
        fn, _ = Activation.get(name)
        fig.add_trace(go.Scatter(x=x_range, y=fn(x_range), name=name))
    
    fig.update_layout(title="Common Activation Functions", xaxis_title="z (weighted sum)", yaxis_title="a (activation)")
    st.plotly_chart(fig, use_container_width=True)
    
    st.info("**ReLU** is efficient for deep networks, **Sigmoid** is great for binary outputs (0-1), and **Tanh** centers data around 0 (-1 to 1).")

# --- Tab 3: Network Architecture ---
with tabs[2]:
    st.markdown('<div class="concept-box"><div class="concept-title">Step 3: Layered Intelligence</div><div class="concept-desc">Layers transform raw input into abstract features. Deep networks extract hierarchical representations.</div></div>', unsafe_allow_html=True)
    
    arch = get_architecture()
    st.write(f"**Architecture Config:** Input (2) → {' → '.join(map(str, arch[1:-1]))} → Output (1)")
    
    # Simple architecture diagram
    layers_x = np.arange(len(arch))
    fig = go.Figure()
    
    for i, count in enumerate(arch):
        nodes_y = np.linspace(-count/2, count/2, count)
        fig.add_trace(go.Scatter(x=[i]*count, y=nodes_y, mode='markers', 
                               marker=dict(size=20, color='#6366f1' if i==0 else ('#ef4444' if i==len(arch)-1 else '#a855f7')),
                               name=f"Layer {i}"))
        
        # Connections to next layer
        if i < len(arch) - 1:
            next_count = arch[i+1]
            next_nodes_y = np.linspace(-next_count/2, next_count/2, next_count)
            for y1 in nodes_y:
                for y2 in next_nodes_y:
                    fig.add_trace(go.Scatter(x=[i, i+1], y=[y1, y2], mode='lines', 
                                           line=dict(width=0.2, color='rgba(255,255,255,0.2)'),
                                           hoverinfo='none', showlegend=False))

    fig.update_layout(showlegend=False, xaxis=dict(visible=False), yaxis=dict(visible=False), height=500)
    st.plotly_chart(fig, use_container_width=True)

# --- Tab 4: Forward Propagation ---
with tabs[3]:
    st.markdown('<div class="concept-box"><div class="concept-title">Step 4: Making Predictions</div><div class="concept-desc">Information flows from left to right. Each layer compresses or expands features to reach a conclusion.</div></div>', unsafe_allow_html=True)
    
    sample_idx = st.slider("Select Sample Data Point", 0, len(X)-1, 0)
    x_sample = X[sample_idx:sample_idx+1]
    y_actual = y[sample_idx]
    
    # Show step-by-step forward pass
    current_input = x_sample
    for i, layer in enumerate(viz_nn.layers):
        with st.expander(f"Layer {i+1} Computation ({layer.activation_name})"):
            z = np.dot(current_input, layer.weights) + layer.biases
            a = layer.activation(z)
            st.write(f"**Weighted Sum (z):**")
            st.code(f"{z}")
            st.write(f"**Activation (a):**")
            st.code(f"{a}")
            current_input = a
            
    st.success(f"Final Prediction: {current_input[0,0]:.4f} | Actual Label: {y_actual[0]}")

# --- Tab 5 & 6: Training (Loss & Backprop) ---
with tabs[4]:
    st.markdown('<div class="concept-box"><div class="concept-title">Step 5: Guided Learning via Loss</div><div class="concept-desc">Training is the process of minimizing the error (Loss). We use Mean Squared Error (MSE) here.</div></div>', unsafe_allow_html=True)
    
    if st.button("🔥 Start Training Session"):
        st.session_state.history = []
        progress_bar = st.progress(0)
        loss_chart = st.empty()
        
        for epoch in range(epochs):
            loss = nn.train_step(X, y)
            st.session_state.history.append(loss)
            
            if epoch % 10 == 0:
                progress_bar.progress((epoch + 1) / epochs)
                loss_df = pd.DataFrame({"Epoch": range(len(st.session_state.history)), "Loss": st.session_state.history})
                fig = go.Figure(go.Scatter(x=loss_df["Epoch"], y=loss_df["Loss"], line=dict(color='#ef4444')))
                fig.update_layout(title=f"Learning Curve (Epoch {epoch})", xaxis_title="Epoch", yaxis_title="MSE Loss")
                loss_chart.plotly_chart(fig, use_container_width=True)
        
        st.session_state.trained_nn = nn
        st.session_state.trained = True

with tabs[5]:
    st.markdown('<div class="concept-box"><div class="concept-title">Step 6: Backpropagation - The Magic</div><div class="concept-desc">Gradient Descent uses the chain rule to figure out how much each weight contributed to the error, flow backwards and adjustments are made.</div></div>', unsafe_allow_html=True)
    
    if st.session_state.trained:
        st.subheader("Layer-wise Gradient Magnitudes")
        for i, layer in enumerate(reversed(viz_nn.layers)):
            if layer.dw is not None:
                grad_mag = np.mean(np.abs(layer.dw))
                st.write(f"**Layer {len(viz_nn.layers)-i}:** {grad_mag:.8f}")
            else:
                st.write(f"**Layer {len(viz_nn.layers)-i}:** Gradient not yet computed.")
        st.info("Watch how the gradients diminish (vanishing gradient) or stay healthy across layers during training.")
    else:
        st.warning("Train the model in Tab 5 to see gradients.")

# --- Tab 7: Model Complexity ---
with tabs[6]:
    st.markdown('<div class="concept-box"><div class="concept-title">Step 7: Visualizing Knowledge</div><div class="concept-desc">The decision boundary shows what the network has "learned". Too complex? Overfitting. Too simple? Underfitting.</div></div>', unsafe_allow_html=True)
    
    if st.session_state.get('trained', False):
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
        
        grid = np.c_[xx.ravel(), yy.ravel()]
        Z = st.session_state.trained_nn.predict(grid)
        Z = Z.reshape(xx.shape)
        
        fig = go.Figure()
        # Decision Boundary
        fig.add_trace(go.Contour(x=xx[0], y=yy[:,0], z=Z, colorscale='RdBu', opacity=0.4, showscale=False))
        # Data points
        fig.add_trace(go.Scatter(x=X[y.flatten()==0, 0], y=X[y.flatten()==0, 1], mode='markers', 
                               marker=dict(color='#ef4444', line=dict(width=1, color='white')), name="Class 0"))
        fig.add_trace(go.Scatter(x=X[y.flatten()==1, 0], y=X[y.flatten()==1, 1], mode='markers', 
                               marker=dict(color='#22c55e', line=dict(width=1, color='white')), name="Class 1"))
        
        fig.update_layout(title="Trained Decision Boundary", xaxis_title="x1", yaxis_title="x2", height=600)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Train the model first to visualize the decision boundary.")

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center; color: #64748b;'>Built with ❤️ for Neural Network Students</div>", unsafe_allow_html=True)
