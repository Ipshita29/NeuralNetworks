import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd
import requests
import io
from nn_engine import NeuralNetwork, Layer, Activation, generate_data, StandardScaler

# Page Setup
st.set_page_config(page_title="Neural Lab Pro", layout="wide", page_icon="🧬")

# Custom CSS
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# --- Sidebar Configuration ---
st.sidebar.markdown("### Model Configuration")

# Dataset Source Logic
data_source = st.sidebar.radio("Dataset Source", ["Standard", "Upload CSV", "Direct URL"])
X, y = None, None

if data_source == "Standard":
    dataset_name = st.sidebar.selectbox("Select Pattern", ["Circles", "Moons", "Linear Clusters"])
    noise_val = st.sidebar.slider("Dataset Noise", 0.0, 0.5, 0.1)
    X, y = generate_data(type=dataset_name.lower().split()[0], noise=noise_val)
elif data_source == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.sidebar.info(f"Loaded {len(df)} rows.")
        cols = df.columns.tolist()
        x1_col = st.sidebar.selectbox("Feature x1", cols, index=0)
        x2_col = st.sidebar.selectbox("Feature x2", cols, index=1 if len(cols)>1 else 0)
        y_col = st.sidebar.selectbox("Label (y)", cols, index=len(cols)-1)
        X = df[[x1_col, x2_col]].values
        y = df[y_col].values.reshape(-1, 1)
elif data_source == "Direct URL":
    st.sidebar.caption("Provide a **Raw CSV URL** (e.g., from raw.githubusercontent.com)")
    sample_url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
    if st.sidebar.button("Use Sample URL (Iris)"):
        url = sample_url
    else:
        url = st.sidebar.text_input("Raw CSV URL", placeholder="https://raw.githubusercontent.com/...")
    
    if url:
        try:
            response = requests.get(url, timeout=5)
            # Security & Format Check
            content_type = response.headers.get("Content-Type", "").lower()
            if "html" in content_type:
                st.sidebar.error("❌ Link is a website/HTML page, not a raw file. Use a 'Raw' link instead.")
            else:
                df = pd.read_csv(io.StringIO(response.text))
                st.sidebar.success("✅ Dataset fetched successfully.")
                cols = df.columns.tolist()
                x1_col = st.sidebar.selectbox("Feature x1 (URL)", cols, index=0)
                x2_col = st.sidebar.selectbox("Feature x2 (URL)", cols, index=1 if len(cols)>1 else 0)
                y_col = st.sidebar.selectbox("Label (y) (URL)", cols, index=len(cols)-1)
                X = df[[x1_col, x2_col]].values
                y = df[y_col].values.reshape(-1, 1)
        except Exception as e:
            st.sidebar.error(f"Fetch failed: {e}")

# Architecture & Optimization
st.sidebar.markdown("---")
st.sidebar.markdown("### Hyperparameters")
num_hidden_layers = st.sidebar.slider("Hidden Layers", 1, 5, 2)
neurons_per_layer = st.sidebar.slider("Neurons per Layer", 1, 32, 8)
activation_fn = st.sidebar.selectbox("Activation", ["Sigmoid", "ReLU", "Tanh"], index=1)
lr = st.sidebar.number_input("Learning Rate", value=0.001, format="%.4f")
epochs = st.sidebar.number_input("Max Epochs", value=500)

if st.sidebar.button("Reset & Retrain"):
    st.session_state.trained = False

# --- Initialization ---
if 'trained' not in st.session_state:
    st.session_state.trained = False
    st.session_state.history = []
    st.session_state.trained_nn = None

# Scaling (Only fit if X is available)
if X is not None:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
else:
    st.stop()

def get_architecture():
    return [2] + [neurons_per_layer] * num_hidden_layers + [1]

# Visualization Model
nn = NeuralNetwork(get_architecture(), lr, activation_fn)
viz_nn = st.session_state.trained_nn if st.session_state.trained else nn

# --- Main Navigation ---
tabs = st.tabs(["Overview", "Exploration", "User Guide"])

# 1. Overview Tab
with tabs[0]:
    st.markdown('<div class="concept-card"><div class="concept-title">Neural Engine Pro</div><div class="concept-body">This dashboard provides a professional environment to study and debug neural network behavior. It uses the Adam optimizer and mini-batch gradient descent for high-accuracy results on custom datasets.</div></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Network Architecture")
        arch = get_architecture()
        fig = go.Figure()
        for i, count in enumerate(arch):
            nodes_y = np.linspace(-count/2, count/2, count)
            fig.add_trace(go.Scatter(x=[i]*count, y=nodes_y, mode='markers', 
                                   marker=dict(size=12, color='#2563eb'), name=f"Layer {i}"))
            if i < len(arch) - 1:
                next_count = arch[i+1]
                next_nodes_y = np.linspace(-next_count/2, next_count/2, next_count)
                for y1 in nodes_y:
                    for y2 in next_nodes_y:
                        fig.add_trace(go.Scatter(x=[i, i+1], y=[y1, y2], mode='lines', 
                                               line=dict(width=0.1, color='#e2e8f0'), hoverinfo='none'))
        fig.update_layout(showlegend=False, xaxis=dict(visible=False), yaxis=dict(visible=False), height=400, margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Current Dataset")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=X[y.flatten()==0, 0], y=X[y.flatten()==0, 1], mode='markers', 
                               marker=dict(color='#ef4444', size=8), name="Class 0"))
        fig.add_trace(go.Scatter(x=X[y.flatten()==1, 0], y=X[y.flatten()==1, 1], mode='markers', 
                               marker=dict(color='#3b82f6', size=8), name="Class 1"))
        fig.update_layout(title="Raw Input Features", margin=dict(l=0,r=0,t=40,b=0), height=400)
        st.plotly_chart(fig, use_container_width=True)

# 2. Exploration Tab
with tabs[1]:
    subtabs = st.tabs(["Neuron Math", "Forward Prop", "Training & Loss", "Boundary Analysis"])
    
    with subtabs[0]:
        st.subheader("Interactive Neuron Computation")
        c1, c2 = st.columns([1, 2])
        with c1:
            w = st.slider("Weight (w)", -5.0, 5.0, 1.0)
            b = st.slider("Bias (b)", -5.0, 5.0, 0.0)
            x_in = st.number_input("Input (x)", value=0.5)
            z_val = w*x_in + b
            act, _ = Activation.get(activation_fn)
            st.code(f"z = w*x + b = {z_val:.4f}\na = {activation_fn}(z) = {act(z_val):.4f}")
        with c2:
            z_range = np.linspace(-10, 10, 100)
            fig = go.Figure(go.Scatter(x=z_range, y=act(z_range), name=activation_fn))
            fig.add_trace(go.Scatter(x=[z_val], y=[act(z_val)], marker=dict(size=12, color='red'), name="Current Point"))
            fig.update_layout(title="Activation Mapping", height=300)
            st.plotly_chart(fig, use_container_width=True)

    with subtabs[1]:
        st.subheader("Forward Propagation Trace")
        sample_idx = st.slider("Select Data Sample", 0, len(X)-1, 0)
        x_sample = X_scaled[sample_idx:sample_idx+1]
        
        current = x_sample
        for i, layer in enumerate(viz_nn.layers):
            with st.expander(f"Layer {i+1} Trace"):
                st.write(f"Weights Shape: {layer.weights.shape}")
                current = layer.forward(current)
                st.write("Activation Output:")
                st.code(f"{current}")

    with subtabs[2]:
        st.subheader("Model Training")
        if st.button("Initialize Optimization"):
            st.session_state.history = []
            progress = st.progress(0)
            loss_area = st.empty()
            
            for epoch in range(epochs):
                loss = nn.train_step(X_scaled, y)
                st.session_state.history.append(loss)
                if epoch % 50 == 0:
                    progress.progress((epoch+1)/epochs)
                    fig = go.Figure(go.Scatter(x=list(range(len(st.session_state.history))), y=st.session_state.history))
                    fig.update_layout(title=f"MSE Loss: {loss:.6f}", xaxis_title="Epoch", yaxis_title="Loss")
                    loss_area.plotly_chart(fig, use_container_width=True)
            
            st.session_state.trained_nn = nn
            st.session_state.trained = True
            st.success("Training Complete")

    with subtabs[3]:
        st.subheader("Decision Boundary Analysis")
        if st.session_state.trained:
            x_min, x_max = X_scaled[:, 0].min() - 0.5, X_scaled[:, 0].max() + 0.5
            y_min, y_max = X_scaled[:, 1].min() - 0.5, X_scaled[:, 1].max() + 0.5
            xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
            grid = np.c_[xx.ravel(), yy.ravel()]
            preds = st.session_state.trained_nn.predict(grid).reshape(xx.shape)
            
            fig = go.Figure()
            fig.add_trace(go.Contour(x=xx[0], y=yy[:,0], z=preds, colorscale='RdBu', opacity=0.4))
            fig.add_trace(go.Scatter(x=X_scaled[y.flatten()==0, 0], y=X_scaled[y.flatten()==1, 1], mode='markers', name="Class 0"))
            fig.update_layout(title="Latent Space Separation", height=600)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Optimize the model in the Training tab to visualize features.")

# 3. User Guide Tab
with tabs[2]:
    with open("user_guide.md") as f:
        st.markdown(f.read())

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: #64748b; font-size: 0.8rem;'>Professional Neural Lab Pro | 2026 Academic Edition</p>", unsafe_allow_html=True)
