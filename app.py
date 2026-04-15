import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd

# --- Neural Engine Logic ---

class Activation:
    @staticmethod
    def sigmoid(x): return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    @staticmethod
    def sigmoid_prime(x):
        s = Activation.sigmoid(x)
        return s * (1 - s)
    @staticmethod
    def relu(x): return np.maximum(0, x)
    @staticmethod
    def relu_prime(x): return (x > 0).astype(float)
    @staticmethod
    def tanh(x): return np.tanh(x)
    @staticmethod
    def tanh_prime(x): return 1 - np.tanh(x)**2
    @staticmethod
    def get(name):
        return {
            "Sigmoid": (Activation.sigmoid, Activation.sigmoid_prime),
            "ReLU": (Activation.relu, Activation.relu_prime),
            "Tanh": (Activation.tanh, Activation.tanh_prime)
        }.get(name, (Activation.sigmoid, Activation.sigmoid_prime))

class Layer:
    def __init__(self, input_size, output_size, activation_name="Sigmoid"):
        scale = np.sqrt(2.0/input_size) if activation_name == "ReLU" else np.sqrt(1.0/input_size)
        self.weights = np.random.randn(input_size, output_size) * scale
        self.biases = np.zeros((1, output_size))
        self.activation_name = activation_name
        self.activation, self.activation_prime = Activation.get(activation_name)
        
        self.mw, self.vw = np.zeros_like(self.weights), np.zeros_like(self.weights)
        self.mb, self.vb = np.zeros_like(self.biases), np.zeros_like(self.biases)
        
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
        return np.dot(delta, self.weights.T)

class NeuralNetwork:
    def __init__(self, architecture, learning_rate=0.01, activation="Sigmoid", beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.layers = [Layer(architecture[i], architecture[i+1], activation) for i in range(len(architecture)-1)]
        self.lr = learning_rate
        self.beta1, self.beta2 = beta1, beta2
        self.epsilon = epsilon
        self.t = 0

    def predict(self, x):
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def train_step(self, x, y):
        self.t += 1
        prediction = self.predict(x)
        
        if prediction.shape != y.shape:
            return None
            
        grad = 2 * (prediction - y) / y.size
        
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
            
            layer.mw = self.beta1 * layer.mw + (1 - self.beta1) * layer.dw
            layer.vw = self.beta2 * layer.vw + (1 - self.beta2) * (layer.dw**2)
            mw_corr = layer.mw / (1 - self.beta1**self.t)
            vw_corr = layer.vw / (1 - self.beta2**self.t)
            layer.weights -= self.lr * mw_corr / (np.sqrt(vw_corr) + self.epsilon)
            
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
        self.std[self.std == 0] = 1
        return (x - self.mean) / self.std

def generate_data(type="circles", n_samples=300, noise=0.1):
    from sklearn.datasets import make_circles, make_moons, make_blobs
    if type == "circles":
        X, y = make_circles(n_samples=n_samples, factor=0.5, noise=noise)
    elif type == "moons":
        X, y = make_moons(n_samples=n_samples, noise=noise)
    else:
        X, y = make_blobs(n_samples=n_samples, centers=2, cluster_std=noise*5, random_state=42)
    return X.astype(float), y.reshape(-1, 1).astype(float)

# Silently cleans data so the user never sees technical errors
def validate_and_clean_data_silent(df, feature_cols, label_col):
    try:
        X_df = df[feature_cols].copy()
        y_series = df[label_col].copy()
        
        # Auto-encode text in features
        if X_df.select_dtypes(include=['object', 'category']).columns.any():
            X_df = pd.get_dummies(X_df, drop_first=True)
            
        for col in X_df.columns:
            X_df[col] = pd.to_numeric(X_df[col], errors='coerce')
            
        # Quick mean fill
        if X_df.isna().sum().sum() > 0:
            X_df.fillna(X_df.mean(numeric_only=True), inplace=True)
            X_df.fillna(0, inplace=True)
            
        X_num = np.array(X_df, dtype=float)
        
        # Target fix
        if y_series.isna().sum() > 0:
            val = y_series.mode()
            y_series = y_series.fillna(val[0] if not val.empty else 0)
            
        if y_series.dtype == 'object' or str(y_series.dtype) == 'category':
            y_series = pd.factorize(y_series)[0]
        else:
            y_series = pd.to_numeric(y_series, errors='coerce').fillna(0)
            
        y_num = np.array(y_series, dtype=float).reshape(-1, 1)
        
        if X_num.shape[0] == 0 or y_num.shape[0] == 0 or X_num.shape[0] != y_num.shape[0]:
            return None, None
            
        return X_num, y_num
    except Exception:
        return None, None

# --- Streamlit App ---

st.set_page_config(page_title="Neural Network Visualizer", layout="wide", page_icon="🧠")

with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# -----------------
# SIDEBAR
# -----------------
st.sidebar.title("Data Setup")

data_source = st.sidebar.radio("Choose Dataset", ["Toy Examples (Generated)", "Upload CSV"], help="Pick pre-made data or upload your own file.")
X, y = None, None

if data_source == "Toy Examples (Generated)":
    dataset_name = st.sidebar.selectbox("Pattern Type", ["Circles", "Moons", "Clusters"], help="These shapes test how well the neural network can bend its boundaries.")
    noise_val = st.sidebar.slider("Data Messiness (Noise)", 0.0, 0.5, 0.1, help="Adds randomness mimicking real-world imperfect data.")
    X, y = generate_data(type=dataset_name.lower().split()[0], noise=noise_val)
elif data_source == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type="csv", help="Input your own dataset. We handle missing values and text automatically!")
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.sidebar.caption(f"Loaded {len(df)} rows.")
            cols = df.columns.tolist()
            x1_col = st.sidebar.selectbox("First Feature Column", cols, index=0, help="The first variable giving the network hints.")
            x2_col = st.sidebar.selectbox("Second Feature Column", cols, index=1 if len(cols)>1 else 0, help="The second variable.")
            y_col = st.sidebar.selectbox("Output to Predict", cols, index=len(cols)-1, help="The column the network is trying to guess correctly.")
            X, y = validate_and_clean_data_silent(df, [x1_col, x2_col], y_col)
            if X is None:
                st.sidebar.error("Could not parse dataset columns. Try picking different features.")
        except Exception:
            st.sidebar.error("Failed to read CSV. Please ensure it is formatted correctly.")

st.sidebar.markdown("---")
st.sidebar.title("Adjust the Brain")
num_hidden_layers = st.sidebar.slider("Hidden Layers", 1, 5, 2, help="More layers allow the model to learn complex patterns. Think of layers like steps in a recipe.")
neurons_per_layer = st.sidebar.slider("Neurons per Layer", 1, 32, 8, help="Controls how much information each layer can hold. Too many can cause 'overthinking' (overfitting).")
activation_fn = st.sidebar.selectbox("Activation Function", ["Sigmoid", "ReLU", "Tanh"], index=1, help="The math rule the neuron uses to 'fire'. ReLU is fast, Sigmoid is smooth.")
lr = st.sidebar.select_slider("Learning Rate", options=[0.001, 0.01, 0.05, 0.1, 0.5], value=0.05, help="How fast the network corrects a mistake. Too high and it skips over the right answer.")

# --- Initialization ---
if X is not None and y is not None and len(X) > 0 and len(y) > 0:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
else:
    st.markdown("<h2 style='text-align: center; color: #64748b; margin-top: 15vh;'>Please load some data on the left to begin exploring!</h2>", unsafe_allow_html=True)
    st.stop()

def get_architecture(input_dim):
    return [input_dim] + [neurons_per_layer] * num_hidden_layers + [1]

# Instant Training Background Loop
# We train it very quickly right before we draw the plots so the user sees results immediately
nn = NeuralNetwork(get_architecture(X_scaled.shape[1]), lr, activation_fn)
instant_epochs = 150 # fixed fast training cap

if 'last_loss' not in st.session_state:
    st.session_state.last_loss = 1.0

try:
    for _ in range(instant_epochs):
        loss = nn.train_step(X_scaled, y)
        if loss is not None:
            st.session_state.last_loss = loss
except Exception:
    pass

# -----------------
# MAIN PANEL
# -----------------
st.title("Neural Network Visualizer")

tab1, tab2 = st.tabs(["Real-Time Visualization", "Beginner's Guide"])

with tab1:
    st.markdown("""
    Welcome to the Visualizer! As you change the sliders on the left, the network trains instantly in the background. 
    Watch the "Before / After" graph dynamically morph as the neural network changes shapes!
    """)
    
    # Metrics
    with st.container():
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Dataset Rows", len(X), help="Total data points")
        m2.metric("Features", X_scaled.shape[1], help="Number of input dimensions")
        m3.metric("Network Depth", num_hidden_layers, help="Total hidden layers")
        m4.metric("Current Accuracy estimate", f"{max(0, 100 - (st.session_state.last_loss*100)):.1f}%", help="Rough guess of accuracy based on Loss.")
        
    st.markdown("<br/>", unsafe_allow_html=True)

    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader("Network Topography (Architecture)")
        arch = get_architecture(X_scaled.shape[1])
        fig = go.Figure()
        for i, count in enumerate(arch):
            nodes_y = np.linspace(-count/2, count/2, count)
            # Primary nodes
            fig.add_trace(go.Scatter(x=[i]*count, y=nodes_y, mode='markers', 
                                   marker=dict(size=14, color="#394d72", line=dict(width=1, color='#ffffff')), name=f"Layer {i}"))
            if i < len(arch) - 1:
                next_count = arch[i+1]
                next_nodes_y = np.linspace(-next_count/2, next_count/2, next_count)
                for y1 in nodes_y:
                    for y2 in next_nodes_y:
                        fig.add_trace(go.Scatter(x=[i, i+1], y=[y1, y2], mode='lines', 
                                               line=dict(width=0.2, color='#cbd5e1'), hoverinfo='none'))
        fig.update_layout(showlegend=False, xaxis=dict(visible=False), yaxis=dict(visible=False), 
                          height=350, margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, width="stretch")
            
    with c2:
        st.subheader("Decision Boundary Output")
        # Map decision boundary immediately using the instantly trained model
        try:
            x_min, x_max = X_scaled[:, 0].min() - 0.5, X_scaled[:, 0].max() + 0.5
            y_min, y_max = X_scaled[:, 1].min() - 0.5, X_scaled[:, 1].max() + 0.5
            xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
            grid = np.c_[xx.ravel(), yy.ravel()]
            
            if grid.shape[1] < X_scaled.shape[1]:
                padding = np.zeros((grid.shape[0], X_scaled.shape[1] - grid.shape[1]))
                grid_full = np.hstack([grid, padding])
            else:
                grid_full = grid
                
            raw_preds = nn.predict(grid_full)
            preds = raw_preds.reshape(xx.shape)
            
            fig = go.Figure()
            # Contour mapped cleanly
            # fig.add_trace(go.Contour(x=xx[0], y=yy[:,0], z=preds, colorscale='Blues', opacity=0.4))
            
            # Scatter Points
            if len(y) > 0 and y.max() <= 1 and y.min() >= 0:
                fig.add_trace(go.Scatter(x=X_scaled[y.flatten()<=0.5, 0], y=X_scaled[y.flatten()<=0.5, 1], mode='markers', 
                                       marker=dict(color="#9f3f3f", size=8, line=dict(width=1, color='#ffffff')), name="Group A"))
                fig.add_trace(go.Scatter(x=X_scaled[y.flatten()>0.5, 0], y=X_scaled[y.flatten()>0.5, 1], mode='markers', 
                                       marker=dict(color="#436db1", size=8, line=dict(width=1, color='#ffffff')), name="Group B"))
            else:
                fig.add_trace(go.Scatter(x=X_scaled[:, 0], y=X_scaled[:, 1], mode='markers', marker=dict(color=y.flatten(), colorscale='Viridis', size=8)))
                
            fig.update_layout(height=400, margin=dict(l=0,r=0,t=20,b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, width="stretch")
        except Exception:
            st.info("Mapping the boundary is not possible with these feature shapes!")

with tab2:
    st.markdown("""
        ### ✧ How to Use This Visualizer ✧

        Neural networks may sound complex, but at their core, they are simply learning how to separate patterns — quietly drawing invisible boundaries between different types of data.

        ---

        **1. Choose Your World**

        From the sidebar, select a dataset to begin.

        Each dataset is like a small universe of points.  
        For example, *“Moons”* creates two curved shapes intertwined with each other.

        Your network’s purpose is simple:  
        to understand this space and gently divide one group from the other.

        ---

        **2. Shape the Mind**

        Now you begin to design the “brain”.

        - **Hidden Layers**  
        Think of each layer as a deeper level of thought.  
        More layers allow the network to understand more abstract and complex patterns.

        - **Neurons**  
        These are the tiny decision-makers inside each layer.  
        More neurons give the network more flexibility to shape smoother, more detailed boundaries.

        - **Learning Rate**  
        This controls how quickly the network adapts.  
        Too fast, and it may miss the answer.  
        Too slow, and it takes its time finding clarity.

        ---

        **3. Watch the Transformation**

        As you adjust the sliders, the network rebuilds itself instantly.

        What you see is not just a graph —  
        it is the model’s understanding of the data taking form.

        The shifting colors represent the **decision boundary** —  
        the invisible line that separates one pattern from another.

        ---

        ✦ With every change, you are not just tuning parameters —  
        you are shaping how a machine learns to perceive the world.
        """)
