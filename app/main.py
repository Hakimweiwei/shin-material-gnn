import sys
import os

# Add root directory to path to allow importing from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import torch
from rdkit import Chem
from rdkit.Chem import Draw

from src.models.gnn_predictor import GNN_Predictor
from src.chemistry.featurizer import smiles_to_graph

st.set_page_config(page_title="Shin-Material ESOL Predictor", page_icon="🧪", layout="centered")

@st.cache_resource
def load_model():
    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Initialize model with the exact architecture that achieved R2 > 0.80
    model = GNN_Predictor(node_dim=5, edge_dim=6, hidden_dim=64, num_layers=3)
    
    # Load weights
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'saved_models', 'best_model.pth'))
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model, device

# Sidebar
st.sidebar.title("🧪 Shin-Material")
st.sidebar.markdown("**GNN Model Confidence Score**")
st.sidebar.metric(label="R² Score", value="0.843")
st.sidebar.metric(label="MAE", value="0.629")
st.sidebar.metric(label="RMSE", value="0.811")
st.sidebar.markdown("---")
st.sidebar.info("Model GNN ini dilatih menggunakan dataset ESOL (Water Solubility) dengan arsitektur GINEConv dan global_add_pool.")

# Main Layout
st.title("Water Solubility (ESOL) Predictor")
st.markdown("Masukkan struktur molekul dalam format **SMILES** untuk memprediksi kelarutannya di dalam air.")

# User Input
smiles_input = st.text_input("SMILES String", value="CCO", help="Contoh: CCO (Etanol), c1ccccc1 (Benzena)")

if smiles_input:
    mol = Chem.MolFromSmiles(smiles_input)
    if mol is None:
        st.error("❌ SMILES tidak valid! Harap periksa kembali format SMILES Anda.")
    else:
        # Visualisasi Kimia
        st.markdown("### Struktur 2D Molekul")
        img = Draw.MolToImage(mol, size=(300, 300))
        st.image(img, caption=f"SMILES: {smiles_input}")
        
        # Tombol Prediksi
        if st.button("Prediksi Kelarutan (ESOL)", type="primary"):
            with st.spinner("Menghitung representasi graf dan memprediksi..."):
                model, device = load_model()
                
                # Featurize
                graph_data = smiles_to_graph(smiles_input)
                
                if graph_data is None:
                    st.error("Gagal melakukan pra-pemrosesan RDKit pada molekul ini.")
                else:
                    # Tambahkan dummy batch agar sesuai format input model
                    graph_data.batch = torch.zeros(graph_data.x.shape[0], dtype=torch.long)
                    graph_data = graph_data.to(device)
                    
                    with torch.no_grad():
                        pred = model(graph_data.x, graph_data.edge_index, graph_data.edge_attr, graph_data.batch)
                        pred_val = pred.item()
                        
                    st.success("Prediksi Berhasil!")
                    st.metric(label="Log Solubility (mols/L)", value=f"{pred_val:.3f}")
                    
                    if pred_val > -2:
                        st.info("💧 Kategori: Sangat Larut / Larut (Highly/Moderately Soluble)")
                    elif pred_val > -4:
                        st.warning("⚖️ Kategori: Kurang Larut (Poorly Soluble)")
                    else:
                        st.error("🧱 Kategori: Sangat Sukar Larut (Insoluble)")
