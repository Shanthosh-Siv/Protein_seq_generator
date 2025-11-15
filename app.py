import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import gradio as gr
import os

# ==============================
# File Paths
# ==============================
csv_path = "cancer_protein_dataset.csv"
model_path = "model/cancer_protein_generator.keras"

# ==============================
# Load Dataset for Label Encoding
# ==============================
df = pd.read_csv(csv_path)

# Tokenizer and detokenizer
amino_acids = "ACDEFGHIKLMNPQRSTVWY"
tokenizer = {aa: idx + 1 for idx, aa in enumerate(amino_acids)}
detokenizer = {v: k for k, v in tokenizer.items()}

# Label Encoding for cancer types
label_encoder = LabelEncoder()
df['cancer_label'] = label_encoder.fit_transform(df['Cancer_Type'])
cancer_types = list(label_encoder.classes_)

# ==============================
# Load Model
# ==============================
model = load_model(model_path)

max_length = 512
sequence_length = max_length - 1
vocab_size = 21

# ==============================
# Protein Generator
# ==============================
def generate_protein(cancer_type, seq_length=200):

    if cancer_type not in label_encoder.classes_:
        return f"❌ Invalid cancer type! Choose from: {cancer_types}", None

    cancer_label = label_encoder.transform([cancer_type])[0]

    seed = "MKTLL"
    seq = [tokenizer.get(aa, 0) for aa in seed]

    for _ in range(seq_length - len(seq)):
        seq_input = np.array(seq[-sequence_length:], dtype=np.int32)
        seq_input = np.pad(seq_input, (0, sequence_length - len(seq_input)), mode='constant')

        seq_input = seq_input.reshape(1, -1)
        label_input = np.array([[cancer_label]])

        pred = model.predict([seq_input, label_input], verbose=0)
        next_token = np.argmax(pred)

        if next_token == 0:
            next_token = np.random.randint(1, vocab_size)

        seq.append(next_token)

    protein_sequence = ''.join(detokenizer.get(tok, 'X') for tok in seq if tok != 0)

    return "✅ Protein generated successfully!", protein_sequence

# ==============================
# Protein Validation
# ==============================
def validate_protein(sequence):

    aa_weights = {
        'A': 89.09, 'C': 121.16, 'D': 133.1, 'E': 147.13, 'F': 165.19,
        'G': 75.07, 'H': 155.16, 'I': 131.18, 'K': 146.19, 'L': 131.18,
        'M': 149.21, 'N': 132.12, 'P': 115.13, 'Q': 146.15, 'R': 174.2,
        'S': 105.09, 'T': 119.12, 'V': 117.15, 'W': 204.23, 'Y': 181.19
    }

    total_len = len(sequence)
    aa_freq = {aa: sequence.count(aa) for aa in aa_weights.keys()}
    mol_weight = sum(aa_weights.get(aa, 0) for aa in sequence)
    hydrophobicity = sum(1 for aa in sequence if aa in set("AILMFWYV")) / total_len

    validation_report = f"""
🧪 Validation Report
--------------------------
📌 Total Length: {total_len}
⚖️ Estimated Molecular Weight: {mol_weight:.2f} Da
💧 Hydrophobic Residue Ratio: {hydrophobicity:.2%}

🔠 Amino Acid Composition:
{aa_freq}

✅ Validation complete.
"""
    return validation_report

# ==============================
# Gradio App Wrapper
# ==============================
def gradio_app(candidate_name, age, gender, role, purpose, cancer_type, seq_length):

    if not candidate_name.strip():
        return "❌ Candidate Name is required!", "", ""
    if age <= 0:
        return "❌ Please enter a valid age.", "", ""

    status, sequence = generate_protein(cancer_type, seq_length)

    if sequence is None:
        return status, "", ""

    validation = validate_protein(sequence)

    status += f" (Candidate: {candidate_name}, {age} years old, {gender} - {role})"

    return status, sequence, validation


# ==============================
# GRADIO UI
# ==============================
with gr.Blocks(title="Protein Generator for Cancer Therapy") as demo:

    gr.Markdown("## 🧬 Protein Generator for Cancer Therapy")
    gr.Markdown("Enter the details below to generate a custom protein sequence for the selected cancer type.")

    with gr.Row():

        with gr.Column():
            candidate_name = gr.Textbox(label="Candidate Name (Required)")
            age = gr.Number(label="Age", value=25, precision=0)
            gender = gr.Dropdown(["Male", "Female", "Other"], label="Gender")
            role = gr.Dropdown(["Student", "Researcher", "Scientist"], label="Role")
            purpose = gr.Textbox(label="Purpose", placeholder="e.g., Research / Study")
            cancer_type = gr.Dropdown(cancer_types, label="Cancer Type")
            seq_length = gr.Slider(100, 500, value=200, label="Sequence Length")

        with gr.Column():
            status_box = gr.Textbox(label="Status")
            protein_box = gr.Textbox(label="Generated Protein Sequence", show_copy_button=True)
            validation_box = gr.Textbox(label="Protein Validation Report", show_copy_button=True)

    generate_btn = gr.Button("Generate Protein")

    generate_btn.click(
        fn=gradio_app,
        inputs=[candidate_name, age, gender, role, purpose, cancer_type, seq_length],
        outputs=[status_box, protein_box, validation_box]
    )

demo.launch()
