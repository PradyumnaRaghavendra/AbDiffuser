# AbDiffuser: Full-Atom Generation of Antibodies

## 🧬 Overview
AbDiffuser implements the paper **"AbDiffuser: Full-Atom Generation of in vitro Functioning Antibodies"** by Martinkus et al.  
This model can generate **novel antibody sequences and 3D structures** using a diffusion-based approach with an innovative **Aligned Protein Mixer (APMixer)** architecture.

---

## ✨ Features
- **Sequence and Structure Generation**: Jointly models antibody sequences and their 3D atomic structures.
- **Physics-Informed Constraints**: Enforces physical constraints on bond lengths and angles.
- **Aligned Protein Mixer (APMixer)**: Efficient neural network architecture specialized for aligned protein families.
- **SE(3) Equivariance**: Ensures predictions are rotation and translation invariant.
- **Position-Specific Residue Frequencies**: Incorporates known amino acid distributions for better modeling.

---

## ⚙️ Installation
```bash
# Clone repository
git clone 
cd abdiffuser

📚 Datasets
The model uses two main datasets:

Paired Observable Antibody Space (pOAS): Contains paired heavy and light chain antibody sequences.

HER2 Binder Dataset: Contains CDR H3 mutants of Trastuzumab labeled as binders or non-binders to HER2.

🚀 Usage
1. Data Preprocessing

# Process OAS dataset
python abdiffuser/scripts/preprocess_data.py \
  --mode oas \
  --oas_dir data/OAS \
  --output data/processed/oas_aligned.pkl \
  --max_sequences 1000 \
  --generate_priors

# Process HER2 dataset
python abdiffuser/scripts/preprocess_data.py \
  --mode her2 \
  --her2_dir data/HER2 \
  --output_dir data/processed
2. Training

# Train on OAS dataset
python abdiffuser/scripts/train_model.py \
  --mode oas \
  --oas_data abdiffuser/data/processed/oas_aligned.pkl \
  --output_dir abdiffuser/experiments/checkpoints \
  --batch_size 4 \
  --num_epochs 10

📂 Project Structure

abdiffuser/
├── data/              # Data storage
│   ├── OAS/           # Paired Observable Antibody Space dataset
│   ├── HER2/          # HER2 binder dataset
│   └── processed/     # Preprocessed data files
│
├── models/            # Model components
│   ├── apmixer.py     # Aligned Protein Mixer
│   ├── diffusion.py   # Diffusion models
│   ├── projection.py  # Physics-informed projections
│   └── priors.py      # Diffusion priors
│
├── utils/             # Utility functions
│   ├── aho_numbering.py
│   ├── data_loader.py
│   └── frames.py
│
├── scripts/           # Training and generation scripts
│   ├── preprocess_data.py
│   ├── train_model.py
│   └── generate.py
│
└── experiments/       # Experimental outputs
    ├── checkpoints/   # Saved model weights
    └── outputs/       # Generated samples
📊 Results
Generated novel antibody sequences along with realistic 3D backbone structures.

Proper folding patterns and amino acid distributions observed, comparable to natural antibodies.


📝 License
This project is licensed under the MIT License – see the LICENSE file for details.

