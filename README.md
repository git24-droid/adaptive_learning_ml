# AdaptML — Adaptive Learning System

ML-powered adaptive quiz platform built with PyTorch, Bayesian Knowledge Tracing, Groq LLM, and Streamlit.

---

## System Architecture

```
Teacher uploads PDF
       │
       ▼
PyMuPDF → text extraction
       │
       ▼
Groq (Llama 3.1) → generates MCQs with options + explanations
       │
       ▼
all-MiniLM-L6-v2 → 384-dim sentence embeddings
       │
       ▼
PyTorch MLP (384→256→128→64→3) → difficulty prediction (easy/medium/hard)
       │
       ▼
Question Bank (JSON) ←→ Teacher can delete unwanted questions
       │
       ▼
Student takes quiz
       │
       ▼
BKT (Bayesian Knowledge Tracing) → per-topic P(know) updated after every answer
       │
       ▼
Adaptive selection → maps P(know) → target difficulty → picks best unasked question
       │
       ▼
Analytics dashboard + ML Stats (confusion matrix, F1, loss curves)
```

---

## Quick Start

### 1. Clone and install

```bash
git clone <your-repo>
cd adaptive_learning

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Add your Groq API key

Get a free key at https://console.groq.com

```bash
cp .env.example .env
# Edit .env and set: GROQ_API_KEY=your_key_here
```

### 3. Train the difficulty classifier (one-time setup)

```bash
python setup.py
```

This will:
- Generate the training dataset (~240 labelled questions)
- Download sentence-transformer (~80 MB, first run only)
- Train the PyTorch MLP for 60 epochs
- Save model to `models/difficulty_model.pt`
- Save training stats (confusion matrix, F1, loss curves) to `models/training_stats.json`

### 4. Launch

```bash
streamlit run app.py
```

Open http://localhost:8501

---

## Features

### 🎓 Teacher Dashboard
- Upload any PDF (lecture notes, textbook chapters, etc.)
- Groq LLM (Llama 3.1) generates multiple-choice questions with 4 options, correct answer, and explanation
- PyTorch MLP classifies each question as easy / medium / hard with a confidence score
- Review all questions — see difficulty badge, ML confidence %, times shown, correct rate
- Delete individual questions or clear the entire bank
- Filter by topic or difficulty

### 📚 Student Quiz
- Enter name → quiz begins immediately
- Adaptive question selection: BKT P(know) per topic drives target difficulty
- Click A/B/C/D buttons — coloured feedback (green = correct, red = wrong)
- Explanation shown after every answer
- Live BKT chart updates after every response
- Session persists across browser refreshes

### 📊 Student Analytics
- Per-student cumulative accuracy chart
- Difficulty breakdown (answered vs correct by easy/medium/hard)
- BKT knowledge heatmap with mastery threshold line
- P(know) trajectory per topic over time
- Full response history table

### 🔬 ML Model Stats
- Model architecture summary
- Interactive confusion matrix (Plotly heatmap)
- Per-class precision / recall / F1 table + grouped bar chart
- Training & validation loss curves
- BKT equations rendered in LaTeX
- Adaptive selection algorithm documented
- Question bank analysis: difficulty distribution, confidence boxplots, topic × difficulty heatmap
- Retrain button to re-run training pipeline

---

## ML Model Details

### Difficulty Classifier

| Property | Value |
|----------|-------|
| Type | Multi-layer Perceptron (MLP) |
| Input | 384-dimensional sentence embedding |
| Architecture | 384 → 256 (BN+ReLU+Drop) → 128 (BN+ReLU+Drop) → 64 (ReLU) → 3 |
| Embedder | `all-MiniLM-L6-v2` (SentenceTransformers) |
| Loss | CrossEntropyLoss |
| Optimizer | Adam (lr=1e-3, weight_decay=1e-4) |
| Scheduler | ReduceLROnPlateau (patience=5) |
| Training | 60 epochs, best checkpoint saved |
| Dataset | ~240 samples, 80/20 stratified split, 2× augmentation |

### Bayesian Knowledge Tracing

```
P(know | correct) = P(know)*(1-P_slip) / [P(know)*(1-P_slip) + (1-P(know))*P_guess]
P(know | wrong)   = P(know)*P_slip / [P(know)*P_slip + (1-P(know))*(1-P_guess)]
P(know_next)      = P(know_updated) + (1-P(know_updated)) * P_learn
```

| Parameter | Value |
|-----------|-------|
| P(init)   | 0.30 |
| P(learn)  | 0.10 |
| P(guess)  | 0.20 |
| P(slip)   | 0.10 |
| Mastery   | 0.85 |

### Adaptive Selection

| P(know) | Target difficulty |
|---------|-----------------|
| < 0.40  | easy            |
| 0.40–0.75 | medium        |
| ≥ 0.75  | hard            |

Selection score: `|diff_distance| + confidence_penalty + freshness + random_tiebreak`

---

## Deploy to Streamlit Cloud

1. Push to GitHub (public or private repo)
2. Go to https://share.streamlit.io → New app → select repo → `app.py`
3. In **Advanced settings → Secrets**, add:
   ```
   GROQ_API_KEY = "your_key_here"
   ```
4. Add a `packages.txt` with:
   ```
   libgl1
   ```
5. Click Deploy

> **Note:** The pre-trained model (`models/difficulty_model.pt`) and `models/training_stats.json` must be committed to the repo, or you add `python setup.py` as a pre-run command.

---

## Project Structure

```
adaptive_learning/
├── app.py                     # Main Streamlit app
├── setup.py                   # First-time training script
├── requirements.txt
├── .env.example
├── .streamlit/
│   └── config.toml
├── data/
│   ├── dataset.json           # Training data (generated by setup.py)
│   ├── question_bank.json     # Teacher's MCQ bank
│   └── sessions.json          # Student session data
├── models/
│   ├── difficulty_model.pt    # Trained PyTorch weights
│   └── training_stats.json    # Confusion matrix, F1, loss curves
└── src/
    ├── __init__.py
    ├── bkt.py                 # Bayesian Knowledge Tracing
    ├── generate_dataset.py    # Training data generator
    ├── train_model.py         # PyTorch MLP training + inference
    ├── teacher_pipeline.py    # PDF → LLM → classifier → bank
    ├── student_pipeline.py    # Session persistence
    └── adaptive.py            # Adaptive question selection
```

---

## Customisation

**Change BKT parameters:** Edit `BKTModel` defaults in `src/bkt.py`

**Change adaptive thresholds:** Edit `_target_difficulty()` in `src/adaptive.py`

**Change model architecture:** Edit `DifficultyClassifier` in `src/train_model.py`, then retrain

**Use a different LLM:** Edit `generate_mcqs_from_text()` in `src/teacher_pipeline.py`

**Add more training data:** Append samples to `SAMPLES` in `src/generate_dataset.py`, then retrain

---

## Requirements

- Python 3.10+
- GROQ_API_KEY (free at console.groq.com)
- ~500 MB disk (model weights + sentence-transformer)
- GPU optional — CPU training takes ~2 min
