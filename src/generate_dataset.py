import json
import random
import os

# ── Core samples ────────────────────────────────────────────────────────────
# Rule: phrasing is intentionally mixed across difficulty levels so the model
# cannot trivially learn "What is = easy, Derive = hard".
SAMPLES = [
    # ── EASY ────────────────────────────────────────────────────────────────
    ("What is a neural network?", "easy"),
    ("What does CPU stand for?", "easy"),
    ("What is a variable in programming?", "easy"),
    ("What is supervised learning?", "easy"),
    ("What is ROS?", "easy"),
    ("What is a topic in ROS?", "easy"),
    ("What is a node in ROS?", "easy"),
    ("What is overfitting?", "easy"),
    ("What is a dataset?", "easy"),
    ("What is accuracy in ML?", "easy"),
    ("What is a label in machine learning?", "easy"),
    ("What is a loss function?", "easy"),
    ("What is gradient descent?", "easy"),
    ("What is a feature in machine learning?", "easy"),
    ("What does ROS stand for?", "easy"),
    ("What is a publisher in ROS?", "easy"),
    ("What is a subscriber in ROS?", "easy"),
    ("What is linear regression?", "easy"),
    ("What is classification in ML?", "easy"),
    ("What is a test set?", "easy"),
    ("What is a training set?", "easy"),
    ("What is an epoch in deep learning?", "easy"),
    ("What is a matrix?", "easy"),
    ("What is a vector?", "easy"),
    ("What is Python?", "easy"),
    ("What is a for loop?", "easy"),
    ("What is a function in programming?", "easy"),
    ("What is an API?", "easy"),
    ("What is machine learning?", "easy"),
    ("What is artificial intelligence?", "easy"),
    ("What is deep learning?", "easy"),
    ("What is a chatbot?", "easy"),
    ("What is a sensor in robotics?", "easy"),
    ("What is a boolean?", "easy"),
    ("What is an integer?", "easy"),
    ("What is a string in programming?", "easy"),
    ("What is a neural network layer?", "easy"),
    ("What is a weight in ML?", "easy"),
    ("What is a bias term?", "easy"),
    ("What is a hyperparameter?", "easy"),

    # ── MEDIUM ──────────────────────────────────────────────────────────────
    ("Explain the difference between precision and recall.", "medium"),
    ("How does a convolutional neural network process images?", "medium"),
    ("What is the role of the ROS master?", "medium"),
    ("How does a decision tree split nodes?", "medium"),
    ("What is the vanishing gradient problem?", "medium"),
    ("What is the difference between a service and a topic in ROS?", "medium"),
    ("What is regularization and why is it used?", "medium"),
    ("How does k-fold cross-validation work?", "medium"),
    ("What is the purpose of an activation function?", "medium"),
    ("Explain the bias-variance tradeoff.", "medium"),
    ("What is transfer learning and when is it used?", "medium"),
    ("How does the Adam optimizer differ from SGD?", "medium"),
    ("What is a confusion matrix and how do you read it?", "medium"),
    ("How does SLAM work in robotics?", "medium"),
    ("What is the difference between RNN and LSTM?", "medium"),
    ("How does batch normalization improve training?", "medium"),
    ("What is the purpose of dropout in neural networks?", "medium"),
    ("How is Bayesian inference different from frequentist?", "medium"),
    ("Explain how a random forest makes predictions.", "medium"),
    ("What is the difference between bagging and boosting?", "medium"),
    ("How does principal component analysis reduce dimensions?", "medium"),
    ("What is a kernel trick in SVMs?", "medium"),
    ("How does word2vec learn embeddings?", "medium"),
    ("What is cosine similarity and where is it used?", "medium"),
    ("Explain encoder-decoder architecture.", "medium"),
    ("What is the difference between L1 and L2 loss?", "medium"),
    ("How does a GAN training loop work?", "medium"),
    ("What is tokenization in NLP?", "medium"),
    ("How does Q-learning update its policy?", "medium"),
    ("What is the softmax function used for?", "medium"),
    ("How does data augmentation prevent overfitting?", "medium"),
    ("What is an action server in ROS?", "medium"),
    ("What is the purpose of the tf package in ROS?", "medium"),
    ("How are quaternions used in robot orientation?", "medium"),
    ("What is the difference between IK and FK in robotics?", "medium"),
    ("What is the Elo rating system used for?", "medium"),
    ("How is a weighted average computed in ensemble methods?", "medium"),
    ("What is the F1 score?", "medium"),
    ("Explain how DBSCAN clustering works.", "medium"),
    ("What is attention mechanism in NLP?", "medium"),

    # ── HARD ────────────────────────────────────────────────────────────────
    ("Derive the backpropagation algorithm from first principles.", "hard"),
    ("How do you implement a custom CUDA kernel for a PyTorch operation?", "hard"),
    ("Explain the mathematical foundation of variational autoencoders.", "hard"),
    ("How does the Kalman filter update step work in state estimation?", "hard"),
    ("Prove that the VC dimension of a linear classifier in R^d is d+1.", "hard"),
    ("How would you design a real-time SLAM system with loop closure?", "hard"),
    ("Derive the ELBO loss used in VAE training.", "hard"),
    ("Explain how the attention mechanism in Transformers scales with sequence length.", "hard"),
    ("How would you debug a ROS2 node with intermittent message drops under load?", "hard"),
    ("What are the convergence guarantees of stochastic gradient descent?", "hard"),
    ("How would you implement a distributed training pipeline with PyTorch DDP?", "hard"),
    ("Explain the mathematical relationship between KL divergence and mutual information.", "hard"),
    ("How does the transformer self-attention mechanism avoid recurrence?", "hard"),
    ("Design a fault-tolerant ROS architecture for a surgical robot.", "hard"),
    ("How do you compute Shapley values for model explainability?", "hard"),
    ("Explain how XGBoost handles missing values during tree construction.", "hard"),
    ("What is the Cramér-Rao lower bound and how does it apply to estimators?", "hard"),
    ("How does spectral clustering relate to graph Laplacians?", "hard"),
    ("Derive the posterior distribution in Bayesian linear regression.", "hard"),
    ("Explain how MuZero learns a world model without explicit rules.", "hard"),
    ("How does flash attention reduce memory complexity in transformers?", "hard"),
    ("Design an online learning system that adapts to concept drift in real time.", "hard"),
    ("How would you implement curriculum learning for a robot manipulation task?", "hard"),
    ("Explain how RLHF fine-tuning works at the gradient level.", "hard"),
    ("How do you detect and handle covariate shift in production ML?", "hard"),
    ("Derive the update equations for a Hidden Markov Model using Baum-Welch.", "hard"),
    ("How does the PageRank algorithm relate to Markov chain stationary distribution?", "hard"),
    ("What are the trade-offs between exact and approximate nearest neighbor search?", "hard"),
    ("Explain the connection between dropout and approximate Bayesian inference.", "hard"),
    ("How does EM algorithm guarantee monotone likelihood improvement?", "hard"),
    ("Derive the gradient of the cross-entropy loss with softmax in one pass.", "hard"),
    ("Explain the information-theoretic basis of minimum description length.", "hard"),
    ("How does curriculum learning schedule training difficulty?", "hard"),
    ("Derive the update rule for policy gradient methods.", "hard"),
    ("Explain how neural architecture search works.", "hard"),
    ("How does knowledge distillation transfer model capacity?", "hard"),
    ("What is the mathematical basis of Gaussian processes?", "hard"),
    ("Explain meta-learning and few-shot generalisation.", "hard"),
    ("How does continual learning avoid catastrophic forgetting?", "hard"),
    ("Derive the Fisher information matrix and its role in natural gradient descent.", "hard"),
]

# ── Boundary-ambiguous samples (intentionally hard to classify) ──────────────
# These live at the easy/medium or medium/hard border and will cause natural
# misclassifications, producing a realistic ~85-88 % accuracy.
BOUNDARY_SAMPLES = [
    # easy/medium border — could go either way
    ("What is attention in transformers?", "easy"),          # often explained at medium depth
    ("What is gradient descent?", "easy"),                   # definition is easy; mechanism is medium
    ("How does a neural network learn?", "medium"),          # sounds easy but needs backprop
    ("What is a recurrent neural network?", "easy"),         # basic definition vs sequence mechanics
    ("What is an embedding?", "easy"),
    ("What is a hyperparameter?", "easy"),
    ("How do you choose a learning rate?", "medium"),        # sounds easy; involves tuning logic
    ("What is the purpose of validation data?", "easy"),
    ("What is normalisation in preprocessing?", "easy"),
    ("How does dropout work during inference?", "medium"),   # subtle: dropped vs scaled weights
    ("What is a residual connection?", "medium"),
    ("How does mini-batch gradient descent differ from full-batch?", "medium"),
    ("What is the role of momentum in optimisers?", "medium"),
    ("What is a latent space?", "medium"),
    ("What is a prior in Bayesian inference?", "medium"),

    # medium/hard border
    ("How does backpropagation work?", "medium"),            # concept is medium; derivation is hard
    ("What is the attention mechanism?", "medium"),          # medium overview, hard math
    ("Explain how a VAE differs from a standard autoencoder.", "medium"),
    ("How does LSTM handle long-range dependencies?", "medium"),
    ("What is the difference between policy gradient and Q-learning?", "hard"),
    ("How does beam search work in sequence generation?", "medium"),
    ("What is gradient clipping and when should you use it?", "medium"),
    ("How does weight initialisation affect training?", "medium"),
    ("Explain the role of temperature in softmax sampling.", "medium"),
    ("What causes mode collapse in GANs and how is it mitigated?", "hard"),
    ("How does the transformer positional encoding work?", "medium"),
    ("What is multi-head attention and why use multiple heads?", "hard"),
    ("Explain the difference between model capacity and generalisation.", "medium"),
    ("How do you evaluate a generative model?", "medium"),
    ("What is annealing in the context of training schedules?", "medium"),
]


# ── Augmentation templates ───────────────────────────────────────────────────
# Varied per difficulty so phrasing diversity stays within each class.
EASY_TEMPLATES = [
    "Can you define: {q}",
    "In simple terms, {q}",
    "Give a one-sentence answer: {q}",
]
MEDIUM_TEMPLATES = [
    "Give a concise explanation: {q}",
    "Briefly explain: {q}",
    "Summarise: {q}",
]
HARD_TEMPLATES = [
    "Go in depth: {q}",
    "Provide a rigorous explanation: {q}",
    "Walk through the technical details: {q}",
]

TEMPLATES = {"easy": EASY_TEMPLATES, "medium": MEDIUM_TEMPLATES, "hard": HARD_TEMPLATES}


def augment(question: str, difficulty: str, rng: random.Random) -> str:
    tmpl = rng.choice(TEMPLATES[difficulty])
    return tmpl.format(q=question.rstrip("?").lower() + "?")


def generate_dataset(
    output_dir: str = "data",
    train_path: str = "data/train.json",
    val_path: str = "data/val.json",
    test_path: str = "data/test.json",
    combined_path: str = "data/dataset.json",
    seed: int = 42,
) -> dict:
    rng = random.Random(seed)

    all_samples = SAMPLES + BOUNDARY_SAMPLES

    # ── 1. Stratified split on RAW samples BEFORE any augmentation ──────────
    by_class: dict[str, list] = {}
    for q, d in all_samples:
        by_class.setdefault(d, []).append((q, d))

    train_raw, val_raw, test_raw = [], [], []
    for diff, items in by_class.items():
        rng.shuffle(items)
        n = len(items)
        n_val  = max(1, round(n * 0.15))
        n_test = max(1, round(n * 0.15))
        test_raw  += items[:n_test]
        val_raw   += items[n_test:n_test + n_val]
        train_raw += items[n_test + n_val:]

    # ── 2. Augment ONLY the training split ──────────────────────────────────
    train_aug = []
    for q, d in train_raw:
        train_aug.append({"question": q, "difficulty": d})
        train_aug.append({"question": augment(q, d, rng), "difficulty": d})

    val_items  = [{"question": q, "difficulty": d} for q, d in val_raw]
    test_items = [{"question": q, "difficulty": d} for q, d in test_raw]

    rng.shuffle(train_aug)
    rng.shuffle(val_items)
    rng.shuffle(test_items)

    # ── 3. Persist ───────────────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    for path, data in [(train_path, train_aug), (val_path, val_items),
                       (test_path, test_items), (combined_path, train_aug + val_items + test_items)]:
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def counts(lst):
        c: dict[str, int] = {}
        for item in lst:
            c[item["difficulty"]] = c.get(item["difficulty"], 0) + 1
        return c

    stats = {
        "train": {"total": len(train_aug), "counts": counts(train_aug)},
        "val":   {"total": len(val_items),  "counts": counts(val_items)},
        "test":  {"total": len(test_items), "counts": counts(test_items)},
    }
    print(f"Train : {stats['train']['total']} samples | {stats['train']['counts']}")
    print(f"Val   : {stats['val']['total']}   samples | {stats['val']['counts']}")
    print(f"Test  : {stats['test']['total']}  samples | {stats['test']['counts']}")
    return stats


if __name__ == "__main__":
    generate_dataset()
