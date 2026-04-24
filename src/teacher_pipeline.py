"""
Teacher pipeline:
  PDF → text extraction → Groq LLM (MCQ generation) → PyTorch difficulty classifier → question bank
"""

import fitz  # PyMuPDF
import json
import os
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer

from src.train_model import (
    load_model,
    predict_with_confidence,
    DifficultyClassifier,
)

load_dotenv()

QUESTION_BANK_PATH = "data/question_bank.json"

# Lazy-loaded singletons
_groq_client: Groq | None = None
_embedder:    SentenceTransformer | None = None
_pt_model:    DifficultyClassifier | None = None


def _get_groq() -> Groq:
    global _groq_client
    if _groq_client is None:
        key = os.environ.get("GROQ_API_KEY", "")
        if not key:
            raise ValueError("GROQ_API_KEY not set. Add it to your .env file.")
        _groq_client = Groq(api_key=key)
    return _groq_client


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


def _get_pt_model() -> DifficultyClassifier:
    global _pt_model
    if _pt_model is None:
        _pt_model = load_model()
    return _pt_model


# ─── PDF extraction ──────────────────────────────────────────────────────────

def extract_text_from_pdf(pdf_path: str, max_chars: int = 8000) -> str:
    doc  = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
        if len(text) >= max_chars:
            break
    return text.strip()[:max_chars]


# ─── MCQ generation via Groq ─────────────────────────────────────────────────

def generate_mcqs_from_text(text: str, num_questions: int = 15) -> list[dict]:
    """
    Returns a list of dicts:
      { question, topic, options: [A,B,C,D], correct_index: 0-3,
        explanation }
    """
    prompt = f"""You are an expert quiz creator. Given the educational text below, generate exactly {num_questions} multiple-choice questions.

TEXT:
{text[:4000]}

STRICT RULES:
- Each question must have exactly 4 answer options (A, B, C, D).
- Exactly one option must be correct.
- Options must be plausible — no obviously wrong answers.
- Identify a short topic label (1-4 words) per question.
- Write a one-sentence explanation for the correct answer.
- Questions should vary in depth: some factual, some conceptual, some applied.
- Do NOT number the options — just provide the strings.

Respond ONLY with a valid JSON array. No explanation, no markdown, no code fences. Raw JSON only.

Format exactly:
[
  {{
    "question": "What is X?",
    "topic": "Topic Label",
    "options": ["Option A text", "Option B text", "Option C text", "Option D text"],
    "correct_index": 0,
    "explanation": "Option A is correct because..."
  }}
]"""

    client   = _get_groq()
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=4000,
    )

    raw = response.choices[0].message.content.strip()

    # Strip markdown fences if the model adds them
    if raw.startswith("```"):
        parts = raw.split("```")
        raw   = parts[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    questions = json.loads(raw)

    # Validate and clean each question
    cleaned = []
    for q in questions:
        if (
            isinstance(q.get("options"), list)
            and len(q["options"]) == 4
            and isinstance(q.get("correct_index"), int)
            and 0 <= q["correct_index"] <= 3
        ):
            cleaned.append(q)

    if not cleaned:
        raise ValueError("LLM returned no valid MCQ questions. Try again.")

    return cleaned


# ─── Full pipeline ────────────────────────────────────────────────────────────

def build_question_bank(pdf_path: str, num_questions: int = 15) -> list[dict]:
    print("Extracting text from PDF …")
    text = extract_text_from_pdf(pdf_path)

    print("Generating MCQs with Groq (Llama 3) …")
    questions = generate_mcqs_from_text(text, num_questions)

    print("Loading PyTorch difficulty classifier …")
    model    = _get_pt_model()
    embedder = _get_embedder()

    print("Predicting difficulty with PyTorch …")
    q_texts  = [q["question"] for q in questions]
    preds    = predict_with_confidence(q_texts, model, embedder)

    # Merge model difficulty predictions into questions
    bank = []
    existing = load_question_bank()
    start_id = max((q["id"] for q in existing), default=-1) + 1

    for i, (q, pred) in enumerate(zip(questions, preds)):
        bank.append({
            "id":              start_id + i,
            "question":        q["question"],
            "topic":           q.get("topic", "General"),
            "options":         q["options"],
            "correct_index":   q["correct_index"],
            "explanation":     q.get("explanation", ""),
            "difficulty":      pred["difficulty"],
            "confidence":      round(pred["confidence"], 3),
            "prob_easy":       round(pred["prob_easy"],   3),
            "prob_medium":     round(pred["prob_medium"], 3),
            "prob_hard":       round(pred["prob_hard"],   3),
            "times_shown":     0,
            "times_correct":   0,
        })

    # Append to existing bank
    full_bank = existing + bank
    save_question_bank(full_bank)

    print(f"\nQuestion bank: {len(full_bank)} total questions ({len(bank)} new)")
    return bank


# ─── Bank I/O ────────────────────────────────────────────────────────────────

def load_question_bank() -> list[dict]:
    if not os.path.exists(QUESTION_BANK_PATH):
        return []
    with open(QUESTION_BANK_PATH) as f:
        return json.load(f)


def save_question_bank(bank: list[dict]):
    os.makedirs("data", exist_ok=True)
    with open(QUESTION_BANK_PATH, "w") as f:
        json.dump(bank, f, indent=2)


def delete_question(question_id: int) -> bool:
    bank = load_question_bank()
    new_bank = [q for q in bank if q["id"] != question_id]
    if len(new_bank) == len(bank):
        return False
    save_question_bank(new_bank)
    return True


def clear_question_bank():
    save_question_bank([])


def update_question_stats(question_id: int, correct: bool):
    bank = load_question_bank()
    for q in bank:
        if q["id"] == question_id:
            q["times_shown"]   = q.get("times_shown",   0) + 1
            q["times_correct"] = q.get("times_correct", 0) + (1 if correct else 0)
            break
    save_question_bank(bank)
