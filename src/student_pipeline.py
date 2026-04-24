import json
import os
from datetime import datetime

SESSIONS_FILE = "data/sessions.json"


def load_sessions() -> dict:
    if not os.path.exists(SESSIONS_FILE):
        return {}
    with open(SESSIONS_FILE) as f:
        return json.load(f)


def save_sessions(sessions: dict):
    os.makedirs("data", exist_ok=True)
    with open(SESSIONS_FILE, "w") as f:
        json.dump(sessions, f, indent=2)


def get_or_create_session(student_name: str) -> dict:
    sessions = load_sessions()
    if student_name not in sessions:
        sessions[student_name] = {
            "name":       student_name,
            "created_at": datetime.now().isoformat(),
            "responses":  [],
            "knowledge":  {},
            "score":      0,
            "total":      0,
        }
        save_sessions(sessions)
    return sessions[student_name]


def record_response(
    student_name:  str,
    question:      dict,
    chosen_index:  int,
    correct:       bool,
    p_know_after:  float,
):
    sessions = load_sessions()
    session  = sessions[student_name]

    session["responses"].append({
        "question_id":    question["id"],
        "question":       question["question"],
        "topic":          question["topic"],
        "difficulty":     question["difficulty"],
        "options":        question["options"],
        "correct_index":  question["correct_index"],
        "chosen_index":   chosen_index,
        "correct":        correct,
        "explanation":    question.get("explanation", ""),
        "p_know_after":   round(p_know_after, 4),
        "timestamp":      datetime.now().isoformat(),
    })

    session["total"] += 1
    if correct:
        session["score"] += 1

    session["knowledge"][question["topic"]] = round(p_know_after, 4)
    sessions[student_name] = session
    save_sessions(sessions)


def delete_session(student_name: str):
    sessions = load_sessions()
    sessions.pop(student_name, None)
    save_sessions(sessions)
