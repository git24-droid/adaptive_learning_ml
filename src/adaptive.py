"""
Adaptive question selection — maps BKT P(know) per topic to
the target difficulty, then picks the best unasked question.

Selection score (lower = better):
  diff_distance   : |question_difficulty_idx - target_difficulty_idx|
  confidence_bonus: prefer high-confidence model predictions
  freshness       : prefer less-shown questions
"""
import random

from src.bkt import StudentKnowledgeTracker

DIFFICULTY_ORDER = ["easy", "medium", "hard"]


def _target_difficulty(p_know: float) -> str:
    if p_know < 0.40:
        return "easy"
    if p_know < 0.75:
        return "medium"
    return "hard"


def select_next_question(
    question_bank:  list[dict],
    tracker:        StudentKnowledgeTracker,
    asked_ids:      set[int],
    strategy:       str = "adaptive",
) -> dict | None:
    available = [q for q in question_bank if q["id"] not in asked_ids]
    if not available:
        return None

    if strategy == "random":
        return random.choice(available)

    scored = []
    for q in available:
        p_know = tracker.get_p_know(q["topic"])
        target = _target_difficulty(p_know)

        diff_distance   = abs(
            DIFFICULTY_ORDER.index(q["difficulty"]) -
            DIFFICULTY_ORDER.index(target)
        )
        # Prefer high-confidence model predictions (lower confidence = less reliable)
        confidence_penalty = 1.0 - q.get("confidence", 0.5)
        freshness          = q.get("times_shown", 0)

        scored.append((diff_distance, confidence_penalty, freshness, random.random(), q))

    scored.sort(key=lambda x: (x[0], x[1], x[2], x[3]))
    return scored[0][4]


def get_performance_summary(
    tracker:   StudentKnowledgeTracker,
    responses: list[dict],
) -> dict:
    if not responses:
        return {}

    total    = len(responses)
    correct  = sum(1 for r in responses if r["correct"])
    accuracy = correct / total if total > 0 else 0

    # Per-topic stats
    topic_stats: dict[str, dict] = {}
    for r in responses:
        t = r["topic"]
        if t not in topic_stats:
            topic_stats[t] = {"correct": 0, "total": 0}
        topic_stats[t]["total"]  += 1
        topic_stats[t]["correct"] += int(r["correct"])

    for t in topic_stats:
        topic_stats[t]["p_know"]   = round(tracker.get_p_know(t), 3)
        topic_stats[t]["accuracy"] = round(
            topic_stats[t]["correct"] / topic_stats[t]["total"], 3
        )

    # Per-difficulty stats
    diff_stats: dict[str, dict] = {"easy": {"c":0,"t":0}, "medium": {"c":0,"t":0}, "hard": {"c":0,"t":0}}
    for r in responses:
        d = r.get("difficulty", "easy")
        diff_stats[d]["t"] += 1
        diff_stats[d]["c"] += int(r["correct"])

    weakest   = sorted(topic_stats.items(), key=lambda x: x[1]["p_know"])[:3]
    strongest = sorted(topic_stats.items(), key=lambda x: x[1]["p_know"], reverse=True)[:3]

    return {
        "total_questions":  total,
        "correct":          correct,
        "accuracy":         round(accuracy * 100, 1),
        "topic_stats":      topic_stats,
        "diff_stats":       diff_stats,
        "weakest_topics":   [(t, s["p_know"]) for t, s in weakest],
        "strongest_topics": [(t, s["p_know"]) for t, s in strongest],
        "mastered_topics":  tracker.mastered_topics(),
    }
