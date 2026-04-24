"""
Bayesian Knowledge Tracing — implemented from scratch.

For each topic we maintain P(know) = probability the student knows it.

After CORRECT answer:
    P(know|correct) = P(know)*(1-p_slip) / [P(know)*(1-p_slip) + (1-P(know))*p_guess]

After WRONG answer:
    P(know|wrong) = P(know)*p_slip / [P(know)*p_slip + (1-P(know))*(1-p_guess)]

Learning transition (applied after every observation):
    P(know_next) = P(know_updated) + (1 - P(know_updated)) * p_learn
"""


class BKTModel:
    def __init__(
        self,
        p_init:  float = 0.3,
        p_learn: float = 0.1,
        p_guess: float = 0.2,
        p_slip:  float = 0.1,
    ):
        self.p_init  = p_init
        self.p_learn = p_learn
        self.p_guess = p_guess
        self.p_slip  = p_slip

    def update(self, p_know: float, correct: bool) -> float:
        if correct:
            numerator   = p_know * (1 - self.p_slip)
            denominator = numerator + (1 - p_know) * self.p_guess
        else:
            numerator   = p_know * self.p_slip
            denominator = numerator + (1 - p_know) * (1 - self.p_guess)

        p_know_obs = (numerator / denominator) if denominator > 1e-9 else p_know
        p_know_new = p_know_obs + (1 - p_know_obs) * self.p_learn
        return float(min(max(p_know_new, 0.0), 1.0))


class StudentKnowledgeTracker:
    def __init__(self, bkt: BKTModel = None):
        self.bkt       = bkt or BKTModel()
        self.knowledge: dict[str, float] = {}

    def get_p_know(self, topic: str) -> float:
        return self.knowledge.get(topic, self.bkt.p_init)

    def update(self, topic: str, correct: bool) -> float:
        p_new = self.bkt.update(self.get_p_know(topic), correct)
        self.knowledge[topic] = p_new
        return p_new

    def mastery_reached(self, topic: str, threshold: float = 0.85) -> bool:
        return self.get_p_know(topic) >= threshold

    def mastered_topics(self, threshold: float = 0.85) -> list[str]:
        return [t for t, p in self.knowledge.items() if p >= threshold]

    def weakest_topics(self, n: int = 3) -> list[tuple[str, float]]:
        return sorted(self.knowledge.items(), key=lambda x: x[1])[:n]

    def strongest_topics(self, n: int = 3) -> list[tuple[str, float]]:
        return sorted(self.knowledge.items(), key=lambda x: x[1], reverse=True)[:n]

    def to_dict(self) -> dict:
        return {"knowledge": self.knowledge}

    @classmethod
    def from_dict(cls, data: dict) -> "StudentKnowledgeTracker":
        t = cls()
        t.knowledge = data.get("knowledge", {})
        return t
