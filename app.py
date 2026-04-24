"""
Adaptive Learning System — Streamlit App
Teacher uploads PDF → LLM generates MCQs → PyTorch classifies difficulty
Student takes adaptive MCQ quiz → BKT tracks knowledge → Analytics dashboard
"""

import json
import os
import tempfile

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit_extras.colored_header import colored_header  # optional, graceful fallback

from src.bkt import BKTModel, StudentKnowledgeTracker
from src.student_pipeline import (
    delete_session,
    get_or_create_session,
    load_sessions,
    record_response,
    save_sessions,
)
from src.adaptive import select_next_question, get_performance_summary
from src.teacher_pipeline import (
    build_question_bank,
    load_question_bank,
    save_question_bank,
    delete_question,
    clear_question_bank,
    update_question_stats,
)
from src.train_model import load_training_stats

# ─── Page config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AdaptML — Adaptive Learning System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Global CSS ──────────────────────────────────────────────────────────────

# ─── Global CSS Upgrade ───────────────────────────────────────────────────────

st.markdown("""
<style>
/* hide default streamlit menu & footer */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

/* Modern Gradient Headers */
h1, h2, h3 {
    background: -webkit-linear-gradient(45deg, #ff6b6b, #4facfe, #00f2fe);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
}

/* difficulty badges with glassmorphism */
.badge-easy   { background: rgba(40, 167, 69, 0.15); color: #28a745; padding: 4px 12px; border-radius: 12px; font-size: 12px; font-weight: bold; border: 1px solid rgba(40, 167, 69, 0.3); }
.badge-medium { background: rgba(255, 193, 7, 0.15); color: #ffc107; padding: 4px 12px; border-radius: 12px; font-size: 12px; font-weight: bold; border: 1px solid rgba(255, 193, 7, 0.3); }
.badge-hard   { background: rgba(220, 53, 69, 0.15); color: #dc3545; padding: 4px 12px; border-radius: 12px; font-size: 12px; font-weight: bold; border: 1px solid rgba(220, 53, 69, 0.3); }

/* option button override with hover animations */
div[data-testid="stButton"] > button {
    width: 100%;
    text-align: left;
    padding: 12px 16px;
    border-radius: 10px;
    font-size: 14px;
    transition: all 0.3s ease;
    border: 1px solid rgba(150, 150, 150, 0.2);
}
div[data-testid="stButton"] > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 15px rgba(0,0,0,0.1);
    border-color: #4facfe;
}

/* correct / wrong feedback bars */
.feedback-correct { background: rgba(40, 167, 69, 0.05); border-left: 5px solid #28a745; padding:12px 16px; border-radius:4px; margin-top:8px; }
.feedback-wrong   { background: rgba(220, 53, 69, 0.05); border-left: 5px solid #dc3545; padding:12px 16px; border-radius:4px; margin-top:8px; }

/* Floating style for Streamlit's native metrics */
div[data-testid="metric-container"] {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(150, 150, 150, 0.15);
    padding: 15px;
    border-radius: 12px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    transition: transform 0.2s ease;
}
div[data-testid="metric-container"]:hover {
    transform: translateY(-3px);
    border-color: rgba(79, 172, 254, 0.5);
}
</style>
""", unsafe_allow_html=True)


# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🧠 AdaptML")
    st.markdown("*ML-powered adaptive learning*")
    st.divider()

    mode = st.radio(
        "Navigation",
        ["🎓 Teacher Dashboard", "📚 Student Quiz", "📊 Student Analytics", "🔬 ML Model Stats"],
        label_visibility="collapsed",
    )

    st.divider()
    bank = load_question_bank()
    st.metric("Questions in bank", len(bank))

    sessions = load_sessions()
    st.metric("Active students",   len(sessions))

    stats = load_training_stats()
    if stats:
        st.metric("Model accuracy", f"{round(stats.get('accuracy',0)*100,1)}%")

    st.divider()
    st.caption("Built with PyTorch · BKT · Groq · Streamlit")


# ══════════════════════════════════════════════════════════════════════════════
# TEACHER DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════

if mode == "🎓 Teacher Dashboard":
    st.title("🎓 Teacher Dashboard")
    st.markdown("Upload your unit PDF → AI generates MCQs → PyTorch classifies difficulty. Review, delete, and publish.")

    # ── Upload & generate ────────────────────────────────────────────────────
    with st.expander("➕ Generate new questions from PDF", expanded=True):
        col1, col2 = st.columns([3, 1])
        with col1:
            uploaded = st.file_uploader("Upload unit PDF", type=["pdf"], key="teacher_upload")
        with col2:
            num_q = st.number_input("Questions to generate", min_value=5, max_value=40, value=15, step=5)

        if uploaded and st.button("🚀 Generate Question Bank", type="primary", use_container_width=True):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded.read())
                tmp_path = tmp.name

            progress = st.progress(0, text="Extracting text from PDF …")
            try:
                progress.progress(15, "Calling Groq LLM (Llama 3) for MCQ generation …")
                new_qs = build_question_bank(tmp_path, num_questions=num_q)
                progress.progress(90, "Finalising question bank …")

                counts = {"easy": 0, "medium": 0, "hard": 0}
                for q in new_qs:
                    counts[q["difficulty"]] += 1
                progress.progress(100, "Done!")

                st.success(f"✅ {len(new_qs)} questions generated and classified by PyTorch!")
                c1, c2, c3 = st.columns(3)
                c1.metric("🟢 Easy",   counts["easy"])
                c2.metric("🟡 Medium", counts["medium"])
                c3.metric("🔴 Hard",   counts["hard"])

            except Exception as e:
                st.error(f"Error: {e}")
            finally:
                os.unlink(tmp_path)

    st.divider()

    # ── Question bank manager ─────────────────────────────────────────────────
    bank = load_question_bank()

    if not bank:
        st.info("📭 No questions in the bank yet. Upload a PDF to get started.")
        st.stop()

    # Filter controls
    col_f1, col_f2, col_f3 = st.columns([2, 2, 1])
    with col_f1:
        topics = ["All"] + sorted(set(q["topic"] for q in bank))
        sel_topic = st.selectbox("Filter by topic", topics, key="t_filter")
    with col_f2:
        diffs  = ["All", "easy", "medium", "hard"]
        sel_diff = st.selectbox("Filter by difficulty", diffs, key="d_filter")
    with col_f3:
        if st.button("🗑 Clear All Questions", use_container_width=True):
            clear_question_bank()
            st.rerun()

    filtered = bank
    if sel_topic != "All": filtered = [q for q in filtered if q["topic"] == sel_topic]
    if sel_diff  != "All": filtered = [q for q in filtered if q["difficulty"] == sel_diff]

    st.markdown(f"### Question Bank — {len(filtered)} / {len(bank)} questions")

    diff_counts = {"easy": 0, "medium": 0, "hard": 0}
    for q in bank:
        diff_counts[q["difficulty"]] += 1

    c1, c2, c3 = st.columns(3)
    c1.metric("🟢 Easy",   diff_counts["easy"])
    c2.metric("🟡 Medium", diff_counts["medium"])
    c3.metric("🔴 Hard",   diff_counts["hard"])

    st.divider()

    # Per-question cards with delete
    for q in filtered:
        with st.container():
            col_q, col_del = st.columns([9, 1])
            with col_q:
                diff_color = {"easy": "🟢", "medium": "🟡", "hard": "🔴"}
                times_shown   = q.get("times_shown",   0)
                times_correct = q.get("times_correct", 0)
                acc_str = f"{round(times_correct/times_shown*100)}%" if times_shown > 0 else "—"

                st.markdown(
                    f"**{diff_color[q['difficulty']]} [{q['difficulty'].upper()}]** "
                    f"*{q['topic']}* "
                    f"<span style='color:#6c757d;font-size:12px'>| Confidence: {round(q.get('confidence',0)*100)}% "
                    f"| Shown: {times_shown} | Correct: {acc_str}</span>",
                    unsafe_allow_html=True,
                )
                st.markdown(f"**Q:** {q['question']}")

                letters = ["A", "B", "C", "D"]
                for i, opt in enumerate(q["options"]):
                    marker = "✅ " if i == q["correct_index"] else "   "
                    st.markdown(f"&nbsp;&nbsp;&nbsp;{marker}**{letters[i]}.** {opt}", unsafe_allow_html=True)

                if q.get("explanation"):
                    st.caption(f"💡 {q['explanation']}")

            with col_del:
                if st.button("🗑", key=f"del_{q['id']}", help="Delete this question"):
                    delete_question(q["id"])
                    st.rerun()

            st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# STUDENT QUIZ
# ══════════════════════════════════════════════════════════════════════════════

elif mode == "📚 Student Quiz":
    st.title("📚 Student Quiz")

    # Init session state keys
    for key, default in [
        ("student_name",  None),
        ("asked_ids",     None),
        ("current_q",     None),
        ("tracker",       None),
        ("last_result",   None),
        ("show_answer",   False),
        ("chosen_idx",    None),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    if st.session_state.asked_ids is None:
        st.session_state.asked_ids = set()

    # ── Login ────────────────────────────────────────────────────────────────
    if not st.session_state.student_name:
        st.markdown("### Enter your name to begin")
        col1, col2 = st.columns([3, 1])
        with col1:
            name = st.text_input("Your name", placeholder="e.g. Alice")
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            start = st.button("Start Quiz ▶", type="primary", use_container_width=True)

        if start and name.strip():
            bank = load_question_bank()
            if not bank:
                st.warning("⚠️ No question bank found. Ask your teacher to upload a PDF first.")
                st.stop()

            st.session_state.student_name = name.strip()
            session = get_or_create_session(st.session_state.student_name)
            tracker = StudentKnowledgeTracker(BKTModel())
            tracker.knowledge = dict(session.get("knowledge", {}))
            st.session_state.tracker   = tracker
            st.session_state.asked_ids = set(r["question_id"] for r in session.get("responses", []))
            st.rerun()

        st.stop()

    # ── Quiz interface ───────────────────────────────────────────────────────
    bank    = load_question_bank()
    tracker = st.session_state.tracker
    session = get_or_create_session(st.session_state.student_name)

    if not bank:
        st.warning("⚠️ No question bank found. Ask your teacher to upload a PDF first.")
        st.stop()

    # Stats bar
    total = session["total"]
    score = session["score"]
    acc   = round(score / total * 100, 1) if total > 0 else 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Student",  st.session_state.student_name)
    c2.metric("Answered", f"{total}")
    c3.metric("Score",    f"{score}/{total}")
    c4.metric("Accuracy", f"{acc}%")

    st.divider()

    # Last result feedback
    if st.session_state.last_result:
        r = st.session_state.last_result
        if r["correct"]:
            st.success(
                f"✅ **Correct!** — P(know) for **{r['topic']}** updated: "
                f"{r['p_know_before']:.2f} → **{r['p_know_after']:.2f}**"
            )
        else:
            letters = ["A","B","C","D"]
            st.error(
                f"❌ **Incorrect.** Correct answer: **{letters[r['correct_idx']]}. {r['correct_text']}** — "
                f"P(know) for **{r['topic']}**: {r['p_know_before']:.2f} → **{r['p_know_after']:.2f}**"
            )
        if r.get("explanation"):
            st.info(f"💡 {r['explanation']}")
        st.session_state.last_result = None

    # Select next question
    if st.session_state.current_q is None:
        q = select_next_question(bank, tracker, st.session_state.asked_ids)
        if q is None:
            st.balloons()
            st.success("🎉 You've answered all available questions! Check your Analytics.")
            if st.button("🔄 Reset and retake"):
                delete_session(st.session_state.student_name)
                for k in ["student_name","asked_ids","current_q","tracker","last_result","show_answer","chosen_idx"]:
                    st.session_state[k] = None
                st.session_state.asked_ids = set()
                st.rerun()
            st.stop()
        st.session_state.current_q  = q
        st.session_state.show_answer = False
        st.session_state.chosen_idx  = None

    q = st.session_state.current_q

    # Question card
    p_know  = tracker.get_p_know(q["topic"])
    target  = "easy" if p_know < 0.4 else ("medium" if p_know < 0.75 else "hard")
    letters = ["A", "B", "C", "D"]

    diff_badge = {"easy": "🟢 EASY", "medium": "🟡 MEDIUM", "hard": "🔴 HARD"}

    st.markdown(
        f"<div style='display:flex;align-items:center;gap:12px;margin-bottom:8px'>"
        f"<span style='font-size:13px;color:#6c757d'>Topic: <b>{q['topic']}</b></span>"
        f"<span style='font-size:12px;color:#6c757d'>| Difficulty: <b>{diff_badge[q['difficulty']]}</b></span>"
        f"<span style='font-size:12px;color:#6c757d'>| P(know): <b>{p_know:.2f}</b></span>"
        f"<span style='font-size:12px;color:#6c757d'>| Q {total+1}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    st.markdown(f"### {q['question']}")
    st.markdown("")

    answered = st.session_state.show_answer

    for i, opt in enumerate(q["options"]):
        if answered:
            if i == q["correct_index"]:
                st.markdown(
                    f"<div style='padding:12px 16px;border-radius:8px;background:#d4edda;color:#155724;border:2px solid #28a745;margin-bottom:6px'>"
                    f"✅ <b>{letters[i]}.</b> {opt}</div>",
                    unsafe_allow_html=True,
                )
            elif i == st.session_state.chosen_idx:
                st.markdown(
                    f"<div style='padding:12px 16px;border-radius:8px;background:#f8d7da;color:#721c24;border:2px solid #dc3545;margin-bottom:6px'>"
                    f"❌ <b>{letters[i]}.</b> {opt}</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"<div style='padding:12px 16px;border-radius:8px;background:#f8f9fa;color:#495057;margin-bottom:6px'>"
                    f"&nbsp;&nbsp;&nbsp;<b>{letters[i]}.</b> {opt}</div>",
                    unsafe_allow_html=True,
                )
        else:
            if st.button(f"{letters[i]}.  {opt}", key=f"opt_{i}", use_container_width=True):
                chosen   = i
                correct  = (i == q["correct_index"])

                p_before = tracker.get_p_know(q["topic"])
                p_after  = tracker.update(q["topic"], correct)

                record_response(st.session_state.student_name, q, chosen, correct, p_after)
                update_question_stats(q["id"], correct)

                st.session_state.asked_ids.add(q["id"])
                st.session_state.chosen_idx  = chosen
                st.session_state.show_answer = True
                st.session_state.last_result = {
                    "correct":       correct,
                    "topic":         q["topic"],
                    "correct_idx":   q["correct_index"],
                    "correct_text":  q["options"][q["correct_index"]],
                    "p_know_before": p_before,
                    "p_know_after":  p_after,
                    "explanation":   q.get("explanation", ""),
                }
                st.rerun()

    if answered:
        st.markdown("")
        if st.button("➡ Next Question", type="primary", use_container_width=True):
            st.session_state.current_q   = None
            st.session_state.show_answer = False
            st.session_state.chosen_idx  = None
            st.rerun()

    st.divider()

    # Live BKT chart
    if tracker.knowledge:
        st.markdown("#### 📈 Live Knowledge State (Bayesian Knowledge Tracing)")
        kdf = pd.DataFrame(
            {"Topic": list(tracker.knowledge.keys()), "P(know)": list(tracker.knowledge.values())}
        ).sort_values("P(know)", ascending=True)

        fig = px.bar(
            kdf, x="P(know)", y="Topic", orientation="h",
            color="P(know)", color_continuous_scale=["#dc3545","#ffc107","#28a745"],
            range_color=[0, 1], text=kdf["P(know)"].apply(lambda v: f"{v:.2f}"),
        )
        fig.update_layout(
            margin=dict(l=0, r=0, t=10, b=10),
            height=max(200, len(tracker.knowledge) * 40 + 60),
            coloraxis_showscale=False,
            xaxis=dict(range=[0, 1]),
        )
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Updated live after every answer using Bayes theorem. Mastery threshold = 0.85")

    # Reset button
    with st.expander("⚙️ Session controls"):
        if st.button("🔄 Reset my session (start over)", use_container_width=True):
            delete_session(st.session_state.student_name)
            for k in ["student_name","asked_ids","current_q","tracker","last_result","show_answer","chosen_idx"]:
                st.session_state[k] = None
            st.session_state.asked_ids = set()
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# STUDENT ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════

elif mode == "📊 Student Analytics":
    st.title("📊 Student Analytics")

    sessions = load_sessions()
    if not sessions:
        st.info("No student sessions recorded yet.")
        st.stop()

    selected = st.selectbox("Select student", sorted(sessions.keys()))
    session  = sessions[selected]
    responses = session.get("responses", [])

    if not responses:
        st.info(f"No responses recorded yet for {selected}.")
        st.stop()

    # Rebuild tracker from saved knowledge
    tracker = StudentKnowledgeTracker(BKTModel())
    tracker.knowledge = dict(session.get("knowledge", {}))
    summary = get_performance_summary(tracker, responses)

    # ── Top metrics ──────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Questions", session["total"])
    c2.metric("Correct",         session["score"])
    c3.metric("Accuracy",        f"{round(session['score']/session['total']*100,1)}%")
    c4.metric("Mastered Topics", len(summary.get("mastered_topics", [])))
    streak = 0
    for r in reversed(responses):
        if r["correct"]: streak += 1
        else:            break
    c5.metric("Current Streak", streak)

    st.divider()

    # ── Accuracy over time ───────────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Accuracy over time")
        cum_acc = []
        for i, r in enumerate(responses):
            correct_so_far = sum(1 for x in responses[:i+1] if x["correct"])
            cum_acc.append(round(correct_so_far / (i+1) * 100, 1))

        fig_acc = go.Figure()
        fig_acc.add_trace(go.Scatter(
            y=cum_acc, x=list(range(1, len(cum_acc)+1)),
            mode="lines+markers", line=dict(color="#4c78a8", width=2),
            name="Cumulative accuracy",
        ))
        fig_acc.update_layout(
            xaxis_title="Question #", yaxis_title="Accuracy %",
            yaxis=dict(range=[0, 105]),
            margin=dict(l=0, r=0, t=10, b=0), height=260,
        )
        st.plotly_chart(fig_acc, use_container_width=True)

    with col2:
        st.markdown("#### Difficulty breakdown")
        ds = summary.get("diff_stats", {})
        diff_df = pd.DataFrame({
            "Difficulty": ["Easy", "Medium", "Hard"],
            "Answered":   [ds.get("easy",{}).get("t",0), ds.get("medium",{}).get("t",0), ds.get("hard",{}).get("t",0)],
            "Correct":    [ds.get("easy",{}).get("c",0), ds.get("medium",{}).get("c",0), ds.get("hard",{}).get("c",0)],
        })
        fig_diff = px.bar(
            diff_df.melt(id_vars="Difficulty", var_name="Type", value_name="Count"),
            x="Difficulty", y="Count", color="Type", barmode="group",
            color_discrete_map={"Answered": "#4c78a8", "Correct": "#54a24b"},
        )
        fig_diff.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=260)
        st.plotly_chart(fig_diff, use_container_width=True)

    # ── BKT knowledge per topic ───────────────────────────────────────────────
    if tracker.knowledge:
        st.markdown("#### Knowledge State per Topic (BKT)")

        kdf = pd.DataFrame(
            {"Topic": list(tracker.knowledge.keys()), "P(know)": list(tracker.knowledge.values())}
        ).sort_values("P(know)", ascending=True)

        fig_bkt = px.bar(
            kdf, x="P(know)", y="Topic", orientation="h",
            color="P(know)", color_continuous_scale=["#dc3545","#ffc107","#28a745"],
            range_color=[0, 1], text=kdf["P(know)"].apply(lambda v: f"{v:.2f}"),
        )
        fig_bkt.add_vline(x=0.85, line_dash="dash", line_color="green",
                          annotation_text="Mastery (0.85)", annotation_position="top right")
        fig_bkt.update_layout(
            margin=dict(l=0, r=0, t=10, b=0),
            height=max(200, len(tracker.knowledge) * 40 + 80),
            coloraxis_showscale=False, xaxis=dict(range=[0, 1.05]),
        )
        fig_bkt.update_traces(textposition="outside")
        st.plotly_chart(fig_bkt, use_container_width=True)

    # ── P(know) over time per topic ───────────────────────────────────────────
    st.markdown("#### P(know) trajectory per topic")
    topic_timeline: dict[str, list] = {}
    for r in responses:
        topic_timeline.setdefault(r["topic"], []).append(r["p_know_after"])

    if len(topic_timeline) <= 6:
        fig_traj = go.Figure()
        for topic, vals in topic_timeline.items():
            fig_traj.add_trace(go.Scatter(
                y=vals, x=list(range(1, len(vals)+1)),
                mode="lines+markers", name=topic,
            ))
        fig_traj.add_hline(y=0.85, line_dash="dash", line_color="green",
                            annotation_text="Mastery threshold")
        fig_traj.update_layout(
            yaxis_title="P(know)", xaxis_title="Attempts on topic",
            yaxis=dict(range=[0, 1.05]),
            margin=dict(l=0, r=0, t=10, b=0), height=300,
        )
        st.plotly_chart(fig_traj, use_container_width=True)
    else:
        sel_topic = st.selectbox("Select topic", sorted(topic_timeline.keys()))
        vals = topic_timeline[sel_topic]
        fig_t = go.Figure(go.Scatter(
            y=vals, x=list(range(1, len(vals)+1)),
            mode="lines+markers", line=dict(color="#4c78a8"),
        ))
        fig_t.add_hline(y=0.85, line_dash="dash", line_color="green")
        fig_t.update_layout(yaxis=dict(range=[0,1.05]), margin=dict(l=0,r=0,t=10,b=0), height=280)
        st.plotly_chart(fig_t, use_container_width=True)

    # ── Full response history ─────────────────────────────────────────────────
    st.markdown("#### Full response history")
    hist_df = pd.DataFrame([{
        "#":          i + 1,
        "Question":   r["question"][:70] + ("…" if len(r["question"]) > 70 else ""),
        "Topic":      r["topic"],
        "Difficulty": r["difficulty"],
        "Correct":    "✅" if r["correct"] else "❌",
        "P(know)":    r["p_know_after"],
        "Timestamp":  r["timestamp"][:19].replace("T", " "),
    } for i, r in enumerate(responses)])

    st.dataframe(hist_df, use_container_width=True, hide_index=True)

    with st.expander("⚙️ Manage session"):
        if st.button(f"🗑 Delete session for {selected}", use_container_width=True):
            delete_session(selected)
            st.success("Session deleted.")
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# ML MODEL STATS
# ══════════════════════════════════════════════════════════════════════════════

elif mode == "🔬 ML Model Stats":
    st.title("🔬 ML Model Stats")
    st.markdown("Everything about the underlying machine learning powering this system.")

    # ── Train / retrain ───────────────────────────────────────────────────────
    with st.expander("🔁 Train / Retrain difficulty classifier"):
        st.markdown(
            "The PyTorch MLP is trained on a curated dataset of labelled questions. "
            "Retrain after changing the dataset or to refresh model weights."
        )
        if st.button("🚀 Train model now", type="primary"):
            from src.generate_dataset import generate_dataset
            from src.train_model import train

            with st.spinner("Generating dataset and training PyTorch model … (1-3 min)"):
                try:
                    generate_dataset()
                    stats_new = train()
                    st.success(f"✅ Training complete! Accuracy: {round(stats_new['accuracy']*100, 1)}%")
                    st.rerun()
                except Exception as e:
                    st.error(f"Training error: {e}")

    stats = load_training_stats()

    if not stats:
        st.warning("No training stats found. Train the model first using the button above.")
        st.stop()

    # ── Architecture overview ─────────────────────────────────────────────────
    st.markdown("### Model Architecture")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
| Component | Details |
|-----------|---------|
| Input dim | 384 (all-MiniLM-L6-v2) |
| Layer 1   | Linear(384→256) + BatchNorm + ReLU + Dropout(0.3) |
| Layer 2   | Linear(256→128) + BatchNorm + ReLU + Dropout(0.3) |
| Layer 3   | Linear(128→64) + ReLU |
| Output    | Linear(64→3) — softmax over {easy, medium, hard} |
| Loss      | CrossEntropyLoss |
| Optimizer | Adam (lr=1e-3, weight_decay=1e-4) |
| Scheduler | ReduceLROnPlateau (patience=5) |
| Epochs    | 60 (best checkpoint saved) |
""")
    with col2:
        c1, c2, c3, c4 = st.columns(2)
        c1.metric("Accuracy",       f"{round(stats['accuracy']*100, 1)}%")
        c2.metric("Train samples",  stats.get("train_samples", "—"))
        c3.metric("Test samples",   stats.get("test_samples",  "—"))
        c4.metric("Epochs trained", stats.get("epochs", "—"))

    st.divider()

    # ── Confusion matrix ─────────────────────────────────────────────────────
    st.markdown("### Confusion Matrix")
    cm = stats.get("confusion_matrix", [])
    if cm:
        cm_arr = np.array(cm)
        labels = ["Easy", "Medium", "Hard"]

        fig_cm = px.imshow(
            cm_arr,
            labels=dict(x="Predicted", y="True", color="Count"),
            x=labels, y=labels,
            color_continuous_scale="Blues",
            text_auto=True,
        )
        fig_cm.update_layout(margin=dict(l=0, r=0, t=30, b=0), height=350)
        st.plotly_chart(fig_cm, use_container_width=True)

    # ── Per-class report ──────────────────────────────────────────────────────
    st.markdown("### Per-class Classification Report")
    report = stats.get("report", {})
    if report:
        rows = []
        for cls in ["easy", "medium", "hard"]:
            r = report.get(cls, {})
            rows.append({
                "Class":     cls.capitalize(),
                "Precision": round(r.get("precision", 0), 3),
                "Recall":    round(r.get("recall",    0), 3),
                "F1-Score":  round(r.get("f1-score",  0), 3),
                "Support":   int(r.get("support",     0)),
            })
        rep_df = pd.DataFrame(rows)
        st.dataframe(rep_df, hide_index=True, use_container_width=True)

        # Grouped bar
        fig_rep = px.bar(
            rep_df.melt(id_vars="Class", value_vars=["Precision","Recall","F1-Score"],
                        var_name="Metric", value_name="Score"),
            x="Class", y="Score", color="Metric", barmode="group",
            color_discrete_sequence=["#4c78a8","#54a24b","#f58518"],
        )
        fig_rep.update_layout(yaxis=dict(range=[0, 1.05]), margin=dict(l=0,r=0,t=10,b=0), height=280)
        st.plotly_chart(fig_rep, use_container_width=True)

    # ── Loss curves ───────────────────────────────────────────────────────────
    st.markdown("### Training & Validation Loss")
    train_losses = stats.get("train_losses", [])
    val_losses   = stats.get("val_losses",   [])
    if train_losses:
        epochs = list(range(1, len(train_losses) + 1))
        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(x=epochs, y=train_losses, name="Train loss",  line=dict(color="#4c78a8")))
        fig_loss.add_trace(go.Scatter(x=epochs, y=val_losses,   name="Val loss",    line=dict(color="#f58518")))
        fig_loss.update_layout(
            xaxis_title="Epoch", yaxis_title="Loss",
            margin=dict(l=0, r=0, t=10, b=0), height=300,
        )
        st.plotly_chart(fig_loss, use_container_width=True)

    st.divider()

    # ── BKT parameters ────────────────────────────────────────────────────────
    st.markdown("### Bayesian Knowledge Tracing (BKT) Parameters")
    bkt_df = pd.DataFrame([
        {"Parameter": "P(init)",              "Value": 0.30, "Meaning": "Prior: P(student already knows topic)"},
        {"Parameter": "P(learn)",             "Value": 0.10, "Meaning": "P(knowledge gained after each question)"},
        {"Parameter": "P(guess)",             "Value": 0.20, "Meaning": "P(correct answer despite not knowing)"},
        {"Parameter": "P(slip)",              "Value": 0.10, "Meaning": "P(wrong answer despite knowing)"},
        {"Parameter": "Mastery threshold",    "Value": 0.85, "Meaning": "P(know) required to mark topic as mastered"},
    ])
    st.dataframe(bkt_df, hide_index=True, use_container_width=True)

    st.markdown("#### BKT update equations")
    st.latex(r"P(\text{know} \mid \text{correct}) = \frac{P(\text{know})(1 - P_\text{slip})}{P(\text{know})(1-P_\text{slip}) + (1-P(\text{know}))P_\text{guess}}")
    st.latex(r"P(\text{know} \mid \text{wrong}) = \frac{P(\text{know}) \cdot P_\text{slip}}{P(\text{know}) \cdot P_\text{slip} + (1-P(\text{know}))(1-P_\text{guess})}")
    st.latex(r"P(\text{know}_\text{next}) = P(\text{know}_\text{updated}) + (1 - P(\text{know}_\text{updated})) \cdot P_\text{learn}")

    st.divider()

    # ── Adaptive selection algorithm ─────────────────────────────────────────
    st.markdown("### Adaptive Question Selection Algorithm")
    st.markdown("""
| P(know) range | Target difficulty | Rationale |
|---------------|------------------|-----------|
| < 0.40        | Easy             | Student is struggling — reinforce fundamentals |
| 0.40 – 0.75   | Medium           | Student is learning — challenge but don't overwhelm |
| ≥ 0.75        | Hard             | Student is proficient — push towards mastery |

**Selection score** (lower = better):  
`score = |difficulty_idx − target_idx| + (1 − model_confidence) + times_shown_bonus + random_tiebreak`

This means: correct difficulty match → high-confidence ML prediction → freshest question.
""")

    # ── Live question bank ML stats ───────────────────────────────────────────
    bank = load_question_bank()
    if bank:
        st.divider()
        st.markdown("### Question Bank ML Analysis")

        bank_df = pd.DataFrame(bank)

        col1, col2 = st.columns(2)

        with col1:
            fig_dist = px.histogram(
                bank_df, x="difficulty", color="difficulty",
                color_discrete_map={"easy":"#28a745","medium":"#ffc107","hard":"#dc3545"},
                title="Difficulty distribution",
                category_orders={"difficulty":["easy","medium","hard"]},
            )
            fig_dist.update_layout(showlegend=False, margin=dict(l=0,r=0,t=40,b=0), height=260)
            st.plotly_chart(fig_dist, use_container_width=True)

        with col2:
            fig_conf = px.box(
                bank_df, x="difficulty", y="confidence", color="difficulty",
                color_discrete_map={"easy":"#28a745","medium":"#ffc107","hard":"#dc3545"},
                title="Model confidence by difficulty",
                category_orders={"difficulty":["easy","medium","hard"]},
            )
            fig_conf.update_layout(showlegend=False, margin=dict(l=0,r=0,t=40,b=0), height=260)
            st.plotly_chart(fig_conf, use_container_width=True)

        # Topic × difficulty heatmap
        if len(bank_df["topic"].unique()) <= 20:
            pivot = bank_df.groupby(["topic","difficulty"]).size().unstack(fill_value=0)
            for col in ["easy","medium","hard"]:
                if col not in pivot.columns:
                    pivot[col] = 0
            pivot = pivot[["easy","medium","hard"]]

            fig_heat = px.imshow(
                pivot.values, x=["Easy","Medium","Hard"], y=pivot.index.tolist(),
                color_continuous_scale="YlGn", text_auto=True,
                labels=dict(x="Difficulty", y="Topic", color="Count"),
                title="Questions per topic × difficulty",
            )
            fig_heat.update_layout(
                margin=dict(l=0,r=0,t=40,b=0),
                height=max(300, len(pivot) * 30 + 80),
            )
            st.plotly_chart(fig_heat, use_container_width=True)
