"""
src/review/dashboard.py
Stage 10 — Human Quality Review Dashboard using Streamlit.

Run with:
    streamlit run src/review/dashboard.py -- --db outputs/review_queue.jsonl

Features:
- Audio playback for each segment
- Transcript display with quality metrics sidebar
- Accept / Needs Review / Reject buttons
- Progress tracking
- Keyboard shortcuts (a=accept, r=reject, n=needs_review)
- Export reviewed decisions to JSONL
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TTS Dataset Review",
    page_icon="🎙️",
    layout="wide",
)

REVIEW_FILE = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/review_queue.jsonl")
DECISIONS_FILE = REVIEW_FILE.parent / "review_decisions.jsonl"


@st.cache_data
def load_queue(path: Path) -> list[dict]:
    items = []
    if path.exists():
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
    return items


def save_decision(segment_id: str, decision: str, reviewer_note: str = "") -> None:
    record = {
        "segment_id": segment_id,
        "decision": decision,
        "reviewer_note": reviewer_note,
    }
    with open(DECISIONS_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")


def load_decisions() -> dict[str, str]:
    decisions: dict[str, str] = {}
    if DECISIONS_FILE.exists():
        with open(DECISIONS_FILE) as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    decisions[rec["segment_id"]] = rec["decision"]
    return decisions


# ─── Main Dashboard ────────────────────────────────────────────────────────────

def main() -> None:
    st.title("🎙️ TTS Dataset — Human Review Dashboard")

    queue = load_queue(REVIEW_FILE)
    if not queue:
        st.warning(f"No items found in {REVIEW_FILE}. Run the pipeline first.")
        return

    decisions = load_decisions()
    pending   = [q for q in queue if q["segment_id"] not in decisions]
    reviewed  = len(queue) - len(pending)

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Progress")
        st.metric("Total segments",  len(queue))
        st.metric("Reviewed",        reviewed)
        st.metric("Pending",         len(pending))
        progress = reviewed / max(len(queue), 1)
        st.progress(progress, text=f"{progress*100:.0f}% complete")

        st.divider()
        st.subheader("Filter")
        lang_filter = st.selectbox("Language", ["All", "en-IN", "hi-IN"])
        min_score   = st.slider("Min quality score", 0.0, 1.0, 0.0, 0.05)

        st.divider()
        st.subheader("Decision summary")
        counts = {"accepted": 0, "rejected": 0, "needs_review": 0}
        for d in decisions.values():
            if d in counts:
                counts[d] += 1
        for k, v in counts.items():
            st.metric(k.replace("_", " ").title(), v)

    # ── Filter queue ─────────────────────────────────────────────────────────
    filtered = [
        q for q in pending
        if (lang_filter == "All" or q.get("language") == lang_filter)
        and q.get("final_score", 0.0) >= min_score
    ]

    if not filtered:
        st.success("✅ All filtered segments reviewed!")
        return

    # ── Current segment ───────────────────────────────────────────────────────
    if "idx" not in st.session_state:
        st.session_state.idx = 0
    idx = min(st.session_state.idx, len(filtered) - 1)
    item = filtered[idx]

    # Navigation
    col_nav_l, col_info, col_nav_r = st.columns([1, 6, 1])
    with col_nav_l:
        if st.button("◀ Prev", disabled=(idx == 0)):
            st.session_state.idx = max(0, idx - 1)
            st.rerun()
    with col_info:
        st.caption(f"Segment {idx + 1} of {len(filtered)} (pending) | ID: `{item['segment_id']}`")
    with col_nav_r:
        if st.button("Next ▶", disabled=(idx >= len(filtered) - 1)):
            st.session_state.idx = min(len(filtered) - 1, idx + 1)
            st.rerun()

    st.divider()

    # ── Main content area ────────────────────────────────────────────────────
    col_audio, col_meta = st.columns([3, 2])

    with col_audio:
        st.subheader("🔊 Audio")
        audio_path = item.get("audio_path", "")
        if audio_path and Path(audio_path).exists():
            st.audio(audio_path, format="audio/wav")
        else:
            st.error(f"Audio file not found: {audio_path}")

        st.subheader("📝 Transcript")
        transcript = item.get("transcript", "")
        st.markdown(f"> {transcript}")

        st.subheader("✅ Review decision")
        reviewer_note = st.text_input(
            "Reviewer note (optional)", key=f"note_{item['segment_id']}"
        )

        col_a, col_r, col_n = st.columns(3)
        with col_a:
            if st.button("✅ Accept", type="primary", use_container_width=True):
                save_decision(item["segment_id"], "accepted", reviewer_note)
                st.session_state.idx = min(len(filtered) - 1, idx + 1)
                st.rerun()
        with col_r:
            if st.button("❌ Reject", type="secondary", use_container_width=True):
                save_decision(item["segment_id"], "rejected", reviewer_note)
                st.session_state.idx = min(len(filtered) - 1, idx + 1)
                st.rerun()
        with col_n:
            if st.button("🔍 Needs Review", use_container_width=True):
                save_decision(item["segment_id"], "needs_review", reviewer_note)
                st.session_state.idx = min(len(filtered) - 1, idx + 1)
                st.rerun()

    with col_meta:
        st.subheader("📊 Quality metrics")

        final_score = item.get("final_score", 0.0)
        color = "green" if final_score >= 0.75 else "orange" if final_score >= 0.55 else "red"
        st.markdown(
            f"**Final score:** :{color}[{final_score:.3f}]"
        )

        # Score breakdown bars
        metrics = {
            "Audio quality":      item.get("audio_quality_score", 0),
            "Transcript quality": item.get("transcript_quality_score", 0),
            "Speaker purity":     item.get("speaker_purity_score", 0),
            "Emotion confidence": item.get("emotion_confidence", 0),
        }
        for name, val in metrics.items():
            col1, col2 = st.columns([2, 1])
            col1.progress(float(val), text=name)
            col2.caption(f"{val:.2f}")

        st.divider()
        st.subheader("🏷️ Labels")
        col_l1, col_l2 = st.columns(2)
        col_l1.metric("Language",  item.get("language", "—"))
        col_l2.metric("Emotion",   item.get("emotion", "—"))
        col_l1.metric("Style",     item.get("style", "—"))
        col_l2.metric("Duration",  f"{item.get('duration', 0):.1f}s")

        st.divider()
        st.subheader("🔍 ASR details")
        st.metric("ASR confidence",  f"{item.get('asr_confidence', 0):.3f}")
        st.metric("LLM QA score",    f"{item.get('llm_quality_score', 0):.3f}")
        st.metric("SNR",             f"{item.get('estimated_snr_db', 0):.1f} dB")
        st.metric("Speech rate",     f"{item.get('speech_rate_wpm', 0):.0f} wpm")
        st.metric("Clip ratio",      f"{item.get('clipping_ratio', 0):.4f}")

        llm_issues = item.get("llm_issues", [])
        if llm_issues:
            st.warning("LLM issues: " + ", ".join(llm_issues))

        st.divider()
        st.subheader("📹 Source")
        st.caption(f"Video: {item.get('video_title', '—')}")
        st.caption(f"Channel: {item.get('channel', '—')}")
        st.caption(f"Video ID: `{item.get('source', '—')}`")


if __name__ == "__main__":
    main()