"""Streamlit frontend for the AI Educational Video Platform.

A thin client over the FastAPI backend: create a job, watch live SSE progress,
review/edit the generated script + code at the HITL gate, approve to render, then
play the finished MP4 and inspect per-stage token/cost/latency traces.
"""

import api_client as api
import streamlit as st

st.set_page_config(page_title="AI Educational Video Platform", page_icon="🎬", layout="wide")

_RUNNING_STATES = {"pending", "running", "approved", "rendering"}
# Egyptian Arabic first so it is the default selection.
_LANG_LABELS = {"egyptian_arabic": "Egyptian Arabic", "en": "English"}

if "job_id" not in st.session_state:
    st.session_state.job_id = None

st.title("🎬 AI Educational Video Platform")
st.caption("Kimi K2.6 · ElevenLabs · Qdrant — topic → narrated code-tutorial video")


# ── Sidebar: new job + recent jobs ───────────────────────────────────────────
with st.sidebar:
    st.header("New video")
    
    mode = st.radio(
        "Video type",
        ["code_tutorial", "web_explainer"],
        format_func=lambda m: "💻 Code tutorial" if m == "code_tutorial" else "🌐 Explain a web page",
    )

    with st.form("new_job", clear_on_submit=False):
        if mode == "web_explainer":
            url = st.text_input("Page URL", placeholder="https://www.noon.com")
            topic = st.text_input("Title", placeholder="noon.com homepage")
        else:
            url = None
            topic = st.text_input("Topic", placeholder="e.g. Python list comprehensions")
        
        language = st.selectbox("Narration language", list(_LANG_LABELS), format_func=_LANG_LABELS.get)
        submitted = st.form_submit_button("Generate", type="primary", use_container_width=True)

    if submitted:
        title = (topic or "").strip() or (url or "").strip()
        if mode == "web_explainer" and not (url or "").strip():
            st.error("Enter a page URL to explain.")
        elif not title:
            st.error("Enter a topic or URL.")
        else:
            try:
                res = api.create_job(title, language, mode=mode, url=url)
                st.session_state.job_id = res["job_id"]
                st.rerun()
            except Exception as exc:
                st.error(f"Failed to start job: {exc}")

    st.divider()
    row = st.columns([3, 1])
    row[0].header("Recent jobs")
    if row[1].button("🔄"):
        st.rerun()
    try:
        jobs = api.list_jobs(20)
    except Exception as exc:  # noqa: BLE001
        jobs = []
        st.warning(f"Backend not reachable:\n{exc}")
    for j in jobs:
        icon = {"done": "✅", "error": "❌", "awaiting_approval": "⏸️", "rejected": "🚫"}.get(j["status"], "⏳")
        if st.button(f"{icon} {j['topic'][:28]}", key=f"job_{j['job_id']}", width="stretch"):
            st.session_state.job_id = j["job_id"]
            st.rerun()


# ── Main panel ───────────────────────────────────────────────────────────────
job_id = st.session_state.job_id
if not job_id:
    st.info("Create a video from the sidebar, or pick a recent job.")
    st.stop()

try:
    status = api.get_status(job_id)
except Exception as exc:  # noqa: BLE001
    st.error(f"Could not load job {job_id}: {exc}")
    st.stop()

st.subheader(f"{status['topic']}")
c = st.columns(4)
c[0].metric("Status", status["status"])
c[1].metric("Step", status["current_step"])
c[2].metric("Language", _LANG_LABELS.get(status["language"], status["language"]))
c[3].metric("Job", job_id[:8])

if status["status"] == "error":
    st.error(f"Pipeline error: {status.get('error')}")

# Live progress via SSE while the job is working.
if status["status"] in _RUNNING_STATES:
    st.markdown("#### Live progress")
    stage_box = st.empty()
    done_box = st.empty()
    with st.spinner("Working… Kimi is a reasoning model (~20-30s per stage)."):
        try:
            for ev in api.stream_events(job_id):
                stage_box.markdown(f"**{ev['status']}** — `{ev['current_step']}`")
                produced = [k for k, v in (ev.get("has") or {}).items() if v]
                done_box.write("Produced: " + (" · ".join(produced) if produced else "—"))
                if ev.get("error"):
                    st.error(ev["error"])
        except Exception as exc:  # noqa: BLE001
            st.warning(f"progress stream ended: {exc}")
    st.rerun()

# HITL review gate.
if status["awaiting_approval"]:
    st.markdown("#### Review & approve")
    review = api.get_review(job_id)
    is_web = review.get("mode") == "web_explainer"
    left, right = st.columns(2)
    with left:
        script = st.text_area("Narration script", review.get("script") or "", height=340)
    with right:
        if is_web:
            shots = review.get("screenshots") or []
            if shots:
                st.image(shots[0], caption="Captured page", width="stretch")
            with st.expander("Page description (Kimi vision)"):
                st.write(review.get("research") or "")
            code = None
        else:
            code = st.text_area("Code", review.get("code") or "", height=340)
            timeline = review.get("timeline") or {}
            st.caption(f"Structured timeline: {len(timeline.get('events', []))} events")
    a, b = st.columns(2)
    if a.button("✅ Approve & render", type="primary", width="stretch"):
        api.approve(job_id, script=script, code=code)
        st.rerun()
    if b.button("❌ Reject", width="stretch"):
        api.reject(job_id, reason="rejected from UI")
        st.rerun()

# Result.
if status["status"] == "done":
    st.markdown("#### Result")
    art = status.get("artifacts", {})
    st.caption(f"Narration duration: {art.get('audio_duration_s', '?')}s")
    try:
        data = api.get_result_bytes(job_id)
        if data:
            st.video(data)
            st.download_button("⬇️ Download MP4", data, file_name=f"{job_id}.mp4", mime="video/mp4")
    except Exception as exc:  # noqa: BLE001
        st.warning(f"video not available: {exc}")

# Artifacts + traces.
with st.expander("📄 Script / code / timeline"):
    review = api.get_review(job_id)
    st.text_area("Script", review.get("script") or "", height=160, disabled=True)
    st.code(review.get("code") or "", language="python")

with st.expander("📊 Traces — tokens · estimated cost · latency"):
    try:
        traces = api.get_traces(job_id)
        st.dataframe(traces.get("rows", []), width="stretch")
        totals = traces.get("totals", {})
        t = st.columns(3)
        t[0].metric("Total tokens", totals.get("total_tokens", 0))
        t[1].metric("Est. cost (USD)", f"${totals.get('est_cost_usd', 0):.4f}")
        t[2].metric("LLM latency", f"{totals.get('latency_ms', 0) / 1000:.1f}s")
    except Exception as exc:  # noqa: BLE001
        st.write(f"no traces: {exc}")
