"""Educational-video generation pipeline.

Stages: research (RAG) -> script -> code -> self-healing sandbox -> structured
timeline -> [HITL approval gate] -> TTS narration -> render. The orchestrator
runs synchronously inside a FastAPI background task (threadpool) and persists
progress + artifacts through ``app.services.video_store``.
"""
