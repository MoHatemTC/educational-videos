# Educational Videos AI Platform

AI pipeline for turning a topic, script, or webpage URL into a narrated educational MP4.

The system supports:

* **Code tutorials**: topic → research → code → sandbox validation/repair → narration → timeline → MP4
* **Web explainers**: URL → browser screenshots → vision description → narration → screenshot video → MP4
* Human review before rendering
* RAG-grounded research
* Timeline-driven code animation
* ElevenLabs TTS
* LangGraph generation workflow
* Celery/Redis-compatible background workers
* Streamlit frontend

---

## API endpoints

| Endpoint                                    | Purpose                              |
| ------------------------------------------- | ------------------------------------ |
| `POST /api/v1/videos/jobs`                  | Create a video job                   |
| `GET /api/v1/videos/jobs`                   | List recent jobs                     |
| `GET /api/v1/videos/jobs/{job_id}`          | Get job status                       |
| `GET /api/v1/videos/jobs/{job_id}/events`   | Stream live progress                 |
| `GET /api/v1/videos/jobs/{job_id}/review`   | Get editable review artifacts        |
| `POST /api/v1/videos/jobs/{job_id}/approve` | Approve and render                   |
| `POST /api/v1/videos/jobs/{job_id}/reject`  | Reject a job                         |
| `GET /api/v1/videos/jobs/{job_id}/result`   | Download final MP4                   |
| `GET /api/v1/videos/jobs/{job_id}/traces`   | View token, cost, and latency traces |

---

## Main components

### LangGraph pipeline

`app/services/pipeline/graph.py` controls generation as a state graph:

```text
load_job
  → research
  → code
  → sandbox
  → sandbox_repair if needed
  → script
  → visual_planning
  → approval_gate
```

For webpage jobs:

```text
load_job
  → web_capture
  → web_describe
  → web_script
  → web_visual_planning
  → approval_gate
```

### Human review

Generated artifacts pause at the approval gate. The reviewer can edit both the narration script and the code.

Rendering starts only after approval.

### Rendering

Code tutorials render with:

```text
timeline events + code + narration audio → PNG frames → MP4
```

Web explainers render with:

```text
screenshots + narration audio → Ken-Burns style video → MP4
```

### Vision recovery

The canonical vision package is:

```text
app/services/pipeline/vision/
```

It handles screenshot-based browser work and recovery from:

* cookie banners
* popups
* layout shifts
* navigation errors

It blocks unsafe bypass attempts for:

* captchas
* login walls
