# Educational Videos AI Platform

An AI-powered platform for generating technical educational videos from text prompts.

This repository is part of the **Sprints AI Engineering Virtual Internship**. The project focuses on building an autonomous educational video-generation system that can transform a simple learning prompt into a structured, validated, and renderable technical video workflow.

---

## Overview

Technical educational videos are expensive and time-consuming to produce manually. A single tutorial often requires scripting, coding, demo preparation, recording, editing, narration, visual synchronization, and repeated updates whenever tools, APIs, or libraries change.

The goal of this project is to automate major parts of that workflow using AI agents, structured outputs, retrieval, code execution, text-to-speech, video synchronization, observability, and evaluation.

At a high level, the platform aims to generate publish-ready technical educational videos from text prompts.

Example prompt:

```text
Create a beginner-friendly tutorial explaining vector databases with a Python demo.
```

Expected output:

```text
A scripted, validated, narrated, visually synchronized technical video with code/demo animations.
```

---

## Project Vision

The platform is designed to support scalable educational content creation.

The long-term vision is an autonomous AI system that can:

* Understand a learning request
* Research and ground the topic
* Generate an educational script
* Produce code examples or UI demo steps
* Convert narration into structured animation events
* Validate generated outputs against strict schemas
* Repair invalid AI outputs automatically
* Generate multilingual narration
* Synchronize audio with visual/code animations
* Render a final educational video
* Track quality, cost, and system behavior through observability tools

---

## Business Problem

Traditional technical video production does not scale well.

### Manual Production Bottlenecks

Creating technical tutorials requires multiple manual steps: scripting, coding, recording, editing, reviewing, and publishing.

### High Cost of Senior Talent

Senior developers, instructors, and DevRel teams often spend valuable time preparing demos, recording walkthroughs, and editing content instead of focusing on core engineering or teaching priorities.

### Difficult Localization

Adapting content to new languages, accents, or markets usually requires re-recording narration and re-syncing visual content.

### Fragile Content Maintenance

Technical videos become outdated quickly when interfaces, libraries, APIs, or workflows change. Traditional automation scripts can also break when UI details change.

---

## Solution

This project explores an AI-native pipeline for creating technical educational videos.

Instead of treating video production as a fully manual process, the system breaks it into structured stages:

```text
User Prompt
   ↓
Research and Retrieval
   ↓
Lesson Script Generation
   ↓
Structured Timeline Planning
   ↓
Code or UI Demo Generation
   ↓
Execution and Validation
   ↓
Repair and Quality Checks
   ↓
TTS Narration
   ↓
Audio/Visual Synchronization
   ↓
Programmatic Video Rendering
   ↓
Final Educational Video
```

The system is designed around structured, testable intermediate outputs so that AI-generated content can be validated, repaired, evaluated, and eventually rendered reliably.

---

## Target Demo

The final demo target is a fully autonomous service that generates technical educational videos from text.

The expected demo highlights include:

### Autonomous AI Pipeline

LangGraph-based agents research, script, and generate technical lesson content from a single user prompt.

### Vision and UI Execution

A vision-enabled agent navigates a simulated browser or interface, executes generated code or UI steps, and verifies the result.

### Programmatic Video Synchronization

The system synchronizes multilingual text-to-speech narration with rendered code typing, highlighting, scrolling, and execution animations.

### Observability and Quality

The backend exposes traces, fallbacks, cost tracking, hallucination checks, and quality reports to make the system debuggable and measurable.

---

## Core Capabilities

Planned and developed capabilities include:

* AI-assisted lesson planning
* Agentic research and retrieval
* Script generation
* Structured output generation
* Pydantic schema validation
* Timeline event planning
* Code-demo generation
* Browser or UI execution agents
* Sandboxed code correction
* Text-to-speech generation
* Audio and visual synchronization
* Programmatic video rendering
* Human-in-the-loop review
* LLM fallback handling
* Observability and tracing
* Quality evaluation and reporting
* Cost and accuracy tracking

---

## High-Level Architecture

The platform is organized around several major layers.

### 1. API Layer

The API layer exposes backend endpoints for receiving generation requests, tracking jobs, and returning generated outputs.

Expected responsibilities:

* Accept user prompts
* Start video-generation workflows
* Track workflow status
* Return generated artifacts
* Expose health and monitoring endpoints

---

### 2. Agent Workflow Layer

The agent workflow layer coordinates multistep AI tasks.

Expected responsibilities:

* Manage LangGraph workflows
* Route work between specialized agents
* Maintain workflow state
* Handle retries and fallbacks
* Coordinate research, scripting, coding, validation, and rendering stages

---

### 3. Retrieval and Knowledge Layer

The retrieval layer grounds generated educational content in relevant source material.

Expected responsibilities:

* Store learning resources
* Retrieve relevant context
* Support agentic RAG
* Improve factual grounding
* Reduce hallucinations

---

### 4. Structured Output Layer

The structured output layer converts free-form model responses into strict machine-readable formats.

Expected responsibilities:

* Define schemas for timeline events
* Validate AI-generated JSON
* Repair invalid outputs
* Measure schema conformance
* Provide reliable instructions for downstream rendering

Example event types:

```text
type
run
highlight
scroll
```

---

### 5. Execution Layer

The execution layer verifies generated code and UI behavior.

Expected responsibilities:

* Run generated code safely
* Detect execution errors
* Support self-correction
* Validate demo steps
* Connect to browser or vision-based execution agents

---

### 6. Media Generation Layer

The media layer turns structured plans into final video assets.

Expected responsibilities:

* Generate narration
* Map script segments to visual events
* Render code animations
* Sync audio with visuals
* Compile final video output

---

### 7. Observability and Evaluation Layer

The evaluation layer tracks system quality, reliability, and cost.

Expected responsibilities:

* Trace LLM calls
* Track token usage
* Track generation cost
* Measure schema conformance
* Measure sequence-level accuracy
* Report repair attempts
* Detect hallucinations
* Support debugging and iteration

---

## Internship Milestones

The project follows a structured 4-week engineering cycle.

### Week 1 — Core Infrastructure

Focus:

* Base architecture
* FastAPI endpoints
* Vector database initialization
* LangGraph state schemas
* Baseline observability

Goal:

```text
A working foundation for the AI video-generation backend.
```

---

### Week 2 — End-to-End Pipeline

Focus:

* First complete workflow execution
* Multi-agent routing
* Agentic RAG integration
* TTS audio generation
* Visual execution planning

Goal:

```text
A working end-to-end path from prompt to structured generation output.
```

---

### Week 3 — Resilience and Human-in-the-Loop

Focus:

* LLM fallbacks
* Sandboxed code self-correction
* Streamlit review interface
* Human-in-the-loop validation
* More reliable repair and recovery behavior

Goal:

```text
A more fault-tolerant system with review and correction mechanisms.
```

---

### Week 4 — Scale and Polish

Focus:

* Asynchronous job processing
* Celery/Redis queues
* Concurrent request handling
* Programmatic video compilation
* Accuracy and cost dashboards
* Final demo preparation

Goal:

```text
A polished demo-ready system capable of generating technical educational videos.
```

---

## Structured Outputs Module

One important part of the platform is the structured-output pipeline.

This module converts narration scripts into validated code-animation timelines. These timelines provide precise instructions for rendering educational videos.

The structured-output workflow includes:

* Script segmentation
* Timeline synthesis
* Pydantic v2 validation
* Discriminated event unions
* Temporal validation
* Repair loops for invalid model output
* Evaluation metrics
* Schema and repair-loop tests

This helps ensure that model outputs are not just readable, but also usable by downstream rendering systems.

---

## Evaluation Goals

The project emphasizes measurable quality.

Important evaluation dimensions include:

* Schema conformance rate
* Mean repair rounds
* Sequence-level accuracy
* Execution success rate
* Hallucination detection
* Token usage
* Cost per generation
* Final video readiness

The system should not only generate content, but also provide evidence that the generated content is valid, accurate, and suitable for rendering.

---

## Expected Final Outcome

The final system should demonstrate a pipeline capable of turning a text prompt into a technical educational video workflow.

A successful demo should show:

* A user submitting a technical lesson prompt
* AI agents researching and planning the lesson
* A generated script and code/demo plan
* Validated structured timeline events
* Code or UI execution
* Narration generation
* Audio and visual synchronization
* Final video compilation
* Observability traces and quality metrics

---

## Technology Areas

The project may involve the following technologies and concepts:

* Python
* FastAPI
* LangGraph
* Pydantic v2
* Vector databases
* Agentic RAG
* Text-to-speech
* Vision-based agents
* Browser automation
* Sandboxed code execution
* Streamlit
* Langfuse
* DeepEval
* Celery
* Redis
* Docker
* GitHub Actions
* Programmatic video rendering

---

## Project Status

This repository is under active development as part of the Sprints AI Engineering Virtual Internship.

Current focus:

```text
Building the foundation for an autonomous AI-powered educational video-generation platform.
```

---

## Contributing

For branch naming, Pull Request rules, and team workflow, see:

```text
.github/CONTRIBUTING.md
```

---

## Contact

Sprints website:

```text
sprints.ai
```

Sprints email:

```text
info@sprints.ai
```
