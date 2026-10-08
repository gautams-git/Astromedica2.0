# AstroMedica Architecture Overview

This document provides a highly detailed explanation of the AstroMedica codebase, a web-based artificial intelligence health consultation application driven by astrological birth chart analysis. Use this overview to understand how data flows through the application, where core logic resides, and how the patient state is saved.

## 📂 Root Directory / Architecture

The codebase is built on **Python (Flask)** in the backend, relying on **Prokerala APIs** for astrological charting, **Gemini APIs (Gemma-3-27b-it model)** for chatbot consultation, and raw **HTML/JS/CSS** for the frontend user interface.

### 🌐 Server & Endpoints
* **`server.py`**: The main Flask application. It defines standard HTTP REST endpoints used by the frontend:
  * `/generate-chart` (POST): Initiates the astrological chart generation.
  * `/start-consultation` (POST): Starts a new consultation session and generates the first AI response.
  * `/send-message` (POST): Handles user messages and returns AI responses.
  * `/combine-output` (POST): Triggers "Module C" to compile diagnoses and chat logs into a final health overview.
  * Static file serving routes (`/`, `/results`, `/results.html`).

### 🧠 Core Astrology & Risk Logic
* **`rules.py`**: Contains all strict Vedic astrology formulations alongside predictive algorithms.
  * Calculates `apply_rule_1` (6th house analysis, computing Rogkaraka planets 1, 2, and 3).
  * Evaluates system doshas, active Dashas (`analyse_dasha`), and Drishtis (`analyse_drishti`).
  * Features `score_health_risks`, computing a prioritized array of the user's highest health risks mapping planets/zodiacs to medical terminology.
* **`knowledge_base.py`**: Hardcoded dictionaries mapping planetary bodies, zodiac rashis, and nakshatras to anatomical organs, diseases, and doshas (Vata, Pitta, Kapha). 
* **`disease_diagnose.py`**: Script processing astrological anomalies and creating immediate risk diagnoses.
* **`check_dasha.py`**: Associated logic file evaluating Vimshottari Mahadasha/Antardasha timing events.

### 🤖 AI / Chatbot Pipeline
* **`chat_module.py`**: Acting as "Module B", this script crafts system prompts and limits the Gemini model parameters:
  * Pre-loads patient contexts from `diagnosis.json` and `processed_logic.json`.
  * Computes strict prompt injections forcing the AI to simulate an empathetic Ayurvedic health advisor who asks highly targeted questions *without* ever using astrological terminology.
  * Employs stateful user `conversation_store` management.
  * Actively creates, updates, and saves conversation contexts into a specialized `chat_log.json` on the disk system per patient.

### 🪝 Data Ingestion (Module A)
* **`jataka_fetch.py`**: Calls the external Prokerala REST API. Generates patient counter ids, fetches the chart (planet alignments, SVG drawing data, birth details, dasha sequences), and dumps `raw_chart.json` and `raw_dasha.json` into the patient's individual directory.
* **`jataka_logic.py`**: The bridge transforming raw Prokerala data payload formats into normalized dictionary artifacts the rest of the ecosystem safely digests.

### 🗃️ Aggregation
* **`combine_output.py`**: The final phase ("Module C"). Joins the generated mathematical probabilities from astrological charts with the dynamic user symptomatic responses gathered by the chat AI, outputting `final_output.json`.

## 📁 Storage Operations

### `patients/` Directory
The system avoids complex relational databases by saving everything directly to flat files within sequentially generated subdirectories (e.g., `patients/001/`, `patients/002/`).

A standard patient folder contains:
* **`raw_chart.json` / `raw_dasha.json`**: Source-of-truth payloads directly pulled from Prokerala API per patient credentials.
* **`processed_logic.json`**: Output from `rules.py` determining Rogkarakas, Dashas, and disease filters.
* **`diagnosis.json`**: Mathematical health probabilities specific to the person.
* **`chart_data.json`**: Consolidated snapshot data.
* **`chat_log.json`**: Ongoing transcript between the client and Gemma AI. Saves on every conversation turn.

### `data/` Directory
Provides statically hosted domain models.
* **`disease_table.json`**: Extended definitions of astrological rules translated to disease logic variables mapping.

### Utility & Logs
* **`counter.txt`**: Generates and persists auto-incrementing numerical IDs (like `031`) for newly submitted patients in `jataka_fetch.py`.

## 🖥️ Frontend layer
* **`astrohealth-landing.html`**: The intake gateway for generating patient IDs, querying birthdates/lat/lng user info, and loading animations.
* **`results.html`**: Heavily formatted Single Page Application (SPA). Capable of reading Dasha states, displaying astrological SVG wheels, organizing risk scores via priority cards, simulating an ongoing consultation chatbot window, and producing the complete final PDF report visually.

## 🧪 Testing Suites
Multiple unit test and integration scripts evaluate API limits and structural stability:
* `test_chat.py`, `test_dasha_accuracy.py`, `test_details.py`, `test_endpoint.py`, `test_flask.py`, `test_gemini.py`, `test_gemini_connection.py`, `test_models.py`, `test_new_logic.py`, `test_output.py`, `test_questions.py`, `test_scoring.py`.
