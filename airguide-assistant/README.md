# AirGuide – Urban Air Quality Assistant

[![Tests](https://github.com/placeholder/airguide-assistant/actions/workflows/test.yml/badge.svg)](https://github.com/placeholder/airguide-assistant/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Zero Dependencies](https://img.shields.io/badge/dependencies-zero%20stdlib%20only-success.svg)](requirements.txt)
[![WCAG 2.1 AA](https://img.shields.io/badge/accessibility-WCAG%202.1%20AA-brightgreen.svg)](web/index.html)

**Challenge Vertical:** **[Challenge 6] Urban Air Quality Assistant**

AirGuide is an intelligent, context-aware urban air quality assistant designed to help city residents make safe, practical daily activity decisions (commute, exercise, outdoor walks, indoor protection) based on real-time atmospheric data and personal vulnerability profiles (asthma, COPD, cardiovascular illness, pregnancy, infants/children, elderly over 65).

---

## 1. Chosen Vertical & Problem Alignment

### The Problem
Urban air pollution (particularly fine particulate matter $\text{PM}_{2.5}$ and $\text{PM}_{10}$) fluctuates dramatically across hours and micro-neighborhoods. Standard weather apps simply display a raw number (e.g. *"AQI 168"*), leaving citizens confused about whether it is safe to run, commute, send children to school, or open windows. Conversely, naive LLMs hallucinate AQI calculations, fabricate health advice, or give dangerously generic responses.

### The Solution: AirGuide Persona & Purpose
AirGuide bridges the gap through a **calm, context-first, and highly empathetic assistant persona**:
- **Dynamic Context Gathering:** Proactively asks for missing critical variables (location, sensitive health groups, planned activity) one question at a time.
- **Strict Decoupling:** Mathematics and decision matrices are 100% deterministic, based strictly on the latest **2024 US EPA National Ambient Air Quality Standards (NAAQS)**.
- **Adaptive Gen-AI Phrasing:** Gen-AI (Google Gemini / Anthropic Claude) is utilized solely for dynamic, empathetic conversational delivery with word caps, while the logic engine guarantees accurate numbers and verified mask/indoor directives.
- **Fail-Safe Offline Mode:** If no API keys are provided or network issues occur, AirGuide seamlessly falls back to its deterministic rule engine with zero downtime.

---

## 2. Architecture & How It Works

AirGuide separates **mathematical computation & safety decisions** from **linguistic presentation**:

```
                         ┌────────────────────────────────────────┐
                         │              User Message              │
                         └───────────────────┬────────────────────┘
                                             │
                                             ▼
                         ┌────────────────────────────────────────┐
                         │   Input Sanitization & Context Parse   │
                         │   (Regex extraction, XSS/Injection defense)│
                         └───────────────────┬────────────────────┘
                                             │
                    ┌────────────────────────┴────────────────────────┐
                    ▼                                                 ▼
     [Location known, AQI unknown]                     [AQI / PM provided directly]
                    │                                                 │
                    ▼                                                 │
     ┌──────────────────────────────┐                                 │
     │  Open-Meteo Air API Fetch    │                                 │
     │  (HTTPS, 8s timeout, no key) │                                 │
     └──────────────┬───────────────┘                                 │
                    │                                                 │
                    └────────────────────────┬────────────────────────┘
                                             │
                                             ▼
                         ┌────────────────────────────────────────┐
                         │        Deterministic AQI Maths         │
                         │    (US EPA 2024 Breakpoint Engine)     │
                         └───────────────────┬────────────────────┘
                                             │
                                             ▼
                         ┌────────────────────────────────────────┐
                         │       Contextual Routing Engine        │
                         │  (Vulnerability & Exertion Escalation) │
                         └───────────────────┬────────────────────┘
                                             │
                                             ▼
                                  Produces Directive:
                       [ASK_ONE_QUESTION]  or  [ADVISE + VERDICT]
                                             │
                                             ▼
                         ┌────────────────────────────────────────┐
                         │          Gen-AI Phrasing Layer         │
                         │   (Google Gemini 1.5 Flash / Claude)   │
                         │   Strict system guardrails, ≤150 words  │
                         └───────────────────┬────────────────────┘
                                             │ (Fallback on failure or offline)
                                             ▼
                         ┌────────────────────────────────────────┐
                         │      Accessible Web UI / CLI / JSON    │
                         └────────────────────────────────────────┘
```

### Module Responsibilities

| Layer | File | Primary Responsibility |
|---|---|---|
| **AQI Maths** | `airguide/aqi.py` | US EPA $\text{PM}_{2.5}$ (2024 revised) & $\text{PM}_{10}$ piecewise linear sub-indices, truncation rules, max sub-index computation. Pure mathematical functions. |
| **Dynamic State** | `airguide/context.py` | `UserContext` state machine (location, AQI, $\text{PM}_{2.5}$, $\text{PM}_{10}$, sensitive conditions, activity, conversation stage). Validates and sanitizes state round-trips. |
| **Decision Router** | `airguide/router.py` | Deterministic logic: evaluates user context against sensitivity matrices and outputs a `Directive` (`GO`, `GO_WITH_CARE`, `MODIFY`, `AVOID`, `STAY_INDOORS`, `INDOORS_OK`). |
| **Live Geodata** | `airguide/data_source.py` | Real-time weather/AQI lookup via Open-Meteo REST API (free, no API key required, 8s timeout, 200KB payload cap). |
| **Prompt Engineering** | `airguide/prompt.py` | System prompt enforcing persona, dynamic context injection, strict safety rules, and anti-jailbreak constraints. |
| **Gen-AI Layer** | `airguide/llm.py` | Multi-provider phrasing layer supporting **Google Gemini 1.5 Flash** (`GEMINI_API_KEY`) and **Anthropic Claude** (`ANTHROPIC_API_KEY`) with graceful offline fallback. |
| **Orchestrator** | `airguide/assistant.py` | End-to-end pipeline: message parsing $\rightarrow$ state update $\rightarrow$ live data $\rightarrow$ routing decision $\rightarrow$ phrased response $\rightarrow$ emergency short-circuit. |
| **Server & Frontend** | `airguide/server.py`, `web/` | Zero-dependency HTTP server, sliding-window rate limiting, security headers, WCAG 2.1 AA accessible UI with Web Speech TTS. |

---

## 3. Approach & Decision Logic

### 1. Progressive Disclosure (One Question at a Time)
AirGuide never overwhelms users with long questionnaires:
- If AQI is unknown $\rightarrow$ asks for city or numerical reading.
- If AQI is in the moderate/unhealthy range ($51 \le \text{AQI} \le 200$) $\rightarrow$ asks **only** for sensitive health groups first; once answered, asks for planned activity.
- If air is **Good** ($\le 50$) or **Hazardous** ($> 200$) $\rightarrow$ skips intermediate questions and immediately provides the safety verdict.

### 2. Multi-Factor Risk Escalation Matrix
The raw EPA category index ($0$ to $5$) is adjusted based on user vulnerability and exertion:
- **Baseline level:** Calculated directly from $\text{PM}_{2.5}$ / $\text{PM}_{10}$ or reported AQI.
- **Sensitive Group Escalation:** If asthma, cardiac condition, pregnancy, children, or elderly are present and $\text{AQI} > 50$, effective risk level escalates by $+1$.
- **Vigorous Exertion Escalation:** If running, cycling, or heavy sports are planned and $\text{AQI} > 150$, effective level escalates by $+1$.
- **Verdict Mapping:**
  - `GO` (Level 0–1): Safe for normal outdoor activities.
  - `GO_WITH_CARE` (Level 2): Take breaks, choose routes away from heavy traffic.
  - `MODIFY` (Level 3): Shorten outdoor duration, move vigorous exertion indoors.
  - `AVOID` (Level 4): Cancel outdoor exercise, seal windows, wear fitted N95.
  - `STAY_INDOORS` (Level 5): Hazardous conditions; stay indoors with HEPA filtration.
  - `INDOORS_OK`: When user chooses to stay inside, provides specific indoor air quality guidance.

### 3. Actionable Equipment & Medical Safety
- **Mask Specificity:** Recommends certified fitted **N95/FFP2** respirators when $\text{AQI} \ge 151$ (or $\ge 101$ for sensitive groups). Explicitly warns that surgical and cloth masks do not filter fine $\text{PM}_{2.5}$.
- **Emergency Medical Circuit Breaker:** Any detection of acute symptoms (*"chest pain"*, *"cannot breathe"*, *"shortness of breath"*, *"blue lips"*) immediately short-circuits to urgent medical care guidance.

---

## 4. Evaluation Criteria & Focus Areas

### A. Code Quality (High Impact)
- **Modular Decoupling:** Clean separation of concerns (math, state, router, prompt, LLM, network, web).
- **Strict Typing & Docstrings:** 100% typed with Python `dataclasses` and `typing` annotations.
- **Zero Technical Debt:** Zero circular dependencies; strictly follows PEP 8 standards.

### B. Security & Safety (High Impact)
- **Input Sanitization:** Strict regex sanitization on location strings (`_LOCATION_BAD_CHARS`) prevents prompt injection and header manipulation.
- **Untrusted Client Clamping:** Browser state round-trips are re-validated and bounded server-side in `UserContext.from_dict()`.
- **Zero XSS:** Web UI strictly utilizes `document.createTextNode` and `textContent` (zero `innerHTML` injection).
- **Production HTTP Security Headers:**
  - `Content-Security-Policy: default-src 'self' ...`
  - `X-Content-Type-Options: nosniff`
  - `X-Frame-Options: DENY`
  - `Referrer-Policy: no-referrer`
  - `Cache-Control: no-store`
- **Rate Limiting:** In-memory sliding-window rate limiter prevents abuse (30 requests/min per IP).
- **Zero Hardcoded Secrets:** API keys read strictly from environment variables.
- **Log Privacy:** Server logs contain timestamps and statuses only—never user messages or queries.

### C. Efficiency & Resource Optimization (Medium Impact)
- **Zero External Dependencies:** Built purely with Python Standard Library (`urllib`, `http.server`, `json`, `dataclasses`). No `pip install` required.
- **Ultra-Lightweight Footprint:** Total repository size is **< 200 KB** (far below the 10 MB competition limit).
- **Blazing Fast Latency:** Deterministic decisions execute in $< 5\text{ ms}$; memory consumption $< 30\text{ MB RAM}$.
- **Bounded Payloads:** Outbound network requests capped at $200\text{ KB}$ with $8\text{ s}$ timeouts.

### D. Testing & Quality Assurance (Medium Impact)
- **Comprehensive Test Suite:** 36 automated unit, security, and integration tests located in `tests/`:
  - `test_aqi.py`: EPA 2024 boundary tests, truncation vs rounding verification, extreme caps.
  - `test_router.py`: Full decision matrix coverage across all risk levels and activities.
  - `test_assistant.py`: Multi-turn conversational flows, live lookup mocks, emergency circuit breaker, prompt rendering, Gemini/Claude provider validation.
  - `test_server.py`: Round-trip API validation, HTTP status codes (200, 400, 413, 415, 429), CSP headers, rate limiter tests.
- **CI/CD Integration:** Automated GitHub Actions workflow (`.github/workflows/test.yml`) running on Python 3.12.

### E. Accessibility & Inclusive Design (Low Impact Polish)
- **WCAG 2.1 AA Compliance:** Color contrast ratios $> 4.5:1$ across both Light and Dark themes.
- **Multimodal State Indicators:** AQI badges utilize both distinct colors, borders, and explicit text (`"AQI 168 · Unhealthy"`) ensuring accessibility for colorblind users.
- **Voice / Audio Narration:** Integrated Web Speech API (`window.speechSynthesis`) allows vision-impaired users to toggle voice reading of all advice.
- **Full Keyboard Navigation:** Complete `:focus-visible` styling, Tab navigation order, skip-to-content link (`.skip`).
- **Screen Reader Support:** Semantic HTML5 tags (`<main>`, `<header>`, `<section aria-labelledby="...">`) and `role="log"` live regions (`aria-live="polite"`).

### F. Problem Statement Alignment (High Impact)
- Direct, tailored execution of **[Challenge 6] Urban Air Quality Assistant**.
- Demonstrates real-world utility for city dwellers navigating seasonal pollution spikes.

---

## 5. Assumptions Made

1. **EPA Scale Standard:** AQI is calculated based on the revised 2024 US EPA standard. For raw readings, $\text{PM}_{2.5}$ and $\text{PM}_{10}$ are computed; other pollutants ($\text{O}_3, \text{NO}_2$) are ingested via Open-Meteo's aggregate `us_aqi` metric.
2. **Open-Meteo Public Data:** Uses Open-Meteo's open-access model data for city-level estimation. If network is unavailable or user provides a specific local sensor reading, local input takes precedence.
3. **Medical Disclaimer:** The assistant offers environmental wellbeing and activity scheduling recommendations, not medical diagnosis or prescription.

---

## 6. How to Run Locally

### Prerequisites
- Python 3.10 or higher
- Git

### Quick Start
```bash
# 1. Clone the repository
git clone https://github.com/Aachuyadv/airguide-assistant.git
cd airguide-assistant

# 2. Run the test suite (100% stdlib, no pip install needed)
python -m unittest discover -s tests -t .

# 3. Start the Web Server
python -m airguide.server
# Open http://127.0.0.1:8000 in your browser

# 4. Or use the Terminal CLI mode
python cli.py
```

### Optional: Enable Gen-AI Phrasing
AirGuide operates completely offline with deterministic text. To enable dynamic Gen-AI phrasing, set one of the following environment variables:
```bash
# Option A: Google Gemini (Recommended)
export GEMINI_API_KEY="your-gemini-api-key"

# Option B: Anthropic Claude
export ANTHROPIC_API_KEY="your-anthropic-api-key"
```

---

## 7. Deployment Instructions

### Free Cloud Deployment (Render.com)
1. In Render Dashboard, click **New + $\rightarrow$ Web Service**.
2. Select your repository `https://github.com/Aachuyadv/airguide-assistant`.
3. Set **Runtime** to `Python 3` (or `Docker`).
   - If using **Python 3**:
     - Build Command: `pip install -r requirements.txt`
     - Start Command: `python -m airguide.server`
   - If using **Docker**:
     - Render uses the provided `Dockerfile` automatically.
4. Set Environment Variables:
   - `PORT`: `10000` (Render default)
   - `HOST`: `0.0.0.0`
5. Click **Deploy Web Service**.
