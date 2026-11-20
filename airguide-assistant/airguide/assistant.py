"""Orchestrator: message parsing -> context update -> live fetch -> routing -> response.

Coordinates deterministic decision rules, optional Gen-AI phrasing, and safety guardrails.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from . import llm
from .aqi import category
from .context import ACTIVITIES, SENSITIVE_GROUPS, UserContext, sanitize_location
from .data_source import DataSourceError, LiveReading, fetch_live
from .prompt import render_system_prompt
from .router import Directive, decide

MAX_MESSAGE_LEN = 500

EMERGENCY_RE = re.compile(
    r"chest (pain|tight)|can'?t breathe|cannot breathe|trouble breathing|"
    r"difficulty breathing|short(ness)? of breath|struggling to breathe|"
    r"faint(ing)?|blue lips",
    re.I,
)
EMERGENCY_REPLY = (
    "Those symptoms can be serious. Please seek urgent medical care now: call your "
    "local emergency number or go to the nearest emergency room. Move to clean, "
    "indoor air while you get help."
)

_AQI_RE = re.compile(r"\baqi\b\D{0,12}(\d{1,3})\b|\b(\d{1,3})\s*aqi\b", re.I)
_PM25_RE = re.compile(r"\bpm\s?2\.?5\b\D{0,12}(\d{1,4}(?:\.\d+)?)", re.I)
_PM10_RE = re.compile(r"\bpm\s?10\b\D{0,12}(\d{1,4}(?:\.\d+)?)", re.I)
_LOC_RE = re.compile(
    r"(?i:\bin|\bat|\bfor|\bnear)\s+"
    r"([A-Z][A-Za-z'.\-]+(?:\s+[A-Z][A-Za-z'.\-]+){0,2})"
)
_LEAD_RE = re.compile(r"^(i live in|i am in|i'm in|in|at|near|it'?s|city[:\s]*)\s+", re.I)

# Pre-compiled regex patterns for CPU efficiency (avoid runtime recompilation)
_SENS_PATTERNS: dict[str, re.Pattern[str]] = {
    "lung": re.compile(r"asthma|copd|lung|bronch|wheez", re.I),
    "heart": re.compile(r"heart|cardiac|cardio", re.I),
    "child": re.compile(r"child|kid|toddler|baby|infant|son\b|daughter", re.I),
    "elderly": re.compile(r"elderly|senior|grand(ma|pa|mother|father)|old age|over 65|65\+", re.I),
    "pregnant": re.compile(r"pregnan|expecting", re.I),
}

_NONE_RE = re.compile(r"^\W*(none|no|nobody|no one|nothing|healthy|all healthy|n/?a)\W*$", re.I)

_ACTIVITY_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("indoor", re.compile(r"indoor|inside|stay(ing)? (in|home)|at home|work from home", re.I)),
    ("exercise", re.compile(r"\brun|jog|cycl|bike|biking|gym|workout|exercise|football|cricket|sport|hike|swim", re.I)),
    ("walk", re.compile(r"walk|stroll|dog|stand|play", re.I)),
    ("commute", re.compile(r"commut|office|school|drive|driving|bus|metro|scooter|work\b", re.I)),
)


def _parse(message: str, ctx: UserContext) -> None:
    """Extract air quality figures, locations, sensitive groups, and activities from message."""
    text = message[:MAX_MESSAGE_LEN]
    lowered = text.lower()
    had_number = False

    # 1. Parse AQI and PM readings
    m = _AQI_RE.search(text)
    if m:
        value = int(m.group(1) or m.group(2))
        if 0 <= value <= 500:
            ctx.aqi_reported, ctx.source, had_number = value, "user", True
            ctx.fetch_failed = False
    for rx, attr in ((_PM25_RE, "pm25"), (_PM10_RE, "pm10")):
        m = rx.search(text)
        if m:
            setattr(ctx, attr, float(m.group(1)))
            had_number = True
            ctx.source = ctx.source or "user"
            ctx.fetch_failed = False
            if ctx.aqi_reported is not None and ctx.source == "live":
                ctx.aqi_reported = None  # recompute from user figures

    # 2. Parse location if no numerical reading was provided
    if not had_number:
        loc = None
        m = _LOC_RE.search(text)
        if m:
            loc = sanitize_location(m.group(1))
        elif ctx.awaiting == "air_reading" and not re.search(r"\d", text) and len(text) <= 40:
            loc = sanitize_location(_LEAD_RE.sub("", text.strip()).rstrip("?.! "))
        if loc:
            if loc != ctx.location:  # new place invalidates old readings
                ctx.aqi_reported = ctx.pm25 = ctx.pm10 = None
            ctx.location = loc

    # 3. Parse sensitive health conditions (pre-compiled patterns)
    found = tuple(g for g in SENSITIVE_GROUPS if _SENS_PATTERNS[g].search(lowered))
    if found:
        ctx.sensitivity = tuple(sorted(set((ctx.sensitivity or ()) + found)))
    elif ctx.awaiting == "sensitivity" and _NONE_RE.match(lowered):
        ctx.sensitivity = ()

    # 4. Parse planned activities (ordered by specificity)
    for name, pattern in _ACTIVITY_PATTERNS:
        if pattern.search(lowered):
            ctx.activity = name
            break


def _apply_live(ctx: UserContext, reading: LiveReading) -> None:
    """Update user context with live atmospheric observations."""
    ctx.location = ctx.location or reading.place
    ctx.pm25, ctx.pm10 = reading.pm25, reading.pm10
    ctx.aqi_reported = reading.aqi
    ctx.source = "live"
    ctx.fetch_failed = False


def fallback_reply(ctx: UserContext, directive: Directive) -> str:
    """Generate high-clarity deterministic response text without LLM."""
    if directive.action == "ASK":
        prefix = ""
        if directive.ask_field == "air_reading" and ctx.fetch_failed:
            prefix = "I couldn't fetch live data for that place. "
        return prefix + directive.text.split("QUESTION_TO_ASK: ")[1].split("\n")[0]
    aqi = ctx.aqi
    where = ctx.location or "your area"
    source = "live data" if ctx.source == "live" else "your reading"
    lines = [
        f"AQI is {aqi} ({category(aqi)}) in {where}, from {source}. "
        f"Plan: {directive.verdict.replace('_', ' ').lower()}."
    ]
    lines += [f"- {p}" for p in directive.points]
    lines.append(f"- Mask: {directive.mask}.")
    return "\n".join(lines)


@dataclass(frozen=True)
class Reply:
    """Encapsulates the final response text, updated context, and directive."""

    text: str
    state: UserContext
    directive: Directive


Fetch = Callable[[str], LiveReading]


def respond(
    message: str,
    state: UserContext | None = None,
    *,
    live_fetch: Fetch | None = fetch_live,
    use_llm: bool = True,
) -> Reply:
    """Orchestrate the end-to-end assistant turn.

    1. Checks emergency symptom circuit breaker.
    2. Parses user input into dynamic context state.
    3. Fetches live data if location is known and AQI is missing.
    4. Evaluates deterministic decision matrix.
    5. Dispatches to Gen-AI phrasing layer with graceful offline fallback.
    """
    ctx = state or UserContext()
    message = (message or "")[:MAX_MESSAGE_LEN].strip()

    if EMERGENCY_RE.search(message):
        return Reply(EMERGENCY_REPLY, ctx, decide(ctx))

    _parse(message, ctx)

    if ctx.location and ctx.aqi is None and live_fetch and not ctx.fetch_failed:
        try:
            _apply_live(ctx, live_fetch(ctx.location))
        except DataSourceError:
            ctx.fetch_failed = True

    directive = decide(ctx)
    ctx.awaiting = directive.ask_field
    ctx.stage = "advising" if directive.action == "ADVISE" else "gathering"

    text = None
    if use_llm and llm.enabled():
        text = llm.generate(render_system_prompt(ctx, directive), message)
    if text is None:
        text = fallback_reply(ctx, directive)
    return Reply(text, ctx, directive)
