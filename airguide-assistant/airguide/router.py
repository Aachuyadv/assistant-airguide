"""Backend decision engine: turns user context into a routing directive.

All logic is deterministic, reproducible, and unit-tested. The LLM (when enabled) only
rephrases the directive; it never calculates risk, escalates, or decides.
"""
from __future__ import annotations

from dataclasses import dataclass

from .aqi import category, level_index
from .context import ACTIVITIES, UserContext

VERDICTS: tuple[str, ...] = ("GO", "GO_WITH_CARE", "MODIFY", "AVOID", "STAY_INDOORS", "INDOORS_OK")

QUESTIONS: dict[str, str] = {
    "air_reading": (
        "Which city or area are you in? You can also type today's AQI "
        "or PM2.5 reading if you have it."
    ),
    "sensitivity": (
        "Does anyone with you have asthma or lung disease, a heart condition, "
        "is pregnant, a child, or over 65? Reply 'none' if not."
    ),
    "activity": (
        "What do you plan to do outdoors: walk, commute, exercise "
        "(run/cycle/sport), or stay indoors?"
    ),
}

_VERDICT_BY_LEVEL: dict[int, str] = {
    0: "GO",
    1: "GO",
    2: "GO_WITH_CARE",
    3: "MODIFY",
    4: "AVOID",
    5: "STAY_INDOORS",
}

_POINTS: dict[str, tuple[str, ...]] = {
    "GO": ("Air is fine for your plan.", "Enjoy outdoor time as normal."),
    "GO_WITH_CARE": (
        "Keep it shorter or lighter than usual.",
        "Choose parks or side streets away from heavy traffic.",
        "Stop and move indoors if you cough or feel tight-chested.",
    ),
    "MODIFY": (
        "Move vigorous activity indoors; keep outdoor time short and light.",
        "Avoid busy roads and rush-hour hours.",
        "Check again in a few hours; air often improves in the afternoon.",
    ),
    "AVOID": (
        "Postpone outdoor exertion or move it indoors.",
        "Close windows; run a HEPA purifier if you have one.",
        "If you must go out, keep it brief and wear a fitted N95/FFP2.",
    ),
    "STAY_INDOORS": (
        "Stay indoors with windows and doors closed.",
        "Use a HEPA purifier or a well-sealed room.",
        "Only make essential trips, wearing a fitted N95/FFP2.",
    ),
    "INDOORS_OK": (
        "Staying in is a good call.",
        "Keep windows closed while air is poor; open them when AQI drops.",
    ),
}


@dataclass(frozen=True)
class Directive:
    """Actionable directive output produced by the deterministic decision router."""

    action: str  # "ASK" or "ADVISE"
    ask_field: str | None
    verdict: str | None
    level: int | None  # AQI category index of the raw reading (0-5)
    effective_level: int | None  # After sensitivity and exertion risk adjustments
    mask: str | None
    points: tuple[str, ...]
    text: str  # Formatted for prompt injection


def _mask(base: int, sensitive: bool, verdict: str) -> str:
    """Determine mask recommendation based on EPA guidance."""
    if verdict == "INDOORS_OK":
        return "Not needed indoors"
    if base >= 3 or verdict in ("AVOID", "STAY_INDOORS"):
        return "Fitted N95/FFP2 for any outdoor time (cloth/surgical masks do not stop PM2.5)"
    if base == 2 or (base == 1 and sensitive):
        return "Optional fitted N95/FFP2 for sensitive people"
    return "Not needed"


def _ask(ctx: UserContext, field: str) -> Directive:
    """Construct an inquiry directive requesting exactly one missing variable."""
    text = (
        "ACTION: ASK_ONE_QUESTION\n"
        f"MISSING_FIELD: {field}\n"
        f"QUESTION_TO_ASK: {QUESTIONS[field]}\n"
        "Do not give advice yet."
    )
    return Directive("ASK", field, None, None, None, None, (), text)


def decide(ctx: UserContext) -> Directive:
    """Evaluate context through multi-factor decision matrix and produce directive.

    Progressive logic:
    - If AQI missing -> ASK air_reading
    - If AQI between 51 and 200 -> progressively ASK sensitivity then activity
    - Good (<=50) and Hazardous (>200) skip questions for immediate safety verdicts.
    """
    aqi = ctx.aqi
    if aqi is None:
        return _ask(ctx, "air_reading")

    base = level_index(aqi)
    # Progressive gathering: ask only what alters the recommendation
    if 1 <= base <= 3:
        if ctx.sensitivity is None:
            return _ask(ctx, "sensitivity")
        if ctx.activity is None:
            return _ask(ctx, "activity")

    sensitive = bool(ctx.sensitivity)
    activity = ctx.activity or "walk"
    exertion = ACTIVITIES.get(activity, 1)

    effective = base
    if base >= 1 and sensitive:
        effective += 1
    if base >= 3 and exertion == 2:
        effective += 1
    effective = min(effective, 5)

    if activity == "indoor" and effective < 5:
        verdict = "INDOORS_OK"
    else:
        verdict = _VERDICT_BY_LEVEL[effective]

    points = _POINTS[verdict]
    mask = _mask(base, sensitive, verdict)
    label = category(aqi)
    where = ctx.location or "your area"
    text = (
        "ACTION: ADVISE\n"
        f"KEY_FIGURES: AQI {aqi} ({label}) in {where}; "
        f"PM2.5={ctx.prompt_fields()['pm25']}; PM10={ctx.prompt_fields()['pm10']}\n"
        f"VERDICT: {verdict}\n"
        f"MASK: {mask}\n"
        "MUST_COVER: " + " | ".join(points)
    )
    return Directive("ADVISE", None, verdict, base, effective, mask, points, text)
