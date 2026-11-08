"""Deterministic answer composition + intent rules (used by the demo gateway and as the live fallback)."""

from __future__ import annotations

import re

from geoai.llm.types import IntentDecision
from geoai.schemas import Domain, Intent

TOPIC_PATTERNS: dict[str, str] = {
    "building": r"building|house|home|structure|roof|facilit|school|clinic",
    "road": r"road|street|highway|avenue|route|bridge|access|traffic",
    "count": r"how many|count|number of|total",
    "area": r"area|extent|how much|hectare|\bha\b|km|size|spread|cover",
    "change": r"change|since|earlier|before|compare|grew|growth|difference|trend|declin|over time",
    "priority": r"priorit|rank|triage|order|first|worst|most affected|hotspot|respond|response|scout",
    "risk": r"risk|threat|expos|danger|endanger|evacuat|vulnerab",
    "smoke": r"smoke|plume|haze",
    "fire": r"fire|flame|blaze|burning",
    "burn": r"burn|char|scar",
    "stress": r"stress|health|ndvi|dry|disease|unhealthy|anomal|vigou?r|crop",
    "zoning": r"zon(e|ing)|conflict|violation|permit",
    "park": r"park|green|open space|walk|accessib",
    "landuse": r"land ?use|land ?cover|urban|built|develop|construct",
    "damage": r"damage|collaps|rubble|debris|destroy",
    "weather": r"weather|wind|humid|temperature",
    "flood": r"flood|water|inundat|river|submerg",
    "summary": r"summar|overview|impact|assess|status|situation",
}


WEIGHT = {"flood": 0.5, "area": 1.0, "summary": 1.0, "landuse": 1.0, "fire": 2.0}


def question_topics(q: str) -> set[str]:
    ql = q.lower()
    return {t for t, pat in TOPIC_PATTERNS.items() if re.search(pat, ql)}


LOCATE_RE = re.compile(
    r"(?:bounding box|bbox|box)\s+(?:for|of Around)?\s*(?:the\s+)?(?P<a>.+)|"
    r"(?:locate|find|where (?:is|are)|show(?: me)?|highlight|detect|ground|mark|identify)\s+(?:the\s+|all\s+)?(?P<b>.+)",
    re.I,
)
INJECTION_RE = re.compile(r"ignore (all |any |the )?(previous|prior|above)|system prompt|reveal .*(key|secret|prompt)|disregard .*instructions|you are now", re.I)
DOMAIN_WORDS = {
    Domain.flood: r"flood|inundat|water level|river|levee",
    Domain.wildfire: r"fire|burn|smoke|wildfire|blaze",
    Domain.earthquake: r"earthquake|quake|collaps|rubble|debris|seismic|aftershock",
    Domain.urban: r"zoning|urban|land ?use|park|construction|development|walkab",
    Domain.agriculture: r"crop|field|ndvi|farm|harvest|irrigat|agricult",
}


def extract_target(q: str) -> str | None:
    m = LOCATE_RE.search(q.strip().rstrip("?.!"))
    if not m:
        return None
    t = (m.group("a") or m.group("b") or "").strip()
    return t or None


def rule_classify(question: str, domain: Domain, has_before: bool) -> IntentDecision:
    q = question.strip()
    ql = q.lower()
    topics = sorted(question_topics(q))
    if INJECTION_RE.search(q):
        return IntentDecision(intent=Intent.out_of_scope, domain=domain, rationale="Request attempts to override instructions.", topics=topics)
    # a question that clearly belongs to another monitoring domain should use that domain's tools
    for d, pat in DOMAIN_WORDS.items():
        if d != domain and re.search(pat, ql) and not any(re.search(p, ql) for dd, p in DOMAIN_WORDS.items() if dd == domain):
            domain = d
            break
    if re.search(r"\b(bounding box|bbox|locate|find|where (is|are)|highlight|detect|mark)\b", ql) or ql.startswith("show"):
        t = extract_target(q)
        if not t:
            return IntentDecision(intent=Intent.locate_object, domain=domain, clarification="Which object should I locate (for example buildings, roads, flooded areas)?", topics=topics)
        return IntentDecision(intent=Intent.locate_object, domain=domain, target=t, rationale="Localisation request.", topics=topics)
    if re.search(r"\b(describe|what is|what's|what do you see|caption|tell me about)\b", ql):
        return IntentDecision(intent=Intent.describe_scene, domain=domain, rationale="Scene description request.", topics=topics)
    if "change" in topics and has_before and not re.search(r"how many|count", ql):
        return IntentDecision(intent=Intent.change_analysis, domain=domain, rationale="Temporal comparison request.", topics=topics)
    if len(ql.split()) < 2:
        return IntentDecision(intent=Intent.domain_assessment, domain=domain, clarification="Could you say what you want to know about this image?", topics=topics)
    return IntentDecision(intent=Intent.domain_assessment if topics or re.search(r"is |are |does|how", ql) else Intent.visual_question,
                          domain=domain, rationale="Question answered from domain measurements.", topics=topics)


def compose_answer(question: str, intent: Intent, findings: list[dict], texts: list[str], max_findings: int = 3) -> str:
    qt = question_topics(question)
    fl = findings
    if intent == Intent.describe_scene:
        picked = [f for f in fl if "describe" in f["topics"]]
        picked += [f for f in fl if "summary" in f["topics"] and f not in picked][:3]
    elif intent == Intent.locate_object:
        picked = [f for f in fl if "locate" in f["topics"]] or fl[:2]
    else:
        def score(f: dict) -> float:
            ft = set(f["topics"])
            return sum(WEIGHT.get(t, 3.0) for t in qt & ft) + 0.5 * ("summary" in ft)

        scored = sorted(enumerate(fl), key=lambda t: (-score(t[1]), t[0]))
        picked = [f for _, f in scored[:max_findings]]
        if qt and not any(qt & set(f["topics"]) for f in picked):
            picked = [f for f in fl if "summary" in f["topics"]][:max_findings]
    parts = []
    for t in texts:
        if t:
            parts.append(t.strip())
    for f in picked:
        txt = f["text"].strip()
        if f["kind"] == "inferred" and not txt.lower().startswith(("(inference)", "suggested")):
            txt = f"(Inference) {txt}"
        elif f["kind"] == "not_determinable":
            txt = f"Not determinable: {txt}"
        parts.append(txt)
    return " ".join(parts) if parts else "I could not derive an answer from the available measurements."
