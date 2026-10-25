"""Evidence-grounded RoadWatch assistant.

Remote path: Gemini via the official `google-genai` SDK using the documented function-calling flow
(client.interactions.create with function declarations; the APPLICATION executes each call and returns a
function_result). Only the five read-only tools in evidence_tools are exposed. Gemini can explain the retrieved figures;
it cannot run SQL or shell commands, create scores, or change a ranking.
Local path: a deterministic keyword router over the same tools, used when the flag is off, the key is missing, a request
limit is hit or the API fails.
Every answer carries the tools' source references; numbers in a remote answer that do not appear in any tool result are
reported in `unverified_numbers`.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

from . import config
from .evidence_tools import CAPS, METRICS, SCHEMAS, EvidenceTools

DEFAULT_MODEL = "gemini-3.8-flash"        # model id used in Google's current function-calling docs; override with GEMINI_MODEL
MAX_QUESTION_CHARS = 600
MAX_TOOL_ROUNDS = 4
TIMEOUT_S = 30

INSTRUCTIONS = (
    "You are the RoadWatch evidence assistant for municipal road inspectors. RoadWatch ranks Brooklyn street blocks by how "
    "likely they are to receive a 311 pothole REPORT next week. Rules: (1) Obtain every number by calling the provided tools; "
    "never estimate, round differently or invent figures. (2) If the tools do not contain the answer, say the evidence is "
    "insufficient. (3) Scores are rankings, not probabilities. Reports are not confirmed defects; never claim defects found, "
    "failures prevented or savings. (4) Always name the capture definition and the target when quoting capture or precision. "
    "(5) You cannot change rankings, issue repair orders, or run queries. Keep answers under 180 words."
)
LIMITATIONS = (
    "Limitations for an inspector: (1) RoadWatch predicts 311 reports, not confirmed defects; reports also reflect who complains. "
    "(2) About 44% of test-period reports could belong to several blocks, so capture depends on the definition used. "
    "(3) The non-adjacent lists win only if an inspection also covers a block's end intersections. "
    "(4) Probabilities and weekly volume forecasts are unreliable in surges such as March 2026; use ranks. "
    "(5) There are no inspection outcomes yet, so nothing here measures defects found, failures prevented or savings."
)


@dataclass
class Answer:
    text: str
    sources: list[str] = field(default_factory=list)
    mode: str = "local"                      # "gemini" | "local"
    tool_calls: list[dict] = field(default_factory=list)
    unverified_numbers: list[str] = field(default_factory=list)
    note: str | None = None


NUM = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?")


def unverified(text: str, tool_results: list[dict]) -> list[str]:
    """Numbers in `text` that appear in no tool result (so they cannot be traced to verified evidence)."""
    pool = json.dumps(tool_results, default=str)
    known = {n.replace(",", "") for n in NUM.findall(pool)}
    out = []
    for n in NUM.findall(text):
        c = n.replace(",", "")
        if c in known or c.rstrip("0").rstrip(".") in known or (len(c) <= 1):
            continue
        if re.fullmatch(r"20\d\d", c) or c in {"100", "95"}:
            continue
        out.append(n)
    return out


# ------------------------------------------------------------------ local fallback
def local_answer(question: str, tools: EvidenceTools, week: str | None = None, capacity: str = "1%",
                 list_key: str | None = None) -> Answer:
    q = question.lower()
    m = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", question)
    week = m.group(1) if m else week
    bid = re.search(r"\bblock\s*#?\s*(\d{3,6})\b|\b(\d{5,6})\b", question)
    calls, srcs, parts = [], [], []

    def run(name, args):
        r = tools.call(name, args)
        calls.append({"tool": name, "args": args})
        srcs.extend(r.get("sources", []))
        return r
    rank_m = re.search(r"#\s*(\d{1,3})\b|\brank\s*(\d{1,3})\b", q)
    if bid and week:
        b = int([g for g in bid.groups() if g][0])
        r = run("get_block_evidence", {"block_id": b, "week": week, "capacity": capacity, **({"list_key": list_key} if list_key else {})})
        if "error" in r:
            parts.append(r["error"])
        else:
            d = r["data"]
            if d.get("on_worklist"):
                parts.append(f"Block {d['block_id']} ({d['street']}, ZIP {d['zip']}) is rank {d['rank']} ({d['priority']}) on the {capacity} "
                             f"worklist for the week of {d['week']}, relative score {d['relative_score']} (a ranking score, not a probability). "
                             f"{d['rationale']} Model drivers: {d['model_drivers']}. This is supporting evidence of report activity, not proof of a defect.")
            else:
                parts.append(f"Block {d['block_id']} is not on the {capacity} worklist for {d['week']}; it ranks {d['within_week_rank_of_all_blocks']} "
                             f"of {d['blocks_ranked']:,} blocks that week.")
    if any(k in q for k in ("compare", "policy", "policies", "strict", "non-adjacent", "nonadjacent", "capture", "differ")):
        period = "march_2026" if ("march" in q or "surge" in q) else ("validation" if "validation" in q else "test")
        r = run("compare_inspection_policies", {"period": period, "capacity": capacity})
        if "error" in r:
            parts.append(r["error"])
        else:
            d = r["data"]
            lines = [f"{x['list']}: fractional {x['fractional']['value_pct']}%, any-candidate {x['any_candidate']['value_pct']}%, "
                     f"intersection-node {x['intersection_node']['value_pct']}%" for x in d["lists"]]
            parts.append(f"Unique-report capture at {capacity} capacity ({period}, {d['weeks']} weeks, {d['reports']:,} reports): " + "; ".join(lines)
                         + f". {d['caveat']}")
    for k in METRICS:
        if re.search(rf"\b{k.replace('_', '[ _-]?')}\b", q) and k not in ("any_candidate",):
            parts.append(f"{k.replace('_', ' ').capitalize()}: " + run("get_metric_definition", {"metric": k})["data"]["definition"])
    if "difference" in q and "precision" in q and "recall" in q:
        parts.append(run("get_metric_definition", {"metric": "report_capture"})["data"]["definition"])
    if "limitation" in q or "caveat" in q or "inspector" in q:
        parts.append(LIMITATIONS)
    if (rank_m or "top" in q) and week and not bid:
        r = run("get_worklist_position", {"week": week, "top_n": 5, "capacity": capacity})
        if "data" in r:
            parts.append("Top of the list: " + "; ".join(f"#{x['rank']} {x['street']} ({x['zip']}, score {x['relative_score']})" for x in r["data"]["top"]))
    if not parts:
        parts.append("I can answer from saved RoadWatch evidence about a specific block (include a block id and week, e.g. "
                     "'why was block 52878 selected on 2026-03-16?'), policy comparisons, metric definitions and limitations. "
                     "The evidence provided does not cover this question.")
    return Answer(" ".join(parts), sorted(set(srcs)), "local", calls)


# ------------------------------------------------------------------ Gemini path
class Assistant:
    def __init__(self, tools: EvidenceTools | None = None, client=None, max_requests: int | None = None):
        self.tools = tools or EvidenceTools()
        self._client = client
        self.max_requests = max_requests if max_requests is not None else int(os.environ.get("ROADWATCH_ASSISTANT_MAX_REQUESTS", "20"))
        self.requests_used = 0
        self._cache: dict[str, Answer] = {}

    def _get_client(self):
        if self._client is None:
            from google import genai                       # imported lazily: never at application start-up
            self._client = genai.Client(api_key=config.secret("GEMINI_API_KEY"))
        return self._client

    @property
    def remote_enabled(self) -> bool:
        return config.flag("gemini") and not config.missing_credentials("gemini")

    def ask(self, question: str, week: str | None = None, capacity: str = "1%", list_key: str | None = None) -> Answer:
        question = (question or "").strip()
        if not question:
            return Answer("Ask a question about a block, a policy or a metric.", mode="local")
        if len(question) > MAX_QUESTION_CHARS:
            return Answer(f"Please keep questions under {MAX_QUESTION_CHARS} characters.", mode="local")
        key = json.dumps([question.lower(), week, capacity, list_key])
        if key in self._cache:
            return self._cache[key]
        ans = self._remote(question, week, capacity, list_key) if self.remote_enabled or self._client is not None else None
        if ans is None:
            ans = local_answer(question, self.tools, week, capacity, list_key)
            if config.flag("gemini") and not self.remote_enabled:
                ans.note = "Gemini is enabled but GEMINI_API_KEY is missing; showing the local evidence answer."
        self._cache[key] = ans
        return ans

    def _remote(self, question, week, capacity, list_key) -> Answer | None:
        if self.requests_used >= self.max_requests:
            a = local_answer(question, self.tools, week, capacity, list_key)
            a.note = f"Session limit of {self.max_requests} Gemini requests reached; showing the local evidence answer."
            return a
        decls = [{"type": "function", "name": n, "description": s["description"], "parameters": s["parameters"]} for n, s in SCHEMAS.items()]
        model = os.environ.get("GEMINI_MODEL", DEFAULT_MODEL)
        ctx = f"Selected forecast week: {week or 'none'}; capacity: {capacity}; list: {list_key or self.tools.default_list}."
        calls, results = [], []
        try:
            client = self._get_client()
            self.requests_used += 1
            inter = client.interactions.create(model=model, input=f"{INSTRUCTIONS}\n\n{ctx}\n\nQuestion: {question}", tools=decls)
            for _ in range(MAX_TOOL_ROUNDS):
                fc = next((s for s in inter.steps if s.type == "function_call"), None)
                if fc is None:
                    break
                res = self.tools.call(fc.name, dict(fc.arguments or {}))        # validated, read-only, application-executed
                calls.append({"tool": fc.name, "args": dict(fc.arguments or {})})
                results.append(res)
                self.requests_used += 1
                inter = client.interactions.create(
                    model=model, tools=decls, previous_interaction_id=inter.id,
                    input=[{"type": "function_result", "name": fc.name, "call_id": fc.id,
                            "result": [{"type": "text", "text": json.dumps(res, default=str)}]}])
            text = (inter.output_text or "").strip()
            if not text:
                return None
        except Exception as e:                                                  # network, auth, quota, SDK changes
            a = local_answer(question, self.tools, week, capacity, list_key)
            a.note = f"Gemini request failed ({type(e).__name__}); showing the local evidence answer."
            return a
        srcs = sorted({s for r in results for s in r.get("sources", [])})
        return Answer(text, srcs, "gemini", calls, unverified(text, results),
                      None if results else "No evidence tool was called; this answer is general and not backed by RoadWatch figures.")
