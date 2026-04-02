import os
import re
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
import streamlit as st


DEFAULT_DATA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "data",
    "processed",
    "step16",
    "all_llm_explanations_gpu.csv",
)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
COMPACT_CONTEXT_PATH = os.path.join(PROJECT_ROOT, "CONTEXT_COMPACT.md")


@dataclass
class QueryContext:
    anomaly_ids: List[str]
    stages: List[str]
    top_n: int
    range_start: Optional[datetime]
    range_end: Optional[datetime]
    wants_compare: bool
    wants_low_conf: bool
    wants_high_conf: bool
    wants_explain: bool


INTENT_KEYWORDS: Dict[str, List[str]] = {
    "recommendation": ["what should i do", "what to do", "fix", "action", "next step", "recommend"],
    "summary": ["summary", "brief", "short", "what happened", "explain"],
    "root_cause": ["root cause", "cause", "why"],
    "path": ["path", "propagation", "route", "flow"],
    "confidence": ["confidence", "confident", "score", "uncertain", "certainty"],
    "meta": ["when", "timestamp", "time", "stage", "which stage"],
}

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"


def safe_float(value, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def extract_anomaly_ids(text: str) -> List[str]:
    if not text:
        return []
    # Accept both 'anom_0' and 'anom 0' and normalize to 'anom_0'
    ids = re.findall(r"anom[_\s-]?(\d+)", text.lower())
    return [f"anom_{num}" for num in ids]


def should_use_implicit_anomaly(query: str) -> bool:
    q = query.lower()
    markers = ["this anomaly", "that anomaly", "this one", "that one", "it", "its"]
    return any(m in q for m in markers)


def resolve_prompt_with_memory(query: str, messages: List[Dict], memory: Dict) -> str:
    if extract_anomaly_ids(query):
        return query

    if not should_use_implicit_anomaly(query):
        return query

    last_anomaly = (memory or {}).get("last_anomaly_id", "")
    if not last_anomaly:
        # Fallback: scan recent messages for the latest anomaly mention.
        for msg in reversed(messages[-8:]):
            ids = extract_anomaly_ids(str(msg.get("content", "")))
            if ids:
                last_anomaly = ids[0]
                break

    if last_anomaly:
        return f"{query.strip()} for {last_anomaly}"
    return query


def build_recent_chat_context(messages: List[Dict], max_items: int = 6, max_chars: int = 2000) -> str:
    if not messages:
        return ""

    recent = messages[-max_items:]
    lines: List[str] = []
    for m in recent:
        role = str(m.get("role", "assistant")).upper()
        content = str(m.get("content", "")).strip()
        if not content:
            continue
        lines.append(f"[{role}] {content}")

    text = "\n".join(lines)
    return text[:max_chars]


def build_memory_context(memory: Dict, max_chars: int = 800) -> str:
    if not memory:
        return ""

    parts: List[str] = []
    last_anomaly = memory.get("last_anomaly_id", "")
    if last_anomaly:
        parts.append(f"last_anomaly_id: {last_anomaly}")

    notes = memory.get("notes", [])
    if notes:
        notes_text = "; ".join([str(n) for n in notes[-8:]])
        parts.append(f"session_notes: {notes_text}")

    text = "\n".join(parts)
    return text[:max_chars]


def update_session_memory(memory: Dict, user_query: str, answer: str) -> Dict:
    if memory is None:
        memory = {"last_anomaly_id": "", "notes": []}

    q = user_query.strip()
    q_lower = q.lower()

    # Manual memory commands.
    if q_lower.startswith("remember:"):
        note = q.split(":", 1)[1].strip()
        if note:
            memory.setdefault("notes", []).append(note)
            memory["notes"] = memory["notes"][-20:]

    if q_lower in {"forget memory", "clear memory"}:
        memory["notes"] = []
        memory["last_anomaly_id"] = ""

    ids = extract_anomaly_ids(user_query) + extract_anomaly_ids(answer)
    if ids:
        memory["last_anomaly_id"] = ids[0]

    return memory


@st.cache_data(show_spinner=False)
def load_reference_context(project_root: str) -> str:
    compact_path = os.path.join(project_root, "CONTEXT_COMPACT.md")

    if os.path.exists(compact_path):
        try:
            with open(compact_path, "r", encoding="utf-8") as file:
                return file.read().strip()
        except OSError:
            return ""

    return ""


def compact_reference_context(text: str, max_chars: int = 1800) -> str:
    if not text:
        return ""

    compact = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(compact) <= max_chars:
        return compact

    return compact[:max_chars]


@st.cache_data(show_spinner=False)
def load_step16_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    for col in ["llm_confidence", "temporal_score", "final_score", "rca_score", "path_confidence_score"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    for text_col in [
        "anomaly_id",
        "stage",
        "best_path",
        "summary",
        "root_cause_explanation",
        "propagation_explanation",
        "confidence_explanation",
        "recommendation",
    ]:
        if text_col in df.columns:
            df[text_col] = df[text_col].fillna("").astype(str)

    return df


def parse_time_range(query: str) -> Tuple[Optional[datetime], Optional[datetime]]:
    matches = re.findall(r"\d{4}-\d{2}-\d{2}(?:\s+\d{2}:\d{2}:\d{2})?", query)
    if len(matches) >= 2:
        start = pd.to_datetime(matches[0], errors="coerce")
        end = pd.to_datetime(matches[1], errors="coerce")
        if pd.notna(start) and pd.notna(end):
            return start.to_pydatetime(), end.to_pydatetime()
    return None, None


def build_query_context(query: str) -> QueryContext:
    text = query.lower()

    anomaly_ids = re.findall(r"anom_\d+", text)
    stages = sorted(set(re.findall(r"\bp[1-6]\b", text)))

    top_n_match = re.search(r"top\s+(\d+)", text)
    top_n = int(top_n_match.group(1)) if top_n_match else 5

    range_start, range_end = parse_time_range(query)

    wants_compare = "compare" in text
    wants_low_conf = any(key in text for key in ["low confidence", "uncertain", "uncertainty", "weak confidence"])
    wants_high_conf = any(key in text for key in ["high confidence", "highest confidence", "most confident"])
    wants_explain = any(key in text for key in ["explain", "why", "what happened", "root cause"]) or bool(anomaly_ids)

    return QueryContext(
        anomaly_ids=anomaly_ids,
        stages=[s.upper() for s in stages],
        top_n=max(1, min(top_n, 50)),
        range_start=range_start,
        range_end=range_end,
        wants_compare=wants_compare,
        wants_low_conf=wants_low_conf,
        wants_high_conf=wants_high_conf,
        wants_explain=wants_explain,
    )


@st.cache_data(show_spinner=False)
def build_confidence_indexes(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    indexes: Dict[str, pd.DataFrame] = {}
    if df.empty or "final_score" not in df.columns:
        return indexes

    indexes["global_low"] = df.sort_values(["final_score", "llm_confidence", "temporal_score"], ascending=[True, True, True])
    indexes["global_high"] = df.sort_values(["final_score", "llm_confidence", "temporal_score"], ascending=[False, False, False])

    if "stage" in df.columns:
        for stage in sorted(df["stage"].dropna().astype(str).str.upper().unique()):
            subset = df[df["stage"].astype(str).str.upper() == stage]
            indexes[f"stage_low_{stage}"] = subset.sort_values(
                ["final_score", "llm_confidence", "temporal_score"],
                ascending=[True, True, True],
            )
            indexes[f"stage_high_{stage}"] = subset.sort_values(
                ["final_score", "llm_confidence", "temporal_score"],
                ascending=[False, False, False],
            )

    return indexes


def apply_filters(df: pd.DataFrame, ctx: QueryContext) -> pd.DataFrame:
    filtered = df.copy()

    if ctx.anomaly_ids and "anomaly_id" in filtered.columns:
        filtered = filtered[filtered["anomaly_id"].str.lower().isin(ctx.anomaly_ids)]

    if ctx.stages and "stage" in filtered.columns:
        filtered = filtered[filtered["stage"].str.upper().isin(ctx.stages)]

    if ctx.range_start and ctx.range_end and "timestamp" in filtered.columns:
        filtered = filtered[(filtered["timestamp"] >= ctx.range_start) & (filtered["timestamp"] <= ctx.range_end)]

    return filtered


def summarize_anomaly_row(row: pd.Series) -> str:
    anomaly_id = row.get("anomaly_id", "unknown")
    stage = row.get("stage", "unknown")
    timestamp = row.get("timestamp", "")
    if pd.notna(timestamp):
        timestamp = pd.to_datetime(timestamp).strftime("%Y-%m-%d %H:%M:%S")
    else:
        timestamp = "unknown"

    best_path = row.get("best_path", "not available")
    summary = row.get("summary", "")
    root = row.get("root_cause_explanation", "")
    propagation = row.get("propagation_explanation", "")
    confidence_expl = row.get("confidence_explanation", "")
    recommendation = row.get("recommendation", "")

    llm_conf = safe_float(row.get("llm_confidence"))
    temporal = safe_float(row.get("temporal_score"))
    final = safe_float(row.get("final_score"))
    rca_score = safe_float(row.get("rca_score"))
    path_conf = safe_float(row.get("path_confidence_score"))

    return (
        f"Anomaly {anomaly_id} (stage {stage}, {timestamp})\n"
        f"- Path: {best_path}\n"
        f"- Scores: llm_confidence={llm_conf:.3f}, temporal_score={temporal:.3f}, rca_score={rca_score:.3f}, path_confidence_score={path_conf:.3f}, final_score={final:.3f}\n"
        f"- Summary: {summary}\n"
        f"- Root cause explanation: {root}\n"
        f"- Propagation explanation: {propagation}\n"
        f"- Confidence note: {confidence_expl}\n"
        f"- Recommendation: {recommendation}"
    )


def detect_query_intents(query: str) -> List[str]:
    text = query.lower()
    intents: List[str] = []

    for intent, keys in INTENT_KEYWORDS.items():
        if any(k in text for k in keys):
            intents.append(intent)

    return intents


def format_anomaly_targeted_response(row: pd.Series, query: str) -> str:
    intents = detect_query_intents(query)

    anomaly_id = row.get("anomaly_id", "unknown")
    stage = row.get("stage", "unknown")
    timestamp = row.get("timestamp", "")
    if pd.notna(timestamp):
        timestamp = pd.to_datetime(timestamp).strftime("%Y-%m-%d %H:%M:%S")
    else:
        timestamp = "unknown"

    lines = [f"Anomaly {anomaly_id} | stage={stage} | timestamp={timestamp}"]

    if not intents:
        # Default: concise answer, not full template dump.
        lines.append(f"Summary: {row.get('summary', 'Not available')}")
        lines.append(f"Path: {row.get('best_path', 'Not available')}")
        lines.append(
            "Scores: "
            f"llm_confidence={safe_float(row.get('llm_confidence')):.3f}, "
            f"temporal_score={safe_float(row.get('temporal_score')):.3f}, "
            f"final_score={safe_float(row.get('final_score')):.3f}"
        )
        lines.append(f"Recommendation: {row.get('recommendation', 'Not available')}")
        return "\n".join(lines)

    if "summary" in intents:
        lines.append(f"Summary: {row.get('summary', 'Not available')}")

    if "root_cause" in intents:
        lines.append(f"Root cause: {row.get('root_cause_explanation', 'Not available')}")

    if "path" in intents:
        lines.append(f"Path: {row.get('best_path', 'Not available')}")
        lines.append(f"Propagation: {row.get('propagation_explanation', 'Not available')}")

    if "confidence" in intents:
        lines.append(
            "Scores: "
            f"llm_confidence={safe_float(row.get('llm_confidence')):.3f}, "
            f"temporal_score={safe_float(row.get('temporal_score')):.3f}, "
            f"final_score={safe_float(row.get('final_score')):.3f}"
        )
        lines.append(f"Confidence note: {row.get('confidence_explanation', 'Not available')}")

    if "recommendation" in intents:
        lines.append(f"Recommendation: {row.get('recommendation', 'Not available')}")

    if "meta" in intents and "stage" not in query.lower() and "timestamp" not in query.lower():
        lines.append(f"Meta: stage={stage}, timestamp={timestamp}")

    return "\n".join(lines)


def compare_stages_response(df: pd.DataFrame, stages: List[str]) -> str:
    if "stage" not in df.columns or "final_score" not in df.columns:
        return "Cannot compare stages because required columns are missing in the current RCA dataset."

    if not stages:
        return "Please mention stages (for example: compare P3 and P5)."

    subset = df[df["stage"].str.upper().isin(stages)].copy()
    if subset.empty:
        return "No rows found for the requested stage comparison in the current RCA dataset."

    agg = (
        subset.groupby("stage", as_index=False)
        .agg(
            anomalies=("anomaly_id", "count"),
            avg_final_score=("final_score", "mean"),
            avg_temporal_score=("temporal_score", "mean"),
            avg_llm_confidence=("llm_confidence", "mean"),
        )
        .sort_values("avg_final_score", ascending=False)
    )

    lines = ["Stage comparison from the current RCA dataset:"]
    for _, r in agg.iterrows():
        lines.append(
            f"- {r['stage']}: anomalies={int(r['anomalies'])}, "
            f"avg_final_score={safe_float(r['avg_final_score']):.3f}, "
            f"avg_temporal_score={safe_float(r['avg_temporal_score']):.3f}, "
            f"avg_llm_confidence={safe_float(r['avg_llm_confidence']):.3f}"
        )

    return "\n".join(lines)


def low_confidence_response(df: pd.DataFrame, top_n: int) -> str:
    if "final_score" not in df.columns:
        return "Cannot compute low-confidence anomalies because final_score is missing."

    subset = df.sort_values("final_score", ascending=True).head(top_n)
    if subset.empty:
        return "No anomalies found in the current RCA dataset."

    lines = [f"Lowest-confidence anomalies (top {top_n}):"]
    for _, row in subset.iterrows():
        lines.append(
            f"- {row.get('anomaly_id', 'unknown')} | stage={row.get('stage', 'unknown')} | "
            f"timestamp={row.get('timestamp', 'unknown')} | final_score={safe_float(row.get('final_score')):.3f} | "
            f"path={row.get('best_path', 'not available')}"
        )
    return "\n".join(lines)


def high_confidence_response(df: pd.DataFrame, top_n: int) -> str:
    if "final_score" not in df.columns:
        return "Cannot compute high-confidence anomalies because final_score is missing."

    subset = df.sort_values("final_score", ascending=False).head(top_n)
    if subset.empty:
        return "No anomalies found in the current RCA dataset."

    lines = [f"Highest-confidence anomalies (top {top_n}):"]
    for _, row in subset.iterrows():
        lines.append(
            f"- {row.get('anomaly_id', 'unknown')} | stage={row.get('stage', 'unknown')} | "
            f"timestamp={row.get('timestamp', 'unknown')} | final_score={safe_float(row.get('final_score')):.3f} | "
            f"path={row.get('best_path', 'not available')}"
        )
    return "\n".join(lines)


def domain_reference_response(query: str) -> Optional[str]:
    q = query.lower()
    if "attack window" in q or ("attack" in q and "window" in q):
        return (
            "Attack windows:\n"
            "- cyber_attack: 2019-12-06 10:30:00 to 2019-12-06 11:20:00\n"
            "- physical_attack: 2019-12-06 12:30:00 to 2019-12-06 13:25:00\n"
            "Related timeline labels:\n"
            "- normal: before 2019-12-06 10:20:00\n"
            "- post_attack: at or after 2019-12-06 13:30:00"
        )

    if "what is anomaly" in q or "define anomaly" in q:
        return (
            "In this project, an anomaly is a time window where process behavior deviates from learned normal patterns "
            "(from stage-wise .Pv sensor reconstruction and RCA pipeline outputs)."
        )

    return None


def is_structured_query(ctx: QueryContext, query: str) -> bool:
    if ctx.wants_compare or ctx.wants_low_conf or ctx.wants_high_conf:
        return True
    if ctx.range_start and ctx.range_end:
        return True
    if ctx.anomaly_ids:
        return True
    q = query.lower()
    if "attack window" in q or "what is anomaly" in q or "define anomaly" in q:
        return True
    return False


def deterministic_answer(df: pd.DataFrame, indexes: Dict[str, pd.DataFrame], query: str) -> str:
    ctx = build_query_context(query)

    domain_answer = domain_reference_response(query)
    if domain_answer:
        return domain_answer

    if ctx.wants_low_conf:
        if ctx.stages:
            stage = ctx.stages[0]
            key = f"stage_low_{stage}"
            return low_confidence_response(indexes.get(key, pd.DataFrame()), ctx.top_n)
        return low_confidence_response(indexes.get("global_low", df), ctx.top_n)

    if ctx.wants_high_conf:
        if ctx.stages:
            stage = ctx.stages[0]
            key = f"stage_high_{stage}"
            return high_confidence_response(indexes.get(key, pd.DataFrame()), ctx.top_n)
        return high_confidence_response(indexes.get("global_high", df), ctx.top_n)

    return answer_query(df, query)


def generic_search_response(df: pd.DataFrame, query: str, top_n: int) -> str:
    tokens = [t for t in re.findall(r"[a-zA-Z0-9_.:-]+", query.lower()) if len(t) >= 3]
    if not tokens:
        return "Please ask a more specific question (example: explain anom_12244 in P5)."

    text_cols = [
        c
        for c in [
            "anomaly_id",
            "stage",
            "best_path",
            "summary",
            "root_cause_explanation",
            "propagation_explanation",
            "recommendation",
        ]
        if c in df.columns
    ]
    if not text_cols:
        return "No searchable text columns available in the current RCA dataset."

    score = pd.Series([0] * len(df), index=df.index)
    for col in text_cols:
        col_text = df[col].astype(str).str.lower()
        for token in tokens:
            score += col_text.str.contains(re.escape(token), na=False).astype(int)

    matched = df[score > 0].copy()
    matched["_score"] = score[score > 0]
    matched = matched.sort_values(["_score", "final_score"], ascending=[False, False]).head(top_n)

    if matched.empty:
        return "Not available in the current RCA dataset for this question."

    lines = [f"Matched anomalies from the RCA dataset (top {len(matched)}):"]
    for _, row in matched.iterrows():
        lines.append(
            f"- {row.get('anomaly_id', 'unknown')} | stage={row.get('stage', 'unknown')} | "
            f"timestamp={row.get('timestamp', 'unknown')} | final_score={safe_float(row.get('final_score')):.3f}"
        )
    return "\n".join(lines)


def _keyword_match_score(df: pd.DataFrame, query: str) -> pd.Series:
    tokens = [t for t in re.findall(r"[a-zA-Z0-9_.:-]+", query.lower()) if len(t) >= 3]
    text_cols = [
        c
        for c in [
            "anomaly_id",
            "stage",
            "best_path",
            "summary",
            "root_cause_explanation",
            "propagation_explanation",
            "confidence_explanation",
            "recommendation",
        ]
        if c in df.columns
    ]

    if not tokens or not text_cols:
        return pd.Series([0] * len(df), index=df.index)

    score = pd.Series([0] * len(df), index=df.index)
    for col in text_cols:
        col_text = df[col].astype(str).str.lower()
        for token in tokens:
            score += col_text.str.contains(re.escape(token), na=False).astype(int)
    return score


def select_context_rows_for_llm(df: pd.DataFrame, query: str, max_rows: int = 8) -> pd.DataFrame:
    if df.empty:
        return df

    score = _keyword_match_score(df, query)
    matched = df[score > 0].copy()

    if matched.empty:
        return df.sort_values("final_score", ascending=False).head(max_rows)

    matched["_score"] = score[score > 0]
    matched = matched.sort_values(["_score", "final_score"], ascending=[False, False])
    return matched.head(max_rows)


def rows_to_context_text(df: pd.DataFrame) -> str:
    context_cols = [
        "anomaly_id",
        "stage",
        "timestamp",
        "best_path",
        "llm_confidence",
        "temporal_score",
        "rca_score",
        "path_confidence_score",
        "final_score",
        "summary",
        "root_cause_explanation",
        "propagation_explanation",
        "confidence_explanation",
        "recommendation",
    ]
    cols = [c for c in context_cols if c in df.columns]

    lines: List[str] = []
    for _, row in df[cols].iterrows():
        row_lines = []
        row_lines.append(f"Anomaly {row.get('anomaly_id', '')} (stage {row.get('stage', '')}, {row.get('timestamp', '')})")
        row_lines.append(f"- Path: {row.get('best_path', '')}")
        row_lines.append(
            f"- Scores: llm_confidence={row.get('llm_confidence', 0.0):.3f}, temporal_score={row.get('temporal_score', 0.0):.3f}, "
            f"rca_score={row.get('rca_score', 0.0):.3f}, path_confidence_score={row.get('path_confidence_score', 0.0):.3f}, final_score={row.get('final_score', 0.0):.3f}"
        )
        row_lines.append(f"- Summary: {row.get('summary', '')}")
        row_lines.append(f"- Root cause explanation: {row.get('root_cause_explanation', '')}")
        row_lines.append(f"- Propagation explanation: {row.get('propagation_explanation', '')}")
        row_lines.append(f"- Confidence note: {row.get('confidence_explanation', '')}")
        row_lines.append(f"- Recommendation: {row.get('recommendation', '')}")
        lines.append("\n".join(row_lines))

    return "\n\n---\n\n".join(lines)


def groq_answer(
    query: str,
    context_df: pd.DataFrame,
    reference_context: str,
    chat_context: str,
    memory_context: str,
    api_key: str,
    model: str,
    timeout_seconds: int = 60,
) -> str:
    context_text = rows_to_context_text(context_df)
    if not context_text.strip():
        return "No matching records found in the current RCA dataset for this query."

    system_prompt = (
        "You are an RCA assistant for SWaT static RCA outputs. "
        "You also have repository reference docs (CODE.md, INFO.md) for domain context. "
        "Use the provided RCA rows for anomaly-specific facts and reference docs for pipeline/domain facts. "
        "Do not invent sensors, stages, scores, timestamps, or events. "
        "If asked for recommendation/fix, return only recommendation and brief rationale. "
        "If information is missing in both provided rows and docs, say 'Not available in current data/docs'. "
        "Keep answers concise and directly aligned with user intent."
    )
    reference_text = compact_reference_context(reference_context, max_chars=1800)

    user_prompt = (
        f"User question:\n{query}\n\n"
        f"Session memory:\n{memory_context}\n\n"
        f"Recent chat context:\n{chat_context}\n\n"
        f"Repository reference context (CODE.md + INFO.md):\n{reference_text}\n\n"
        f"RCA rows:\n{context_text}\n\n"
        "Return the smallest useful answer for the question."
    )

    payload = {
        "model": model,
        "temperature": 0.1,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            response = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=timeout_seconds)
            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                wait_seconds = float(retry_after) if retry_after and retry_after.isdigit() else (1.5 * (attempt + 1))
                if attempt < max_attempts - 1:
                    time.sleep(wait_seconds)
                    continue
                return "Groq is rate-limited right now. Please retry in a few seconds."

            response.raise_for_status()
            data = response.json()
            choices = data.get("choices", [])
            if not choices:
                return "Groq returned no answer."
            return str(choices[0].get("message", {}).get("content", "")).strip() or "Groq returned an empty answer."
        except requests.RequestException:
            if attempt < max_attempts - 1:
                time.sleep(1.5 * (attempt + 1))
                continue
            return "Groq request failed. Please retry."

    return "Groq request failed. Please retry."


def answer_query(df: pd.DataFrame, query: str) -> str:
    ctx = build_query_context(query)
    filtered = apply_filters(df, ctx)

    if filtered.empty:
        return "No matching records found in the current RCA dataset for this query."

    if ctx.wants_compare:
        return compare_stages_response(filtered, ctx.stages)

    if ctx.wants_low_conf:
        return low_confidence_response(filtered, ctx.top_n)

    if ctx.anomaly_ids:
        rows = filtered.head(ctx.top_n)
        blocks = [format_anomaly_targeted_response(r, query) for _, r in rows.iterrows()]
        return "\n\n".join(blocks)

    if ctx.wants_explain:
        # For explanation queries without explicit anomaly_id, return best match only.
        top = filtered.sort_values("final_score", ascending=False).head(1)
        if top.empty:
            return "No matching explanation found in the current RCA dataset."
        row = top.iloc[0]
        return format_anomaly_targeted_response(row, query)

    if ctx.range_start and ctx.range_end:
        lines = [
            f"Found {len(filtered)} anomalies between {ctx.range_start} and {ctx.range_end}.",
            f"Showing top {min(ctx.top_n, len(filtered))} by final_score:",
        ]
        top = filtered.sort_values("final_score", ascending=False).head(ctx.top_n)
        for _, row in top.iterrows():
            lines.append(
                f"- {row.get('anomaly_id', 'unknown')} | stage={row.get('stage', 'unknown')} | "
                f"timestamp={row.get('timestamp', 'unknown')} | final_score={safe_float(row.get('final_score')):.3f} | "
                f"path={row.get('best_path', 'not available')}"
            )
        return "\n".join(lines)

    return generic_search_response(filtered, query, ctx.top_n)


def main() -> None:
    st.set_page_config(page_title="SWaT RCA Assistant", layout="centered")
    st.markdown(
        """
        <style>
        /* Responsive container for all content */
        .main > div:first-child {
            max-width: 480px;
            margin: 40px auto 0 auto;
            background: #181c23;
            border-radius: 16px;
            padding: 32px 32px 24px 32px;
            color: #fff;
            box-shadow: 0 4px 24px rgba(0,0,0,0.18);
        }
        .app-subtitle {
            color: #9fb4ca;
            margin-top: -0.25rem;
            margin-bottom: 0.7rem;
            letter-spacing: 0.01em;
            font-size: 1.08rem;
        }
        .small-note {
            color: #8ea3b8;
            font-size: 0.92rem;
            margin-top: -0.1rem;
        }
        .stButton > button, .stDownloadButton > button {
            width: 100%;
            margin-bottom: 16px;
            padding: 16px 0;
            border-radius: 8px;
            font-size: 1.1rem;
            background: #23272f;
            color: #fff;
            border: none;
            transition: background 0.2s;
        }
        .stButton > button:hover, .stDownloadButton > button:hover {
            background: #2d313a;
        }
        .stTextInput > div > input, .stChatInputContainer textarea {
            padding: 16px;
            border-radius: 8px;
            font-size: 1.1rem;
            background: #23272f;
            color: #fff;
            border: 1px solid #23272f;
        }
        @media (max-width: 600px) {
            .main > div:first-child {
                max-width: 100vw;
                margin: 0;
                border-radius: 0;
                padding: 40px 12vw 32px 8vw; /* More left margin */
                min-height: 100vh;
            }
            .app-subtitle { font-size: 1.01rem; }
            .small-note { font-size: 0.98rem; }
            .stButton > button, .stDownloadButton > button {
                font-size: 1.2rem;
                padding: 20px 0;
                margin-bottom: 22px;
            }
            .stTextInput > div > input, .stChatInputContainer textarea {
                font-size: 1.2rem;
                padding: 20px;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("SWaT Incident Insight Assistant")
    st.markdown(
        '<div class="app-subtitle">Interactive root-cause copilot for anomaly explanations and investigation support.</div>',
        unsafe_allow_html=True,
    )

    data_path = DEFAULT_DATA_PATH


    # Minimal, centered, mobile-friendly UI: Quick Actions always visible above chat
    st.markdown("<div style='text-align:center; margin-bottom: 1rem;'><b>Quick Actions</b></div>", unsafe_allow_html=True)
    quick_col1, quick_col2 = st.columns(2)
    with quick_col1:
        if st.button("Explain anom_4914", use_container_width=True):
            st.session_state.queued_prompt = "explain anom_4914"
        if st.button("Top 10 low conf P5", use_container_width=True):
            st.session_state.queued_prompt = "top 10 low confidence anomalies in P5"
        
    with quick_col2:
        if st.button("What is attack window", use_container_width=True):
            st.session_state.queued_prompt = "what is the attack window"
        if st.button("Compare P3 and P5", use_container_width=True):
            st.session_state.queued_prompt = "compare P3 and P5"

    st.markdown("### Chat")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "answer_cache" not in st.session_state:
        st.session_state.answer_cache = {}
    if "memory" not in st.session_state:
        st.session_state.memory = {}
    if "queued_prompt" not in st.session_state:
        st.session_state.queued_prompt = ""

    # (Optional) Download chat button can be placed at the bottom if needed
    df = load_step16_csv(data_path)
    confidence_indexes = build_confidence_indexes(df)
    reference_context = load_reference_context(PROJECT_ROOT)

    groq_api_key = os.getenv("GROQ_API_KEY", "").strip()
    groq_model = os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL).strip() or DEFAULT_GROQ_MODEL
    use_groq = bool(groq_api_key)


    # Remove metrics and make layout dynamic for all screens
    st.markdown(
        '<div class="small-note">Grounded on RCA rows + compact repository context. Use Quick Actions for faster analysis flows.</div>',
        unsafe_allow_html=True,
    )



    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "answer_cache" not in st.session_state:
        st.session_state.answer_cache = {}
    if "queued_prompt" not in st.session_state:
        st.session_state.queued_prompt = ""
    if "memory" not in st.session_state:
        st.session_state.memory = {"last_anomaly_id": "", "notes": []}

    import re, json
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            # Pretty-print anomaly explanation JSON for all assistant messages
            if msg["role"] == "assistant":
                content = msg["content"]
                json_match = re.search(r'\{[\s\S]*\}', content)
                if json_match:
                    try:
                        data = json.loads(json_match.group(0))
                        st.markdown("**Anomaly Explanation**")
                        display_order = [
                            "anomaly_id", "stage", "timestamp", "best_path", "path_confidence_score",
                            "llm_confidence", "temporal_score", "rca_score", "final_score", "summary",
                            "root_cause_explanation", "propagation_explanation", "confidence_explanation", "recommendation"
                        ]
                        # Check if all fields are 'Not available' or empty
                        all_na = all(
                            (str(data.get(k, '')).strip().lower() in {"not available", "", "none"})
                            for k in display_order
                        )
                        if all_na:
                            st.warning("No data found for the requested anomaly. All fields are marked as 'Not available'.")
                        for k in display_order:
                            if k in data:
                                st.markdown(f"**{k.replace('_', ' ').capitalize()}**: {data[k]}")
                        for k, v in data.items():
                            if k not in display_order:
                                st.markdown(f"**{k.replace('_', ' ').capitalize()}**: {v}")
                        meta = msg.get("meta", {})
                        meta_bits = []
                        if meta.get("engine"):
                            meta_bits.append(f"engine={meta['engine']}")
                        if meta.get("cached"):
                            meta_bits.append("cached")
                        if meta.get("fallback"):
                            meta_bits.append("fallback")
                        if meta_bits:
                            st.caption(" | ".join(meta_bits))
                        continue
                    except Exception:
                        pass
            # Default: show as markdown
            st.markdown(msg["content"])
            meta = msg.get("meta", {})
            if msg["role"] == "assistant" and meta:
                meta_bits = []
                if meta.get("engine"):
                    meta_bits.append(f"engine={meta['engine']}")
                if meta.get("cached"):
                    meta_bits.append("cached")
                if meta.get("fallback"):
                    meta_bits.append("fallback")
                if meta_bits:
                    st.caption(" | ".join(meta_bits))

    prompt = st.chat_input("Ask about anomalies, root cause, stage comparison, or confidence...")
    if not prompt and st.session_state.queued_prompt:
        prompt = st.session_state.queued_prompt
        st.session_state.queued_prompt = ""

    if prompt:
        resolved_prompt = resolve_prompt_with_memory(prompt, st.session_state.messages, st.session_state.memory)
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        cache_key = f"groq={use_groq}|model={groq_model}|q={resolved_prompt.strip().lower()}"
        cached_answer = st.session_state.answer_cache.get(cache_key)
        cached = False
        engine_used = "Deterministic"
        used_fallback = False

        ctx = build_query_context(resolved_prompt)
        # Always use LLM for anomaly explanations (by anomaly_id or explain intent)
        is_anomaly_explain = bool(ctx.anomaly_ids) or ctx.wants_explain

        import json

        if cached_answer:
            answer = cached_answer
            cached = True
            engine_used = "cache"
        else:
            with st.spinner("Thinking..."):
                if use_groq and is_anomaly_explain:
                    # Always send all available fields for the anomaly to the LLM
                    if ctx.anomaly_ids:
                        filtered = apply_filters(df, ctx)
                        rows = filtered.head(ctx.top_n)
                    else:
                        filtered = apply_filters(df, ctx)
                        rows = filtered.sort_values("final_score", ascending=False).head(1)
                    # Build explicit LLM prompt
                    context_rows = rows
                    chat_context = build_recent_chat_context(st.session_state.messages, max_items=6, max_chars=2000)
                    memory_context = build_memory_context(st.session_state.memory, max_chars=800)
                    anomaly_info = rows_to_context_text(context_rows)
                    # Instruct LLM to answer in JSON format for anomaly explanation
                    user_prompt = (
                        f"User question: {resolved_prompt}\n\n"
                        f"This is the input. This is the information I have about the anomaly/anomalies.\n"
                        f"Please answer using only this information.\n"
                        f"Return your answer in the following JSON format (fill all fields, use 'Not available' if missing):\n"
                        '{\n'
                        '  "anomaly_id": "...",\n'
                        '  "stage": "...",\n'
                        '  "timestamp": "...",\n'
                        '  "best_path": "...",\n'
                        '  "rca_score": "...",\n'
                        '  "final_score": "...",\n'
                        '  "summary": "...",\n'
                        '  "root_cause_explanation": "...",\n'
                        '  "propagation_explanation": "...",\n'
                        '  "recommendation": "..."\n'
                        '}'
                        f"\n\n{anomaly_info}"
                    )
                    answer = groq_answer(
                        user_prompt,
                        context_rows,
                        reference_context,
                        chat_context,
                        memory_context,
                        groq_api_key,
                        groq_model,
                    )
                    engine_used = "Groq API"
                    if answer.startswith("Groq is rate-limited") or answer.startswith("Groq request failed") or answer.startswith("Groq error"):
                        used_fallback = True
                        engine_used = "Deterministic"
                        answer = "LLM unavailable. No answer generated."
                elif use_groq and not is_structured_query(ctx, resolved_prompt):
                    engine_used = "Groq API"
                    context_rows = select_context_rows_for_llm(df, resolved_prompt, max_rows=4)
                    chat_context = build_recent_chat_context(st.session_state.messages, max_items=6, max_chars=2000)
                    memory_context = build_memory_context(st.session_state.memory, max_chars=800)
                    answer = groq_answer(
                        resolved_prompt,
                        context_rows,
                        reference_context,
                        chat_context,
                        memory_context,
                        groq_api_key,
                        groq_model,
                    )
                    if answer.startswith("Groq is rate-limited") or answer.startswith("Groq request failed") or answer.startswith("Groq error"):
                        used_fallback = True
                        engine_used = "Deterministic"
                        answer = deterministic_answer(df, confidence_indexes, resolved_prompt)
                        answer = f"{answer}\n\n_Note: Groq unavailable; deterministic fallback used._"
                else:
                    engine_used = "Deterministic"
                    answer = deterministic_answer(df, confidence_indexes, resolved_prompt)

            st.session_state.answer_cache[cache_key] = answer

        st.session_state.memory = update_session_memory(st.session_state.memory, resolved_prompt, answer)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "meta": {
                    "engine": engine_used,
                    "cached": cached,
                    "fallback": used_fallback,
                },
            }
        )

        with st.chat_message("assistant"):
            # Always pretty-print JSON if the answer is in JSON format and this is an anomaly explanation
            if is_anomaly_explain:
                try:
                    import re
                    json_match = re.search(r'\{[\s\S]*\}', answer)
                    if json_match:
                        json_str = json_match.group(0)
                        data = json.loads(json_str)
                        st.markdown("**Anomaly Explanation**")
                        display_order = [
                            "anomaly_id", "stage", "timestamp", "best_path", "path_confidence_score",
                            "llm_confidence", "temporal_score", "rca_score", "final_score", "summary",
                            "root_cause_explanation", "propagation_explanation", "confidence_explanation", "recommendation"
                        ]
                        for k in display_order:
                            if k in data:
                                st.markdown(f"**{k.replace('_', ' ').capitalize()}**: {data[k]}")
                        for k, v in data.items():
                            if k not in display_order:
                                st.markdown(f"**{k.replace('_', ' ').capitalize()}**: {v}")
                        # Always return after pretty-printing
                        meta_bits = [f"engine={engine_used}"]
                        if cached:
                            meta_bits.append("cached")
                        if used_fallback:
                            meta_bits.append("fallback")
                        st.caption(" | ".join(meta_bits))
                        return
                except Exception:
                    pass
            # Fallback: show as plain markdown if not anomaly explain or not JSON
            st.markdown(answer)
            meta_bits = [f"engine={engine_used}"]
            if cached:
                meta_bits.append("cached")
            if used_fallback:
                meta_bits.append("fallback")
            st.caption(" | ".join(meta_bits))


if __name__ == "__main__":
    main()
