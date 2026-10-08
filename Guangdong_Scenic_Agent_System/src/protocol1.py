"""JSON protocol models and serializers for the scenic customer service agent."""

from datetime import datetime, timezone
import json
from typing import Any, Iterable


PROTOCOL_NAME = "scenic_customer_service"
PROTOCOL_VERSION = "1.0"
SUCCESS_CODE = 0
INVALID_REQUEST_CODE = 1001
QUERY_ERROR_CODE = 2001


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _value(record: dict, *keys: str) -> Any:
    for key in keys:
        value = record.get(key)
        if value is not None and value != "":
            return value
    return None


def _json_value(value: Any) -> Any:
    """Convert common database/model scalar values into JSON-safe values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except Exception:
            pass
    return str(value)


def normalize_spot(record: dict) -> dict:
    """Map SQL rows and retrieval metadata to one public scenic-spot shape."""
    return {
        "spot_name": _json_value(_value(record, "spot_name", "name")),
        "city": _json_value(_value(record, "city")),
        "level": _json_value(_value(record, "level", "spot_level")),
        "address": _json_value(_value(record, "address", "location", "location_val")),
        "ticket": _json_value(_value(record, "ticket", "ticket_val")),
        "open_time": _json_value(_value(record, "open_time", "open_time_val")),
        "telephone": _json_value(_value(record, "telephone", "tel")),
    }


def _merge_spot(existing: dict, incoming: dict) -> dict:
    for key, value in incoming.items():
        if (existing.get(key) is None or existing.get(key) == "") and value not in (None, ""):
            existing[key] = value
    return existing


def normalize_spots(records: Iterable[dict] | None, structured_spots: Any = None) -> list[dict]:
    """Deduplicate scenic spots while preserving the order returned by retrieval."""
    normalized = []
    by_name = {}

    candidates = []
    if isinstance(structured_spots, dict):
        candidates.append(structured_spots)
    elif isinstance(structured_spots, (list, tuple)):
        candidates.extend(structured_spots)
    if records:
        candidates.extend(record for record in records if isinstance(record, dict))

    for record in candidates:
        if isinstance(record, str):
            record = {"spot_name": record}
        if not isinstance(record, dict):
            continue
        spot = normalize_spot(record)
        name = spot["spot_name"]
        if not name:
            continue
        if name in by_name:
            _merge_spot(by_name[name], spot)
        else:
            by_name[name] = spot
            normalized.append(spot)
    return normalized


def normalize_sources(records: Iterable[dict] | None) -> list[dict]:
    """Expose source identity without returning raw chunks or internal scores."""
    sources = []
    seen = set()
    for record in records or []:
        if not isinstance(record, dict):
            continue
        source = {
            "spot_name": _json_value(_value(record, "spot_name", "name")),
            "source_url": _json_value(_value(record, "source_url", "url")),
            "source_domain": _json_value(_value(record, "source_domain")),
            "chunk_index": _json_value(_value(record, "chunk_index")),
        }
        if not any(source[key] is not None for key in ("source_url", "source_domain")):
            continue
        identity = tuple(source.values())
        if identity not in seen:
            seen.add(identity)
            sources.append(source)
    return sources


def validate_request(user_query: str, session_id: str | None = None) -> None:
    if not isinstance(user_query, str) or not user_query.strip():
        raise ValueError("user_query must be a non-empty string")
    if session_id is not None and (not isinstance(session_id, str) or not session_id.strip()):
        raise ValueError("session_id must be a non-empty string when provided")


def build_success_response(
    *,
    request_id: str,
    user_query: str,
    session_id: str,
    answer: str,
    docs: list | None = None,
    query_meta: dict | None = None,
) -> dict:
    query_meta = query_meta or {}
    docs = docs or []
    route = query_meta.get("route", "unknown")
    answer_type = query_meta.get("answer_type", "text")

    return {
        "protocol": PROTOCOL_NAME,
        "version": PROTOCOL_VERSION,
        "timestamp": _timestamp(),
        "request_id": request_id,
        "code": SUCCESS_CODE,
        "message": "success",
        "data": {
            "session_id": session_id,
            "query": user_query,
            "answer": answer or "",
            "answer_type": answer_type,
            "route": route,
            "route_confidence": query_meta.get("confidence"),
            "route_reason": query_meta.get("reason", []),
            "spots": normalize_spots(docs, query_meta.get("spots")),
            "sources": normalize_sources(docs),
            "suggestions": query_meta.get("suggestions", []),
        },
        "error": None,
    }


def build_error_response(
    *,
    request_id: str,
    code: int,
    message: str,
    error_type: str,
    detail: str | None = None,
    session_id: str | None = None,
    retryable: bool = False,
) -> dict:
    return {
        "protocol": PROTOCOL_NAME,
        "version": PROTOCOL_VERSION,
        "timestamp": _timestamp(),
        "request_id": request_id,
        "code": code,
        "message": message,
        "data": {"session_id": session_id} if session_id else None,
        "error": {
            "type": error_type,
            "detail": detail,
            "retryable": retryable,
        },
    }


def to_json(response: dict) -> str:
    """Serialize a protocol response for transport over HTTP, sockets, or logs."""
    return json.dumps(response, ensure_ascii=False, separators=(",", ":"))


__all__ = [
    "INVALID_REQUEST_CODE",
    "QUERY_ERROR_CODE",
    "build_error_response",
    "build_success_response",
    "to_json",
    "validate_request",
]
