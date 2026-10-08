"""Persistent operational counters and sanitized errors, separate from scenic data."""
import calendar
import json
import logging
import os
import threading
import uuid
from contextvars import ContextVar
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import psutil

from config import ROOT_PATH
from file_store import write_json

LOCAL_TZ = ZoneInfo("Asia/Shanghai")
_lock = threading.RLock()
_request_context = ContextVar("operations_request_context", default={})
_last_prune_day = None


def _ops_root():
    from user_account import USER_ROOT_DIR
    return Path(USER_ROOT_DIR) / "operations"


def _prune_old_files(now):
    global _last_prune_day
    if _last_prune_day == now.date():
        return
    root = _ops_root()
    for folder, pattern, retention in ((root / "errors", "*.jsonl", 90),
                                       (root / "requests", "*.json", 400),
                                       (root / "usage", "*.json", 400)):
        cutoff = now.date() - timedelta(days=retention)
        for path in folder.glob(pattern) if folder.is_dir() else ():
            try:
                file_day = datetime.strptime(path.stem, "%Y-%m-%d").date()
                if file_day < cutoff:
                    path.unlink()
            except (OSError, ValueError):
                continue
    _last_prune_day = now.date()


def set_request_context(user_id, request_id, stage="chat"):
    return _request_context.set({"user_id": user_id, "request_id": request_id, "stage": stage})


def reset_request_context(token):
    _request_context.reset(token)


def _error_category(error_type, message):
    text = str(message).lower()
    if "超时" in text or "timeout" in text:
        return "模型超时"
    if "证书" in text or "ssl" in text:
        return "TLS/证书"
    if "密钥" in text or "拒绝访问" in text or "401" in text:
        return "模型认证"
    if "余额" in text or "额度" in text or "402" in text:
        return "模型额度"
    if "限流" in text or "429" in text:
        return "模型限流"
    if "模型名称" in text or "服务地址" in text or "配置" in text:
        return "模型配置"
    if "连接" in text or "解析模型服务地址" in text or "联网" in text:
        return "模型连接"
    if "serviceerror" in str(error_type).lower():
        return "请求/账户"
    return "服务内部错误"


def _now():
    return datetime.now(LOCAL_TZ)


def _usage_path(day):
    return _ops_root() / "usage" / f"{day}.json"


def record_usage(model, prompt_tokens=None, completion_tokens=None):
    context = _request_context.get()
    user_id = context.get("user_id")
    if not user_id:
        return
    reported = isinstance(prompt_tokens, int) and isinstance(completion_tokens, int)
    with _lock:
        now = _now()
        _prune_old_files(now)
        path = _usage_path(now.date().isoformat())
        entries = {}
        if path.is_file():
            try:
                entries = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                entries = {}
        row = entries.setdefault(user_id, {"input_tokens": 0, "output_tokens": 0, "calls": 0,
                                           "reported_calls": 0, "unreported_calls": 0, "models": {}})
        row["calls"] += 1
        if reported:
            row["input_tokens"] += max(0, prompt_tokens)
            row["output_tokens"] += max(0, completion_tokens)
            row["reported_calls"] += 1
        else:
            row["unreported_calls"] += 1
        model_row = row["models"].setdefault(str(model or "unknown"), {"input_tokens": 0, "output_tokens": 0, "calls": 0})
        model_row["calls"] += 1
        if reported:
            model_row["input_tokens"] += max(0, prompt_tokens)
            model_row["output_tokens"] += max(0, completion_tokens)
        try:
            write_json(path, entries)
        except OSError:
            logging.getLogger(__name__).exception("Unable to persist token usage metrics")


def record_error(error_type, status, message, *, stage=None, user_id=None, request_id=None):
    context = _request_context.get()
    now = _now()
    event = {
        "id": uuid.uuid4().hex,
        "timestamp": now.isoformat(),
        "username": user_id or context.get("user_id", "system"),
        "request_id": request_id or context.get("request_id", ""),
        "stage": stage or context.get("stage", "request"),
        "error_type": str(error_type)[:80],
        "category": _error_category(error_type, message),
        "status": int(status) if status is not None else 500,
        "message": str(message)[:500],
    }
    path = _ops_root() / "errors" / f"{now.date().isoformat()}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with _lock:
        _prune_old_files(now)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, ensure_ascii=False) + "\n")
    return event


def record_request_result(outcome, duration_ms):
    now = _now()
    path = _ops_root() / "requests" / f"{now.date().isoformat()}.json"
    with _lock:
        _prune_old_files(now)
        data = {}
        if path.is_file():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                data = {}
        data["total"] = int(data.get("total", 0)) + 1
        data["succeeded"] = int(data.get("succeeded", 0)) + int(outcome == "succeeded")
        data["failed"] = int(data.get("failed", 0)) + int(outcome == "failed")
        data["duration_ms_total"] = int(data.get("duration_ms_total", 0)) + max(0, int(duration_ms))
        write_json(path, data)


def request_report(days=7):
    today = _now().date()
    totals = {"total": 0, "succeeded": 0, "failed": 0, "duration_ms_total": 0}
    days = max(1, min(int(days), 90))
    for offset in range(days):
        path = _ops_root() / "requests" / f"{today - timedelta(days=offset)}.json"
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for field in totals:
            totals[field] += int(data.get(field, 0))
    totals["average_duration_ms"] = round(totals["duration_ms_total"] / totals["total"]) if totals["total"] else 0
    totals["days"] = days
    totals.pop("duration_ms_total")
    return totals


def _read_error_events(days):
    today = _now().date()
    events = []
    for offset in range(max(1, min(int(days), 90))):
        path = _ops_root() / "errors" / f"{today - timedelta(days=offset)}.jsonl"
        if not path.is_file():
            continue
        try:
            with path.open(encoding="utf-8") as stream:
                for line in stream:
                    try:
                        events.append(json.loads(line))
                    except ValueError:
                        continue
        except OSError:
            continue
    return sorted(events, key=lambda item: item.get("timestamp", ""), reverse=True)


def error_report(days=7, limit=100):
    events = _read_error_events(days)
    grouped = {}
    for event in events:
        key = f"{event.get('category', event.get('error_type', 'Error'))} · {event.get('status', 500)}"
        grouped[key] = grouped.get(key, 0) + 1
    return {"days": max(1, min(int(days), 90)), "total": len(events), "by_type": grouped,
            "errors": events[:max(1, min(int(limit), 200))]}


def usage_report(period="today"):
    now = _now()
    if period == "today":
        days = [now.date()]
    else:
        first = now.date().replace(day=1)
        days = [first + timedelta(days=i) for i in range(calendar.monthrange(now.year, now.month)[1])]
    totals = {}
    for day in days:
        path = _usage_path(day.isoformat())
        if not path.is_file():
            continue
        try:
            entries = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for username, row in entries.items():
            target = totals.setdefault(username, {"input_tokens": 0, "output_tokens": 0, "calls": 0,
                                                   "reported_calls": 0, "unreported_calls": 0})
            for field in target:
                target[field] += int(row.get(field, 0))
    return {"period": period, "start": days[0].isoformat(), "end": min(days[-1], now.date()).isoformat(), "users": totals}


def system_snapshot(queue_size=0, engine_ready=False, engine_error=None):
    process = psutil.Process(os.getpid())
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage(str(ROOT_PATH))
    return {
        "timestamp": _now().isoformat(),
        "uptime_seconds": int(_now().timestamp() - process.create_time()),
        "cpu_percent": psutil.cpu_percent(interval=None),
        "memory_percent": memory.percent,
        "memory_used_bytes": memory.used,
        "memory_total_bytes": memory.total,
        "process_memory_bytes": process.memory_info().rss,
        "disk_percent": disk.percent,
        "disk_free_bytes": disk.free,
        "queue_size": int(queue_size),
        "engine_ready": bool(engine_ready),
        "engine_error": engine_error,
    }
