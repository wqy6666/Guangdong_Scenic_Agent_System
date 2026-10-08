# -*- coding: utf-8 -*-
import time
import socket
import ssl
from urllib.parse import urlsplit
from security_controls import validate_model_endpoint
from operations_monitor import record_usage
from openai import OpenAI, APIError, BadRequestError, APITimeoutError, APIConnectionError
from logger import logger
from service_errors import LLMServiceError, LLMOutputError, strict_llm_errors

MAX_RETRY = 3


def _model_error_message(error):
    """Classify failures without exposing provider bodies, URLs or credentials."""
    causes = []
    current = error
    while current is not None and all(current is not item for item in causes):
        causes.append(current)
        current = current.__cause__
    status = getattr(error, "status_code", None)
    logger.warning(
        "LLM请求失败: type=%s status=%s causes=%s",
        type(error).__name__, status,
        ", ".join(f"{type(item).__name__}(winerror={getattr(item, 'winerror', None)},errno={getattr(item, 'errno', None)})" for item in causes),
    )
    if any(isinstance(item, PermissionError) or getattr(item, "winerror", None) == 10013 or getattr(item, "errno", None) in (13, 10013) for item in causes):
        return "后端进程被系统禁止联网。请停止旧服务，并在你自己的终端重新运行 start_web.ps1，再重试本次提问。"
    if isinstance(error, APITimeoutError):
        return "模型服务响应超时，请稍后重试；若持续超时，请检查网络或更换模型。"
    if any(isinstance(item, ssl.SSLCertVerificationError) for item in causes):
        return "模型服务的安全证书校验失败，请检查系统时间、代理或证书配置。"
    if any(isinstance(item, socket.gaierror) for item in causes):
        return "无法解析模型服务地址，请检查地址拼写和网络连接。"
    if isinstance(error, APIConnectionError):
        return "无法连接模型服务，请检查服务地址、网络或代理后重试。"
    if status in (401, 403):
        return "模型服务拒绝访问，请检查 API 密钥与权限。"
    if status == 402:
        return "模型服务账户余额不足，请检查账户余额后重试。"
    if status == 429:
        return "模型服务限流或额度不足，请稍后重试或检查账户额度。"
    if status in (400, 404):
        return "模型调用失败，请检查服务地址和模型名称。"
    if status and status >= 500:
        return "模型服务暂时不可用，请稍后重试。"
    return "模型请求失败，请检查模型配置后重试。"

# 统一输出格式约束：注入所有面向用户的生成提示词（净化器是确定性兜底，这里是概率性前置约束）
OUTPUT_FORMAT_RULE = (
    "\n【输出格式要求】以纯文本客服话术输出：禁止使用markdown符号（不要用**加粗**、不要#标题、"
    "不要`代码标记`、不要表格语法），分点直接用“1.”或“-”开头；"
    "列举推荐或筛选结果时用“为您推荐的景区有：”“符合条件的有：”等陈述句式，禁止用“有哪些”等疑问句式表达结果。\n"
)

def generate_answer_by_api(messages, llm_cfg: dict, user_query: str = "", timeout: int = 25, max_tokens: int = None,
                           *, purpose: str = "answer") -> str:
    """
    改用传入的用户独立llm_cfg配置，不再读取config全局密钥
    """
    api_key = llm_cfg.get("api_key", "")
    base_url = validate_model_endpoint(llm_cfg.get("base_url", ""))
    model_name = llm_cfg.get("model_name", "")
    verifying = purpose == "verification"
    rewriting = purpose == "rewrite"
    strict = strict_llm_errors.get() or verifying or rewriting
    options = {"max_tokens": max_tokens} if max_tokens else {}
    # These bounded stages need usable output, not a reasoning-only token budget.
    # Keep the ordinary answer's model behavior and other providers unchanged.
    if (verifying or rewriting) and urlsplit(base_url).hostname == "api.deepseek.com":
        options["extra_body"] = {"thinking": {"type": "disabled"}}

    # 每次调用动态实例client，不使用全局单例
    started = time.perf_counter()
    with OpenAI(
        api_key=api_key,
        base_url=base_url,
        **({"max_retries": 0} if verifying or rewriting else {})
    ) as client:
        last_err = None
        for retry_idx in range(MAX_RETRY):
            try:
                with client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    stream=True,
                    stream_options={"include_usage": True},
                    timeout=timeout,
                    **options
                ) as stream_response:
                    chunks = []
                    reasoning_chars = 0
                    finish_reason = None
                    prompt_tokens = completion_tokens = reasoning_tokens = None
                    for chunk in stream_response:
                        if chunk.choices:
                            choice = chunk.choices[0]
                            if choice.delta.content:
                                chunks.append(choice.delta.content)
                            reasoning_chars += len(getattr(choice.delta, "reasoning_content", None) or "")
                            if choice.finish_reason:
                                finish_reason = choice.finish_reason
                        usage = getattr(chunk, "usage", None)
                        if usage is not None:
                            prompt_tokens = getattr(usage, "prompt_tokens", None)
                            completion_tokens = usage.completion_tokens
                            reasoning_tokens = getattr(getattr(usage, "completion_tokens_details", None), "reasoning_tokens", None)
                record_usage(model_name, prompt_tokens, completion_tokens)
                full_text = "".join(chunks)
                safe_finish = finish_reason if finish_reason in (None, "stop", "length", "content_filter", "tool_calls", "insufficient_system_resource", "aborted") else "unknown"
                logger.info("LLM输出摘要: stage=%s duration_ms=%.0f finish_reason=%s content_chars=%d reasoning_chars=%d completion_tokens=%s reasoning_tokens=%s",
                            purpose, (time.perf_counter() - started) * 1000, safe_finish, len(full_text), reasoning_chars, completion_tokens, reasoning_tokens)
                if strict:
                    if finish_reason == "length":
                        raise LLMOutputError("模型输出达到长度上限，回答未完成，请重试。", "length")
                    if finish_reason not in (None, "stop"):
                        raise LLMOutputError("模型未正常完成回答，请稍后重试。", "incomplete")
                    if not full_text.strip():
                        message = ("模型只返回了思考内容，未生成正式回答，请重试。" if reasoning_chars
                                   else "模型返回了空回答，请稍后重试。")
                        raise LLMOutputError(message, "empty")
                return full_text
            except (APIError, BadRequestError) as e:
                last_err = e
                record_usage(model_name)
                if strict:
                    raise LLMServiceError(_model_error_message(e)) from None
                err_msg = str(e)
                if "System protection triggered by request burst" in err_msg or "request burst" in err_msg:
                    logger.warning("触发API限流保护")
                    return "当前大模型服务请求过于频繁，请稍后再提问。"
                logger.warning(f"LLM调用异常，重试{retry_idx+1}/{MAX_RETRY}, sleep={2 ** retry_idx}, err={err_msg}")
                time.sleep(2 ** retry_idx)
    logger.error(f"LLM调用全部重试失败 err={str(last_err)}")
    return "抱歉，大模型服务暂时繁忙，请稍后再提问。"


def build_chat_messages(user_query: str, docs: list, chat_history: list, session_ctx: dict | None = None) -> list:
    doc_text = "\n".join([d.get("chunk_text", "") for d in docs])
    messages = []
    # chat_history截断最近5轮（每轮=user+assistant共2条消息，5轮=10条；滑窗本身已保证上限）
    for his in chat_history[-10:]:
        messages.append(his)
    extra_prompt = ""
    if session_ctx:
        import json
        pref = session_ctx.get("user_preference", {})
        alias_map = session_ctx.get("session_alias_map", {})
        if pref:
            extra_prompt += f"\n【用户记忆偏好】{json.dumps(pref, ensure_ascii=False)}"
        if alias_map:
            alias_lines = [f"{k}→{v}" for k, v in alias_map.items()]
            extra_prompt += "\n【本次对话别名映射】\n" + "\n".join(alias_lines)
    sys_msg = {
        "role": "system",
        "content": f"你是广东省A级景区客服，严格使用参考资料回答，禁止编造。参考资料：\n{doc_text}{extra_prompt}{OUTPUT_FORMAT_RULE}"
    }
    messages.insert(0, sys_msg)
    messages.append({"role": "user", "content": user_query})
    return messages
