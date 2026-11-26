"""Zero-dependency HTTP adapters for OpenAI, Anthropic, Gemini, and Ollama with real token cost accounting."""
import json, os, re, urllib.request
from typing import Any
from .syntax import AgentDecl, TigoError, TypeRef

# shortcut: hardcoded per-1M-token (input, output) USD rates; update or read from a pricing table as vendors change rates.
RATES_PER_M: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "claude-3-5-haiku": (0.80, 4.00),
    "claude-3-5-sonnet": (3.00, 15.00),
    "claude-sonnet": (3.00, 15.00),
    "gemini-2.0-flash": (0.10, 0.40),
    "gemini-2.5-flash": (0.15, 0.60),
    "gemini-2.5-pro": (1.25, 10.00),
}
DEFAULT_STEP_COST = 0.002


def parse_model_spec(spec: str) -> tuple[str, str]:
    if ":" in spec:
        prov, model = spec.split(":", 1)
        if prov in ("openai", "anthropic", "gemini", "ollama"):
            return prov, model
    if spec.startswith("claude-"):
        return "anthropic", spec
    if spec.startswith("gemini-"):
        return "gemini", spec
    return "openai", spec


def compute_cost(provider: str, model: str, in_toks: int, out_toks: int) -> float:
    if provider == "ollama":
        return 0.0
    if in_toks == 0 and out_toks == 0:
        return DEFAULT_STEP_COST
    for prefix, (in_r, out_r) in RATES_PER_M.items():
        if model.startswith(prefix):
            return round((in_toks * in_r + out_toks * out_r) / 1_000_000, 6)
    return round((in_toks * 1.0 + out_toks * 3.0) / 1_000_000, 6)


def _post_json(url: str, payload: dict, headers: dict[str, str]) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _parse_response_text(text: str, expected: TypeRef | None) -> Any:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n|\n```$", "", text).strip()
    if expected and expected.base != "string":
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text  # let validate_type trigger the auto-repair loop
    return text


def call_provider(
    ag: AgentDecl,
    history: list[dict],
    schema: dict,
    expected: TypeRef | None,
    post_fn: Any = _post_json,
) -> tuple[Any, float] | None:
    """Call configured model (with fallback). Returns (value, cost_usd), or None if no credentials for offline synth."""
    models = [ag.config["model"]]
    if "fallback" in ag.config:
        models.append(ag.config["fallback"])

    sys_msg = ag.config.get("system", "You are a helpful AI agent.")
    if expected and expected.base != "string":
        sys_msg += f"\nRespond ONLY with JSON matching this schema: {json.dumps(schema)}"

    last_err: Exception | None = None
    attempted = False
    for spec in models:
        prov, model = parse_model_spec(spec)
        try:
            if prov == "openai":
                key = os.environ.get("OPENAI_API_KEY") or os.environ.get("TIGO_API_KEY")
                if not key:
                    continue
                attempted = True
                url = os.environ.get("TIGO_API_URL", "https://api.openai.com/v1/chat/completions")
                data = post_fn(
                    url,
                    {"model": model, "messages": [{"role": "system", "content": sys_msg}] + history, "temperature": 0},
                    {"Authorization": f"Bearer {key}"},
                )
                text = data["choices"][0]["message"]["content"]
                u = data.get("usage", {})
                cost = compute_cost(prov, model, u.get("prompt_tokens", 0), u.get("completion_tokens", 0))
                return _parse_response_text(text, expected), cost

            if prov == "anthropic":
                key = os.environ.get("ANTHROPIC_API_KEY")
                if not key:
                    continue
                attempted = True
                msgs = [{"role": "user" if m["role"] == "tool" else m["role"], "content": m["content"]} for m in history]
                data = post_fn(
                    "https://api.anthropic.com/v1/messages",
                    {"model": model, "max_tokens": 2048, "system": sys_msg, "messages": msgs},
                    {"x-api-key": key, "anthropic-version": "2023-06-01"},
                )
                text = data["content"][0]["text"]
                u = data.get("usage", {})
                cost = compute_cost(prov, model, u.get("input_tokens", 0), u.get("output_tokens", 0))
                return _parse_response_text(text, expected), cost

            if prov == "gemini":
                key = os.environ.get("GEMINI_API_KEY")
                if not key:
                    continue
                attempted = True
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
                contents = [
                    {"role": "model" if m["role"] == "assistant" else "user", "parts": [{"text": m["content"]}]}
                    for m in history
                ]
                data = post_fn(
                    url,
                    {"systemInstruction": {"parts": [{"text": sys_msg}]}, "contents": contents},
                    {},
                )
                text = data["candidates"][0]["content"]["parts"][0]["text"]
                u = data.get("usageMetadata", {})
                cost = compute_cost(prov, model, u.get("promptTokenCount", 0), u.get("candidatesTokenCount", 0))
                return _parse_response_text(text, expected), cost

            if prov == "ollama":
                attempted = True
                host = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
                data = post_fn(
                    f"{host}/api/chat",
                    {"model": model, "messages": [{"role": "system", "content": sys_msg}] + history, "stream": False},
                    {},
                )
                text = data["message"]["content"]
                return _parse_response_text(text, expected), 0.0
        except Exception as err:
            last_err = err

    if attempted:
        raise TigoError(f"LLM request failed across models {models}: {last_err}")
    return None
