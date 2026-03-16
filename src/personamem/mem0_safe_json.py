"""
Safe JSON parsing for Mem0 LLM responses that may be truncated or malformed.

Mem0 parses LLM output as JSON; when the model hits max_tokens or produces
invalid JSON (e.g. unterminated string), parsing fails and that user gets no memories.
This module adds a repair step: try normal parse, on failure attempt to fix
truncated JSON and retry, then fall back to empty result.

Apply the patch by importing this module before creating Memory:
    import mem0_safe_json  # noqa: F401
    from mem0 import Memory
"""

from __future__ import annotations

import json
import re


def _repair_truncated_memory_json(raw: str) -> str:
    """
    Heuristic repair for truncated JSON of the form {"memory": [{"id": ..., "text": "...", "event": ...}, ...]}.
    - Close an unterminated string (add "), then close any open arrays/objects.
    """
    s = raw.strip()
    if not s:
        return "{}"
    # If it looks like we're inside an unclosed string (odd number of unescaped " in the tail),
    # try to close it and the structure.
    # Find last position that could be start of truncated content (e.g. after last complete "event": "ADD"}
    # Simple approach: append closing delimiters to try to get valid JSON.
    # Try to find the last complete object in the "memory" array: ..., "event": "ADD"} or "NONE"}
    # Find last complete memory entry: ends with "event": "X"}
    last_complete = max(
        ((s.rfind(suf), suf) for suf in (
            '"event": "ADD"}', '"event": "UPDATE"}', '"event": "DELETE"}', '"event": "NONE"}'
        ) if s.rfind(suf) != -1),
        key=lambda x: x[0],
        default=(-1, None),
    )
    if last_complete[0] != -1:
        idx, suf = last_complete
        end = idx + len(suf)
        truncated = s[:end] + "]}"  # close "memory" array and root object
        try:
            parsed = json.loads(truncated)
            if isinstance(parsed, dict) and "memory" in parsed:
                return truncated
        except json.JSONDecodeError:
            pass
    # Fallback: try closing an unterminated string at the end and then close brackets
    # Count open braces/brackets
    open_braces = 0
    open_brackets = 0
    in_string = False
    escape = False
    i = 0
    while i < len(s):
        c = s[i]
        if escape:
            escape = False
            i += 1
            continue
        if c == '\\' and in_string:
            escape = True
            i += 1
            continue
        if in_string:
            if c == '"':
                in_string = False
            i += 1
            continue
        if c == '"':
            in_string = True
            i += 1
            continue
        if c == '{':
            open_braces += 1
        elif c == '}':
            open_braces -= 1
        elif c == '[':
            open_brackets += 1
        elif c == ']':
            open_brackets -= 1
        i += 1
    # If we might be inside a string (in_string True), add a quote to close it
    if in_string:
        s = s + '"'
    # Close any open brackets ( ] then ]...) then braces ( } then }...)
    s = s + "]" * max(0, open_brackets) + "}" * max(0, open_braces)
    return s


def safe_parse_memory_actions_response(response: str) -> dict:
    """
    Parse Mem0 memory-actions JSON from LLM response. Uses remove_code_blocks,
    then json.loads; on JSONDecodeError (e.g. unterminated string from truncation),
    attempts repair and retry. Returns {} on total failure.
    """
    if not response or not response.strip():
        return {}
    # Use remove_code_blocks from mem0 (avoid circular import by patching at runtime)
    from mem0.memory.utils import remove_code_blocks
    text = remove_code_blocks(response).strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    repaired = _repair_truncated_memory_json(text)
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        return {}


def _patch_mem0_memory_main():
    """Patch Memory._add_to_vector_store to use safe_parse when json.loads fails on memory-actions JSON."""
    import mem0.memory.main as main_mod

    Memory = main_mod.Memory
    if getattr(Memory, "_mem0_safe_json_patched", False):
        return

    _orig_add = Memory._add_to_vector_store

    def _wrapper(self, messages, metadata, filters, infer):
        _json_orig = main_mod.json
        class _SafeJson:
            @staticmethod
            def loads(s, *a, **k):
                if isinstance(s, str) and '"memory"' in s and '"event"' in s:
                    try:
                        return json.loads(s, *a, **k)
                    except json.JSONDecodeError:
                        return safe_parse_memory_actions_response(s)
                return json.loads(s, *a, **k)

            def __getattr__(self, name):
                return getattr(json, name)

        main_mod.json = _SafeJson()
        try:
            return _orig_add(self, messages, metadata, filters, infer)
        finally:
            main_mod.json = _json_orig

    Memory._add_to_vector_store = _wrapper
    Memory._mem0_safe_json_patched = True


# Apply patch on import
_patch_mem0_memory_main()
