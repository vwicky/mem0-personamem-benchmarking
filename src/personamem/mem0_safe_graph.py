"""
Safe graph entity parsing patch for Mem0.

Some LLM tool-call payloads returned during graph extraction contain malformed
entity items (for example {"": None} or missing "entity_type"), which can
trigger noisy KeyError logs and reduce extraction quality for that turn.

This patch monkey-patches MemoryGraph._retrieve_nodes_from_data to:
- validate each extracted entity item,
- skip malformed items instead of raising,
- keep processing the remaining valid entities.

Apply by importing this module before creating Memory:
    import mem0_safe_graph  # noqa: F401
"""

from __future__ import annotations

from typing import Any, Dict
import threading


_MALFORMED_ENTITY_COUNT = 0
_MALFORMED_ENTITY_LOCK = threading.Lock()


def get_malformed_entity_skip_count() -> int:
    """Return total number of skipped malformed graph entities."""
    with _MALFORMED_ENTITY_LOCK:
        return _MALFORMED_ENTITY_COUNT


def _patch_mem0_graph_memory() -> None:
    import mem0.memory.graph_memory as gm

    MemoryGraph = gm.MemoryGraph
    if getattr(MemoryGraph, "_mem0_safe_graph_patched", False):
        return

    def _safe_retrieve_nodes_from_data(self, data, filters):
        """Robust entity extraction that tolerates malformed tool-call entries."""
        tools = [gm.EXTRACT_ENTITIES_TOOL]
        if self.llm_provider in ["azure_openai_structured", "openai_structured"]:
            tools = [gm.EXTRACT_ENTITIES_STRUCT_TOOL]

        search_results = self.llm.generate_response(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a smart assistant who understands entities and their "
                        f"types in a given text. If user message contains self reference "
                        f"such as 'I', 'me', 'my' etc. then use {filters['user_id']} as "
                        "the source entity. Extract all the entities from the text. "
                        "***DO NOT*** answer the question itself if the given text is a question."
                    ),
                },
                {"role": "user", "content": data},
            ],
            tools=tools,
        )

        entity_type_map: Dict[str, str] = {}
        malformed_count = 0

        for tool_call in (search_results or {}).get("tool_calls", []):
            if tool_call.get("name") != "extract_entities":
                continue

            args: Dict[str, Any] = tool_call.get("arguments") or {}
            entities = args.get("entities") or []
            if not isinstance(entities, list):
                malformed_count += 1
                continue

            for item in entities:
                if not isinstance(item, dict):
                    malformed_count += 1
                    continue

                entity = item.get("entity")
                entity_type = item.get("entity_type")
                if not isinstance(entity, str) or not entity.strip():
                    malformed_count += 1
                    continue
                if not isinstance(entity_type, str) or not entity_type.strip():
                    malformed_count += 1
                    continue

                entity_type_map[entity] = entity_type

        if malformed_count:
            global _MALFORMED_ENTITY_COUNT
            with _MALFORMED_ENTITY_LOCK:
                _MALFORMED_ENTITY_COUNT += malformed_count
            gm.logger.warning(
                "Skipped %s malformed graph entities from LLM tool-calls.",
                malformed_count,
            )

        entity_type_map = {
            k.lower().replace(" ", "_"): v.lower().replace(" ", "_")
            for k, v in entity_type_map.items()
        }
        gm.logger.debug("Entity type map: %s", entity_type_map)
        return entity_type_map

    MemoryGraph._retrieve_nodes_from_data = _safe_retrieve_nodes_from_data
    MemoryGraph._mem0_safe_graph_patched = True


_patch_mem0_graph_memory()

