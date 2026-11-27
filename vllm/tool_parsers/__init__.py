# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.tool_parsers.abstract_tool_parser import (
    ToolParser,
    ToolParserManager,
)

__all__ = ["ToolParser", "ToolParserManager"]


"""
Register a lazy module mapping.

Example:
    ToolParserManager.register_lazy_module(
        name="qwen3_xml",
        module_path="vllm.tool_parsers.qwen3_engine_tool_parser",
        class_name="Qwen3EngineToolParser",
    )
"""


_TOOL_PARSERS_TO_REGISTER = {
    "hermes": (
        "hermes_tool_parser",
        "Hermes2ProToolParser",
    ),
    "qwen3_coder": (
        "qwen3_engine_tool_parser",
        "Qwen3EngineToolParser",
    ),
    # TODO: Qwen2.5/QwQ/Qwen3 JSON tool calls. Aliased to `hermes` for now;
    # a dedicated parser should treat the template's newline before and
    # between `<tool_call>` blocks as framing, as the Rust frontend does.
    "qwen3_json": (
        "hermes_tool_parser",
        "Hermes2ProToolParser",
    ),
    "qwen3_xml": (
        "qwen3_engine_tool_parser",
        "Qwen3EngineToolParser",
    ),
}


def register_lazy_tool_parsers():
    for name, (file_name, class_name) in _TOOL_PARSERS_TO_REGISTER.items():
        module_path = f"vllm.tool_parsers.{file_name}"
        ToolParserManager.register_lazy_module(name, module_path, class_name)


register_lazy_tool_parsers()
