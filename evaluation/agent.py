"""ReAct evaluation with skill, document, and Python tools."""
from __future__ import annotations

import ast
from dataclasses import dataclass
import json
import logging
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from evaluation.skills import SkillBundle, SkillFile

_RUN_PYTHON_TIMEOUT_SECONDS = 30

@dataclass(frozen=True)
class QwenAgentReactResult:
    response: str
    conversation: list[dict[str, Any]]
    n_turns: int
    metadata: dict[str, Any]
    solution_code: str = ""
    solution_path: str = ""
    final_response: str = ""

@dataclass(frozen=True)
class CodeToolContext:
    code_prompt: str
    entry_point: str
    public_tests: str
    work_dir: str
    timeout_seconds: int = _RUN_PYTHON_TIMEOUT_SECONDS

@dataclass(frozen=True)
class QAContext:
    question: str
    context: str
    max_doc_chars: int = 5000

class SkillFileStore:
    def __init__(self, root: Path | None, files: dict[str, SkillFile]) -> None:
        self.root = root
        self.files = dict(sorted(files.items()))
        self.calls: list[dict[str, Any]] = []

    @classmethod
    def from_skill(cls, skill: str | SkillBundle) -> "SkillFileStore":
        if isinstance(skill, SkillBundle):
            return cls(skill.root, skill.files)
        item = SkillFile(
            path="skill.md",
            content=skill,
            title=_extract_title(skill, "skill.md"),
            description=_extract_description(skill),
        )
        return cls(None, {"skill.md": item})

    def list_files(self) -> list[dict[str, str]]:
        rows = [
            {
                "path": item.path,
                "title": item.title,
                "description": item.description,
            }
            for item in self.files.values()
        ]
        self.calls.append({"tool": "list_skill_files", "observation": rows})
        return rows

    def read_file(self, path: str) -> dict[str, str]:
        clean = self._validate_path(path)
        if clean == "__path_outside_skill_directory__":
            result = {"error": "path_outside_skill_directory", "path": str(path or "")}
            self.calls.append({"tool": "read_skill_file", "args": {"path": path}, "observation": result})
            return result
        if clean not in self.files:
            result = {"error": "unknown_skill_file", "path": clean}
            self.calls.append({"tool": "read_skill_file", "args": {"path": path}, "observation": result})
            return result
        item = self.files[clean]
        if self._was_successfully_read(clean):
            result = {
                "path": item.path,
                "already_read": True,
                "message": (
                    "This skill file was already read in this task. Use the previously returned "
                    "content and provide the final answer; do not call read_skill_file on this "
                    "path again."
                ),
            }
        else:
            result = {"path": item.path, "content": item.content}
        self.calls.append({"tool": "read_skill_file", "args": {"path": path}, "observation": result})
        return result

    def _was_successfully_read(self, path: str) -> bool:
        for call in self.calls:
            if call.get("tool") != "read_skill_file":
                continue
            obs = call.get("observation") or {}
            if obs.get("path") == path and "content" in obs and not obs.get("error"):
                return True
        return False

    def search_files(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        terms = _terms(query)
        scored: list[tuple[float, SkillFile]] = []
        for item in self.files.values():
            score = _score_file(item, terms)
            if score > 0:
                scored.append((score, item))
        scored.sort(key=lambda pair: (-pair[0], pair[1].path))
        rows = [
            {
                "path": item.path,
                "title": item.title,
                "description": item.description,
                "score": round(score, 3),
            }
            for score, item in scored[: max(1, int(top_k))]
        ]
        self.calls.append(
            {
                "tool": "search_skill_files",
                "args": {"query": query, "top_k": top_k},
                "observation": rows,
            }
        )
        return rows

    def manifest_text(self) -> str:
        lines = []
        for item in self.files.values():
            suffix = f" — {item.description}" if item.description else ""
            lines.append(f"- `{item.path}`: {item.title}{suffix}")
        return "\n".join(lines)

    def _validate_path(self, path: str) -> str:
        raw = str(path or "").strip().replace("\\", "/")
        if not raw or raw.startswith("/") or ".." in Path(raw).parts:
            return "__path_outside_skill_directory__"
        return raw

class QAContextStore:
    def __init__(self, context: QAContext) -> None:
        self.context = context
        self.docs = _parse_qa_docs(context.context)
        self.calls: list[dict[str, Any]] = []

    def list_titles(self) -> list[dict[str, Any]]:
        rows = [
            {
                "title": doc["title"],
                "sentence_count": len(doc["sentences"]),
                "preview": doc["content"][:240],
            }
            for doc in self.docs
        ]
        self.calls.append({"tool": "list_context_titles", "observation": rows})
        return rows

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        terms = _terms(query)
        scored: list[tuple[float, dict[str, Any]]] = []
        for doc in self.docs:
            haystack = f"{doc['title']} {doc['content']}".lower()
            score = sum(1.0 for term in terms if term in haystack)
            if any(term in doc["title"].lower() for term in terms):
                score += 2.0
            if score > 0:
                scored.append((score, doc))
        scored.sort(key=lambda pair: (-pair[0], pair[1]["title"]))
        rows = [
            {
                "title": doc["title"],
                "score": round(score, 3),
                "preview": doc["content"][:400],
            }
            for score, doc in scored[: max(1, int(top_k))]
        ]
        self.calls.append({"tool": "search_context", "args": {"query": query, "top_k": top_k}, "observation": rows})
        return rows

    def read_doc(self, title: str) -> dict[str, Any]:
        query = str(title or "").strip().lower()
        doc = None
        for candidate in self.docs:
            if candidate["title"].lower() == query:
                doc = candidate
                break
        if doc is None:
            for candidate in self.docs:
                if query and query in candidate["title"].lower():
                    doc = candidate
                    break
        if doc is None:
            result = {"error": "unknown_context_title", "title": str(title or "")}
        else:
            content = doc["content"][: max(1, int(self.context.max_doc_chars))]
            result = {"title": doc["title"], "content": content, "sentence_count": len(doc["sentences"])}
        self.calls.append({"tool": "read_context_doc", "args": {"title": title}, "observation": result})
        return result

    def has_context_call(self) -> bool:
        return any(call.get("tool") in {"search_context", "read_context_doc"} for call in self.calls)

def build_qwen_agent_llm_config(
    *,
    max_output_tokens: int,
    target_config: dict[str, Any] | None = None,
    seed: int | None = None,
) -> Any:
    from evaluation.model import is_gemma4_model

    if target_config is None:
        from evaluation.model import get_target_config_snapshot

        target_config = get_target_config_snapshot()
    cfg = target_config
    generate_cfg: dict[str, Any] = {
        "use_raw_api": bool(cfg.get("use_raw_api", True)),
        "max_tokens": min(int(max_output_tokens), int(cfg["max_tokens"])),
        "request_timeout": cfg["timeout_seconds"],
        "max_retries": 5,
        "max_input_tokens": max(1, int(cfg["truncate_prompt_tokens"]) - 2048)
        if cfg["truncate_prompt_tokens"]
        else 0,
    }
    extra_body: dict[str, Any] = {}
    if seed is not None:
        generate_cfg["seed"] = seed
    if cfg.get("enable_thinking") is not None:
        extra_body["chat_template_kwargs"] = {"enable_thinking": bool(cfg["enable_thinking"])}
    for key in ("temperature", "top_p", "top_k"):
        if cfg.get(key) is not None:
            generate_cfg[key] = cfg[key]
    if cfg.get("min_p") is not None:
        extra_body["min_p"] = cfg["min_p"]
    if extra_body:
        generate_cfg["extra_body"] = extra_body
    if cfg.get("provider") == "anthropic" or is_gemma4_model(cfg["deployment"]):
        return _build_target_llm(cfg["deployment"], generate_cfg)
    return {
        "model": cfg["deployment"],
        "model_type": os.environ.get("QWEN_AGENT_MODEL_TYPE", "oai"),
        "model_server": cfg["base_url"],
        "api_key": cfg["api_key"] or "EMPTY",
        "generate_cfg": generate_cfg,
    }

def _build_target_llm(model: str, generate_cfg: dict[str, Any]) -> Any:
    from qwen_agent.llm.base import BaseChatModel
    from qwen_agent.llm.schema import ASSISTANT, Message

    class _TargetLLM(BaseChatModel):
        def __init__(self) -> None:
            super().__init__({
                "model": model,
                "model_type": "evaluation",
                "generate_cfg": generate_cfg,
            })

        @staticmethod
        def _payload(messages: list[Any]) -> list[dict[str, Any]]:
            return [
                message.model_dump(exclude_none=True) if hasattr(message, "model_dump") else dict(message)
                for message in messages
            ]

        def _chat_no_stream(self, messages: list[Any], generate_cfg: dict[str, Any]) -> list[Any]:
            from evaluation.model import chat_target_messages
            from evaluation.model import is_gemma4_model

            if is_gemma4_model(model):
                from evaluation.model import chat_target_messages as gemma_chat

                payload = self._payload(messages)
                for message in payload:
                    message.pop("extra", None)
                message, usage = gemma_chat(
                    messages=payload,
                    max_completion_tokens=int(generate_cfg.get("max_tokens", 4096)),
                    timeout=int(generate_cfg.get("request_timeout", 900)),
                    seed=generate_cfg.get("seed"),
                    tools=self._eval_tools,
                    tool_choice="auto",
                    return_message=True,
                )
                text = message.content or ""
                if message.tool_calls:
                    call = message.tool_calls[0].function
                    text += f"\nAction: {call.name}\nAction Input: {call.arguments}\n"
                usage["native_message"] = message.metadata["choice0"]["message"]
                return [Message(role=ASSISTANT, content=text,
                                reasoning_content=usage.get("reasoning_content"), extra=usage)]
            text, _ = chat_target_messages(
                messages=self._payload(messages),
                max_completion_tokens=int(generate_cfg.get("max_tokens", 4096)),
                timeout=int(generate_cfg.get("request_timeout", 900)),
            )
            return [Message(role=ASSISTANT, content=text)]

        def _chat_stream(
            self,
            messages: list[Any],
            delta_stream: bool,
            generate_cfg: dict[str, Any],
        ) -> Any:
            yield self._chat_no_stream(messages, generate_cfg)

        def _chat_with_functions(
            self,
            messages: list[Any],
            functions: list[dict[str, Any]],
            stream: bool,
            delta_stream: bool,
            generate_cfg: dict[str, Any],
            lang: str,
        ) -> Any:
            if stream:
                return self._chat_stream(messages, delta_stream, generate_cfg)
            return self._chat_no_stream(messages, generate_cfg)

    return _TargetLLM()

def run_qwen_agent_react(
    *,
    system: str,
    user: str,
    skill_resource: str | SkillBundle,
    max_output_tokens: int,
    code_context: CodeToolContext | None = None,
    qa_context: QAContext | None = None,
    agent_cls: Any | None = None,
    max_agent_steps: int = 4,
    max_no_tool_retries: int = 3,
    max_context_tool_retries: int = 2,
    max_solution_repair_retries: int = 1,
    target_config: dict[str, Any] | None = None,
    seed: int = 42,
) -> QwenAgentReactResult:
    from evaluation.model import ContextWindowExceeded

    _quiet_qwen_agent_logging()
    store = SkillFileStore.from_skill(skill_resource)
    qa_store = QAContextStore(qa_context) if qa_context is not None else None
    use_native_qwen_tools = agent_cls is None
    solution_tool: _WriteCodeSolutionTool | None = None
    public_tests_tool: _RunPublicTestsTool | None = None
    solution_kind = ""
    if code_context is not None:
        solution_tool = _WriteCodeSolutionTool(code_context)
        public_tests_tool = _RunPublicTestsTool(code_context, solution_tool)
        solution_kind = "bigcodebench"
    tools = _build_tools(
        store,
        code_context,
        qa_store,
        native_qwen_tools=use_native_qwen_tools,
        solution_tool=solution_tool,
        public_tests_tool=public_tests_tool,
    )
    bot_cls = agent_cls or _load_react_chat_cls()
    last: list[dict[str, Any]] = []
    manifest = store.manifest_text()
    default_skill_path = _default_skill_path(store)
    initial_prompt = _with_required_skill_tool_instruction(
        user,
        manifest,
        default_skill_path,
        require_solution_tool=solution_tool is not None,
        solution_kind=solution_kind,
        require_public_tests=public_tests_tool is not None,
        require_qa_context=qa_store is not None,
    )
    conversation: list[dict[str, Any]] = [{"role": "user", "content": initial_prompt}]
    no_tool_retries = 0
    context_tool_retries = 0
    solution_repair_retries = 0
    final_answer_retries = 0
    final_code_block_solution_attempts = 0
    forced_default_skill_read = False
    final_response_ready = False
    final_response = ""
    final_response_source = "invalid"
    attempt_count = 0
    context_window_error = ""
    retry_stages: list[str] = []
    max_agent_steps = max(1, int(max_agent_steps))
    while True:
        attempt_count += 1
        attempt_max_output_tokens = (
            min(int(max_output_tokens), 1024)
            if retry_stages and retry_stages[-1] == "final_answer"
            else int(max_output_tokens)
        )
        bot = bot_cls(
            llm=build_qwen_agent_llm_config(
                max_output_tokens=attempt_max_output_tokens,
                target_config=target_config,
                seed=seed,
            ),
            system_message=system,
            function_list=tools,
            name="skill_react",
            description="Skill evaluation with ReAct",
            max_llm_calls_per_run=max_agent_steps,
        )
        _quiet_qwen_agent_logging()
        attempt_last: list[dict[str, Any]] = []
        try:
            for response in bot.run(messages=conversation):
                attempt_last = _normalise_messages(response)
        except ContextWindowExceeded as exc:
            if not attempt_last and not any(message["role"] == "assistant" for message in conversation):
                raise
            context_window_error = str(exc)
        if attempt_last or not context_window_error:
            last = attempt_last
        files_read = _files_read(store.calls)
        final_text = _final_text(last)
        model_responses = getattr(bot, "_eval_model_responses", [])
        if (final_text and attempt_last) or model_responses:
            message = {"role": "assistant", "content": final_text}
            if model_responses:
                message["extra"] = {"model_responses": model_responses}
                reasoning = "\n".join(
                    response.get("reasoning_content") or "" for response in model_responses
                )
                if getattr(bot, "_eval_gemma4", False):
                    message["reasoning_content"] = reasoning
            conversation.append(message)
        answer_text = final_text
        if getattr(bot, "_eval_gemma4", False) and solution_tool is None:
            answer_text = ""
            if model_responses:
                native = model_responses[-1]
                extra = native.get("extra") or {}
                if (extra.get("thinking_complete") and not extra.get("tool_calls")
                        and not bot._detect_tool(native["content"])[0]):
                    answer_text = native["content"]
        final_response_ready, final_response, final_response_source = _normalise_final_response(
            answer_text,
            system=system,
            user=user,
            solution_required=solution_tool is not None,
        )
        if context_window_error:
            break
        context_ready = qa_store is None or qa_store.has_context_call()
        public_tests_ready = public_tests_tool is None or public_tests_tool.current_solution_ok()
        solution_ready = solution_tool is None or (bool(solution_tool.solution_code) and public_tests_ready)
        if files_read and solution_tool is not None and public_tests_tool is not None and not solution_ready:
            if _submit_final_python_block_if_present(final_text, solution_tool, public_tests_tool):
                final_code_block_solution_attempts += 1
                public_tests_ready = public_tests_tool is None or public_tests_tool.current_solution_ok()
                solution_ready = bool(solution_tool.solution_code) and public_tests_ready
        if (
            solution_tool is not None
            and public_tests_tool is not None
            and solution_tool.solution_code
            and public_tests_tool.tested_solution_code != solution_tool.solution_code
        ):
            public_tests_tool.call({})
            public_tests_ready = public_tests_tool.current_solution_ok()
            solution_ready = bool(solution_tool.solution_code) and public_tests_ready
        if files_read and context_ready and solution_ready and final_response_ready:
            break
        if model_responses and (model_responses[-1].get("extra") or {}).get("finish_reason") == "length":
            break
        if not files_read:
            if no_tool_retries >= max(0, int(max_no_tool_retries)) and not forced_default_skill_read:
                forced_default_skill_read = True
                observation = store.read_file(default_skill_path)
                retry_prompt = _with_forced_default_skill_read_instruction(
                    default_skill_path,
                    observation,
                    require_solution_tool=solution_tool is not None,
                    solution_kind=solution_kind,
                    require_public_tests=public_tests_tool is not None,
                )
                conversation.append({"role": "user", "content": retry_prompt})
                retry_stages.append("forced_default_skill_read")
                continue
            if no_tool_retries >= max(0, int(max_no_tool_retries)):
                break
            no_tool_retries += 1
            retry_prompt = _with_no_tool_retry_instruction(
                manifest,
                default_skill_path,
                require_solution_tool=solution_tool is not None,
                solution_kind=solution_kind,
                require_public_tests=public_tests_tool is not None,
            )
            conversation.append({"role": "user", "content": retry_prompt})
            retry_stages.append("no_tool")
            continue
        if qa_store is not None and not context_ready:
            if context_tool_retries >= max(0, int(max_context_tool_retries)):
                break
            context_tool_retries += 1
            retry_prompt = _with_context_tool_retry_instruction(qa_store.context.question)
            conversation.append({"role": "user", "content": retry_prompt})
            retry_stages.append("context_tool")
            continue
        if solution_tool is not None and not solution_ready:
            if solution_repair_retries >= max(0, int(max_solution_repair_retries)):
                break
            solution_repair_retries += 1
            retry_prompt = _with_solution_retry_instruction(
                solution_tool,
                public_tests_tool=public_tests_tool,
                solution_kind=solution_kind,
            )
            conversation.append({"role": "user", "content": retry_prompt})
            retry_stages.append("solution")
            continue
        if not final_response_ready:
            if solution_tool is None and final_answer_retries < 2:
                final_answer_retries += 1
                conversation.append({"role": "user", "content": _with_final_answer_retry_instruction()})
                retry_stages.append("final_answer")
                continue
            break
        break
    final_text = _final_text(last)
    tool_calls = _serialise_tool_calls(store.calls)
    if qa_store is not None:
        tool_calls.extend(_serialise_tool_calls(qa_store.calls))
    if solution_tool is not None:
        tool_calls.extend(solution_tool.serialised_calls())
    if public_tests_tool is not None:
        tool_calls.extend(public_tests_tool.serialised_calls())
    metadata = {
        "agent_backend": "qwen_agent_react",
        "skill_source_mode": "file_structure" if isinstance(skill_resource, SkillBundle) else "monolithic",
        "files_read": _files_read(store.calls),
        "tool_calls": tool_calls,
        "required_skill_tool_used": bool(store.calls),
        "required_skill_file_read": bool(_files_read(store.calls)),
        "no_tool_retries": no_tool_retries,
        "context_tool_retries": context_tool_retries,
        "final_answer_retries": final_answer_retries,
        "final_response_ready": final_response_ready,
        "final_response_source": final_response_source,
        "final_response_chars": len(final_response),
        "forced_default_skill_read": forced_default_skill_read,
        "final_code_block_solution_attempts": final_code_block_solution_attempts,
        "max_agent_steps": max_agent_steps,
        "seed": seed,
        "attempt_count": attempt_count,
        "retry_stages": retry_stages,
        "tool_call_count": len(tool_calls),
        "agent_step_limit_hit": (
            _react_trace_action_count(final_text) >= max_agent_steps and not final_response_ready
        ),
    }
    if solution_tool is not None:
        metadata.update(
            {
                "solution_written": bool(solution_tool.solution_code),
                "solution_ast_ok": bool(solution_tool.solution_code),
                "solution_path": solution_tool.solution_path,
                "solution_attempts": len(solution_tool.calls),
                "solution_repair_retries": solution_repair_retries,
                "solution_last_error": solution_tool.last_error,
            }
        )
    if public_tests_tool is not None:
        metadata.update(
            {
                "test_feedback_mode": public_tests_tool.feedback_mode,
                "test_feedback_ok": public_tests_tool.current_solution_ok(),
                "public_tests_ok": public_tests_tool.public_tests_ok,
                "public_tests_available": public_tests_tool.public_tests_available,
                "public_tests_attempts": len(public_tests_tool.calls),
                "public_tests_last_error": public_tests_tool.last_error,
                "smoke_check_ok": public_tests_tool.smoke_check_ok,
            }
        )
    if qa_store is not None:
        metadata.update(
            {
                "context_tool_used": qa_store.has_context_call(),
                "context_title_count": len(qa_store.docs),
                "context_searches": sum(1 for call in qa_store.calls if call.get("tool") == "search_context"),
                "context_docs_read": sum(1 for call in qa_store.calls if call.get("tool") == "read_context_doc"),
            }
        )
    if context_window_error:
        metadata.update(context_window_exhausted=True, context_window_error=context_window_error)
    return QwenAgentReactResult(
        response=final_text,
        conversation=conversation,
        n_turns=len(tool_calls),
        metadata=metadata,
        solution_code=solution_tool.solution_code if solution_tool is not None else "",
        solution_path=solution_tool.solution_path if solution_tool is not None else "",
        final_response=final_response,
    )

def _quiet_qwen_agent_logging() -> None:
    for name in ("qwen_agent", "qwen_agent.llm", "qwen_agent.llm.base", "qwen_agent_logger"):
        logging.getLogger(name).setLevel(logging.WARNING)

def _with_required_skill_tool_instruction(
    user: str,
    manifest: str,
    default_skill_path: str,
    *,
    require_solution_tool: bool = False,
    solution_kind: str = "",
    require_public_tests: bool = False,
    require_qa_context: bool = False,
) -> str:
    solution_part = ""
    if require_solution_tool:
        solution_part = _solution_tool_instruction(solution_kind, require_public_tests=require_public_tests)
    context_part = ""
    if require_qa_context:
        context_part = (
            "\n\n## Required HotpotQA Context Tools\n"
            "After reading at least one skill file, use search_context or read_context_doc on the provided "
            "context before giving the final answer. HotpotQA is multi-hop: inspect evidence from the "
            "context tools and return the short answer in <answer>...</answer>. Do not answer from memory."
        )
    return (
        f"{user.strip()}\n\n"
        "## Required ReAct Skill-Tool Use\n"
        "Available skill file paths:\n"
        f"{manifest}\n\n"
        "You must call read_skill_file on at least one skill file before giving the final answer. "
        "You may use list_skill_files or search_skill_files first, but list/search alone is insufficient. "
        f"If you are unsure which file is relevant, first read `{default_skill_path}`. "
        "Call read_skill_file with one of the exact paths listed above. "
        "Use exactly this tool-call format: Action: read_skill_file, then Action Input: {\"path\": \"...\"}. "
        "Immediately after reading the skill and before calling any other tool, explain in a brief Thought "
        "whether it is useful for the current problem and which specific parts you will use. Ignore irrelevant "
        "parts, then solve the task. "
        "Do not put a tool name inside <answer> tags. Do not answer directly without reading a skill file."
        f"{context_part}"
        f"{solution_part}"
    )

def _with_no_tool_retry_instruction(
    manifest: str,
    default_skill_path: str,
    *,
    require_solution_tool: bool = False,
    solution_kind: str = "",
    require_public_tests: bool = False,
) -> str:
    solution_part = ""
    if require_solution_tool:
        solution_part = _solution_retry_suffix(solution_kind, require_public_tests=require_public_tests)
    return (
        "Available skill file paths:\n"
        f"{manifest}\n\n"
        "Previous attempt did not read any skill file. "
        "Your next assistant message must call read_skill_file before any final answer. "
        "Do not answer the question yet. Do not explain that you should use a tool; use the tool. "
        "Use this exact first action if no other file is clearly better:\n"
        "Action: read_skill_file\n"
        f"Action Input: {{\"path\": \"{default_skill_path}\"}}"
        f"{solution_part}"
    )

def _with_forced_default_skill_read_instruction(
    default_skill_path: str,
    observation: dict[str, str],
    *,
    require_solution_tool: bool = False,
    solution_kind: str = "",
    require_public_tests: bool = False,
) -> str:
    content = observation.get("content") or ""
    solution_part = ""
    if require_solution_tool:
        solution_part = _forced_solution_suffix(solution_kind, require_public_tests=require_public_tests)
    return (
        "## Mandatory Skill Read\n"
        "The previous attempts did not produce a valid tool-call format. "
        f"The harness has now executed read_skill_file for `{default_skill_path}`. "
        "Use this skill content before answering:\n\n"
        f"{content}\n\n"
        "Now answer the original question. Do not answer without using the skill content above."
        f"{solution_part}"
    )

def _with_solution_retry_instruction(
    solution_tool: "_WriteCodeSolutionTool",
    *,
    public_tests_tool: "_RunPublicTestsTool | None" = None,
    solution_kind: str = "",
) -> str:
    if solution_tool.last_error:
        if solution_tool.last_error_kind == "syntax":
            reason = (
                "The previous solution.py candidate failed Python syntax validation.\n"
                f"Syntax error: {solution_tool.last_error[:1200]}\n\n"
            )
        elif solution_tool.last_error_kind == "public_tests":
            reason = (
                "The previous solution.py candidate failed BigCodeBench public tests.\n"
                f"Public test error: {solution_tool.last_error[:1200]}\n\n"
            )
        elif solution_tool.last_error_kind == "smoke_check":
            reason = (
                "The previous solution.py candidate failed the BigCodeBench smoke check.\n"
                f"Smoke check error: {solution_tool.last_error[:1200]}\n\n"
            )
        else:
            reason = (
                "The previous solution.py candidate failed validation.\n"
                f"Validation error: {solution_tool.last_error[:1200]}\n\n"
            )
    elif public_tests_tool is not None and not public_tests_tool.ok:
        reason = (
            "The previous attempt wrote solution.py but did not call run_public_tests successfully. "
            "Call run_public_tests after write_solution and revise if it reports an error.\n\n"
        )
    else:
        reason = (
            "The previous attempt read a skill file but did not call write_solution. "
            "A final markdown code block is not a valid ReAct code artifact.\n\n"
        )
    suffix = _solution_retry_suffix(solution_kind, require_public_tests=public_tests_tool is not None)
    return (
        f"{reason}"
        "Call write_solution again with the complete corrected Python source for solution.py. "
        "Do not answer with markdown or explanation."
        f"{suffix}"
    )

def _with_context_tool_retry_instruction(question: str) -> str:
    return (
        "The previous attempt read a skill file but did not inspect the HotpotQA context with tools. "
        "Your next assistant message must call search_context or read_context_doc before any final answer. "
        "Use the context to answer this question:\n"
        f"{question.strip()}\n\n"
        "Example action format:\n"
        "Action: search_context\n"
        "Action Input: {\"query\": \"main entities from the question\", \"top_k\": 5}"
    )

def _with_final_answer_retry_instruction() -> str:
    return (
        "The previous response did not contain a valid final answer. "
        "Use your completed reasoning and reply with exactly one line: "
        "<answer>final answer</answer>. Do not call another tool or repeat the reasoning."
    )

def _solution_tool_instruction(solution_kind: str, *, require_public_tests: bool) -> str:
    if solution_kind == "bigcodebench":
        public_part = (
            " After write_solution succeeds, call run_public_tests. It runs public tests when they are "
            "available and otherwise performs only a non-leaking smoke check. If run_public_tests "
            "reports a failure, revise the full code and call write_solution and run_public_tests again."
            if require_public_tests
            else ""
        )
        return (
            "\n\n## Required BigCodeBench Solution Tools\n"
            "After reading at least one skill file, call write_solution with the complete Python source "
            "for solution.py. Do not rely on a final markdown code block."
            f"{public_part}"
        )
    return ""

def _solution_retry_suffix(solution_kind: str, *, require_public_tests: bool) -> str:
    if solution_kind == "bigcodebench":
        public_part = (
            " Then call run_public_tests before finalizing; it may be a smoke check if no public tests are available."
            if require_public_tests
            else ""
        )
        return f"\nProvide complete solution.py source for the requested function and keep the required entry point.{public_part}"
    return ""

def _forced_solution_suffix(solution_kind: str, *, require_public_tests: bool) -> str:
    if solution_kind == "bigcodebench":
        public_part = " Then call run_public_tests." if require_public_tests else ""
        return (
            "\nFor this BigCodeBench task, now call write_solution with the complete Python source "
            f"for solution.py.{public_part} Do not use a final markdown code block as the solution."
        )
    return ""

def _final_response_ready(
    text: str,
    *,
    system: str,
    user: str,
    solution_required: bool,
) -> bool:
    return _normalise_final_response(
        text,
        system=system,
        user=user,
        solution_required=solution_required,
    )[0]

def _normalise_final_response(
    text: str,
    *,
    system: str,
    user: str,
    solution_required: bool,
) -> tuple[bool, str, str]:
    stripped = str(text or "").strip()
    if solution_required:
        return bool(stripped), stripped, "solution" if stripped else "invalid"
    if not stripped:
        return False, "", "invalid"
    if _requires_answer_tag(system, user):
        region = _after_last_think_block(stripped).strip()
        candidate = _final_response_candidate(region)
        if candidate is None:
            return False, "", "invalid"
        answer = _last_concrete_answer_tag(candidate)
        if answer:
            return True, f"<answer>{answer}</answer>", "answer_tag"
        answer = _plain_final_answer(region)
        if answer:
            return True, f"<answer>{answer}</answer>", "final_answer_marker"
        return False, "", "invalid"
    candidate = _final_response_candidate(stripped)
    if candidate is None:
        return False, "", "invalid"
    if _requires_python_code_block(system, user):
        has_code_block = "```python" in candidate.lower()
        return has_code_block, candidate if has_code_block else "", "python_code_block"
    return True, candidate, "final_text"

def _final_response_candidate(text: str) -> str | None:
    nonempty_lines = [line for line in text.splitlines() if line.strip()]
    if nonempty_lines and re.match(
        r"(?i)^\s*(?:Thought|Action|Action\s+Input|Observation|Final\s+Answer)\s*:\s*$",
        nonempty_lines[-1],
    ):
        return None
    trace_markers = list(re.finditer(r"(?im)^\s*(?:Action|Action\s+Input|Observation)\s*:", text))
    final_markers = list(re.finditer(r"(?im)^\s*Final\s+Answer\s*:\s*", text))
    if not trace_markers:
        if final_markers:
            tail = text[final_markers[-1].end() :].strip()
            return tail or None
        return text
    last_trace = max(marker.start() for marker in trace_markers)
    final_after_trace = [marker for marker in final_markers if marker.start() > last_trace]
    if not final_after_trace:
        return None
    tail = text[final_after_trace[-1].end() :].strip()
    return tail or None

def _submit_final_python_block_if_present(
    text: str,
    solution_tool: "_WriteCodeSolutionTool",
    public_tests_tool: "_RunPublicTestsTool | None",
) -> bool:
    code = _extract_final_python_code_block(text)
    if not code:
        return False
    solution_tool.call({"code": code})
    if public_tests_tool is not None and solution_tool.solution_code:
        public_tests_tool.call({})
    return True

def _extract_final_python_code_block(text: str) -> str:
    candidate = _final_response_candidate(_after_last_think_block(text).strip())
    if candidate is None:
        return ""
    fenced = re.findall(r"```(?:python)?\s*(.*?)```", candidate, flags=re.IGNORECASE | re.DOTALL)
    return fenced[-1].strip() if fenced else ""

def _has_concrete_answer_tag(text: str) -> bool:
    return bool(_last_concrete_answer_tag(_after_last_think_block(text)))

def _last_concrete_answer_tag(text: str) -> str:
    matches = re.findall(
        r"<answer\b[^>]*>\s*([^<>]*?)\s*</answer\s*>",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    for answer in reversed(matches):
        normalized = re.sub(r"\s+", " ", answer).strip()
        if not _is_placeholder_answer(normalized):
            return normalized
    return ""

def _plain_final_answer(text: str) -> str:
    markers = list(re.finditer(r"(?im)^\s*Final\s+Answer\s*:\s*", text))
    if not markers:
        return ""
    tail = text[markers[-1].end() :].strip()
    if not tail:
        return ""
    line = tail.splitlines()[0].strip()
    if re.match(r"(?i)^\s*(?:Thought|Action|Action\s+Input|Observation)\s*:", line):
        return ""
    if _is_placeholder_answer(line) or "<answer" in line.lower():
        return ""
    return line

def _is_placeholder_answer(answer: str) -> bool:
    normalized = re.sub(r"\s+", " ", str(answer or "")).strip().lower()
    return normalized in {"", "...", "…", "[answer]", "answer", "final answer", "your answer"}

def _react_trace_action_count(text: str) -> int:
    return len(re.findall(r"(?im)^\s*Action\s*:", str(text or "")))

def _after_last_think_block(text: str) -> str:
    lower = text.lower()
    last_think_end = lower.rfind("</think>")
    if last_think_end == -1:
        return text
    return text[last_think_end + len("</think>") :]

def _requires_answer_tag(system: str, user: str) -> bool:
    return "<answer>" in f"{system}\n{user}".lower()

def _requires_python_code_block(system: str, user: str) -> bool:
    text = f"{system}\n{user}".lower()
    return "```python" in text or "python code block" in text or "single python code block" in text

def _build_tools(
    store: SkillFileStore,
    code_context: CodeToolContext | None,
    qa_store: QAContextStore | None = None,
    *,
    native_qwen_tools: bool,
    solution_tool: "_WriteCodeSolutionTool | None" = None,
    public_tests_tool: "_RunPublicTestsTool | None" = None,
) -> list[Any]:
    tools: list[Any] = [
        _ListSkillFilesTool(store),
        _ReadSkillFileTool(store),
        _SearchSkillFilesTool(store),
    ]
    if qa_store is not None:
        tools.extend(
            [
                _ListContextTitlesTool(qa_store),
                _SearchContextTool(qa_store),
                _ReadContextDocTool(qa_store),
            ]
        )
    if code_context is not None:
        if solution_tool is not None:
            tools.append(solution_tool)
        if public_tests_tool is not None:
            tools.append(public_tests_tool)
    if not native_qwen_tools:
        return tools
    BaseTool = _load_base_tool_cls()
    return [_to_qwen_tool(tool, BaseTool) for tool in tools]

def _load_react_chat_cls() -> Any:
    from qwen_agent.agents.react_chat import ReActChat
    from qwen_agent.llm.schema import ASSISTANT, Message

    class _ReActChat(ReActChat):
        def __init__(self, *args: Any, max_llm_calls_per_run: int = 4, **kwargs: Any) -> None:
            from evaluation.model import is_gemma4_model
            from qwen_agent.llm.oai import TextChatAtOAI

            super().__init__(*args, **kwargs)
            self._eval_max_llm_calls_per_run = max(1, int(max_llm_calls_per_run))
            self._eval_gemma4 = is_gemma4_model(self.llm.model)
            self._eval_model_responses = []
            if isinstance(self.llm, TextChatAtOAI) and "qwen3" in self.llm.model.lower():
                def chat_with_usage(messages, delta_stream, generate_cfg):
                    response = self.llm._chat_complete_create(
                        model=self.llm.model,
                        messages=self.llm.convert_messages_to_dicts(messages),
                        stream=False, **generate_cfg,
                    )
                    choice = response.choices[0]
                    yield [Message(
                        role=ASSISTANT, content=choice.message.content or "",
                        reasoning_content=getattr(choice.message, "reasoning_content", None),
                        extra={"finish_reason": choice.finish_reason,
                               "completion_tokens": response.usage.completion_tokens,
                               "prompt_tokens": response.usage.prompt_tokens},
                    )]

                self.llm._chat_stream = chat_with_usage
            if self._eval_gemma4:
                self.llm._eval_tools = [
                    {"type": "function", "function": tool.function}
                    for tool in self.function_map.values()
                ]
            if not self.llm.use_raw_api or self._eval_gemma4:
                self.extra_generate_cfg.pop("stop", None)

        def _run(self, messages: list[Any], lang: str = "en", **kwargs: Any) -> Any:
            text_messages = self._prepend_react_prompt(messages, lang=lang)
            num_llm_calls_available = self._eval_max_llm_calls_per_run
            response = "Thought: "
            while num_llm_calls_available > 0:
                num_llm_calls_available -= 1
                if self._eval_gemma4 and not response.endswith("\n"):
                    response += "\n"
                output = []
                if self.llm.use_raw_api:
                    for output in self._call_llm(messages=text_messages):
                        if output:
                            yield [Message(role=ASSISTANT, content=response + output[-1].content)]
                else:
                    output = self._call_llm(messages=text_messages, stream=False)
                    if output:
                        yield [Message(role=ASSISTANT, content=response + output[-1].content)]
                if not output:
                    break
                archived = output[-1].model_dump()
                archived["extra"] = dict(archived.get("extra") or {})
                request_cfg = {**self.llm.generate_cfg, **self.extra_generate_cfg}
                archived["extra"].setdefault("max_tokens", request_cfg["max_tokens"])
                self._eval_model_responses.append(archived)
                response += output[-1].content
                has_action, action, action_input, thought = self._detect_tool(output[-1].content)
                if not has_action:
                    break
                observation = self._call_tool(action, action_input, messages=messages, **kwargs)
                if action == "read_skill_file":
                    observation = (
                        f"{observation}\nThe skill is now read. In the next Thought, briefly state whether it is "
                        "useful and which specific parts apply, then immediately issue the next Action in the "
                        "same response. An analysis-only response is incomplete. Ignore irrelevant parts and "
                        "do not read the skill again."
                    )
                observation = f"\nObservation: {observation}\nThought: "
                response += observation
                yield [Message(role=ASSISTANT, content=response)]
                if (not text_messages[-1].content.endswith("\nThought: ")) and (not thought.startswith("\n")):
                    text_messages[-1].content += "\n"
                if action_input.startswith("```"):
                    action_input = "\n" + action_input
                text_messages[-1].content += thought + f"\nAction: {action}\nAction Input: {action_input}" + observation

        def _detect_tool(self, text: str) -> tuple[bool, str, str, str]:
            has_action, action, action_input, thought = _detect_xml_tool_call(text, set(self.function_map.keys()))
            if has_action:
                return has_action, action, action_input, thought
            return super()._detect_tool(text)

    return _ReActChat

def _default_skill_path(store: SkillFileStore) -> str:
    for path in ("SKILL.md", "skill.md"):
        if path in store.files:
            return path
    return next(iter(store.files))

def _load_base_tool_cls() -> Any:
    from qwen_agent.tools.base import BaseTool
    return BaseTool

class _SimpleTool:
    name = ""
    description = ""
    parameters: dict[str, Any] = {"type": "object", "properties": {}, "required": []}

    @property
    def function(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }

    def _verify_json_format_args(self, params: str | dict) -> dict[str, Any]:
        if isinstance(params, str):
            args = self._parse_text_args(params)
        else:
            args = dict(params or {})
        for key in self.parameters.get("required", []):
            if key not in args:
                raise ValueError(f"Parameter {key} is required.")
        return args

    def _parse_text_args(self, params: str) -> dict[str, Any]:
        raw = params.strip()
        if not raw:
            return {}
        fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", raw, re.IGNORECASE | re.DOTALL)
        if fenced:
            raw = fenced.group(1).strip()
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            try:
                parsed, _ = json.JSONDecoder().raw_decode(raw)
            except json.JSONDecodeError:
                parsed = None
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, str):
            raw = parsed.strip()
        fields = set((self.parameters.get("properties") or {}).keys())
        required = list(self.parameters.get("required", []))
        for key in sorted(fields | set(required)):
            match = re.match(rf"^\s*{re.escape(key)}\s*[:=]\s*(.+?)\s*$", raw, re.DOTALL)
            if match:
                return {key: _strip_arg_value(match.group(1))}
        if len(required) == 1:
            return {required[0]: _strip_arg_value(raw)}
        return {}

def _to_qwen_tool(simple_tool: _SimpleTool, base_tool_cls: Any) -> Any:
    class _WrappedQwenTool(base_tool_cls):
        name = simple_tool.name
        description = simple_tool.description
        parameters = simple_tool.parameters

        def __init__(self) -> None:
            super().__init__()
            self.simple_tool = simple_tool

        def call(self, params: str | dict, **kwargs: Any) -> str:
            return self.simple_tool.call(params, **kwargs)

    return _WrappedQwenTool()

def _detect_xml_tool_call(text: str, tool_names: set[str]) -> tuple[bool, str, str, str]:
    for match in _iter_tag_tool_matches(text):
        action = match.group(1).strip()
        if action not in tool_names:
            continue
        action_input = _clean_action_input(_first_text_group(match, 2))
        thought = text[: match.start()].rstrip()
        return True, action, action_input, thought
    for match in _iter_tag_only_tool_matches(text):
        action = match.group(1).strip()
        if action not in {"list_skill_files", "read_skill_file"} or action not in tool_names:
            continue
        thought = text[: match.start()].rstrip()
        return True, action, "{}", thought
    for match in _iter_implicit_json_action_matches(text):
        parsed = _parse_leading_json_object(match.group(1))
        if not isinstance(parsed, dict):
            continue
        action = str(parsed.get("name") or parsed.get("action") or "read_skill_file")
        if action != "read_skill_file" or action not in tool_names or "path" not in parsed:
            continue
        action_input = _clean_action_input(match.group(1))
        thought = text[: match.start()].rstrip()
        return True, action, action_input, thought
    for match in _iter_action_block_matches(text):
        action = match.group(1).strip()
        if action not in tool_names:
            continue
        action_input = _clean_action_input(match.group(2))
        thought = text[: match.start()].rstrip()
        return True, action, action_input, thought
    natural = _detect_natural_read_skill_file(text)
    if natural is not None and "read_skill_file" in tool_names:
        action_input, start = natural
        thought = text[:start].rstrip()
        return True, "read_skill_file", action_input, thought
    return False, "", "", text

def _clean_action_input(value: str) -> str:
    raw = value.strip()
    if raw.startswith("{"):
        try:
            _, index = json.JSONDecoder().raw_decode(raw)
        except json.JSONDecodeError:
            return raw
        return raw[:index].strip()
    return raw

def _parse_leading_json_object(value: str) -> dict[str, Any] | None:
    raw = value.strip()
    if not raw.startswith("{"):
        return None
    try:
        parsed, _ = json.JSONDecoder().raw_decode(raw)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None

def _iter_tag_tool_matches(text: str) -> Any:
    pattern = re.compile(
        r"<(?:answer|action)>\s*([A-Za-z_][\w.-]*)\s*</(?:answer|action)>\s*"
        r"(?:"
        r"<action_input>\s*(.*?)\s*</action_input>"
        r"|<input>\s*(.*?)\s*</input>"
        r"|(?:Action\s+Input)\s*:\s*(.*?)(?=\n\s*(?:Observation|Thought|Final Answer)\s*:|</?answer\b|</?action\b|</?think\b|\Z)"
        r")",
        re.IGNORECASE | re.DOTALL,
    )
    return pattern.finditer(text)

def _iter_tag_only_tool_matches(text: str) -> Any:
    pattern = re.compile(
        r"<answer>\s*([A-Za-z_][\w.-]*)\s*</answer>(?=\s*(?:</think>|$))",
        re.IGNORECASE | re.DOTALL,
    )
    return pattern.finditer(text)

def _iter_implicit_json_action_matches(text: str) -> Any:
    pattern = re.compile(
        r"<action>\s*(\{.*?\})\s*(?=</action>|<observation\b|</?answer\b|Thought\s*:|Final Answer\s*:|\Z)",
        re.IGNORECASE | re.DOTALL,
    )
    return pattern.finditer(text)

def _detect_natural_read_skill_file(text: str) -> tuple[str, int] | None:
    match = re.search(r"\bread_skill_file\b(?P<body>.{0,240})", text, re.IGNORECASE | re.DOTALL)
    if not match:
        return None
    body = match.group("body") or ""
    path_match = re.search(
        r"\bpath\b\s*(?:is|=|:)?\s*[\"'`]?([A-Za-z0-9_.\-/]+)[\"'`]?",
        body,
        re.IGNORECASE,
    )
    if path_match:
        return json.dumps({"path": path_match.group(1)}), match.start()
    return "{}", match.start()

def _iter_action_block_matches(text: str) -> Any:
    pattern = re.compile(
        r"(?m)^[^\n]*\bAction\s*:\s*([A-Za-z_][\w.-]*)\s*$\s*"
        r"^[^\n]*\bAction\s+Input\s*:\s*(.*?)(?=\n\s*(?:Observation|Thought|Final Answer)\s*:|\Z)",
        re.IGNORECASE | re.DOTALL,
    )
    return pattern.finditer(text)

def _first_text_group(match: re.Match[str], start: int) -> str:
    for idx in range(start, len(match.groups()) + 1):
        value = match.group(idx)
        if value is not None:
            return value
    return ""

def _is_empty_tool_args(params: str | dict) -> bool:
    if isinstance(params, dict):
        return not params or not str(params.get("path") or "").strip()
    raw = str(params or "").strip()
    return raw in {"", "{}"}

def _strip_arg_value(value: str) -> str:
    return value.strip().strip('"').strip("'")

class _ListSkillFilesTool(_SimpleTool):
    name = "list_skill_files"
    description = "List available skill files with path, title, and short description."
    parameters = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    def __init__(self, store: SkillFileStore) -> None:
        self.store = store

    def call(self, params: str | dict, **kwargs: Any) -> str:
        return json.dumps(self.store.list_files(), ensure_ascii=False)

class _ReadSkillFileTool(_SimpleTool):
    name = "read_skill_file"
    description = "Read one skill file by relative path. Use paths returned by list_skill_files or search_skill_files."
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Relative path of the skill file to read.",
            }
        },
        "required": ["path"],
    }

    def __init__(self, store: SkillFileStore) -> None:
        self.store = store

    def call(self, params: str | dict, **kwargs: Any) -> str:
        if _is_empty_tool_args(params):
            args = {"path": _default_skill_path(self.store)}
        else:
            args = self._verify_json_format_args(params)
        return json.dumps(self.store.read_file(str(args.get("path") or "")), ensure_ascii=False)

class _SearchSkillFilesTool(_SimpleTool):
    name = "search_skill_files"
    description = "Search skill files for task-relevant terms and return ranked paths."
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query describing the needed skill.",
            },
            "top_k": {
                "type": "integer",
                "description": "Maximum number of files to return.",
            },
        },
        "required": ["query"],
    }

    def __init__(self, store: SkillFileStore) -> None:
        self.store = store

    def call(self, params: str | dict, **kwargs: Any) -> str:
        args = self._verify_json_format_args(params)
        top_k = int(args.get("top_k") or 5)
        return json.dumps(self.store.search_files(str(args["query"]), top_k=top_k), ensure_ascii=False)

class _ListContextTitlesTool(_SimpleTool):
    name = "list_context_titles"
    description = "List HotpotQA context document titles with short previews."
    parameters = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    def __init__(self, store: QAContextStore) -> None:
        self.store = store

    def call(self, params: str | dict, **kwargs: Any) -> str:
        return json.dumps(self.store.list_titles(), ensure_ascii=False)

class _SearchContextTool(_SimpleTool):
    name = "search_context"
    description = "Search the HotpotQA context documents for entities or phrases from the question."
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Entity, phrase, or relation to search for in the context.",
            },
            "top_k": {
                "type": "integer",
                "description": "Maximum number of matching context documents to return.",
            },
        },
        "required": ["query"],
    }

    def __init__(self, store: QAContextStore) -> None:
        self.store = store

    def call(self, params: str | dict, **kwargs: Any) -> str:
        args = self._verify_json_format_args(params)
        top_k = int(args.get("top_k") or 5)
        return json.dumps(self.store.search(str(args["query"]), top_k=top_k), ensure_ascii=False)

class _ReadContextDocTool(_SimpleTool):
    name = "read_context_doc"
    description = "Read one HotpotQA context document by title."
    parameters = {
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": "Title returned by list_context_titles or search_context.",
            }
        },
        "required": ["title"],
    }

    def __init__(self, store: QAContextStore) -> None:
        self.store = store

    def call(self, params: str | dict, **kwargs: Any) -> str:
        args = self._verify_json_format_args(params)
        return json.dumps(self.store.read_doc(str(args["title"])), ensure_ascii=False)

class _WriteCodeSolutionTool(_SimpleTool):
    name = "write_solution"
    description = (
        "Write the final BigCodeBench solution.py. The code must be complete Python source "
        "containing the requested entry point and no markdown or ReAct transcript text."
    )
    parameters = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Complete Python source for solution.py.",
            }
        },
        "required": ["code"],
    }

    def __init__(self, context: CodeToolContext) -> None:
        self.context = context
        self.solution_code = ""
        self.solution_path = ""
        self.last_error = ""
        self.last_error_kind = ""
        self.calls: list[dict[str, Any]] = []

    def call(self, params: str | dict, **kwargs: Any) -> str:
        try:
            args = self._verify_json_format_args(params)
        except ValueError as exc:
            payload = self._record_failure(str(exc), syntax_ok=False, error_kind="validation")
            return json.dumps(payload, ensure_ascii=False)
        code = _normalise_code_solution(str(args.get("code") or ""), self.context)
        if not code.strip():
            payload = self._record_failure("Parameter code is required.", syntax_ok=False, error_kind="validation")
            return json.dumps(payload, ensure_ascii=False)
        try:
            tree = ast.parse(code, filename="solution.py")
        except SyntaxError as exc:
            payload = self._record_failure(_format_syntax_error(exc), syntax_ok=False, error_kind="syntax")
            return json.dumps(payload, ensure_ascii=False)
        entry_error = _validate_code_entry_point(tree, self.context.entry_point)
        if entry_error:
            payload = self._record_failure(entry_error, syntax_ok=True, error_kind="validation")
            return json.dumps(payload, ensure_ascii=False)
        work_dir = Path(self.context.work_dir).resolve()
        work_dir.mkdir(parents=True, exist_ok=True)
        path = work_dir / "solution.py"
        path.write_text(code.rstrip() + "\n", encoding="utf-8")
        self.solution_code = code.rstrip() + "\n"
        self.solution_path = str(path)
        self.last_error = ""
        self.last_error_kind = ""
        payload = {
            "ok": True,
            "syntax_ok": True,
            "solution_path": self.solution_path,
            "entry_point": self.context.entry_point,
            "line_count": len(self.solution_code.splitlines()),
        }
        self.calls.append(
            {
                "tool": self.name,
                "args": {"code_chars": len(self.solution_code), "line_count": payload["line_count"]},
                "observation": payload,
            }
        )
        return json.dumps(payload, ensure_ascii=False)

    def serialised_calls(self) -> list[dict[str, Any]]:
        return _serialise_tool_calls(self.calls)

    def record_test_feedback_failure(
        self,
        error: str,
        *,
        error_kind: str,
        candidate_path: str = "",
    ) -> dict[str, Any]:
        self.last_error = error
        self.last_error_kind = error_kind
        payload = {
            "ok": False,
            "syntax_ok": True,
            "error": error,
            "solution_path": self.solution_path,
        }
        if candidate_path:
            payload["candidate_path"] = candidate_path
        self.calls.append(
            {
                "tool": self.name,
                "args": {},
                "observation": payload,
            }
        )
        return payload

    def record_public_test_failure(self, error: str, *, candidate_path: str = "") -> dict[str, Any]:
        return self.record_test_feedback_failure(
            error,
            error_kind="public_tests",
            candidate_path=candidate_path,
        )

    def _record_failure(
        self,
        error: str,
        *,
        syntax_ok: bool,
        error_kind: str,
        candidate_path: str = "",
    ) -> dict[str, Any]:
        self.solution_code = ""
        self.solution_path = ""
        self.last_error = error
        self.last_error_kind = error_kind
        payload = {
            "ok": False,
            "syntax_ok": syntax_ok,
            "error": error,
            "solution_path": "",
        }
        if candidate_path:
            payload["candidate_path"] = candidate_path
        self.calls.append(
            {
                "tool": self.name,
                "args": {},
                "observation": payload,
            }
        )
        return payload

class _RunPublicTestsTool(_SimpleTool):
    name = "run_public_tests"
    description = (
        "Run available public tests for the current BigCodeBench solution.py. If no public tests "
        "are provided, run a non-leaking smoke check that imports the solution script."
    )
    parameters = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    def __init__(self, context: CodeToolContext, solution_tool: _WriteCodeSolutionTool) -> None:
        self.context = context
        self.solution_tool = solution_tool
        self.ok = False
        self.tested_solution_code = ""
        self.public_tests_available = bool(str(context.public_tests or "").strip())
        self.feedback_mode = "public_tests" if self.public_tests_available else "smoke_only"
        self.public_tests_ok: bool | None = False if self.public_tests_available else None
        self.smoke_check_ok: bool | None = None if self.public_tests_available else False
        self.last_error = ""
        self.calls: list[dict[str, Any]] = []

    def current_solution_ok(self) -> bool:
        return bool(self.ok and self.solution_tool.solution_code and self.tested_solution_code == self.solution_tool.solution_code)

    def call(self, params: str | dict, **kwargs: Any) -> str:
        if not self.solution_tool.solution_code:
            payload = self._record_failure("Call write_solution with valid code before run_public_tests.")
            return json.dumps(payload, ensure_ascii=False)
        script = self._write_test_script()
        try:
            completed = subprocess.run(
                [sys.executable, str(script)],
                cwd=self.context.work_dir,
                capture_output=True,
                text=True,
                timeout=max(5, int(self.context.timeout_seconds)),
                start_new_session=True,
            )
            stdout = completed.stdout[-4000:]
            stderr = completed.stderr[-4000:]
            ok = completed.returncode == 0
        except subprocess.TimeoutExpired as exc:
            stdout = (exc.stdout or "")[-4000:] if isinstance(exc.stdout, str) else ""
            stderr = f"Timed out after {self.context.timeout_seconds}s."
            ok = False
            completed = None
        if ok:
            self._set_ok_status(True)
            payload = {
                "ok": True,
                "returncode": 0,
                "stdout": stdout,
                "stderr": stderr,
                "public_tests_available": self.public_tests_available,
                "test_feedback_mode": self.feedback_mode,
                "public_tests_ok": self.public_tests_ok,
                "smoke_check_ok": self.smoke_check_ok,
            }
            self.ok = True
            self.tested_solution_code = self.solution_tool.solution_code
            self.last_error = ""
            self.calls.append({"tool": self.name, "args": {}, "observation": payload})
            return json.dumps(payload, ensure_ascii=False)
        returncode = completed.returncode if completed is not None else 124
        error = stderr or stdout or f"run_public_tests failed with rc={returncode}"
        error_kind = "public_tests" if self.public_tests_available else "smoke_check"
        self.solution_tool.record_test_feedback_failure(
            error,
            error_kind=error_kind,
            candidate_path=self.solution_tool.solution_path,
        )
        payload = self._record_failure(error, returncode=returncode, stdout=stdout, stderr=stderr)
        return json.dumps(payload, ensure_ascii=False)

    def serialised_calls(self) -> list[dict[str, Any]]:
        return _serialise_tool_calls(self.calls)

    def _write_test_script(self) -> Path:
        work_dir = Path(self.context.work_dir).resolve()
        work_dir.mkdir(parents=True, exist_ok=True)
        script = work_dir / "public_tests.py"
        if self.public_tests_available:
            body = (
                f"{self.solution_tool.solution_code.rstrip()}\n\n"
                f"{self.context.public_tests.rstrip()}\n\n"
                "if __name__ == '__main__':\n"
                "    import unittest\n"
                "    unittest.main(verbosity=0)\n"
            )
        else:
            body = f"{self.solution_tool.solution_code.rstrip()}\n"
        script.write_text(body, encoding="utf-8")
        return script

    def _record_failure(
        self,
        error: str,
        *,
        returncode: int = 2,
        stdout: str = "",
        stderr: str = "",
    ) -> dict[str, Any]:
        self._set_ok_status(False)
        self.ok = False
        self.tested_solution_code = ""
        self.last_error = error
        payload = {
            "ok": False,
            "returncode": returncode,
            "stdout": stdout,
            "stderr": stderr or error,
            "public_tests_available": self.public_tests_available,
            "test_feedback_mode": self.feedback_mode,
            "public_tests_ok": self.public_tests_ok,
            "smoke_check_ok": self.smoke_check_ok,
        }
        self.calls.append({"tool": self.name, "args": {}, "observation": payload})
        return payload

    def _set_ok_status(self, ok: bool) -> None:
        if self.public_tests_available:
            self.public_tests_ok = ok
            self.smoke_check_ok = None
        else:
            self.public_tests_ok = None
            self.smoke_check_ok = ok

def _normalise_solution_code(code: str) -> str:
    raw = code.strip()
    fenced = re.findall(r"```(?:python)?\s*(.*?)```", raw, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        raw = fenced[-1].strip()
    jsonish = re.match(r"^\{\s*[\"']?code[\"']?\s*:\s*(.*)\}\s*$", raw, flags=re.DOTALL)
    if jsonish:
        raw = _strip_arg_value(jsonish.group(1).strip().rstrip(","))
    return raw

def _normalise_code_solution(code: str, context: CodeToolContext) -> str:
    raw = _normalise_solution_code(code)
    entry = str(context.entry_point or "").strip()
    prompt = str(context.code_prompt or "").rstrip()
    if entry and prompt and not _code_text_mentions_entry_definition(raw, entry) and _looks_like_function_body(raw):
        body = raw.rstrip()
        if body and not body.startswith((" ", "\t")):
            body = "\n".join(("    " + line if line.strip() else line) for line in body.splitlines())
        return f"{prompt}\n{body}\n"
    return raw

def _code_text_mentions_entry_definition(code: str, entry_point: str) -> bool:
    pattern = rf"\b(?:def|async\s+def|class)\s+{re.escape(entry_point)}\b"
    return bool(re.search(pattern, code))

def _looks_like_function_body(code: str) -> bool:
    lines = [line for line in code.splitlines() if line.strip()]
    if not lines:
        return False
    first = lines[0]
    return first.startswith((" ", "\t")) or all(
        re.match(r"\s*(?:return|if|for|while|try|with|raise|import|from|[A-Za-z_]\w*\s*=)", line)
        for line in lines[: min(3, len(lines))]
    )

def _validate_code_entry_point(tree: ast.AST, entry_point: str) -> str | None:
    entry = str(entry_point or "").strip()
    if not entry:
        return None
    for node in getattr(tree, "body", []):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == entry:
            return None
    return f"solution.py must define the required entry point `{entry}`."

def _format_syntax_error(exc: SyntaxError) -> str:
    location = f"line {exc.lineno}, offset {exc.offset}" if exc.lineno else "unknown location"
    text = (exc.text or "").strip()
    suffix = f": {text}" if text else ""
    return f"SyntaxError at {location}: {exc.msg}{suffix}"

def _normalise_messages(messages: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in messages or []:
        if isinstance(item, dict):
            rows.append(item)
        elif hasattr(item, "model_dump"):
            rows.append(item.model_dump())
        else:
            rows.append({"role": "assistant", "content": str(item)})
    return rows

def _final_text(messages: list[dict[str, Any]]) -> str:
    for item in reversed(messages):
        content = item.get("content")
        if isinstance(content, str) and content.strip():
            return content
    return ""

def _files_read(calls: list[dict[str, Any]]) -> list[str]:
    paths: list[str] = []
    for call in calls:
        if call.get("tool") != "read_skill_file":
            continue
        obs = call.get("observation") or {}
        if obs.get("error") or "content" not in obs:
            continue
        path = str(obs.get("path") or "")
        if path and path not in paths:
            paths.append(path)
    return paths

def _serialise_tool_calls(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for call in calls:
        obs = call.get("observation")
        rows.append(
            {
                "tool": call.get("tool"),
                "args": call.get("args", {}),
                "observation": json.dumps(obs, ensure_ascii=False),
            }
        )
    return rows

def _extract_title(content: str, fallback: str) -> str:
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip() or fallback
    return fallback

def _extract_description(content: str) -> str:
    for line in content.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            return stripped[:240]
    return ""

def _parse_qa_docs(context: str) -> list[dict[str, Any]]:
    raw = str(context or "").strip()
    docs: list[dict[str, Any]] = []
    pattern = re.compile(
        r"\[DOC\]\s*(?:\[TLE\]\s*(.*?)\s*)?\[PAR\]\s*(.*?)(?=\s*\[DOC\]\s*(?:\[TLE\]|\[PAR\])|\Z)",
        flags=re.IGNORECASE | re.DOTALL,
    )
    for idx, match in enumerate(pattern.finditer(raw), start=1):
        title = re.sub(r"\s+", " ", (match.group(1) or f"Document {idx}")).strip()
        content = re.sub(r"\s+", " ", (match.group(2) or "")).strip()
        if not content:
            continue
        docs.append({"title": title, "content": content, "sentences": _split_sentences(content)})
    if docs:
        return docs
    content = re.sub(r"\s+", " ", raw).strip()
    if not content:
        content = "No context was provided."
    return [{"title": "Context", "content": content, "sentences": _split_sentences(content)}]

def _split_sentences(text: str) -> list[str]:
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]
    return sentences or ([text.strip()] if text.strip() else [])

def _terms(text: str) -> list[str]:
    return [part for part in re.findall(r"[A-Za-z0-9_#./+-]+", text.lower()) if len(part) >= 2]

def _score_file(item: SkillFile, query_terms: list[str]) -> float:
    haystack = f"{item.path} {item.title} {item.description} {item.content}".lower()
    score = 0.0
    for term in query_terms:
        if term in haystack:
            score += 1.0
            if term in item.path.lower() or term in item.title.lower():
                score += 1.5
    return score
