from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from http.client import HTTPException
from typing import Any, Callable, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen

from langchain_core.messages import BaseMessage
from langchain_core.tools import BaseTool


LOOKAHEAD_SUB_AGENT_METADATA_KEY = "lookahead_sub_agent"
LOOKAHEAD_SUB_AGENT_SYSTEM_PROMPT_KEY = "lookahead_sub_agent_system_prompt"
LOOKAHEAD_SUB_AGENT_CONTROL_KEY = "lookahead_sub_agent_control"


@dataclass(slots=True)
class LookaheadMainState:
    usage: int
    soft_limit: int
    hard_limit: int
    ready: bool = False


@dataclass(slots=True)
class LookaheadState:
    pending_task_id: str | None = None
    pending_signature: tuple[int, ...] | None = None
    pending_metadata: dict[str, Any] = field(default_factory=dict)

    def clear(self) -> None:
        self.pending_task_id = None
        self.pending_signature = None
        self.pending_metadata.clear()


@dataclass(slots=True)
class LookaheadTask:
    session_id: str
    task_id: str
    strategy: str
    source_text: str | None = None
    control: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class LookaheadArtifact:
    task_id: str
    strategy: str
    metadata: dict[str, Any] = field(default_factory=dict)


class LookaheadDispatcher(Protocol):
    def trigger(self, task: LookaheadTask) -> bool: ...

    def append(self, task: LookaheadTask) -> bool: ...

    def commit(self, task: LookaheadTask) -> LookaheadArtifact | None: ...

    def clear(self, session_id: str, task_id: str | None = None) -> None: ...


def message_signature(messages: list[BaseMessage]) -> tuple[int, ...]:
    return tuple(id(message) for message in messages)


def build_sub_agent_text(
    *,
    system_prompt: str,
    task_text: str,
    prefix_text: str = "",
) -> str:
    parts: list[str] = []
    if system_prompt.strip():
        parts.append(f"System: {system_prompt.strip()}")
    if prefix_text.strip():
        parts.append(f"Context:\n{prefix_text.strip()}")
    if task_text.strip():
        parts.append(f"Human: {task_text.strip()}")
    return "\n".join(parts)


def build_sub_agent_tool_metadata(
    *,
    system_prompt: str,
    prefix_text: str | None = None,
    include_memory_prefix: bool | None = None,
    control: dict[str, Any] | None = None,
) -> dict[str, Any]:
    normalized_system_prompt = system_prompt.strip()
    if not normalized_system_prompt:
        msg = "sub-agent system_prompt must not be empty"
        raise ValueError(msg)
    metadata: dict[str, Any] = {
        LOOKAHEAD_SUB_AGENT_METADATA_KEY: {
            "system_prompt": normalized_system_prompt,
        }
    }
    if prefix_text is not None:
        metadata[LOOKAHEAD_SUB_AGENT_METADATA_KEY]["prefix_text"] = str(prefix_text)
    if include_memory_prefix is not None:
        metadata[LOOKAHEAD_SUB_AGENT_METADATA_KEY]["include_memory_prefix"] = bool(
            include_memory_prefix
        )
    if control:
        metadata[LOOKAHEAD_SUB_AGENT_METADATA_KEY]["control"] = dict(control)
    return metadata


def get_sub_agent_warmup_spec(
    metadata: dict[str, Any] | None,
) -> tuple[str, dict[str, Any] | None] | None:
    metadata = metadata or {}
    nested_spec = metadata.get(LOOKAHEAD_SUB_AGENT_METADATA_KEY)
    if isinstance(nested_spec, dict):
        system_prompt = str(nested_spec.get("system_prompt", "")).strip()
        control = nested_spec.get("control")
        normalized_control = dict(control) if isinstance(control, dict) else {}
        if "prefix_text" in nested_spec:
            normalized_control["prefix_text"] = str(nested_spec.get("prefix_text", ""))
        if "include_memory_prefix" in nested_spec:
            normalized_control["include_memory_prefix"] = bool(
                nested_spec.get("include_memory_prefix")
            )
        if system_prompt:
            return system_prompt, normalized_control or None

    system_prompt = str(
        metadata.get(LOOKAHEAD_SUB_AGENT_SYSTEM_PROMPT_KEY, "")
    ).strip()
    if not system_prompt:
        return None
    control = metadata.get(LOOKAHEAD_SUB_AGENT_CONTROL_KEY)
    return system_prompt, control if isinstance(control, dict) else None


def with_sub_agent_lookahead(
    tool: BaseTool,
    *,
    system_prompt: str,
    prefix_text: str | None = None,
    include_memory_prefix: bool | None = None,
    control: dict[str, Any] | None = None,
) -> BaseTool:
    metadata = dict(tool.metadata or {})
    metadata.update(
        build_sub_agent_tool_metadata(
            system_prompt=system_prompt,
            prefix_text=prefix_text,
            include_memory_prefix=include_memory_prefix,
            control=control,
        )
    )
    return tool.model_copy(update={"metadata": metadata})


def should_trigger(
    main_state: LookaheadMainState,
    la_state: LookaheadState,
) -> bool:
    return main_state.usage >= main_state.soft_limit and la_state.pending_task_id is None


def should_commit(main_state: LookaheadMainState) -> bool:
    return main_state.usage >= main_state.hard_limit


@dataclass
class SGLangLookaheadDispatcher:
    transport: Callable[[dict[str, Any]], dict[str, Any] | None]
    endpoint: str = "/lookahead/control"

    def trigger(self, task: LookaheadTask) -> bool:
        payload = self._build_payload("warmup", task)
        response = self.transport(payload) or {}
        return bool(response.get("accepted", response.get("success", True)))

    def append(self, task: LookaheadTask) -> bool:
        payload = self._build_payload("append", task)
        response = self.transport(payload) or {}
        return bool(response.get("accepted", response.get("success", True)))

    def commit(self, task: LookaheadTask) -> LookaheadArtifact | None:
        payload = self._build_payload("commit", task)
        response = self.transport(payload) or {}
        artifact = response.get("artifact")
        if not artifact:
            return None
        return LookaheadArtifact(
            task_id=str(artifact.get("task_id", task.task_id)),
            strategy=str(artifact.get("strategy", task.strategy)),
            metadata=dict(artifact.get("metadata", {})),
        )

    def clear(self, session_id: str, task_id: str | None = None) -> None:
        payload: dict[str, Any] = {
            "endpoint": self.endpoint,
            "action": "clear",
            "session_id": session_id,
        }
        if task_id is not None:
            payload["task_id"] = task_id
        self.transport(payload)

    def _build_payload(self, action: str, task: LookaheadTask) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "endpoint": self.endpoint,
            "action": action,
            "session_id": task.session_id,
            "task_id": task.task_id,
            "strategy": task.strategy,
            "control": dict(task.control),
        }
        if task.source_text is not None:
            payload["text"] = task.source_text
        return payload


def _default_urlopen(request: Request, timeout: float):
    return urlopen(request, timeout=timeout)


@dataclass
class SGLangControlPlaneClient:
    """Minimal HTTP client for the SGLang lookahead/session control plane."""

    base_url: str
    timeout: float = 10.0
    headers: dict[str, str] = field(default_factory=dict)
    opener: Callable[[Request, float], Any] = _default_urlopen
    lookahead_endpoint: str = "/lookahead/control"
    open_session_endpoint: str = "/open_session"
    close_session_endpoint: str = "/close_session"

    def create_dispatcher(self) -> SGLangLookaheadDispatcher:
        return SGLangLookaheadDispatcher(transport=self.transport)

    def open_lookahead_session(
        self,
        capacity_of_str_len: int,
        session_id: str | None = None,
    ) -> "SGLangLookaheadSession":
        opened_session_id = self.open_session(
            capacity_of_str_len=capacity_of_str_len,
            session_id=session_id,
        )
        return SGLangLookaheadSession(
            client=self,
            session_id=opened_session_id,
            dispatcher=self.create_dispatcher(),
        )

    def transport(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        request_payload = dict(payload)
        endpoint = str(request_payload.pop("endpoint", self.lookahead_endpoint))
        response = self._post_json(endpoint, request_payload)
        if response is None:
            return None
        if not isinstance(response, dict):
            msg = f"Expected JSON object response from {endpoint}, got {type(response).__name__}."
            raise RuntimeError(msg)
        return response

    def open_session(
        self,
        capacity_of_str_len: int,
        session_id: str | None = None,
    ) -> str:
        payload: dict[str, Any] = {"capacity_of_str_len": capacity_of_str_len}
        if session_id is not None:
            payload["session_id"] = session_id
        response = self._post_json(self.open_session_endpoint, payload)
        if not isinstance(response, str):
            msg = (
                f"Expected string session id response from {self.open_session_endpoint}, "
                f"got {type(response).__name__}."
            )
            raise RuntimeError(msg)
        return response

    def close_session(self, session_id: str) -> None:
        self._post_json(self.close_session_endpoint, {"session_id": session_id})

    def _post_json(self, endpoint: str, payload: dict[str, Any]) -> Any:
        request = Request(
            url=self._resolve_url(endpoint),
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                **self.headers,
            },
            method="POST",
        )
        try:
            with self.opener(request, self.timeout) as response:
                raw = response.read()
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            msg = f"SGLang control-plane request to {endpoint} failed with {exc.code}: {detail}"
            raise RuntimeError(msg) from exc
        except URLError as exc:
            msg = f"Failed to reach SGLang control plane at {self._resolve_url(endpoint)}: {exc.reason}"
            raise RuntimeError(msg) from exc
        except (HTTPException, OSError) as exc:
            msg = (
                f"SGLang control-plane request to {self._resolve_url(endpoint)} "
                f"failed before a response was returned: {exc}"
            )
            raise RuntimeError(msg) from exc
        if not raw:
            return None
        try:
            return json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            msg = (
                f"Failed to decode JSON response from {self._resolve_url(endpoint)}: "
                f"{exc.msg}"
            )
            raise RuntimeError(msg) from exc

    def _resolve_url(self, endpoint: str) -> str:
        return urljoin(self.base_url.rstrip("/") + "/", endpoint.lstrip("/"))


@dataclass
class SGLangLookaheadSession:
    client: SGLangControlPlaneClient
    session_id: str
    dispatcher: SGLangLookaheadDispatcher

    def bind_memory(self, memory: Any) -> Any:
        memory.lookahead_dispatcher = self.dispatcher
        memory.lookahead_session_id = self.session_id
        return memory

    def warm_text(
        self,
        *,
        strategy: str,
        text: str,
        task_id: str | None = None,
        control: dict[str, Any] | None = None,
    ) -> str | None:
        warmed_task_id = task_id or uuid.uuid4().hex
        accepted = self.dispatcher.trigger(
            LookaheadTask(
                session_id=self.session_id,
                task_id=warmed_task_id,
                strategy=strategy,
                source_text=text,
                control=dict(control or {}),
            )
        )
        if not accepted:
            return None
        return warmed_task_id

    def append_text(
        self,
        *,
        strategy: str,
        task_id: str,
        text: str,
        control: dict[str, Any] | None = None,
    ) -> bool:
        return self.dispatcher.append(
            LookaheadTask(
                session_id=self.session_id,
                task_id=task_id,
                strategy=strategy,
                source_text=text,
                control=dict(control or {}),
            )
        )

    def commit_text(
        self,
        *,
        strategy: str,
        task_id: str,
        text: str | None = None,
        control: dict[str, Any] | None = None,
    ) -> LookaheadArtifact | None:
        return self.dispatcher.commit(
            LookaheadTask(
                session_id=self.session_id,
                task_id=task_id,
                strategy=strategy,
                source_text=text,
                control=dict(control or {}),
            )
        )

    def warm_sub_agent(
        self,
        *,
        system_prompt: str,
        task_text: str,
        prefix_text: str = "",
        task_id: str | None = None,
        control: dict[str, Any] | None = None,
    ) -> str | None:
        merged_control = dict(control or {})
        merged_control["sub_agent_system_prompt"] = system_prompt
        return self.warm_text(
            strategy="sub_agent",
            text=self._build_sub_agent_text(
                system_prompt=system_prompt,
                prefix_text=prefix_text,
                task_text=task_text,
            ),
            task_id=task_id,
            control=merged_control,
        )

    def commit_sub_agent(
        self,
        *,
        system_prompt: str,
        task_text: str,
        prefix_text: str = "",
        task_id: str,
        control: dict[str, Any] | None = None,
    ) -> LookaheadArtifact | None:
        merged_control = dict(control or {})
        merged_control["sub_agent_system_prompt"] = system_prompt
        return self.commit_text(
            strategy="sub_agent",
            task_id=task_id,
            text=self._build_sub_agent_text(
                system_prompt=system_prompt,
                prefix_text=prefix_text,
                task_text=task_text,
            ),
            control=merged_control,
        )

    @staticmethod
    def _build_sub_agent_text(
        *,
        system_prompt: str,
        task_text: str,
        prefix_text: str = "",
    ) -> str:
        return build_sub_agent_text(
            system_prompt=system_prompt,
            prefix_text=prefix_text,
            task_text=task_text,
        )

    def close(self) -> None:
        self.client.close_session(self.session_id)
