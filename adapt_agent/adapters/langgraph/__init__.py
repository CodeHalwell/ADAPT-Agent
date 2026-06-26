"""LangGraph adapter for ADAPT-Agent.

This adapter wraps a compiled LangGraph graph (or any object exposing an
``invoke`` method) with ADAPT-Agent's security and governance primitives:
a :class:`~adapt_agent.security.Firewall`, an
:class:`~adapt_agent.adversarial.AdversarialDefense`, a
:class:`~adapt_agent.core.PolicyEnforcer`, an
:class:`~adapt_agent.observability.AgentObserver` and a
:class:`~adapt_agent.core.Middleware` pipeline.

The adapter is intentionally *structural*: it integrates with anything that
looks like a compiled LangGraph graph (a callable ``invoke(state) -> state``)
rather than importing LangGraph at module load time. The optional ``langgraph``
dependency is only needed at runtime to build the graph you pass in; importing
this module never imports ``langgraph``.

Example
-------
>>> from adapt_agent.adapters import LangGraphAdapter
>>> from adapt_agent.security import Firewall
>>> fw = Firewall(max_content_length=10_000)
>>> fw.add_blocked_pattern(r"(?i)ignore previous instructions")
>>> adapter = LangGraphAdapter(firewall=fw)
>>> guarded = adapter.wrap_agent(compiled_graph)   # doctest: +SKIP
>>> guarded.execute({"messages": [{"role": "user", "content": "hi"}]})  # doctest: +SKIP
"""

import uuid
from typing import Any, Optional

from adapt_agent.adapters.base import BaseAdapter
from adapt_agent.adversarial import AdversarialDefense
from adapt_agent.core.middleware import Middleware
from adapt_agent.core.policy import PolicyEnforcer
from adapt_agent.core.types import Agent, AgentState
from adapt_agent.exceptions import AdapterError, SecurityBlockedError
from adapt_agent.observability import AgentObserver
from adapt_agent.security.firewall import Firewall


def _extract_texts(data: Any) -> list[str]:
    """Best-effort extraction of human-readable text from a LangGraph payload.

    LangGraph state is typically a dict that may contain a ``messages`` list or
    arbitrary string fields. We collect strings so security controls can scan
    them without assuming a fixed schema.
    """
    texts: list[str] = []

    def _walk(value: Any, depth: int = 0) -> None:
        if depth > 6:  # bound recursion (defensive against pathological nesting)
            return
        if isinstance(value, str):
            texts.append(value)
        elif isinstance(value, dict):
            for v in value.values():
                _walk(v, depth + 1)
        elif isinstance(value, (list, tuple)):
            for v in value:
                _walk(v, depth + 1)
        else:
            # LangChain message objects expose a ``.content`` attribute.
            content = getattr(value, "content", None)
            if isinstance(content, str):
                texts.append(content)

    _walk(data)
    return texts


class LangGraphAdapter(BaseAdapter):
    """Adapter for integrating ADAPT-Agent with LangGraph agents.

    Args:
        config: Optional adapter configuration dictionary.
        firewall: Optional :class:`Firewall` used to screen inputs/outputs.
        defense: Optional :class:`AdversarialDefense` used to detect prompt
            injection and jailbreak attempts on inputs.
        policy_enforcer: Optional :class:`PolicyEnforcer` evaluated against the
            extracted agent state before execution.
        observer: Optional :class:`AgentObserver` used to trace executions.
        middleware: Optional :class:`Middleware` pipeline applied to the input
            payload (pre) and the result payload (post).
        agent_id: Stable identifier used in traces and policy checks.
        block_on_violation: When ``True`` (default), a firewall/defense hit or a
            ``block`` policy action raises :class:`SecurityBlockedError`. When
            ``False`` the execution proceeds but the threats are still recorded.
    """

    def __init__(
        self,
        config: Optional[dict[str, Any]] = None,
        *,
        firewall: Optional[Firewall] = None,
        defense: Optional[AdversarialDefense] = None,
        policy_enforcer: Optional[PolicyEnforcer] = None,
        observer: Optional[AgentObserver] = None,
        middleware: Optional[Middleware] = None,
        agent_id: str = "langgraph-agent",
        block_on_violation: bool = True,
    ):
        super().__init__(config)
        self.firewall = firewall
        self.defense = defense
        self.policy_enforcer = policy_enforcer
        self.observer = observer
        self.middleware = middleware
        self.agent_id = agent_id
        self.block_on_violation = block_on_violation

    def validate_agent(self, agent: Any) -> bool:
        """Return ``True`` if ``agent`` looks like a runnable LangGraph graph."""
        return callable(getattr(agent, "invoke", None))

    def wrap_agent(self, agent: Any) -> Agent:
        """Wrap a compiled LangGraph graph with ADAPT-Agent capabilities.

        Args:
            agent: A compiled LangGraph graph (anything with an ``invoke`` method).

        Returns:
            An object implementing the :class:`~adapt_agent.core.types.Agent`
            protocol (``execute`` and ``get_state``).

        Raises:
            AdapterError: If ``agent`` does not expose a callable ``invoke``.
        """
        if not self.validate_agent(agent):
            raise AdapterError(
                "LangGraphAdapter.wrap_agent expects a compiled LangGraph graph "
                "with a callable 'invoke' method. Compile your graph with "
                "graph.compile() before wrapping it."
            )
        return _WrappedLangGraphAgent(agent, self)

    def extract_state(self, agent: Any) -> AgentState:
        """Extract an :class:`AgentState` from a graph or a raw state payload.

        Accepts either a state mapping (e.g. ``{"messages": [...], ...}``) or a
        stateful graph exposing ``get_state``. Extraction is best-effort and
        always returns a well-formed :class:`AgentState`.
        """
        raw: Any = agent
        get_state = getattr(agent, "get_state", None)
        if callable(get_state):
            try:
                snapshot = get_state(self.config.get("graph_config", {}))
                raw = getattr(snapshot, "values", snapshot)
            except Exception:
                raw = {}

        if not isinstance(raw, dict):
            raw = {}

        messages = raw.get("messages", [])
        if not isinstance(messages, list):
            messages = []

        context = {k: v for k, v in raw.items() if k != "messages"}

        state: AgentState = {"messages": messages, "context": context}
        if "trust_score" in raw and isinstance(raw["trust_score"], (int, float)):
            state["trust_score"] = float(raw["trust_score"])
        if "policy_violations" in raw and isinstance(raw["policy_violations"], list):
            state["policy_violations"] = raw["policy_violations"]
        return state

    def inject_middleware(self, agent: Any, middleware: Any) -> Any:
        """Attach a middleware pipeline and return a freshly wrapped agent.

        Args:
            agent: A compiled LangGraph graph.
            middleware: A :class:`~adapt_agent.core.Middleware` instance.

        Returns:
            A wrapped agent whose ``execute`` runs the middleware pipeline.
        """
        if not isinstance(middleware, Middleware):
            raise AdapterError("inject_middleware expects an adapt_agent.core.Middleware instance.")
        self.middleware = middleware
        return self.wrap_agent(agent)

    # -- internal helpers shared with the wrapped agent ------------------------

    def _screen_input(self, payload: dict[str, Any]) -> list[str]:
        """Run firewall + adversarial defense over a payload, returning threats."""
        threats: list[str] = []
        texts = _extract_texts(payload)
        for text in texts:
            if self.firewall is not None and not self.firewall.check_input(text):
                threats.append("firewall")
            if self.defense is not None:
                analysis = self.defense.analyze_input(text)
                if not analysis["is_safe"]:
                    threats.extend(analysis["threats_detected"])
        return threats

    def _screen_output(self, payload: Any) -> list[str]:
        """Run firewall over an output payload, returning threats."""
        threats: list[str] = []
        if self.firewall is None:
            return threats
        for text in _extract_texts(payload):
            if not self.firewall.check_output(text):
                threats.append("firewall")
        return threats


class _WrappedLangGraphAgent:
    """A LangGraph graph wrapped with ADAPT-Agent governance.

    Implements the :class:`~adapt_agent.core.types.Agent` protocol.
    """

    def __init__(self, graph: Any, adapter: LangGraphAdapter):
        self._graph = graph
        self._adapter = adapter
        self._last_state: AgentState = {"messages": [], "context": {}}

    def execute(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Run the wrapped graph with governance applied.

        Order of operations: input screening -> policy check -> pre-middleware
        -> ``graph.invoke`` (traced) -> post-middleware -> output screening.

        Raises:
            SecurityBlockedError: If a control blocks the input/output (and
                ``block_on_violation`` is enabled).
        """
        adapter = self._adapter
        trace_id = uuid.uuid4().hex

        # 1. Input screening.
        in_threats = adapter._screen_input(input_data)
        if in_threats and adapter.block_on_violation:
            raise SecurityBlockedError("Input blocked by security controls", in_threats)

        # 2. Policy enforcement against the extracted state.
        self._last_state = adapter.extract_state(input_data)
        if adapter.policy_enforcer is not None:
            violations = adapter.policy_enforcer.check_state(self._last_state)
            if violations and adapter.block_on_violation:
                blocking = []
                for v in violations:
                    rule = adapter.policy_enforcer.get_rule(v)
                    if rule is not None and rule.get("action") == "block":
                        blocking.append(v)
                if blocking:
                    raise SecurityBlockedError(
                        "Input blocked by policy", [f"policy:{v}" for v in blocking]
                    )

        # 3. Pre-middleware.
        payload = input_data
        if adapter.middleware is not None:
            payload = adapter.middleware.process_input(input_data)

        # 4. Traced execution.
        if adapter.observer is not None:
            adapter.observer.start_trace(trace_id, adapter.agent_id, "langgraph.invoke")
        try:
            result = self._graph.invoke(payload)
        except Exception as exc:
            if adapter.observer is not None:
                adapter.observer.end_trace(trace_id, status="error", result=str(exc))
            raise
        if adapter.observer is not None:
            adapter.observer.end_trace(trace_id, status="completed")

        # 5. Post-middleware.
        if adapter.middleware is not None:
            wrapped = adapter.middleware.process_output({"result": result})
            result = wrapped["result"]

        # 6. Output screening.
        out_threats = adapter._screen_output(result)
        if out_threats and adapter.block_on_violation:
            raise SecurityBlockedError("Output blocked by security controls", out_threats)

        if isinstance(result, dict):
            self._last_state = adapter.extract_state(result)
        return result if isinstance(result, dict) else {"result": result}

    def get_state(self) -> AgentState:
        """Return the most recently observed agent state."""
        return self._last_state


__all__ = ["LangGraphAdapter"]
