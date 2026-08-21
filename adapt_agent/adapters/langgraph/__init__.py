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

from adapt_agent.adapters._governed import GovernedAdapter, _extract_texts


class LangGraphAdapter(GovernedAdapter):
    """Adapter for integrating ADAPT-Agent with LangGraph agents.

    Wrap a **compiled** LangGraph graph -- or any object exposing a callable
    ``invoke(state) -> state``. See :class:`~adapt_agent.adapters._governed.GovernedAdapter`
    for the full constructor signature; every security control is optional and
    keyword-only.
    """

    framework_name = "LangGraph"
    run_method_names = ("invoke",)
    async_run_method_names = ("ainvoke", "invoke")
    operation = "langgraph.invoke"


__all__ = ["LangGraphAdapter", "_extract_texts"]
