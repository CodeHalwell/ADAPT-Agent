"""Native governance hooks: plug ADAPT-Agent into a framework's own chain.

An adapter (:mod:`adapt_agent.adapters`) wraps an agent from the *outside*. That
is the right tool when a framework has no interception point of its own, but it
has two costs in a real deployment:

* **It only sees the boundary.** Wrapping a multi-agent graph -- a
  ``WorkflowBuilder(...).build().as_agent()``, a supervisor with four
  specialists -- governs the raw request going in and the final answer coming
  out. It cannot apply a different rule to the specialist that reads untrusted
  email than to the one that drafts the reply, because from the outside they are
  one object.
* **It is another layer to fight.** The wrapper has to own the call, so it
  competes with the workflow runtime for control of execution and with any
  middleware the application already runs.

Every framework here already has an interception point, and most are async by
contract. Plugging in *there* is better on four counts at once: governance nests
per-agent inside a graph, it is async-native, it does not fight the runtime, and
it composes with the middleware/callbacks an app already stacks.

============================  ==========================================  ======================
Framework                     Native seam                                 Factory
============================  ==========================================  ======================
Microsoft Agent Framework     ``Agent(middleware=[...])``                 :func:`~adapt_agent.integrations.agent_framework.governance_middleware`
Google ADK                    ``LlmAgent(before_model_callback=...)``     :func:`~adapt_agent.integrations.google_adk.governance_callbacks`
OpenAI Agents SDK             ``Agent(input_guardrails=[...])``           :func:`~adapt_agent.integrations.openai_agents.governance_guardrails`
Claude Agent SDK              ``ClaudeAgentOptions(hooks={...})``         :func:`~adapt_agent.integrations.claude_agent.governance_hooks`
LangGraph                     ``create_react_agent(pre_model_hook=...)``  :func:`~adapt_agent.integrations.langgraph.governance_hooks`
CrewAI                        ``Crew(before_kickoff_callbacks=[...])``    :func:`~adapt_agent.integrations.crewai.governance_callbacks`
Pydantic AI                   ``@agent.output_validator`` (output only)   :func:`~adapt_agent.integrations.pydantic_ai.governance_output_validator`
============================  ==========================================  ======================

Every factory takes the same controls (``firewall``, ``defense``,
``policy_enforcer``, ``block_on_violation``, ``agent_id``) and shares one
implementation, :class:`~adapt_agent.core.governance.GovernanceGate` -- so the
rules do not drift between frameworks, or between this and the adapters.

``agent_id`` is worth setting per agent: it is named in the raised
:class:`~adapt_agent.exceptions.SecurityBlockedError`, which is how you tell
*which* specialist in a graph refused.

Importing this package imports no framework. Each sub-module imports its SDK
lazily, and only for the parts that genuinely need a framework type (building an
ADK refusal response, wrapping an OpenAI guardrail). The hooks themselves are
duck-typed, so they can be unit-tested without the SDK installed.

Pydantic AI is the one partial: it has a native *output* validator but no
pre-run hook, so screen its input with an adapter or ``aexecute``. See that
module's docstring.
"""

from adapt_agent.core.governance import GovernanceGate

__all__ = ["GovernanceGate"]
