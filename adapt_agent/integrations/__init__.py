"""Native governance bindings: plug ADAPT-Agent into a framework's own seams.

An adapter (:mod:`adapt_agent.adapters`) wraps an agent from the *outside*: it
sees the request going in and the final answer coming out. Two costs follow in
a real deployment:

* **By the time the answer is screened, the agent has already acted.** Every
  tool has run, and a tool result carrying an injected instruction has already
  been read by the next model call. Stopping an agent from *doing* something has
  to happen at the tool call.
* **It only sees the boundary.** Wrapping a multi-agent graph -- a supervisor
  with four specialists, a ``WorkflowBuilder(...).build().as_agent()`` --
  governs the run as one object, so it cannot apply a different rule to the
  specialist that reads untrusted mail than to the one that drafts the reply.

Every framework here has interception points around each model call, each tool
call and the run as a whole. The bindings in this package plug the same
governance into them: per agent inside a graph, async-native, without fighting
the runtime, and composable with the middleware an app already stacks.

=========================  ==================================================  ====================
Framework                  What it governs                                     Factory (start here)
=========================  ==================================================  ====================
LangGraph / LangChain v1   model and tool calls of a ``create_agent`` agent    :func:`~adapt_agent.integrations.langgraph.governance_middleware`
Pydantic AI                input, tools (MCP too), approvals, streamed output  :func:`~adapt_agent.integrations.pydantic_ai.governance_capability`
CrewAI (1.15+)             tools, models, steps of every crew in a ``with``    :func:`~adapt_agent.integrations.crewai.governance_hooks`
OpenAI Agents SDK          tool calls, handoffs, agents-as-tools, MCP          :func:`~adapt_agent.integrations.openai_agents.govern_tools`
Claude Agent SDK           prompt, tool calls and results, sub-agents too      :func:`~adapt_agent.integrations.claude_agent.governance_hooks`
Google ADK                 model and tool calls of every agent in an ``App``   :func:`~adapt_agent.integrations.google_adk.governance_plugin`
Microsoft Agent Framework  input, every model call, every tool (MCP too)       :func:`~adapt_agent.integrations.agent_framework.governance_bundle`
=========================  ==================================================  ====================

Each module also keeps narrower or older seams: a ``ToolNode`` wrapper, a
run-level callback handler and the deprecated ``create_react_agent`` hooks
(LangGraph); an output validator (Pydantic AI); kickoff callbacks and task
guardrails (CrewAI); run hooks, a model-input filter, stream governance and
entry-agent guardrails (OpenAI Agents); ``can_use_tool`` and native CLI tool
options (Claude); per-agent callbacks (ADK); separate chat and function
middleware (MAF). ``docs/integrations.md`` compares them, including what each
seam cannot see.

All of them implement one contract on
:class:`~adapt_agent.core.governance.GovernanceGate`:
:meth:`~adapt_agent.core.governance.GovernanceGate.check_tool_call` and
:meth:`~adapt_agent.core.governance.GovernanceGate.check_tool_result` return a
:class:`~adapt_agent.core.governance.ToolDecision` -- ``"allow"``, ``"block"``
or ``"ask"`` -- from a :class:`~adapt_agent.core.governance.ToolPolicy`
(allow / deny / approval globs), firewall and adversarial screening of the
arguments or result, and tool-scoped policy rules (``action="require_approval"``
included). A binding maps a block to the framework's refusal or abort
(``on_tool_violation="refuse"`` or ``"raise"``) and an ask to the framework's
approval mechanism; where it has none, the ask fails closed to a block
(:meth:`~adapt_agent.core.governance.ToolDecision.resolve`).
``block_on_violation=False`` makes any binding report-only.

Every factory takes the same controls (``firewall``, ``defense``,
``policy_enforcer``, ``block_on_violation``, ``agent_id``, ``tool_policy``) or a
shared ``gate=``, so the rules do not drift between frameworks, or between a
binding and an adapter. ``agent_id`` is worth setting per agent: it is named in
the raised :class:`~adapt_agent.exceptions.SecurityBlockedError` and in every
span, which is how you tell *which* specialist in a graph refused.

Importing this package imports no framework. Each sub-module imports its SDK
lazily, only where a factory has to build a framework object (a LangChain
middleware class, an ADK plugin, OpenAI guardrails, Claude hook matchers), and
raises :class:`ImportError` naming the extra to install when it is missing.
Callbacks that need no framework type are duck-typed, so they can be
unit-tested without the SDK installed.
"""

from adapt_agent.core.governance import GovernanceGate, ToolDecision, ToolPolicy

__all__ = ["GovernanceGate", "ToolDecision", "ToolPolicy"]
