# Native framework hooks

An [adapter](adapters.md) wraps an agent from the outside. That is the right
tool when a framework has no interception point of its own, but it has two
costs in a real deployment.

**It only sees the boundary.** Wrapping a multi-agent graph — a
`WorkflowBuilder(...).build().as_agent()`, a supervisor with four specialists —
governs the raw request going in and the final answer coming out. It cannot
apply a different rule to the specialist reading untrusted email than to the
intake router, because from the outside they are one object.

**It is another layer to fight.** The wrapper owns the call, so it competes with
the workflow runtime for control of execution and with any middleware the
application already runs.

Every framework below already has an interception point, and most are async by
contract. `adapt_agent.integrations` plugs the same governance into it, which is
better on four counts at once: governance nests per agent inside a graph, it is
async-native, it does not fight the runtime, and it composes with the
middleware an app already stacks.

## The matrix

| Framework | Native seam | Factory |
| --- | --- | --- |
| Microsoft Agent Framework | `Agent(middleware=[...])` | `agent_framework.governance_middleware()` |
| Google ADK | `LlmAgent(before_model_callback=...)` | `google_adk.governance_callbacks()` |
| OpenAI Agents SDK | `Agent(input_guardrails=[...])` | `openai_agents.governance_guardrails()` |
| OpenAI Agents SDK — handoff target | `Agent(hooks=...)` | `openai_agents.governance_agent_hooks()` |
| Claude Agent SDK | `ClaudeAgentOptions(hooks={...})` | `claude_agent.governance_hooks()` |
| LangGraph | `create_react_agent(pre_model_hook=...)` | `langgraph.governance_hooks()` |
| CrewAI | `Crew(before_kickoff_callbacks=[...])` | `crewai.governance_callbacks()` |
| Pydantic AI | `@agent.output_validator` (output only) | `pydantic_ai.install_governance()` |

Every factory takes the same controls — `firewall`, `defense`,
`policy_enforcer`, `block_on_violation`, `agent_id` — or a shared `gate=`, and
all of them run one `GovernanceGate`. The rules cannot drift between frameworks,
or between these hooks and the adapters.

Importing `adapt_agent.integrations` imports no framework. Each sub-module
imports its SDK lazily and only where a framework *type* is genuinely required
(building an ADK refusal response, wrapping an OpenAI guardrail object).

## Microsoft Agent Framework

```python
from adapt_agent.integrations.agent_framework import governance_middleware

agent = chat_client.create_agent(
    instructions="...",
    middleware=[
        usage_middleware("nos"),                              # the app's own
        governance_middleware(firewall=fw, agent_id="nos"),   # composes with it
    ],
)
```

MAF middleware is `async (context, call_next)`, so this path never touches the
sync/async bridge and works unchanged inside a running event loop.

## Google ADK

```python
from adapt_agent.integrations.google_adk import governance_callbacks

agent = LlmAgent(name="intake", model="gemini-2.0-flash",
                 **governance_callbacks(firewall=fw, agent_id="intake"))
```

`on_block="refuse"` returns an `LlmResponse` that short-circuits the model
instead of raising, so one blocked branch does not abort a whole agent tree.
The default `"raise"` matches every other ADAPT-Agent entry point.

## LangGraph

```python
from adapt_agent.integrations.langgraph import governance_hooks

agent = create_react_agent(model, tools, **governance_hooks(firewall=fw))
```

These fire on **every** model call inside the graph — including one whose input
came from a tool result rather than from the user, which is exactly where
injected content arrives. A policy rule here also sees the real graph state, so
it may reference any key the graph carries, not just the `messages`/`context` an
adapter exposes. For a hand-built `StateGraph`, add `governance_node(gate)` as a
node yourself.

## Claude Agent SDK

```python
from adapt_agent.integrations.claude_agent import governance_hooks

options = ClaudeAgentOptions(hooks=governance_hooks(firewall=fw))
```

`UserPromptSubmit` screens the prompt; `PreToolUse` screens **tool inputs**,
which an outer wrapper cannot reach at all; `PostToolUse` screens **tool
results**. A Claude agent loops through many tool calls inside one `query()`,
and whatever a tool fetched from the open web comes back through `PostToolUse` —
`PreToolUse` alone catches an injection only if the model copies it into a
*subsequent* tool call, so an agent that simply reads a page and answers would
never have it screened.

`matcher=` is a **tool name** (`"Bash"`, `"Write|Edit"`), so it is applied to
`PreToolUse`/`PostToolUse` only. Attaching one to `UserPromptSubmit` would
describe a prompt event that can never match, silently disabling prompt
screening.

Hooks return `{"decision": "block", "reason": ...}`, so the model sees the
refusal and can respond to it rather than the run dying.

## OpenAI Agents SDK

```python
from adapt_agent.integrations.openai_agents import governance_guardrails

agent = Agent(name="triage", instructions="...",
              **governance_guardrails(firewall=fw, agent_id="triage"))
```

**Guardrails cover the entry point, not a handoff target.** The SDK runs input
guardrails for the *starting* agent only (`run.py` gates them on `current_turn
== 0`), so a specialist reached by a handoff never runs its own — verified by
driving a real handoff against the installed SDK:

```text
input guardrails that ran: ['triage']
handoff happened: True (final agent: specialist)
specialist's own input guardrail ran: False
```

For those agents use `governance_agent_hooks()`, which binds to
`AgentHooks.on_llm_start` — per agent, per model call, so it also screens tool
results on their way back to the model — and to `on_end`, so the specialist's
own answer is screened too. Pass `inner=` to keep the app's own
lifecycle hooks running.

```python
from adapt_agent.integrations.openai_agents import governance_agent_hooks

specialist = Agent(name="specialist", instructions="...",
                   hooks=governance_agent_hooks(firewall=fw, agent_id="specialist"))
```

Tripping a guardrail raises the SDK's own `InputGuardrailTripwireTriggered`, so
a handoff chain reports the block the way it reports any other. The threat list
travels on `exc.guardrail_result.output.output_info["threats"]`.

## CrewAI

```python
from adapt_agent.integrations.crewai import governance_callbacks

crew = Crew(agents=[...], tasks=[...], **governance_callbacks(firewall=fw))
```

Crew callbacks fire once per kickoff. For per-task screening inside a long run,
add `governance_guardrail(gate)` to the `Task` objects handling untrusted
content — it returns `(ok, message)` and CrewAI retries on `False`.

## Pydantic AI

```python
from adapt_agent.integrations.pydantic_ai import install_governance

install_governance(agent, firewall=fw, agent_id="triage")
```

This is the one partial. Pydantic AI has a native *output* validator but no
pre-run hook, so `install_governance` covers outputs only, and passing it a
`policy_enforcer` or `defense` raises rather than accepting a control it cannot
honour. Screen inputs with
[`PydanticAIAdapter`](adapters.md) or by calling `gate.review_input(...)` before
`agent.run`. Using both is the recommended setup.

A structured `output_type` is screened field by field: the gate walks Pydantic
model and dataclass fields, so an injection smuggled into one field of a
structured answer is still caught.

## Report-only rollout

`block_on_violation=False` scans and records without refusing anything, which is
how you shake out false positives before enforcing:

```python
gate = GovernanceGate(firewall=fw, block_on_violation=False)
threats = gate.review_input(payload)   # returns instead of raising
```
