# Framework Guides

ADAPT-Agent governs and optimizes agents built with seven frameworks. Each guide
below is a self-contained walkthrough, from a tiny guarded agent to a governed,
optimized multi-agent system, with runnable offline examples under
[`examples/<framework>/`](https://github.com/CodeHalwell/ADAPT-Agent/tree/main/examples).

Every framework gets the same three things:

- **Governance inside the run (native binding).** A factory in
  `adapt_agent.integrations.<framework>` binds one `GovernanceGate` to the
  framework's own extension seams -- middleware, hooks, plugins, guardrails,
  callbacks -- so **every model call and every tool call** is screened where it
  happens: a denied tool never runs, a poisoned tool result never reaches the
  next model call, and an `"ask"` uses the framework's own approval mechanism.
  This is the layer to start with; it is the only one that can stop a tool.
- **Governance at the boundary (adapter).** A `GovernedAdapter` subclass wraps the
  agent so each `execute` / `aexecute` runs the pipeline -- input screening
  (`Firewall` + `AdversarialDefense`) → policy (`PolicyEnforcer`) →
  pre-middleware → traced run (`AgentObserver`) → post-middleware → output
  screening. It sees the run's input and final result only.
- **Training (offline).** `OptimizableAgent` turns the agent (or a whole system)
  into a tunable search space, scored against a `GoldenDataset`, which the
  optimizers and an `LLMJudge` improve in place.

The bindings and the adapter can share one gate (`gate=adapter.gate`, or a
`GovernanceGate` you build), so the same `Firewall`, `AdversarialDefense`,
`PolicyEnforcer` and `ToolPolicy` decide in both. Put the tool policy on the gate
you share: a factory given `gate=` takes only `agent_id` besides it.

## The native bindings at a glance

Every cell below is taken from the binding's code and its guide. "Approvals"
is what a gate `"ask"` (a `ToolPolicy(require_approval=...)` match or a
`require_approval` policy rule) becomes; where that cannot complete, every
binding turns the ask into a block rather than letting the call run.

| Framework | Binding factory | Tool seam | Approvals | Run-level option |
|-----------|-----------------|-----------|-----------|------------------|
| [LangGraph / LangChain v1](langgraph.md) | `langgraph.governance_middleware(...)` for `create_agent(middleware=[...])`; `governance_tool_wrapper(...)` for a hand-built `ToolNode` | `AgentMiddleware.wrap_tool_call` / `ToolNode(wrap_tool_call=...)` | LangGraph `interrupt()`, resumed with `Command(resume=...)`; needs a checkpointer | `governance_callback_handler(...)` in `config={"callbacks": [...]}` reaches every model and tool call of the run, sub-agents included -- abort-only |
| [Pydantic AI](pydantic_ai.md) | `pydantic_ai.governance_capability(...)` in `Agent(capabilities=[...])` | capability `before_tool_execute` / `after_tool_execute` | `ApprovalRequired` → the run returns `DeferredToolRequests`; resume with `DeferredToolResults` | `agent.run(..., capabilities=[governance])` for one run; a delegate agent's run does not inherit it |
| [CrewAI](crewai.md) | `crewai.governance_hooks(...)` as a context manager (or `.install()`) | `crewai.hooks` `pre_tool_call` / `post_tool_call` | `approve=True` (CrewAI's console prompt) or a callable; `approve=None` fails closed | the hooks are process state: every crew, agent and flow run inside the block (or after `install()`) is covered |
| [OpenAI Agents SDK](openai_agents.md) | `openai_agents.govern_tools(agent, ...)` | `ToolInputGuardrail` / `ToolOutputGuardrail` on every `FunctionTool` and `MCPServer` | the SDK's `needs_approval` → `RunResult.interruptions` → `RunState.approve` / `reject` → resume | `Runner.run(..., hooks=governance_run_hooks(...))` fires for every agent the run reaches, handoff targets included |
| [Claude Agent SDK](claude_agent.md) | `claude_agent.governance_hooks(...)` in `ClaudeAgentOptions(hooks=...)` | `PreToolUse` / `PostToolUse` hooks (typed `permissionDecision`; never `"allow"`) | `permissionDecision: "ask"` → the SDK's `can_use_tool` (`governance_can_use_tool(inner=...)`) or the CLI prompt | hooks on the options cover the main agent and every sub-agent; there is nothing to attach per agent |
| [Google ADK](google_adk.md) | `google_adk.governance_plugin(...)` in `App(plugins=[...])`; `governance_callbacks(...)` per `LlmAgent` | plugin / `before_tool_callback` / `after_tool_callback` | ADK tool confirmation (`tool_context.request_confirmation`), answered with an `adk_request_confirmation` function response | the plugin itself: ADK calls it for every agent and tool in the tree (sub-agents, `AgentTool`s, `Workflow` nodes) |
| [Microsoft Agent Framework](microsoft_agent_framework.md) | `agent_framework.governance_bundle(...)` in `Agent(..., middleware=[...])` (agent + chat + function middleware) | function middleware, once per tool call | MAF `function_approval_request` on `AgentResponse.user_input_requests`, resumed on the same `AgentSession`; no session → block | `agent.run(..., middleware=[bundle])` governs one run of an agent you did not build |

### Streaming, MCP and versions

A framework that streams model output hands tokens to the consumer before the
complete response can be screened. Where the binding has an option that holds a
response back until it has passed, its name and default are below; without one,
a blocked response is stopped from reaching the final result, but tokens
already streamed have been seen.

| Framework | Stream screening (binding option, default) | MCP tools | Extra / declared floor |
|-----------|---------------------------------------------|-----------|------------------------|
| LangGraph | `governance_middleware(screen_stream=False)`; `True` tags the model call `nostream`, so `stream_mode="messages"` gets each message whole, after screening. `LangGraphAdapter.stream` / `astream` screen before release. | `langchain-mcp-adapters` tools are ordinary `BaseTool`s and go through the same seam (not exercised by the test suite) | `adapt-agent[langgraph]`: `langgraph>=1.0`, `langchain>=1.0` |
| Pydantic AI | `governance_capability(buffer_streams=True)`: a streamed response is drained, screened, then replayed | `capabilities.MCP(...)` / `MCPToolset` tools hit the same tool hooks (verified with an in-process server) | `adapt-agent[pydantic-ai]`: `pydantic-ai>=2.0` |
| CrewAI | none -- CrewAI's hooks fire after a call completes, so no hook can hold a chunk; `CrewAIAdapter` drains a `stream=True` crew and returns the output only after screening | `Agent(mcps=[...])` tools are `BaseTool`s and run through the same hooks | `adapt-agent[crewai]`: `crewai>=1.0`, but the hooks need **CrewAI 1.15+** (`crewai.hooks`; `tool_inspection_supported()` reports it) |
| OpenAI Agents SDK | `govern_stream(result, ...)` over `Runner.run_streamed` holds each model response's text events until it is screened (opt in by using it); `OpenAIAgentsAdapter.run_streamed(...)` returns one | MCP servers' tool guardrails, set by `govern_tools`; a hosted MCP tool without `on_approval_request` is not covered | `adapt-agent[openai-agents]`: `openai-agents>=0.20` |
| Claude Agent SDK | none -- the SDK has no seam that holds text back; `ClaudeAgentSDKAdapter` drains the stream and screens it before returning | `mcp__<server>__<tool>` tools pass through the same hooks; `native_tool_options(policy, mcp_servers=...)` derives the CLI's own allow/deny lists | `adapt-agent[claude-agent]`: `claude-agent-sdk>=0.2` |
| Google ADK | `governance_plugin(buffer_stream=False)`; `True` screens SSE partials as an aggregate and releases nothing until it passes (`stream_holdback=N` releases all but the last `N` characters) | `McpToolset` tools go through the tool callbacks (verified with a real stdio server); `mcp_toolset_kwargs(...)` hides denied tools from the model | `adapt-agent[google-adk]`: `google-adk>=2.0` |
| Microsoft Agent Framework | `governance_bundle(buffer_streams=True)`: every update is held until the response passed; `False` stops the stream at the update that completes a match | MCP tools surface as function tools and go through the function middleware (verified with a stdio server); provider-hosted tools do not | `adapt-agent[microsoft-agent-framework]`: `agent-framework-core>=1.0` |

The floors are the versions `pyproject.toml` declares. The CI job that installs
each framework at its floor (`frameworks-floor`) is non-blocking until it has been
seen green, so a floor is declared rather than proven; the guides are written
against the latest releases. Install **one framework per environment**: they
constrain each other's dependencies, and `adapt-agent[all]` leaves CrewAI out for
that reason.

### Stated gaps

Each guide has a "what is and is not covered" section; the headline limits:

- **LangGraph** -- middleware is per agent and does not inherit into sub-agents
  (use the callback handler as a run-wide net); `ToolMessage.artifact`, the
  system prompt and custom `get_stream_writer()` output are not screened.
- **Pydantic AI** -- a delegate agent's run needs the capability passed to it;
  `install_governance` on an existing agent is output-only.
- **CrewAI** -- A2A delegation (`Agent(a2a=...)`) has no hook and is not
  governed; streamed chunks cannot be held back; `governance_guardrail` does not
  survive a checkpoint restore.
- **OpenAI Agents SDK** -- input guardrails run for the starting agent only (use
  `governance_run_hooks`); server-side tools (`WebSearchTool`,
  `FileSearchTool`, `CodeInterpreterTool`, `ImageGenerationTool`) have no
  client-side seam, and `ComputerTool` / `LocalShellTool` no per-call approval.
- **Claude Agent SDK** -- not verified end to end (the CLI cannot run offline);
  a block is always a model-visible refusal (no raise mode); a built-in tool whose
  output schema constrains a flagged field may keep its original result.
- **Google ADK** -- live (bidi) runs are not buffered and cannot pause for an
  approval; an `AgentTool` with `include_plugins=False` escapes the plugin;
  non-text content is not screened; ADK wraps a plugin's exception in
  `RuntimeError` (`reraise_security_blocks()` / `find_security_error()`, or the
  adapter, recover it).
- **Microsoft Agent Framework** -- provider-hosted tools run outside the function
  loop; system instructions are not screened; function middleware placed after the
  bundle can change arguments after the gate approved them; A2A and
  `as_mcp_server()` are untested. Output screening judges what the agent *says*:
  function-call arguments in a response are left to the function layer.

## What a refusal reports

Every binding and adapter reports the same threat labels (on
`SecurityBlockedError.threats`, `ToolDecision.threats` and the observer):

| Label | Raised by |
|-------|-----------|
| `firewall` | a blocked pattern, custom filter or length cap of the `Firewall` |
| `prompt_injection`, `jailbreak`, `role_spoofing`, `custom_pattern` | `AdversarialDefense` |
| `content_too_long` | `AdversarialDefense` input over its `max_content_length` (nothing scanned) |
| `unscannable` | content the gate could not fully walk (too deep, too many nodes, an unreadable object) -- blocked rather than passed unseen |
| `payload_too_large` | a payload over the gate's `max_payload_chars` budget |
| `policy:<rule>` | a `block` (or, on a tool call, `require_approval`) policy rule |
| `tool:denied:<name>`, `tool:not_allowed:<name>`, `tool:approval:<name>` | the `ToolPolicy` deny list, allow-list and approval list |

## The adapters

| Framework | Adapter | Wrap target |
|-----------|---------|-------------|
| LangGraph | `LangGraphAdapter` | a compiled graph (`.invoke` / `.stream`) |
| Microsoft Agent Framework | `MicrosoftAgentFrameworkAdapter` | an `agent_framework.Agent` (`.run`; called `ChatAgent` before 1.0), or `workflow.as_agent()` |
| Google ADK | `GoogleADKAdapter` | a `Runner`, an `App`, or a callable driving a `Runner` |
| Pydantic AI | `PydanticAIAdapter` | an `Agent` (`.run_sync` / `.run`) |
| CrewAI | `CrewAIAdapter` | a `Crew` (`.kickoff` / `.akickoff`) |
| OpenAI Agents SDK | `OpenAIAgentsAdapter` | an `Agent`, driven through `Runner` |
| Claude Agent SDK | `ClaudeAgentSDKAdapter` | the SDK's `query` function |

An adapter forwards extra keyword arguments of `execute(input, **run_kwargs)` /
`aexecute(...)` to the framework's run method. With pre-middleware configured it
requires a dict input and raises `AdapterError` for anything else -- including a
LangGraph `Command` -- because the middleware cannot screen what it cannot read.

Each guide covers, in detail: what you wrap, installing the extra and the
import-safety guarantee, governance inside the run and at the boundary,
policy/observability/trust/taint, what the optimizer introspects for that
framework (and how to declare the knobs it can't), evaluation, the judge
(including adversarial mode), the optimizers, tool/skill ablation, the
declarative YAML training config, multi-agent orchestration, and a
pitfalls/FAQ section.

New to the library? Start with the [Quick Start](../quickstart.md), then the
[Native Framework Hooks](../integrations.md), [Framework Adapters](../adapters.md)
and [Optimization & Evaluation](../optimization.md) overviews, then your
framework's guide above.
