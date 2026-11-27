# Microsoft Agent Framework

This page is a complete guide to using
[Microsoft Agent Framework](https://github.com/microsoft/agent-framework) (MAF)
with **ADAPT-Agent**. It is written against `agent-framework-core` 1.26 and
assumes you are new to both, so every concept is explained the first time it
appears.

ADAPT-Agent (**A**dversarial **D**efense & **P**olicy **T**raining for LLM
agents) has two independent halves:

- **Guard (runtime).** Screen what goes into and out of every model call, decide
  every tool call before it runs and every tool result before the model reads
  it, enforce policy rules, and record traces. For MAF this binds to the
  framework's own middleware seams (§3); an outer wrapper is also available
  (§4).
- **Train (offline).** Turn an agent or a workflow into a tunable search space,
  score it against a golden dataset, and let an optimizer search for a better
  configuration (§6).

Both halves are **import-safe**: importing `adapt_agent` never imports
`agent_framework`. The framework is only needed when you build an agent.

---

## 1. What Microsoft Agent Framework is

MAF is Microsoft's unified successor to **Semantic Kernel** and **AutoGen**.
The names below are the 1.x ones; several changed from the pre-1.0 previews.

| Concept | 1.x name | Pre-1.0 name |
|---------|----------|--------------|
| The runnable agent | `agent_framework.Agent` | `ChatAgent` |
| Its result | `AgentResponse` (`.text`, `.messages`, `.user_input_requests`) | `AgentRunResponse` |
| A streamed update | `AgentResponseUpdate` | `AgentRunResponseUpdate` |
| A chat message / its parts | `Message` / `Content` | `ChatMessage` / `*Content` classes |
| Build an agent from a client | `Agent(client, instructions, ...)` or `client.as_agent(...)` | `client.create_agent(...)` |
| Conversation state | `AgentSession` (`agent.create_session()`) | `AgentThread` |

- **Chat clients** talk to a model. Provider clients ship in separate packages:
  `OpenAIChatClient` is in `agent-framework-openai` (`agent_framework.openai`),
  Azure, Anthropic, Ollama and others have their own.
- **`agent.run(messages, *, stream=False, session=None, middleware=None,
  tools=None, options=None, ...)`** returns an awaitable resolving to an
  `AgentResponse` -- or, with `stream=True`, a `ResponseStream` you iterate for
  `AgentResponseUpdate`s and then `await stream.get_final_response()`. `run` is
  no longer declared `async def` (it returns an awaitable object), so wrap it
  when you need a coroutine: `asyncio.run(_go())` with `async def _go(): return
  await agent.run(...)`.
- **Tools** are functions decorated with `@tool` (or plain callables);
  `@tool(approval_mode="always_require")` makes every call pause for approval.
- **Middleware** comes in three kinds -- agent, chat, function -- described in §3.
- **Workflows** (`WorkflowBuilder`) compose agents and executors into a graph;
  `workflow.as_agent()` turns one into a single agent. The Magentic, group-chat,
  handoff and sequential orchestrations are in the separate
  `agent-framework-orchestrations` package (`agent_framework.orchestrations`).

The smallest real agent:

```python
from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient   # pip install agent-framework-openai

agent = Agent(OpenAIChatClient(), "You are a concise assistant.", name="assistant")
response = await agent.run("What is the capital of France?")
print(response.text)                                   # -> "Paris"
```

---

## 2. Installing

```bash
pip install 'adapt-agent[microsoft-agent-framework]'    # or: pip install agent-framework-core
```

`adapt_agent.integrations.agent_framework` and
`MicrosoftAgentFrameworkAdapter` import `agent_framework` lazily, only when a
real MAF type is needed (MAF's middleware decorators, `Content` for an approval
request, `MiddlewareBundle`). Everything in this guide that needs a provider
client, MCP server or orchestration needs that package too.

---

## 3. Governing inside the agent: `governance_bundle`

### 3.1 The one call

```python
from agent_framework import Agent, tool
from adapt_agent import AdversarialDefense, Firewall
from adapt_agent.core.governance import ToolPolicy
from adapt_agent.integrations.agent_framework import governance_bundle

firewall = Firewall()
firewall.add_blocked_pattern(r"(?i)ignore (all|previous) instructions")

bundle = governance_bundle(
    firewall=firewall,
    defense=AdversarialDefense(),
    tool_policy=ToolPolicy(deny=["delete_*"], require_approval=["issue_refund"]),
    on_tool_violation="refuse",      # or "raise"
    agent_id="support",
)

agent = Agent(client, "You are a support agent.", tools=[...], middleware=[bundle])
```

`governance_bundle(...)` returns a MAF `MiddlewareBundle` of three middleware
that MAF installs as one indivisible unit (it refuses to install part of a
bundle). It takes the same controls as every ADAPT-Agent binding -- `gate`,
`firewall`, `defense`, `policy_enforcer`, `block_on_violation`, `agent_id`,
`tool_policy` -- plus `on_tool_violation`, `approvals`, `screen_output`,
`screen_model_calls`, `buffer_streams` and `observer`. Pass `gate=` to share one
configured `GovernanceGate` across several agents; each binding's `agent_id`
still names the agent in errors and spans.

`MiddlewareBundle` is marked experimental by MAF; on a release without it the
three middleware come back as a plain list, which works the same way.

The same bundle works:

- **per agent** -- `Agent(..., middleware=[bundle])`;
- **per run** -- `agent.run(..., middleware=[bundle])`, the run-level seam: use it
  when one code path must govern agents it did not build;
- **inside a workflow** -- give each agent its own bundle (§3.8).

### 3.2 What each layer sees

Verified against the installed release with an offline chat client:

| Layer | MAF seam | Runs | Sees | Does |
|-------|----------|------|------|------|
| agent | `@agent_middleware` | once per `agent.run` | the caller's **new** messages only (not session history, not context-provider content) | screens input; evaluates `policy_enforcer` once per run; screens the final response |
| chat | `@chat_middleware` | once per **model call**, inside the tool loop | the full request: history, `ContextProvider` messages (RAG, memory), tool results of the previous step | screens every non-`assistant`, non-`system` message; screens each response before MAF acts on it or persists it |
| function | `@function_middleware` | once per **tool call** (MCP tools included) | tool name, arguments, result | `check_tool_call` before, `check_tool_result` after |

Two of those facts are why the chat layer exists:

- **Context-provider content never reaches the agent layer.** A RAG provider
  that injects a poisoned document is only visible to the chat layer.
- **MAF persists a response to the session before the agent middleware regains
  control.** A response blocked only at the agent layer was still in the
  session's history on the next turn, so the next model call read it. The chat
  layer blocks first, and the blocked response is never persisted.

### 3.3 Tool calls

Before a tool runs the gate decides on its name (`ToolPolicy`), its arguments
(firewall + adversarial defense over every string, keys included) and any
tool-scoped policy rule. After it returns, the result is screened as *input*,
because it is about to become the model's input.

| Decision | `on_tool_violation="refuse"` (default) | `on_tool_violation="raise"` |
|----------|------------------------------------------|------------------------------|
| block (call) | tool does not run; the model receives `decision.reason` as the tool result -- e.g. `Tool call 'delete_account' refused by ADAPT-Agent [support]: tool:denied:delete_account.` -- and the run continues | the run aborts; the caller gets `SecurityBlockedError` with `.threats` |
| block (result) | the result is replaced by `Tool result from 'read_ticket' withheld by ADAPT-Agent [support]: firewall.` before the next model call | the run aborts before the next model call |
| ask | §3.4 | §3.4 |

**Why refuse is the default.** MAF's tool loop is built to carry on after a
failed tool: the model sees the failure and can explain it or try something
else. Refusing keeps that behaviour, and is exactly as safe -- the tool never
runs, the poisoned result never reaches the model.

**How raise really aborts.** An ordinary exception raised in MAF function
middleware is converted into a tool *error* and fed back to the model -- the run
continues. Only `MiddlewareFailure` aborts the loop. In raise mode ADAPT-Agent
raises an error that is both a `MiddlewareFailure` and a `SecurityBlockedError`,
so MAF aborts and you still write `except SecurityBlockedError as exc:
exc.threats`. This holds inside workflows too (§3.8).

The refusal and placeholder texts carry threat *labels* only, never the
arguments or result content they withhold.

**Placement.** MAF runs function middleware in list order. Put the bundle after
any argument-repair function middleware of your own: arguments changed after
the gate approved them would reach the tool unscreened. (`governance_middleware`,
§3.9, always installs its tool layer innermost.)

### 3.4 Approvals ("ask")

A tool matching `ToolPolicy(require_approval=[...])`, or a tool-scoped policy
rule with `action="require_approval"`, produces an **ask**. The function layer
turns it into a MAF `function_approval_request`; the run pauses and returns an
`AgentResponse` with empty text and the request on `user_input_requests`:

```python
session = agent.create_session()
paused = await agent.run("Refund order A1", session=session)
(request,) = paused.user_input_requests
request.function_call.name                       # 'issue_refund'
request.additional_properties["reason"]          # content-free, for the approver
request.additional_properties["adapt_agent_threats"]

answer = Message(role="user", contents=[request.to_function_approval_response(True)])
resumed = await agent.run([answer], session=session)   # the tool runs now
```

- **Resume on the same `AgentSession`.** MAF records the request in the session
  and binds the response to it -- including the arguments -- so an edited resume
  message cannot swap them. Without a session MAF ignores approval responses it
  cannot bind, so an ask with no session **fails closed**: it is refused like a
  block (`Tool call 'shell' needs approval ...`).
- On the approved replay the gate re-checks the call: an approval upgrades an
  ask to an allow, never a block to an allow.
- A rejection (`to_function_approval_response(False)`) never runs the tool; MAF
  tells the model the call was rejected.
- `approvals=False` treats every ask as a block.
- A tool declared `@tool(approval_mode="always_require")` is paused by MAF
  itself before the gate sees it; on the approved replay the gate still
  decides.

### 3.5 Streaming

```python
stream = agent.run("hi", stream=True)
async for update in stream:          # nothing arrives until the response passed
    print(update.text, end="")
final = await stream.get_final_response()
```

With `buffer_streams=True` (default) both output checks ride the stream's
gates with buffering on (`stream_buffer_updates`): updates are held until the
response has passed screening, so a blocked response is never delivered -- the
iteration raises `SecurityBlockedError` instead. The cost is time-to-first-token:
the consumer receives the response only once it is complete.

`buffer_streams=False` screens the accumulated text as each update arrives and
stops the stream at the update that completes a match. **The text before that
update has already been delivered.** Use it only where a partial leak is
acceptable.

On a release whose contexts have no stream gates, a streamed run is refused
(`SecurityBlockedError(["stream:unscreenable"])`) rather than passed through.

### 3.6 MCP tools

MCP tools surface as MAF function tools, so their calls go through the function
layer like any other -- verified with a real `MCPStdioTool` connected to a local
stdio server: a denied MCP tool never runs and an injected MCP result is
withheld. To also narrow what the model is *offered*:

```python
from agent_framework import MCPStdioTool
from adapt_agent.integrations.agent_framework import mcp_tool_options

policy = ToolPolicy(deny=["wipe*"], require_approval=["send_*"])
files = MCPStdioTool(
    "files", command="uvx", args=["my-mcp-server"],
    **mcp_tool_options(policy, tool_names=["lookup", "wipe_all", "send_mail"]),
)
```

- `allowed_tools` -- the server's tools the policy does not refuse. MAF matches
  exact names, so glob patterns need `tool_names=`; without them the helper
  raises rather than return a filter that filters nothing.
- `approval_mode` -- a callable MAF evaluates per discovered tool:
  `"always_require"` for tools the policy sends for approval.
- `sampling_approval_callback` -- only with `gate=`: approves a server's
  sampling request (the server asking *your* model to generate) only if it
  passes the gate's input screening. MAF denies sampling by default.

Names here are the server's raw tool names. With `tool_name_prefix=` the
function layer sees the prefixed name; write the policy to match it.

Tools executed by the *provider* (hosted tools, a service-side MCP connector,
code interpreter) do not run in MAF's function loop and are **not** seen by the
function layer.

### 3.7 Policy state

A fail-closed `PolicyEnforcer` (the default) treats a rule it cannot evaluate as
violated, so the state each hook builds is deliberate:

| Hook | `state` keys a rule can rely on |
|------|----------------------------------|
| agent layer (once per run) | `messages` (the caller's new messages), `context` (`{}`), and every key of the run's `AgentSession.state` |
| function layer (tool-scoped rules) | `tool` (`{"name", "arguments"}`), `messages` (`[]`), `context`, and every key of `AgentSession.state` |
| chat layer | policy is not evaluated (content screening only), so a `warn`/`log` rule does not re-fire on every tool step |

Keep a trust score or account flag in the session:

```python
session = agent.create_session()
session.state["trust_score"] = 0.2
await agent.run("...", session=session)   # a rule "state['trust_score'] < 0.5" blocks
```

Without a session, rules see only `messages` and `context`.

### 3.8 Workflows and multi-agent teams

Give each agent in a workflow its own bundle. A shared gate with per-agent ids
makes every refusal name the agent:

```python
gate = GovernanceGate(firewall=firewall, tool_policy=ToolPolicy(deny=["shell"]))
intake = Agent(c1, "Route.", name="intake", middleware=[governance_bundle(gate=gate, agent_id="intake")])
coder = Agent(c2, "Code.", name="coder", tools=[shell],
              middleware=[governance_bundle(gate=gate, agent_id="coder")])
workflow = WorkflowBuilder(start_executor=intake).add_edge(intake, coder).build()
team = workflow.as_agent()
```

Verified: a refused tool inside `coder` never runs, and in raise mode the
`SecurityBlockedError` (naming `[coder]`) reaches the caller of `team.run`.
Wrapping `team` with the adapter (§4) governs only its outer boundary; the
per-agent bundles are what govern the specialists' model and tool calls.

After a run that raised, the workflow is in `WorkflowRunState.FAILED` and its
`as_agent()` wrapper refuses to start again -- rebuild the workflow.

### 3.9 The original factory: `governance_middleware`

`governance_middleware(...)` keeps its name and signature and now also takes
`tool_policy`, `on_tool_violation`, `approvals`, `govern_tools`,
`screen_model_calls` and `buffer_streams`. It returns a single agent middleware
that, for each run it governs, appends the chat and function layers to the
run's `client_kwargs["middleware"]` (the list MAF hands the chat client) --
innermost, after your own. Existing code therefore gains tool governance and
stream-safe output screening with no change. Pass `govern_tools=False,
screen_model_calls=False` for the old boundary-only behaviour.

The layers are also available individually: `governance_chat_middleware(...)`
and `governance_function_middleware(...)` -- for example on a chat client
(`client = SomeChatClient(middleware=[...])`), which accepts only chat and
function middleware.

### 3.10 Report-only rollout

`block_on_violation=False` lets everything through -- inputs, outputs, denied
tools, poisoned results -- while the gate still computes every decision
(`decision.report_only` is set). Spans are recorded as usual. Use it to measure
false positives before enforcing.

### 3.11 Observability

Pass `observer=AgentObserver()` to record a span per run
(`agent_framework.run`), per model call (`agent_framework.model_call`) and per
tool call (`agent_framework.tool`). A streamed span stays open until the stream
has been consumed and screened, and closes as `error` when it was blocked.
MAF's own OpenTelemetry instrumentation
(`agent_framework.observability.configure_otel_providers` /
`enable_instrumentation`) is independent and composes with this.

### 3.12 What is and is not covered

Covered: caller input, every model request (history, context providers, tool
results), every model response, every tool call and result run by MAF's
function loop (MCP included), approvals, streams, workflows, run-level
installation.

Not covered, by design or by limitation:

- **System instructions** are developer content and are not screened.
- **Provider-hosted tools** run outside MAF's function loop (§3.6).
- **Unbuffered streams** can deliver the text before a match (§3.5).
- **An ask needs an `AgentSession`**; without one it is refused (§3.4).
- **Function middleware placed after the bundle** can change arguments after
  the gate approved them (§3.3).
- **MAF's experimental security features** (`agent_framework.security`:
  `SecureAgentConfig`, `PolicyEnforcementFunctionMiddleware`) and the
  AGENT-HOOKS-0.1 bundle (`create_agent_hooks_middleware`, needs the separate
  `agent-hooks-sdk`) are independent controls. ADAPT-Agent does not ship an
  AGENT-HOOKS `Interceptor`: its protocol lives in `agent-hooks-sdk`, not in
  the core package. They can be stacked; MAF documents that middleware placed
  before an enforcement bundle runs outside its boundary.
- **A2A and `Agent.as_mcp_server()`** serve runs through `agent.run`, so the
  agent's own middleware applies; this binding has not been tested through
  either transport.

---

## 4. Governing the boundary: `MicrosoftAgentFrameworkAdapter`

The adapter wraps a whole agent (or `workflow.as_agent()`) and runs ADAPT-Agent's
6-step pipeline around each call: input screening, policy, pre-middleware, the
traced run, post-middleware, output screening.

```python
from adapt_agent.adapters import MicrosoftAgentFrameworkAdapter

adapter = MicrosoftAgentFrameworkAdapter(
    firewall=firewall,
    defense=AdversarialDefense(),
    policy_enforcer=PolicyEnforcer(),
    observer=AgentObserver(),
    agent_id="support",
    block_on_violation=True,
)
guarded = adapter.wrap_agent(agent)
out = guarded.execute({"messages": [{"role": "user", "content": "Hello"}]})
out["result"].text
```

- `execute(payload, **run_kwargs)` (and `await aexecute(...)` in async code)
  forward keyword arguments to `agent.run`: `session=`, `stream=True`,
  `middleware=[bundle]` (run-level governance), `tools=`, `options=`.
- A payload of plain `{"role", "content"}` dicts is reduced to its latest user
  message. A payload whose `messages` contain MAF `Message`/`Content` objects is
  passed through as a message list -- which is how an approval is resumed.
- The result is `{"result": <AgentResponse>}`. A run paused for approval also
  carries `"user_input_requests"`; a streamed run is drained, screened, and
  returned as `{"result": <final AgentResponse>, "updates": [...]}`.

```python
session = agent.create_session()
paused = guarded.execute({"prompt": "Refund order A1"}, session=session)
answers = [r.to_function_approval_response(True) for r in paused["user_input_requests"]]
guarded.execute({"messages": [Message(role="user", contents=answers)]}, session=session)
```

The adapter sees only the boundary. A tool call, a tool result or a
context-provider document inside the run is invisible to it; use the bundle
(§3) for those, alone or together with the adapter.

`block_on_violation=False` turns the adapter into an audit tool: threats are
detected and recorded but nothing raises. See
[`01_basic_guarded.py`](../../examples/microsoft_agent_framework/01_basic_guarded.py)
and [`02_policy_observability_trust.py`](../../examples/microsoft_agent_framework/02_policy_observability_trust.py).

---

## 5. Policy, adversarial defense, trust & taint

`PolicyEnforcer` evaluates conditions in a **safe expression language** (no
`eval`; a restricted AST: indexing, `in`, comparisons, boolean operators, no
function calls). In the adapter pipeline a rule sees
`state["messages"]` (the payload's message dicts) and `state["context"]`
(everything else in the payload); inside the agent it sees the keys in §3.7.

```python
policy = PolicyEnforcer()
policy.add_rule(
    name="no_password_leak",
    description="Never let the user request stored passwords.",
    condition="'password' in state['messages'][0]['content']",
    action="block",
)
policy.add_rule(
    name="refunds_need_a_human",
    description="Refund tool calls need approval.",
    condition="tool['name'] == 'issue_refund'",
    action="require_approval",      # a tool-scoped rule: evaluated per tool call
)
```

`AdversarialDefense()` adds heuristic prompt-injection detection on input;
`TrustManager` and `TaintTracker` are standalone helpers you run alongside --
see example 2.

---

## 6. Optimization (offline "training")

### 6.1 What gets introspected for an `Agent`

`detect(agent)` returns `"microsoft_agent_framework"` and `introspect(agent)`
returns the agent's tunable parameters: its instructions (`PROMPT`), sampling
options such as `temperature` from `default_options` (`HYPERPARAM`), and its
`tools` (`TOOL`, with drop-one ablation candidates).

```python
from adapt_agent.optimization.introspection import detect, introspect

detect(agent)        # "microsoft_agent_framework"
for p in introspect(agent):
    print(p.name, p.kind.value)
```

### 6.2 Evaluate and optimize

```python
from adapt_agent.optimization import (
    CoordinateAscentOptimizer, EvaluationHarness, GoldenDataset, LLMJudge,
    OptimizableAgent, exact_match,
)

async def _ask(q):
    return (await agent.run(q)).text

target = OptimizableAgent.from_agent(agent, runner=lambda q: asyncio.run(_ask(q)),
                                     component_name="capital_expert")
data = GoldenDataset.from_list([{"input": "What is the capital of France?", "expected": "Paris"}])
judge = LLMJudge(my_completion_fn)               # any str -> str callable, or a provider
harness = EvaluationHarness(metrics=[exact_match(), judge.as_metric("quality")],
                            primary_metric="exact_match")
result = CoordinateAscentOptimizer(harness, judge=judge, seed=0).optimize(target, data)
result.best_config                                # applied in place
```

`make_default_optimizer(...)` runs the full pipeline (few-shot, prompts,
models/hyperparameters/routing, tools/skills); `GridSearchOptimizer`,
`RandomSearchOptimizer`, `EvolutionaryOptimizer` and
`BootstrapFewShotOptimizer` are the single strategies. For offline runs back
the judge with a deterministic callable, as the examples do.

### 6.3 Tool ablation and explicit knobs

A real `Agent` keeps its tools in `default_options["tools"]`. Declare a wider
pool, including tools it does not have yet:

```python
target.add_tool_parameter(
    "researcher.tool_pool",                       # distinct from the introspected name
    kind=ParameterKind.TOOL,
    getter=lambda: list(researcher.default_options.get("tools") or []),
    setter=lambda v: researcher.default_options.__setitem__("tools", list(v)),
    candidate_tools=[web_search, wiki, arxiv],
)
```

Knobs no attribute exposes -- workflow routing limits -- are explicit
`Parameter(kind=ParameterKind.ROUTING, getter=..., setter=...)` whose setter
**rebuilds the workflow**.

### 6.4 The YAML path

`run_training("train.yaml")` runs a whole optimization from a declarative file;
see [`magentic.train.yaml`](../../examples/microsoft_agent_framework/magentic.train.yaml)
and Part D of example 4.

---

## 7. Multi-agent teams

A workflow built with `WorkflowBuilder` (or an orchestration from
`agent-framework-orchestrations`, such as `MagenticBuilder`) becomes one agent
with `workflow.as_agent()`. For governance, put a bundle on each participant
(§3.8) and, if you like, the adapter around the team (§4). For optimization:

1. **`as_agent()` exposes no knobs.** Register the underlying agents as
   `components` and run the workflow agent as the runner:
   `OptimizableAgent.from_components(components={...}, runner=team.run)`.
2. **Routing limits are not auto-discovered.** Declare them as `ROUTING`
   parameters whose setter rebuilds the workflow.
3. **A workflow agent keeps its conversation across runs.** Each `run` adds to
   the same history; a responder or metric that reads the *first* user message
   sees the first question forever.

See [`04_magentic_team_and_training.py`](../../examples/microsoft_agent_framework/04_magentic_team_and_training.py).

---

## 8. Common pitfalls / FAQ

- **`AttributeError: module 'agent_framework' has no attribute 'ChatAgent'`.**
  Renamed to `Agent` in 1.0; `AgentRunResponse` is now `AgentResponse`, and
  clients expose `as_agent(...)` rather than `create_agent(...)`.
- **`ImportError` for `agent_framework.openai` / `orchestrations` / `a2a`.**
  Those are separate packages: `agent-framework-openai`,
  `agent-framework-orchestrations`, `agent-framework-a2a`.
- **`asyncio.run(agent.run(...))` raises "a coroutine was expected".**
  `run` returns an awaitable object; wrap it in an `async def`.
- **My function middleware raised, and the run carried on.** MAF turns an
  ordinary function-middleware exception into a tool error for the model. Only
  `MiddlewareFailure` aborts (§3.3).
- **An approval never resumes.** Resume on the same `AgentSession` the paused
  run used (§3.4).
- **A blocked answer showed up in the next turn.** That happens with agent-layer
  screening alone; keep `screen_model_calls=True` (§3.2).
- **A streamed answer arrives all at once.** That is the buffering that keeps a
  blocked answer from being delivered (§3.5).
- **The workflow agent refuses to run after a block.** Rebuild the workflow
  (§3.8).
- **Policy rule never fires, or always blocks.** Check the keys the hook
  provides (§3.7); a fail-closed enforcer blocks on a missing key.
- **Do I need an API key?** No: every example uses an offline chat client.

---

## 9. The examples

All four drive real MAF objects with an offline chat client (no API key, no
network) and exit cleanly when MAF is not installed:

1. [`01_basic_guarded.py`](../../examples/microsoft_agent_framework/01_basic_guarded.py)
   -- the bundle on a real `Agent`: clean tool run, blocked injection, denied
   tool, withheld tool result, approval round trip, blocked stream, the adapter.
2. [`02_policy_observability_trust.py`](../../examples/microsoft_agent_framework/02_policy_observability_trust.py)
   -- policy + defense + observer + middleware + trust/taint in audit mode, then
   a policy rule on `AgentSession.state`.
3. [`03_evaluate_and_optimize.py`](../../examples/microsoft_agent_framework/03_evaluate_and_optimize.py)
   -- introspect, evaluate and optimize one agent.
4. [`04_magentic_team_and_training.py`](../../examples/microsoft_agent_framework/04_magentic_team_and_training.py)
   + [`magentic.train.yaml`](../../examples/microsoft_agent_framework/magentic.train.yaml)
   -- a five-agent workflow governed per agent and as a unit, trained as a whole
   and per agent, with tool ablation and the YAML path.
