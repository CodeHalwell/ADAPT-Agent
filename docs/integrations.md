# Native framework bindings

An [adapter](adapters.md) wraps an agent from the outside. It sees the request
going in and the final answer coming out, and nothing in between. Two things
follow in a real deployment:

**By the time the answer is screened, the agent has already acted.** Every tool
has run, every e-mail has been sent, and a tool result carrying an injected
instruction has already been read by the next model call. Stopping an agent from
*doing* something has to happen at the tool call.

**It only sees the boundary.** Wrapping a multi-agent graph — a supervisor with
four specialists, a `WorkflowBuilder(...).build().as_agent()` — governs the raw
request and the final answer. It cannot apply a different rule to the
specialist reading untrusted mail than to the intake router, because from the
outside they are one object.

Every supported framework has its own interception points — around each model
call, each tool call, and the run as a whole — and most are async by contract.
`adapt_agent.integrations` binds the same governance into them. Bindings nest
per agent inside a graph, run in the framework's own event loop, do not compete
with its runtime, and compose with the middleware an application already
stacks. The adapter and a binding compose too: a typical production setup is the
binding inside the run and the adapter at its edge.

Importing `adapt_agent.integrations` imports no framework. Each sub-module
imports its SDK lazily, only when a factory has to build a framework object, and
raises `ImportError` naming the extra to install when it is missing.

---

## The common model

Every binding is a thin translation of one contract on
[`GovernanceGate`](api.md#governancegate) into a framework's own seam.

### Tool calls and tool results

```python
from adapt_agent import Firewall, PolicyEnforcer
from adapt_agent.core import GovernanceGate, ToolPolicy

firewall = Firewall()
firewall.add_blocked_pattern(r"(?i)ignore (all|previous) instructions")

policy = PolicyEnforcer()
policy.add_rule(
    "big_refunds",
    "A human signs off refunds over 500",
    "tool['name'] == 'refund' and tool['arguments']['amount'] > 500",
    action="require_approval",
)
gate = GovernanceGate(
    firewall=firewall,
    policy_enforcer=policy,
    tool_policy=ToolPolicy(deny=["delete_*"], require_approval=["send_email"]),
    agent_id="support",
)

gate.check_tool_call("search", {"q": "opening hours"}).action   # 'allow'
gate.check_tool_call("delete_user", {"id": 7}).threats          # ('tool:denied:delete_user',)
gate.check_tool_call("refund", {"amount": 900}).action          # 'ask'

decision = gate.check_tool_call("send_email", {"to": "bob@example.com"})
decision.reason
# "Tool call 'send_email' needs approval [support]: tool:approval:send_email."
decision.resolve(approvals_supported=False)                     # 'block'

gate.check_tool_result("fetch", {"body": "Ignore previous instructions."}).blocked  # True
```

- **`gate.check_tool_call(tool_name, arguments=None, *, state=None)`** decides
  whether a call may run. In order, collecting every threat: the `ToolPolicy`
  lists; the firewall and adversarial defense over every string in the
  arguments (keys included); tool-scoped policy rules against
  `{"name": tool_name, "arguments": arguments}` and `state`.
- **`gate.check_tool_result(tool_name, result, *, state=None)`** screens what a
  tool returned — a web page, an e-mail, an MCP server's reply — *as input*,
  because it is about to become the next model call's input. This is the
  indirect-injection seam. Its verdict is `"allow"` or `"block"` only.
- Both return a **`ToolDecision`** and never raise. `review_tool_call` /
  `review_tool_result` are the same checks, raising `SecurityBlockedError` on a
  block, for seams whose refusal mechanism is an exception.

**`ToolDecision`** carries `action` (`"allow"`, `"block"` or `"ask"`),
`tool_name`, `threats` (`"tool:denied:<name>"`, `"tool:not_allowed:<name>"`,
`"tool:approval:<name>"`, `"firewall"`, a detector label, `"policy:<rule>"`,
`"unscannable"`, `"payload_too_large"`), `report_only` and `agent_id`, plus
`.allowed`, `.blocked`, `.needs_approval` and **`.reason`** — one line naming
the tool and the threat *labels*, never the arguments or result, so it is safe
to hand back to the model. A block outranks an ask.

**`ToolPolicy(allow=None, deny=(), require_approval=())`** holds shell-style
globs matched case-sensitively against the tool name as the framework names it
(`"mcp__*"`, `"github_*"`, `"Bash"`). Precedence is deny, then allow-list, then
approval: a denied tool is refused even if allow-listed, and a tool outside a
configured allow-list is refused rather than sent for approval. `allow=None`
restricts nothing; `allow=[]` refuses every tool. A bare string is rejected
(`deny="Bash"` would otherwise mean the four patterns `B`, `a`, `s`, `h`).

**Tool-scoped policy rules.** A `PolicyEnforcer` rule whose condition names
`tool` is evaluated only by `PolicyEnforcer.check_tool` (which the gate calls),
and skipped by `check_state` / `check_message` — so tool rules can live on the
same enforcer as state rules. Such a rule may combine `tool['name']`,
`tool['arguments']` and `state[...]`. The gate acts on a violated tool rule
whose action is `"block"` or **`"require_approval"`** (an ask); `add_rule` refuses `require_approval` on a rule that
does not name `tool`, because a framework can pause a tool call but not a
message that has already arrived. Under the default fail-closed enforcer, a
tool rule reading a `state` key the binding does not supply counts as violated;
each framework guide lists the keys its seams provide.

### Block: refuse or raise

`on_tool_violation` chooses what a blocked call or result does:

- **`"refuse"`** — the tool does not run (or its result is replaced by a
  content-free placeholder), the model receives `decision.reason` as the
  tool's result, and the run continues. The model can explain or try something
  else, and the other tool calls of the same turn are not thrown away.
- **`"raise"`** — the run aborts and the caller gets `SecurityBlockedError`
  with `.threats`. Where a framework swallows ordinary exceptions from its
  hooks (CrewAI) or turns them into a tool error the model reads (MAF function
  middleware), the binding raises a subclass of
  both the framework's abort exception and `SecurityBlockedError`, so the abort
  is real and `except SecurityBlockedError` still catches it.

The default is `"refuse"` everywhere except LangGraph and Pydantic AI, whose
default is `"raise"` (each framework guide gives its reasoning). The Claude Agent
SDK has no abort path from a hook, so `on_tool_violation="raise"` is refused
there with a `ValueError` rather than accepted and ignored.

### Ask: the framework's own approval mechanism, or a block

An `"ask"` — a `require_approval` glob or rule — maps onto each framework's
approval mechanism, so the tool has not run when the human is asked. Where the
run pauses and later resumes, the call is screened again on resume: an approval
satisfies the *ask* and nothing else, so arguments the firewall would block are
still blocked.

Where there is no way to pause — the framework has none, the run is not set up
for it (no checkpointer, no session), or the binding was built with
`approvals=False` — the binding calls `decision.resolve(approvals_supported=False)`
and the ask **fails closed to a block**. An ask never becomes an allow.

### Report-only rollout

`block_on_violation=False` makes the whole binding report-only. Every decision
that would have blocked or asked comes back as `"allow"` with
`report_only=True` and its threats still listed, so nothing is refused and
everything is recorded (on the binding's `observer=`, or its own event list).
That is how you measure false positives before enforcing:

```python
audit = GovernanceGate(firewall=firewall, tool_policy=ToolPolicy(deny=["delete_*"]),
                       block_on_violation=False)
decision = audit.check_tool_call("delete_user", {"id": 7})
decision.action, decision.report_only, decision.threats
# ('allow', True, ('tool:denied:delete_user',))
```

### One gate, many agents

Every factory takes the same controls — `firewall`, `defense`,
`policy_enforcer`, `block_on_violation`, `agent_id`, `tool_policy` — or a shared
`gate=`. The rules cannot drift between frameworks, or between a binding and an
adapter. Set `agent_id` per agent: it is named in every refusal, error and span,
which is how you tell *which* specialist in a graph refused.

When you pass `gate=`, the controls passed alongside it are **ignored** —
including `tool_policy=` — and only `agent_id` is applied (by building a gate
with the same controls under the new label). Put the tool policy on the gate
itself: `GovernanceGate(..., tool_policy=ToolPolicy(...))`.

### What every gate refuses to guess about

A payload the scanners cannot read in full — nested deeper than the walk goes,
more objects than its node budget, or holding a one-shot stream it would have to
consume — yields the threat `"unscannable"` and is blocked like any finding. A
payload whose distinct text exceeds the gate's `max_payload_chars` (4,000,000 by
default) is not scanned and yields `"payload_too_large"`. Both apply only when a
scanner (firewall or defense) is configured. See [Security](security.md).

---

## Support matrix

| Framework | Start here: the tool-call binding | Native seam | Also available |
| --- | --- | --- | --- |
| LangGraph / LangChain v1 | `langgraph.governance_middleware()` — every model call and tool call of a `create_agent` agent | `create_agent(middleware=[...])` | `governance_tool_wrapper()` for a hand-built `ToolNode`; `governance_callback_handler()` (run-level); `governance_node()`; `governance_hooks()` for the deprecated `create_react_agent` |
| Pydantic AI | `pydantic_ai.governance_capability()` — input, tool calls (MCP included), approvals, streamed and final output | `Agent(capabilities=[...])` or `agent.run(..., capabilities=[...])` | `install_governance()` / `governance_output_validator()` (output only) |
| CrewAI (1.15+) | `crewai.governance_hooks()` — every tool call and result, model call, task / flow step and run boundary | `crewai.hooks` interception layer, scoped by `with` | `install_hooks()`; `governance_callbacks()` (kickoff in/out); `governance_guardrail()` (per task) |
| OpenAI Agents SDK | `openai_agents.govern_tools()` — every tool call and result, handoff targets, agents-as-tools and MCP servers included | `ToolInputGuardrail` / `ToolOutputGuardrail`, `needs_approval` | `governance_run_hooks()` (every agent's model calls); `governance_model_input_filter()`; `govern_stream()`; `governance_guardrails()` (entry agent); `governance_agent_hooks()`; `governance_tool_guardrails()` |
| Claude Agent SDK | `claude_agent.governance_hooks()` — prompt, every tool call and tool result, sub-agents included | `ClaudeAgentOptions(hooks=...)` | `governance_can_use_tool()` (approver); `native_tool_options()` (CLI tool lists) |
| Google ADK | `google_adk.governance_plugin()` — every model call and tool call of every agent in the run | `App(plugins=[...])` | `governance_callbacks()` (one `LlmAgent`); `gate_confirmation()`; `mcp_toolset_kwargs()`; `reraise_security_blocks()` / `find_security_error()` |
| Microsoft Agent Framework | `agent_framework.governance_bundle()` — the caller's input, every model call, every tool call (MCP included) | `Agent(middleware=[...])` or `agent.run(..., middleware=[...])` | `governance_middleware()` (installs the same layers); `governance_chat_middleware()`; `governance_function_middleware()`; `mcp_tool_options()` |

## Comparison

| Framework | Tool seam | Approval ("ask") | Run-level coverage | Stream screening | MCP | Known gaps (details per section) |
| --- | --- | --- | --- | --- | --- | --- |
| LangGraph | middleware `wrap_tool_call`; `ToolNode(wrap_tool_call=...)` | LangGraph `interrupt()`, resumed with `Command(resume=...)`; needs a checkpointer, else block | `governance_callback_handler()` in `config["callbacks"]`: every model and tool call of the run, abort-only (an ask is a block) | `screen_stream=True` (default off); `LangGraphAdapter.stream` / `astream` | same `ToolNode` path; **unverified** | MCP untested; tokens streamed before a block without `screen_stream`; middleware is per agent |
| Pydantic AI | `before_tool_execute` / `after_tool_execute` | `ApprovalRequired` → `DeferredToolRequests`, resumed with `DeferredToolResults`; a run that cannot pause raises `SecurityBlockedError` | capability per run (`run(capabilities=[...])`); a delegate agent's run does not inherit it | `buffer_streams=True` (default): held at the model until screened | same hooks; verified with an in-process MCP server | system prompts not screened; durable-execution wrappers untested |
| CrewAI | `pre_tool_call` / `post_tool_call` hooks | `approve=`: `True` (console prompt), a callable, or `None` (default: block) | `with governance_hooks(...)` covers every crew, agent and flow in the block | none in the hooks; `CrewAIAdapter` drains `stream=True` crews before returning | same tool hook; pinned offline with `MCPNativeTool` | A2A delegation not hooked; streamed chunks cannot be held outside the adapter; needs CrewAI 1.15+ |
| OpenAI Agents SDK | tool input / output guardrails on every `FunctionTool` and `MCPServer` | `needs_approval` → `result.interruptions` → `RunState.approve` / `reject` → resume | `governance_run_hooks()` on `Runner.run(hooks=...)`: every agent's model calls | `govern_stream()` / `OpenAIAgentsAdapter.run_streamed()`; otherwise fail-open | server guardrails; an MCP ask needs the server's `require_approval`, else block | hosted server-side tools, `ComputerTool`, `LocalShellTool` have no per-call seam; Pydantic-plugin approval quirk |
| Claude Agent SDK | `PreToolUse` / `PostToolUse` hooks | `permissionDecision: "ask"` → `can_use_tool` or the CLI prompt; `approvals=False` makes it a deny | hooks on `ClaudeAgentOptions` cover the main agent and every sub-agent | none; `ClaudeAgentSDKAdapter` drains the stream before returning | same hooks (`mcp__<server>__<tool>`); verified at the hook level only | **behaviour against the real CLI unverified**; a built-in tool may reject a replaced result; no raise mode |
| Google ADK | `before_tool_callback` / `after_tool_callback` | `tool_context.request_confirmation` → `adk_request_confirmation`; live runs and `approvals=False` block | plugin on the `App`: every agent, sub-agent, `AgentTool` and `Workflow` node | `buffer_stream=True` (default off), optional `stream_holdback` | same callbacks; verified with a real stdio MCP server | live / bidi runs not buffered; non-text content not screened |
| Microsoft Agent Framework | function middleware | `function_approval_request` on `user_input_requests`; needs an `AgentSession`, else refused | bundle per agent or per run (`agent.run(middleware=[bundle])`); one bundle per agent in a workflow | `buffer_streams=True` (default) | function middleware; verified with a real `MCPStdioTool` | provider-hosted tools bypass function middleware; unbuffered streams leak the prefix |

"Verified" means exercised against the installed framework by
`tests/test_real_<framework>.py` (offline, with a scripted model); "unverified"
means the seam is the same code path but nothing in the test matrix drives it.

---

## LangGraph / LangChain v1

```python
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from adapt_agent.core import ToolPolicy
from adapt_agent.integrations.langgraph import governance_middleware

agent = create_agent(
    model,
    tools=[search, send_email, drop_table],
    middleware=[
        governance_middleware(
            firewall=firewall,
            tool_policy=ToolPolicy(deny=["drop_*"], require_approval=["send_email"]),
            on_tool_violation="refuse",   # default "raise"
            agent_id="support",
        )
    ],
    checkpointer=InMemorySaver(),         # only needed for approvals
)
```

`governance_middleware` screens each model request before the model is called
and each response before it is written to the graph state; it governs each tool
call (`check_tool_call`) and result (`check_tool_result`). An ask pauses the run
with `interrupt()` in the shape `HumanInTheLoopMiddleware` uses, so an approval
UI built for it renders unchanged; resume with
`agent.invoke(Command(resume={"type": "approve"}), config)`. Edited arguments
are re-screened. Without a checkpointer, or with `approvals=False`, an ask is a
block. Policy rules see the graph state's keys, `state['context']` from the
run's `Runtime.context`, and (tool rules) `tool['name']` / `tool['arguments']`.

- **Hand-built graphs:** `ToolNode(tools, **governance_tool_wrapper(...))` gives
  the same tool behaviour; `governance_node(gate)` is a node that screens the
  state's messages.
- **Run level:** `agent.invoke(inputs, {"callbacks": [governance_callback_handler(...)]})`
  screens every model and tool call of every agent the run starts, middleware or
  not. A callback cannot substitute a result, so it aborts on any block and an
  ask is a block.
- **Deprecated constructor:** `create_react_agent(model, tools, **governance_hooks(...))`
  still works (`pre_model_hook` / `post_model_hook`), with
  `ToolNode(tools, **governance_tool_wrapper(...))` for its tools.
  `create_agent` has no hook parameters; passing `governance_hooks(...)` to it is
  a `TypeError`.

**Known gaps.** MCP tools loaded with `langchain-mcp-adapters` run through the
same `ToolNode`, but that package is not in the test matrix, so MCP coverage is
**unverified**. With `stream_mode="messages"` and `screen_stream=False` (the
default), tokens streamed before a block have been seen by the consumer;
`screen_stream=True` delivers each message whole after screening.
Middleware does not inherit into sub-agents — use the callback handler. Not
screened: `ToolMessage.artifact`, the system prompt, and custom stream output
(`get_stream_writer()`) except through `LangGraphAdapter.stream`. Async
interrupts need Python 3.11+. Full guide:
[frameworks/langgraph.md](frameworks/langgraph.md).

## Pydantic AI

```python
from adapt_agent.core import ToolPolicy
from adapt_agent.integrations.pydantic_ai import governance_capability

governance = governance_capability(
    firewall=firewall,
    defense=defense,
    policy_enforcer=policy,
    tool_policy=ToolPolicy(deny=["delete_*"], require_approval=["send_email"]),
    agent_id="support-bot",
)
agent = Agent(model, tools=[...], capabilities=[governance])     # per agent
result = await other_agent.run(prompt, capabilities=[governance])  # or per run
```

Pydantic AI is fully covered through its capability API. Input is screened at
`before_model_request` (the prompt, `message_history`, every tool result);
tool calls at `before_tool_execute` / `after_tool_execute`, MCP tools included;
model output at `after_model_request` and structured output at
`after_output_validate`. An ask raises `ApprovalRequired`, so an agent with
`output_type=[str, DeferredToolRequests]` returns the pending calls and resumes
with `DeferredToolResults(approvals={call_id: True})`; a run that cannot pause
aborts with `SecurityBlockedError`. With `buffer_streams=True` (default) a
streamed response is held at the model until screened, so a blocked answer is
never delivered — at the price of token-by-token delivery. Policy rules read
the run's `deps` (mapping, dataclass or Pydantic model), or what
`policy_state=lambda ctx: {...}` returns.

`install_governance(agent, ...)` remains **output only**: Pydantic AI has no
public API to add a capability to an agent that already exists, so pass the
capability per run instead. It refuses `policy_enforcer`, `defense` and
`tool_policy` with a `ValueError`.

**Known gaps.** System prompts and instructions are not screened. A delegate
agent called from a tool does not inherit the parent's capability — give it the
capability too. Durable-execution wrappers (Temporal, DBOS, Prefect) are
untested. Full guide: [frameworks/pydantic_ai.md](frameworks/pydantic_ai.md).

## CrewAI

```python
from adapt_agent.core import ToolPolicy
from adapt_agent.integrations.crewai import governance_hooks

hooks = governance_hooks(
    firewall=firewall,
    tool_policy=ToolPolicy(deny=["delete_*"], require_approval=["send_email"]),
    approve=None,              # None: an ask is a block; True: console prompt; or a callable
)
with hooks:                    # scoped to this block, leaks nowhere else
    crew.kickoff(inputs={"topic": "AI"})
```

The binding uses CrewAI's interception layer (`crewai.hooks`, **CrewAI 1.15+**;
`tool_inspection_supported()` says whether the installed release has it). Every
crew, agent and flow running inside the block is covered: tool calls and
results (MCP tools included), model calls, task and flow steps, and the run
boundary. `on_tool_violation="refuse"` (default) skips the tool and tells the
model why; `"raise"` aborts `kickoff` with `SecurityBlockedError`. A blocked
model input or output, or run boundary, always raises. CrewAI swallows every
hook exception except `HookAborted`, so the binding raises `GovernanceBlocked`
(both a `HookAborted` and a `SecurityBlockedError`). `hooks.events` keeps a
bounded audit trail. `install_hooks(...)` registers process-wide instead.
`CrewAIAdapter(hooks=True, tool_policy=..., on_tool_violation=..., approve=...)`
runs each `execute` inside the same hooks.

`governance_callbacks()` (kickoff inputs and crew output, no tool coverage)
survives a CrewAI checkpoint restore: in a new process set
`CREWAI_DESERIALIZE_CALLBACKS=1` and call `governance_callbacks(..., agent_id=<same id>)`
before `Crew.from_checkpoint(...)`; otherwise the restored callback refuses with
`governance:gate_not_registered` instead of running ungoverned.

**Known gaps.** **A2A delegation** (`Agent(a2a=...)`) emits events only and has
no hook, so it is **not governed** by this binding. CrewAI's hooks fire when a
model call completes, so **streamed chunks cannot be held**: a blocked response
is aborted, but chunks already emitted were delivered — use `CrewAIAdapter`,
which drains a `stream=True` crew and returns only after screening. Hooks and
`governance_guardrail` are not part of a checkpoint (re-enter the hooks around
the resumed kickoff). The `crewai` extra's floor is 1.0, older than the hooks
API. Full guide: [frameworks/crewai.md](frameworks/crewai.md).

## OpenAI Agents SDK

```python
from agents import Agent, Runner
from adapt_agent.core import ToolPolicy
from adapt_agent.integrations.openai_agents import govern_tools, governance_run_hooks

triage = Agent(name="triage", instructions="Route.", handoffs=[billing, support])
govern_tools(triage, firewall=firewall,
             tool_policy=ToolPolicy(deny=["delete_account"], require_approval=["refund"]))
result = await Runner.run(triage, text, hooks=governance_run_hooks(firewall=firewall))
if result.interruptions:              # a tool is waiting for a human
    state = result.to_state()
    for item in result.interruptions:
        state.approve(item)           # or state.reject(item)
    result = await Runner.run(triage, state)
```

`govern_tools(target, ...)` attaches tool input/output guardrails in place to an
`Agent`, an `MCPServer`, a tool or a list, following **handoffs**,
**agents-as-tools** and **`mcp_servers`**. `"refuse"` (default) is the SDK's
`reject_content`; `"raise"` aborts with `ToolInputGuardrailTripwireTriggered` /
`ToolOutputGuardrailTripwireTriggered`. An ask installs a gate-driven
`needs_approval`, so the run stops with `result.interruptions`; the guardrails
let an ask through only when the SDK has an approval on record for that exact
call. The SDK runs input guardrails for the *starting* agent only, so
`governance_run_hooks()` on `Runner.run(hooks=...)` is what screens every
agent's model calls, a handoff target's included. A block raised from a function
tool's hooks reaches you wrapped in `agents.UserError`;
`find_security_block(exc)` returns ours. `govern_stream(Runner.run_streamed(...), ...)`
holds each response's text events until screened.
`governance_model_input_filter(on_violation="redact")` withholds flagged items
from model input instead of aborting.

**Known gaps.** Tools OpenAI runs **server-side** (hosted web search, file
search, code interpreter, image generation, a hosted shell, a `HostedMCPTool`
without `on_approval_request`) have **no client-side seam** before or after the
call; `govern_tools` warns about them, and their output is screened only on the
way to the next model call. **`ComputerTool` and `LocalShellTool`** have no
per-call seam: the run hooks apply the `ToolPolicy` by name in `on_tool_start`,
but their arguments are not visible, and the results of shell, apply-patch,
custom and computer tools can only abort the run, not be replaced. **With any
Pydantic plugin installed** (logfire's is common), the SDK never evaluates a
callable `needs_approval` for a decorated tool, so **every call of a tool the
gate can ask about pauses for approval** — erring towards asking, never towards
running unseen; set `PYDANTIC_DISABLE_PLUGINS=__all__` for argument-dependent
approval rules. Without `govern_stream`, a streamed run is fail-open. An agent
behind `Agent.as_tool` runs a nested run that does not inherit the outer run's
hooks. SDK tracing spans are not mirrored into `AgentObserver`. Full guide:
[frameworks/openai_agents.md](frameworks/openai_agents.md).

## Claude Agent SDK

```python
from claude_agent_sdk import ClaudeAgentOptions, query
from adapt_agent.core import ToolPolicy
from adapt_agent.integrations.claude_agent import governance_hooks, native_tool_options

policy = ToolPolicy(allow=["Read", "Grep", "mcp__docs__search"], require_approval=["Write"])
options = ClaudeAgentOptions(
    hooks=governance_hooks(firewall=firewall, tool_policy=policy, agent_id="assistant",
                           state=lambda: {"trust_score": session.trust}),
    **native_tool_options(policy, mcp_servers=["docs"]),   # the CLI enforces the lists too
)
async for message in query(prompt="...", options=options):
    ...
```

`UserPromptSubmit` screens the prompt, and plain state rules run there (`state=`
supplies what they read). `PreToolUse` runs `check_tool_call` and answers with
the SDK's typed `permissionDecision`: `"deny"` for a block, `"ask"` for an
ask — never `"allow"`, which would bypass the user's own permission settings
(an allow returns nothing). `PostToolUse` runs `check_tool_result`, and on a
block replaces the output through `updatedToolOutput` with each flagged string
swapped for a placeholder, keeping keys and types. Matchers carry
`on_failure="block"` and a timeout, so a hook that fails or times out refuses.
Hooks on the options cover every sub-agent too. `governance_can_use_tool()` is
for the approver role only — the SDK calls it only for calls it would prompt
for — and never turns an ask into an allow. `native_tool_options()` derives
`disallowed_tools`, `tools` and `strict_mcp_config` so the CLI never offers a
denied tool; it never sets `allowed_tools` (which auto-approves), and raises on
a pattern the CLI cannot express. `ClaudeAgentSDKAdapter` drains the message
stream before returning, so a blocked answer is never delivered.

**Known gaps.** The CLI cannot run offline, so **behaviour against the real CLI
is unverified**: the tests drive the hook callables with inputs shaped like the
SDK's hook-input types and validate every output against its output types. A
**built-in tool may reject a replaced result** whose shape does not match its
output schema, in which case the SDK keeps the original output (and a CLI that
ignores `updatedToolOutput` keeps it too); deny such a tool in `ToolPolicy` if
that is not acceptable. `PostToolUseFailure` text cannot be replaced. Nothing in
the SDK can hold back an assistant message already streamed. Full guide:
[frameworks/claude_agent.md](frameworks/claude_agent.md).

## Google ADK

```python
from google.adk.apps import App
from google.adk.runners import InMemoryRunner
from adapt_agent.core import ToolPolicy
from adapt_agent.integrations.google_adk import governance_plugin

app = App(
    name="support",
    root_agent=router,   # sub-agents, AgentTools and their tools are covered too
    plugins=[governance_plugin(
        firewall=firewall,
        tool_policy=ToolPolicy(deny=["shell"], require_approval=["refund"]),
    )],
)
runner = InMemoryRunner(app=app)
```

ADK runs a plugin's callbacks for every agent in the tree and every tool, MCP
tools included. The plugin screens the user message, every model request and
response, and every tool call and result. `on_block` (model traffic) defaults to
`"raise"`; `"refuse"` answers with a refusal `LlmResponse` so the rest of the
tree keeps running. `on_tool_violation` defaults to `"refuse"`: the tool does
not run and the model receives `{"error": decision.reason}`. An ask calls
`tool_context.request_confirmation(...)`; the run ends with an
`adk_request_confirmation` call for your client, and ADK re-runs the original
call on a confirmed answer. ADK re-raises plugin exceptions as
`RuntimeError("Error in plugin ...")`: wrap the run in
`reraise_security_blocks()`, use `find_security_error(exc)`, or run it through
`GoogleADKAdapter`, which unwraps them. `governance_callbacks()` gives the same
hooks for one `LlmAgent` (in addition to the plugin, not instead of it). With
`buffer_stream=True` an SSE stream is screened as an aggregate, so a pattern
split across chunks is caught and no part of a blocked answer is released;
`stream_holdback=N` releases all but the last `N` characters as it goes.

**Known gaps.** **Live / bidi runs (`run_live`) are not buffered**, and an ask
there fails closed. Unbuffered SSE (`buffer_stream=False`, the default) delivers
chunks before the aggregate is screened. An `AgentTool` with
`include_plugins=False` runs without the plugin. Non-text content (images,
audio, files) is not screened. Full guide:
[frameworks/google_adk.md](frameworks/google_adk.md).

## Microsoft Agent Framework

```python
from agent_framework import Agent
from adapt_agent.core import ToolPolicy
from adapt_agent.integrations.agent_framework import governance_bundle

agent = Agent(
    client,
    "You are a support agent.",
    tools=[...],
    middleware=[
        usage_middleware("nos"),     # the app's own
        governance_bundle(firewall=firewall, tool_policy=ToolPolicy(deny=["delete_*"]),
                          agent_id="nos"),
    ],
)
```

One bundle installs three layers: agent middleware (the caller's input, once per
run, and the final response), chat middleware (every model request — history,
context-provider content, tool results — and every response *before* MAF
persists it to the session), and function middleware (every tool call and
result, MCP tools included). It refuses or aborts blocked calls, withholds
poisoned results, routes an ask to MAF's `function_approval_request` (resume on
the same `AgentSession` with `to_function_approval_response(True)`), and with
`buffer_streams=True` (default) holds a stream until the response passed
screening. `governance_middleware(...)` keeps its signature and now installs the
same chat and function layers. `mcp_tool_options()` narrows what an MCP tool
offers the model.

**Known gaps.** **Provider-hosted tools** (hosted tools, a service-side MCP
connector, code interpreter) run outside MAF's function loop and **bypass the
function middleware**. An ask without an `AgentSession` is refused.
`buffer_streams=False` delivers the text before a match. Function middleware
placed after the bundle can change arguments after the gate approved them. A2A
and `Agent.as_mcp_server()` transports are untested. Full guide:
[frameworks/microsoft_agent_framework.md](frameworks/microsoft_agent_framework.md).
