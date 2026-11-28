# Microsoft Agent Framework + ADAPT-Agent examples

[Microsoft Agent Framework](https://github.com/microsoft/agent-framework) (MAF)
is Microsoft's unified successor to Semantic Kernel and AutoGen. Its runnable
object is an **`agent_framework.Agent`** (called `ChatAgent` before 1.0), built
from a chat client: `Agent(client, instructions, tools=[...])`. `await
agent.run(...)` returns an `AgentResponse` with the answer on `.text`; with
`stream=True` it returns a `ResponseStream`. Multi-agent workflows are built
with `WorkflowBuilder` (or `MagenticBuilder` from the separate
`agent-framework-orchestrations` package) and exposed as one agent with
`workflow.as_agent()`.

These examples drive **real MAF objects** -- agents, tools, sessions,
approvals, streams, workflows. Only the model is replaced: `_offline.py`
provides a genuine MAF chat client whose replies come from a local function,
so **every example runs with no API key and no network**. Without
`agent-framework-core` installed each example prints an install hint and exits.

## Install

```bash
pip install 'adapt-agent[microsoft-agent-framework]'   # or: pip install agent-framework-core
```

`adapt_agent` itself never imports `agent_framework` at import time.

## The ladder

| # | File | What it teaches |
|---|------|-----------------|
| 1 | [`01_basic_guarded.py`](./01_basic_guarded.py) | `governance_bundle(...)` on a real `Agent`: a clean tool run, a blocked prompt injection, a denied tool that never runs, a poisoned tool result withheld from the model, an approval that pauses and resumes on the session, a leaking streamed answer that is never delivered -- and the outer `MicrosoftAgentFrameworkAdapter`. |
| 2 | [`02_policy_observability_trust.py`](./02_policy_observability_trust.py) | `PolicyEnforcer` (block + warn rules), `Middleware`, `TrustManager`, `TaintTracker` in audit mode (`block_on_violation=False`), then a policy rule inside the agent that reads `AgentSession.state`. |
| 3 | [`03_evaluate_and_optimize.py`](./03_evaluate_and_optimize.py) | Wrap one `Agent` in `OptimizableAgent`, see `introspect(agent)` discover its knobs, score with `EvaluationHarness` + an offline `LLMJudge`, and optimize the prompt with `CoordinateAscentOptimizer`. |
| 4 | [`04_magentic_team_and_training.py`](./04_magentic_team_and_training.py) | Five agents in a workflow, each carrying a bundle that shares one gate, the team wrapped by the adapter via `as_agent()`, then trained as a whole (5 agents as `components` + explicit ROUTING knobs that rebuild the workflow) and per agent, with tool ablation and the YAML `run_training` path ([`magentic.train.yaml`](./magentic.train.yaml)). |

## Run them

```bash
python examples/microsoft_agent_framework/01_basic_guarded.py
python examples/microsoft_agent_framework/02_policy_observability_trust.py
python examples/microsoft_agent_framework/03_evaluate_and_optimize.py
python examples/microsoft_agent_framework/04_magentic_team_and_training.py
```

Each script prints what it ran, what passed, what
was blocked and why (the `SecurityBlockedError` threat labels), and exits 0. The
optimization examples print an `OptimizationResult` line such as
`baseline=0.200, best=0.900, ..., significant=False (p=0.0625, n=4)`: the gain is
applied, but on a handful of rows no outcome can pass the paired significance
test, so `improved` stays `False` -- use a larger golden set for a real verdict.

## What ADAPT-Agent introspects for an `Agent`

`detect(agent)` returns `"microsoft_agent_framework"`; `introspect(agent)`
yields the agent's instructions (`PROMPT`), its sampling options such as
`temperature` (`HYPERPARAM`) and its `tools` (`TOOL`, with drop-one ablation
candidates). Routing limits and the `as_agent()` wrapper expose no knobs --
example 4 shows how to declare those as explicit `ROUTING` parameters whose
setter rebuilds the workflow.

See the full guide: [`docs/frameworks/microsoft_agent_framework.md`](../../docs/frameworks/microsoft_agent_framework.md).
