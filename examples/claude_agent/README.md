# Claude Agent SDK examples

A four-step ladder showing how to **guard** and **train** a
[Claude Agent SDK](https://github.com/anthropics/claude-agent-sdk-python) agent
with ADAPT-Agent, climbing from a single guarded call to a governed,
optimizable multi-agent pipeline.

Every example runs **with no API key and no network**: each substitutes a tiny
stand-in for the SDK's `query` async generator (and, for the training examples,
a deterministic LLM-judge stub). Importing ADAPT-Agent never imports
`claude_agent_sdk`, so the only thing you change to go live is the agent you pass
to `wrap_agent` / the `components` you optimize.

## The Claude Agent SDK in one paragraph

The SDK is driven by a single async function:

```python
from claude_agent_sdk import query, ClaudeAgentOptions

async for message in query(
    prompt="What is the capital of France?",
    options=ClaudeAgentOptions(
        system_prompt="Answer with only the city name.",
        model="claude-opus-4-8",
        allowed_tools=[],            # permission allow-list
        max_turns=1,
        permission_mode="default",   # default | acceptEdits | plan | bypassPermissions
    ),
):
    ...   # AssistantMessage(content=[TextBlock(text=...)]) ... then a ResultMessage
```

`query` is an **async generator** that streams message objects.
`ClaudeAgentSDKAdapter` wraps that function: it derives the prompt from your
payload's latest user message, calls `query(prompt=...)`, and **drains the async
stream to a list** so `execute()` stays synchronous while the firewall scans
every text block. Custom tools are defined with the `@tool` decorator and
exposed through `create_sdk_mcp_server(...)`; the agent's configuration
(`system_prompt`, `model`, `allowed_tools`, `max_turns`, `permission_mode`) lives
on the `ClaudeAgentOptions` object, which is exactly what ADAPT-Agent
introspects.

## Install

```bash
pip install 'adapt-agent[claude-agent]'   # or: pip install claude-agent-sdk
```

(The examples ship stand-ins, so you can run them before installing the SDK.)

## The ladder

| # | File | What it adds |
|---|------|--------------|
| 1 | [`01_basic_guarded.py`](01_basic_guarded.py) | Smallest guarded agent: wrap `query` with a `Firewall`, run a safe input, then a prompt-injection input that raises `SecurityBlockedError`. |
| 2 | [`02_policy_observability_trust.py`](02_policy_observability_trust.py) | Add `PolicyEnforcer` (a block rule + a warn rule), `AdversarialDefense`, `AgentObserver` traces and `Middleware`; show `block_on_violation=False` (audit mode), plus standalone `TrustManager` and `TaintTracker`. |
| 3 | [`03_evaluate_and_optimize.py`](03_evaluate_and_optimize.py) | Wrap one agent's `ClaudeAgentOptions` in `OptimizableAgent`, build a `GoldenDataset`, score with `EvaluationHarness` (`exact_match` + `token_f1` + an offline `LLMJudge`), run `CoordinateAscentOptimizer`, and print `introspect()`-discovered knobs and baseline -> best. |
| 4 | [`04_multi_agent_and_training.py`](04_multi_agent_and_training.py) | A researcher -> writer -> reviewer orchestrator (three `ClaudeAgentOptions`), optimized as ONE system with `make_default_optimizer` (incl. tool/skill drop-one ablation and judge `recommendations`), a parallel YAML-config path via `run_training`, and the whole pipeline guarded as a single unit. |

Plus [`claude_agent.train.yaml`](claude_agent.train.yaml) — the declarative,
real-world training config that example 4's "Path B" mirrors.

## Run them

```bash
python examples/claude_agent/01_basic_guarded.py
python examples/claude_agent/02_policy_observability_trust.py
python examples/claude_agent/03_evaluate_and_optimize.py
python examples/claude_agent/04_multi_agent_and_training.py
```

## What gets introspected

For a `ClaudeAgentOptions` object, ADAPT-Agent discovers:

| Field | Parameter kind |
|-------|----------------|
| `system_prompt` (string, or a preset's `append`) | `PROMPT` |
| `model` | `MODEL` |
| `allowed_tools` | `TOOL` (drop-one ablation when ≥2 tools) |
| `disallowed_tools` | `TOOL` |
| `max_turns` | `HYPERPARAM` (bounds 1–100) |
| `permission_mode` | `ROUTING` (candidates: `default`/`acceptEdits`/`plan`/`bypassPermissions`) |

See [`docs/frameworks/claude_agent.md`](../../docs/frameworks/claude_agent.md)
for the full, teach-everything walkthrough.
