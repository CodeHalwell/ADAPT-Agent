# Google ADK + ADAPT-Agent examples

A four-step ladder from a single guarded [Google ADK](https://google.github.io/adk-docs)
agent to a governed, optimized multi-agent system. Each script is self-contained.

ADAPT-Agent has two halves, both framework-agnostic:

- **Guard (runtime):** wrap an agent so every call runs a six-step security +
  observability pipeline (`GoogleADKAdapter`).
- **Train (offline):** turn an agent (or whole system) into a tunable search
  space and optimize it against a golden dataset, scored by an LLM-as-judge.

## The core ADK shape

ADK does not run an agent by calling a method on the agent object. You build a
`Runner` (bound to a session service), open a session, and feed it a message:

```python
from google.adk.runners import InMemoryRunner
from google.genai import types

runner = InMemoryRunner(agent=my_agent, app_name="app")
msg = types.Content(role="user", parts=[types.Part(text="hi")])
for event in runner.run(user_id="u", session_id="s", new_message=msg):
    if event.is_final_response():
        print(event.content.parts[0].text)
```

Because that call needs `user_id` / `session_id` / `new_message` arguments the
adapter does not know about, the wrap target for `GoogleADKAdapter` is a
**callable you write** that performs the run and returns its events:

```python
from adapt_agent.adapters import GoogleADKAdapter
from adapt_agent import Firewall

def run(payload):
    text = payload["messages"][-1]["content"]
    msg = types.Content(role="user", parts=[types.Part(text=text)])
    return runner.run(user_id="u", session_id="s", new_message=msg)

guarded = GoogleADKAdapter(firewall=Firewall()).wrap_agent(run)
guarded.execute({"messages": [{"role": "user", "content": "hi"}]})
```

The adapter drains the events (sync generator, async generator, or list) and
screens the text inside each event's `content.parts`.

## Running

```bash
# 1 & 2 need the framework installed (they build a real LlmAgent/Runner):
pip install 'adapt-agent[google-adk]'
python examples/google_adk/01_basic_guarded.py
python examples/google_adk/02_policy_observability_trust.py

# 3 & 4 are fully offline (no API key, no network, no extra needed):
python examples/google_adk/03_evaluate_and_optimize.py
python examples/google_adk/04_multi_agent_and_training.py
```

> Examples 2-4 use a small ADK-*shaped* stand-in object so they run without a
> live model. ADAPT-Agent's introspection recognises an ADK `LlmAgent` purely by
> duck typing (it never imports `google.adk`), so the stand-in is faithful and a
> real agent is a drop-in replacement.

## The ladder

### `01_basic_guarded.py`
The smallest guarded ADK agent. Builds a real `LlmAgent` + `InMemoryRunner`
(behind an import guard), wraps the run-callable with a `Firewall`, blocks a
prompt-injection input with `SecurityBlockedError`, then tries a safe input
against the live model.

### `02_policy_observability_trust.py`
Adds the full control stack on a Google ADK agent: `PolicyEnforcer` (a block
rule), `AdversarialDefense`, `AgentObserver` (traces), and pre/post `Middleware`.
Shows `block_on_violation=False` (threats recorded, not raised) and the
standalone `TrustManager` and `TaintTracker` helpers. Fully offline.

### `03_evaluate_and_optimize.py`
Wraps a single ADK agent in `OptimizableAgent`, builds a `GoldenDataset`, scores
it with an `EvaluationHarness` (a metric + an offline `LLMJudge.as_metric`), runs
`CoordinateAscentOptimizer`, and prints `introspect(agent)`'s discovered knobs
plus baseline-to-best improvement. Fully offline.

### `04_multi_agent_and_training.py`
A coordinator `LlmAgent` with two specialist `sub_agents`, wrapped as ONE
governed/optimizable unit. `make_default_optimizer` runs the full pipeline
(few-shot to prompts to models/hparams to tools/skills) with the adversarial
judge proposing new tools on `result.recommendations`. Also shows the parallel
**YAML training** path (`run_training`) mirroring
[`google_adk.train.yaml`](./google_adk.train.yaml). Fully offline.

## What gets introspected for a Google ADK agent

| Attribute | Kind | Notes |
|---|---|---|
| `instruction`, `global_instruction` | PROMPT | only when plain strings (not instruction-provider callables) |
| `model` | MODEL | a string id, or a model object's `model` / `model_name` |
| `generate_content_config.temperature` / `top_p` / `max_output_tokens` | HYPERPARAM | bounds clamped to provider max with a warning |
| `tools` | TOOL | drop-one ablation candidates when 2+ tools |
| `sub_agents` | ROUTING | walked **recursively**; nested knobs namespaced under the parent |
