# Framework Guides

ADAPT-Agent wraps and optimizes agents built with seven frameworks. Each guide
below is a self-contained, teach-everything walkthrough that takes you from a tiny
guarded agent to a governed, optimized multi-agent system — pairing prose with
runnable examples under [`examples/<framework>/`](https://github.com/CodeHalwell/ADAPT-Agent/tree/main/examples).

Every framework shares the **same** two capabilities:

- **Guard (runtime):** wrap your agent in a `GovernedAdapter` so each call runs the
  six-step pipeline — input screening (`Firewall` + `AdversarialDefense`) → policy
  (`PolicyEnforcer`) → pre-middleware → traced run (`AgentObserver`) →
  post-middleware → output screening.
- **Train (offline):** turn your agent (or whole system) into a tunable search
  space with `OptimizableAgent`, score it against a `GoldenDataset`, and let the
  optimizers + `LLMJudge` improve its prompts, models, hyperparameters, and
  tool/skill allow-lists — in place.

| Framework | Adapter | Wrap target | Guide |
|-----------|---------|-------------|-------|
| LangGraph | `LangGraphAdapter` | compiled graph (`.invoke`) | [langgraph.md](langgraph.md) |
| Microsoft Agent Framework | `MicrosoftAgentFrameworkAdapter` | `ChatAgent` (`.run`) | [microsoft_agent_framework.md](microsoft_agent_framework.md) |
| Google ADK | `GoogleADKAdapter` | callable driving a `Runner` | [google_adk.md](google_adk.md) |
| Pydantic AI | `PydanticAIAdapter` | `Agent` (`.run_sync`/`.run`) | [pydantic_ai.md](pydantic_ai.md) |
| CrewAI | `CrewAIAdapter` | `Crew` (`.kickoff`) | [crewai.md](crewai.md) |
| OpenAI Agents SDK | `OpenAIAgentsAdapter` | `Agent` (via `Runner`) | [openai_agents.md](openai_agents.md) |
| Claude Agent SDK | `ClaudeAgentSDKAdapter` | the `query` function | [claude_agent.md](claude_agent.md) |

Each guide covers, in detail: what you wrap, installing the extra and the
import-safety guarantee, the full guard pipeline, policy/observability/trust/taint,
what the optimizer introspects for that framework (and how to declare the knobs it
can't), evaluation, the judge (including adversarial mode), every optimizer,
tool/skill ablation, the declarative YAML training config, multi-agent
orchestration, and a pitfalls/FAQ section.

New to the library? Start with the [Quick Start](../quickstart.md), then the
[Framework Adapters](../adapters.md) and [Optimization & Evaluation](../optimization.md)
overviews, then your framework's guide above.
