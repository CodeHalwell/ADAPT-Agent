"""Agent optimization: evaluate against a golden dataset and improve in place.

This subsystem turns *any* agent -- a single "mega" agent, six specialist
agents, an orchestrator with sub-agents, or a multi-step workflow, across any
supported framework -- into a tunable unit, measures it against a golden
dataset, and searches for a better configuration of its prompts, few-shot
examples, models, hyperparameters, routing knobs, and tool allow-lists.

The pieces:

* :class:`AgentOptimizer` -- runtime performance-metrics collector (legacy /
  observability helper).
* :class:`Parameter` / :class:`SearchSpace` -- the tunable knobs.
* :class:`OptimizableAgent` -- wraps your agent code as ``run`` + a search space.
* :class:`GoldenDataset` / :class:`Example` -- the evaluation data.
* :class:`LLMJudge` -- model-graded scoring *and* prompt improvement, used at
  every stage and backed by a pluggable completion function (no LLM dependency).
* metrics + :class:`EvaluationHarness` -- scoring and aggregation.
* proposers + optimizers -- candidate generation and search strategies.

Quick start::

    from adapt_agent.optimization import (
        GoldenDataset, EvaluationHarness, LLMJudge, OptimizableAgent,
        CoordinateAscentOptimizer, exact_match,
    )

    data = GoldenDataset.from_list([{"input": "2+2", "expected": "4"}, ...])
    judge = LLMJudge(complete=my_llm)            # my_llm: str -> str
    harness = EvaluationHarness([exact_match(), judge.as_metric()],
                                primary_metric="llm_judge")
    agent = OptimizableAgent.from_agent(my_pydantic_ai_agent)
    result = CoordinateAscentOptimizer(harness, judge=judge).optimize(agent, data)
    print(result)   # baseline vs best, with the best config applied in place
"""

from adapt_agent.optimization.dataset import Example, GoldenDataset
from adapt_agent.optimization.evaluation import (
    EvaluationHarness,
    EvaluationReport,
    ExampleResult,
    resolve_runner,
)
from adapt_agent.optimization.judge import CompletionFn, JudgeVerdict, LLMJudge
from adapt_agent.optimization.metrics import (
    BUILTIN_METRICS,
    Metric,
    contains,
    exact_match,
    get_metric,
    jaccard,
    json_subset,
    levenshtein_ratio,
    numeric_close,
    regex_match,
    token_f1,
)
from adapt_agent.optimization.optimizers import (
    BootstrapFewShotOptimizer,
    CoordinateAscentOptimizer,
    EvolutionaryOptimizer,
    GridSearchOptimizer,
    OptimizationResult,
    Optimizer,
    PipelineOptimizer,
    RandomSearchOptimizer,
    Trial,
    make_default_optimizer,
)
from adapt_agent.optimization.parameters import Parameter, ParameterKind, SearchSpace
from adapt_agent.optimization.performance import AgentOptimizer
from adapt_agent.optimization.proposers import (
    CandidateProposer,
    FewShotProposer,
    LLMProposer,
    NumericProposer,
    PromptMutationProposer,
    ProposalContext,
    Proposer,
    default_proposers,
)
from adapt_agent.optimization.providers import (
    AnthropicProvider,
    CallableProvider,
    EchoProvider,
    ModelProvider,
    OpenAIProvider,
    as_provider,
    available_providers,
    get_provider,
    register_provider,
)
from adapt_agent.optimization.target import OptimizableAgent, wrap

__all__ = [
    # legacy runtime metrics
    "AgentOptimizer",
    # parameters
    "Parameter",
    "ParameterKind",
    "SearchSpace",
    # data
    "Example",
    "GoldenDataset",
    # judge
    "LLMJudge",
    "JudgeVerdict",
    "CompletionFn",
    # providers (provider-agnostic model access)
    "ModelProvider",
    "CallableProvider",
    "EchoProvider",
    "AnthropicProvider",
    "OpenAIProvider",
    "get_provider",
    "as_provider",
    "available_providers",
    "register_provider",
    # metrics
    "Metric",
    "exact_match",
    "contains",
    "regex_match",
    "token_f1",
    "jaccard",
    "numeric_close",
    "json_subset",
    "levenshtein_ratio",
    "get_metric",
    "BUILTIN_METRICS",
    # evaluation
    "EvaluationHarness",
    "EvaluationReport",
    "ExampleResult",
    "resolve_runner",
    # target
    "OptimizableAgent",
    "wrap",
    # proposers
    "Proposer",
    "ProposalContext",
    "CandidateProposer",
    "NumericProposer",
    "PromptMutationProposer",
    "FewShotProposer",
    "LLMProposer",
    "default_proposers",
    # optimizers
    "Optimizer",
    "OptimizationResult",
    "Trial",
    "GridSearchOptimizer",
    "RandomSearchOptimizer",
    "CoordinateAscentOptimizer",
    "BootstrapFewShotOptimizer",
    "EvolutionaryOptimizer",
    "PipelineOptimizer",
    "make_default_optimizer",
]
