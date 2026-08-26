"""Offline, deterministic tests for ``adapt_agent.optimization.target``."""

import pytest

from adapt_agent.optimization.parameters import Parameter, ParameterKind
from adapt_agent.optimization.target import OptimizableAgent, wrap

# -- helpers ------------------------------------------------------------------


def _dict_param(name, store, key, *, kind=ParameterKind.PROMPT, candidates=None):
    """A Parameter backed by a plain dict so we can verify live mutation."""
    return Parameter(
        name=name,
        kind=kind,
        value=store.get(key),
        candidates=candidates,
        getter=lambda: store.get(key),
        setter=lambda v: store.__setitem__(key, v),
    )


# -- from_callable ------------------------------------------------------------


def test_from_callable_with_explicit_parameters():
    store = {"prompt": "hello"}
    p = _dict_param("agent.prompt", store, "prompt", candidates=["hello", "hi"])
    agent = OptimizableAgent.from_callable(lambda x: x, parameters=[p], name="cb")
    assert agent.name == "cb"
    assert [param.name for param in agent.parameters] == ["agent.prompt"]
    assert agent.run("input") == "input"


def test_from_callable_rejects_non_callable_runner():
    with pytest.raises(TypeError):
        OptimizableAgent.from_callable("not callable")


# -- from_agent ---------------------------------------------------------------


def test_from_agent_uses_resolve_runner():
    class Agent:
        def run(self, x):
            return f"ran:{x}"

    target = OptimizableAgent.from_agent(Agent(), name="single")
    assert target.run("q") == "ran:q"
    assert list(target.components) == ["agent"]


def test_from_agent_with_explicit_runner_and_component_name():
    sentinel = object()
    target = OptimizableAgent.from_agent(
        sentinel, runner=lambda x: ("out", x), component_name="thing"
    )
    assert target.run(5) == ("out", 5)
    assert list(target.components) == ["thing"]
    assert target.components["thing"] is sentinel


def test_from_agent_falls_back_to_framework_runner_for_openai_shaped_agent(monkeypatch):
    """An agent with no run method is driven through the framework machinery.

    An OpenAI Agents ``Agent`` is a configuration object executed by
    ``Runner.run_sync(agent, input)``; previously ``from_agent`` raised
    TypeError for it and every caller hand-wrote the runner lambda.
    """

    class RunResult:
        def __init__(self, final_output):
            self.final_output = final_output

    class FakeRunner:
        def run_sync(self, agent, input_data, **kwargs):
            return RunResult(f"{agent.instructions} -> {input_data}")

    monkeypatch.setattr(
        "adapt_agent.optimization.runners._load_openai_agents_runner", lambda: FakeRunner()
    )

    class OpenAIShapedAgent:
        def __init__(self):
            self.name = "Solo Agent"
            self.instructions = "You answer."
            self.tools = []
            self.handoffs = []

    agent = OpenAIShapedAgent()
    target = OptimizableAgent.from_agent(agent, name="solo")
    # Driven via the delegated runner, with the final output extracted to text.
    assert target.run("q") == "You answer. -> q"
    # Introspection still sees the framework object's knobs (namespaced under
    # the component name from_agent registered the object as).
    assert any(p.name == "agent.solo_agent.instructions" for p in target.parameters)


def test_from_agent_still_raises_for_unrunnable_unknown_objects():
    with pytest.raises(TypeError):
        OptimizableAgent.from_agent(object())


# -- from_components ----------------------------------------------------------


def test_from_components_infers_runner_when_one_runnable():
    class Runnable:
        def run(self, x):
            return f"r:{x}"

    plain_data = {"some": "config"}  # not runnable (dict is not callable)
    target = OptimizableAgent.from_components({"runner_obj": Runnable(), "data": plain_data})
    assert target.run("z") == "r:z"


def test_from_components_infers_runner_from_plain_callable():
    # The single runnable component is a bare callable (exercises the
    # callable branch of _is_runnable / resolve_runner).
    def fn(x):
        return f"fn:{x}"

    target = OptimizableAgent.from_components({"fn": fn, "data": {"k": 1}})
    assert target.run("a") == "fn:a"


def test_from_components_zero_runnable_raises():
    with pytest.raises(ValueError):
        OptimizableAgent.from_components({"a": object(), "b": object()})


def test_from_components_multiple_runnable_raises():
    class R:
        def run(self, x):  # pragma: no cover - never invoked
            return x

    with pytest.raises(ValueError):
        OptimizableAgent.from_components({"a": R(), "b": R()})


def test_from_components_explicit_runner_overrides_inference():
    class R:
        def run(self, x):  # pragma: no cover - should not be used
            return "wrong"

    target = OptimizableAgent.from_components({"a": R(), "b": R()}, runner=lambda x: "right")
    assert target.run("x") == "right"


# -- collision between declared and introspected ------------------------------


def test_declared_param_collides_with_introspected_raises():
    # Build a fake framework object the CrewAI introspector recognises: it needs
    # an ``agents`` list and a ``kickoff`` method. Each agent contributes
    # parameters namespaced under the component name.
    class FakeAgent:
        def __init__(self):
            self.role = "Researcher"
            self.goal = "find things"
            self.backstory = "experienced"

    class FakeCrew:
        def __init__(self):
            self.agents = [FakeAgent()]

        def kickoff(self, x):  # makes it runnable
            return x

    crew = FakeCrew()
    # ``kickoff`` is not a runner shape resolve_runner recognises, so supply an
    # explicit runner; we only care about the introspected parameters here.
    introspected = OptimizableAgent.from_agent(crew, runner=crew.kickoff, component_name="crew")
    names = {p.name for p in introspected.parameters}
    assert names, "expected the CrewAI-like introspector to find parameters"

    # Pick an introspected name and try to declare a colliding explicit param.
    collide = next(iter(names))
    dup = Parameter(name=collide, kind=ParameterKind.PROMPT, value="x")
    with pytest.raises(ValueError):
        OptimizableAgent.from_agent(
            crew, runner=crew.kickoff, component_name="crew", parameters=[dup]
        )


def test_introspected_params_appear_from_fake_crew():
    class FakeAgent:
        def __init__(self, role):
            self.role = role
            self.goal = "g"
            self.backstory = "b"

    class FakeCrew:
        def __init__(self):
            self.agents = [FakeAgent("A")]

        def kickoff(self, x):
            return x

    crew = FakeCrew()
    target = OptimizableAgent.from_agent(crew, runner=crew.kickoff, component_name="crew")
    names = [p.name for p in target.parameters]
    # All introspected names are namespaced under the component.
    assert names
    assert all(n.startswith("crew.") for n in names)


# -- run / __call__ -----------------------------------------------------------


def test_run_and_call_are_equivalent():
    agent = OptimizableAgent.from_callable(lambda x: x + 1)
    assert agent.run(1) == 2
    assert agent(1) == 2


# -- snapshot / apply / restore round-trip ------------------------------------


def test_snapshot_apply_restore_mutates_live_state():
    store = {"prompt": "v0", "temp": 0.1}
    p_prompt = _dict_param("agent.prompt", store, "prompt", candidates=["v0", "v1", "v2"])
    p_temp = _dict_param(
        "agent.temp",
        store,
        "temp",
        kind=ParameterKind.HYPERPARAM,
    )
    agent = OptimizableAgent.from_callable(lambda x: store["prompt"], parameters=[p_prompt, p_temp])

    snap = agent.snapshot()
    assert snap == {"agent.prompt": "v0", "agent.temp": 0.1}

    agent.apply({"agent.prompt": "v1", "agent.temp": 0.9})
    # Live dict mutated, and the runner closes over it.
    assert store["prompt"] == "v1"
    assert store["temp"] == 0.9
    assert agent.run(None) == "v1"

    agent.restore(snap)
    assert store["prompt"] == "v0"
    assert store["temp"] == 0.1
    assert agent.run(None) == "v0"


def test_apply_ignores_unknown_keys():
    store = {"prompt": "a"}
    p = _dict_param("agent.prompt", store, "prompt", candidates=["a", "b"])
    agent = OptimizableAgent.from_callable(lambda x: x, parameters=[p])
    agent.apply({"does.not.exist": "ignored", "agent.prompt": "b"})
    assert store["prompt"] == "b"


def test_apply_skips_readonly_parameter():
    # A parameter without a setter is read-only and should be skipped silently.
    ro = Parameter(name="ro", kind=ParameterKind.MODEL, value="m", getter=lambda: "m")
    agent = OptimizableAgent.from_callable(lambda x: x, parameters=[ro])
    agent.apply({"ro": "other"})  # no exception
    assert agent.search_space["ro"].read() == "m"


# -- parameters_of_kind / add_parameter ---------------------------------------


def test_parameters_of_kind():
    store = {"p": "x", "m": "gpt"}
    p1 = _dict_param("a.p", store, "p", kind=ParameterKind.PROMPT)
    p2 = _dict_param("a.m", store, "m", kind=ParameterKind.MODEL)
    agent = OptimizableAgent.from_callable(lambda x: x, parameters=[p1, p2])
    prompts = agent.parameters_of_kind(ParameterKind.PROMPT)
    assert [p.name for p in prompts] == ["a.p"]
    models = agent.parameters_of_kind(ParameterKind.MODEL)
    assert [p.name for p in models] == ["a.m"]


def test_add_parameter_after_construction():
    agent = OptimizableAgent.from_callable(lambda x: x)
    assert len(agent.parameters) == 0
    agent.add_parameter(Parameter(name="new", kind=ParameterKind.ROUTING, value=1))
    assert [p.name for p in agent.parameters] == ["new"]


def test_add_parameter_duplicate_raises():
    p = Parameter(name="dup", kind=ParameterKind.PROMPT, value=1)
    agent = OptimizableAgent.from_callable(lambda x: x, parameters=[p])
    with pytest.raises(ValueError):
        agent.add_parameter(Parameter(name="dup", kind=ParameterKind.PROMPT, value=2))


# -- describe -----------------------------------------------------------------


def test_describe_shape():
    store = {"prompt": "x"}
    opt = _dict_param("a.prompt", store, "prompt", candidates=["x", "y", "z"])
    ro = Parameter(
        name="a.readonly",
        kind=ParameterKind.HYPERPARAM,
        value=0.5,
        bounds=(0.0, 1.0),
        component="a",
    )
    agent = OptimizableAgent.from_callable(lambda x: x, parameters=[opt, ro], name="desc")
    d = agent.describe()
    assert d["name"] == "desc"
    assert d["components"] == []
    assert d["n_parameters"] == 2
    assert d["n_optimizable"] == 1  # only the one with a setter
    assert len(d["parameters"]) == 2

    by_name = {p["name"]: p for p in d["parameters"]}
    assert by_name["a.prompt"]["kind"] == "prompt"
    assert by_name["a.prompt"]["optimizable"] is True
    assert by_name["a.prompt"]["n_candidates"] == 3
    assert by_name["a.prompt"]["bounds"] is None
    assert by_name["a.readonly"]["optimizable"] is False
    assert by_name["a.readonly"]["n_candidates"] is None
    assert by_name["a.readonly"]["bounds"] == (0.0, 1.0)
    assert by_name["a.readonly"]["component"] == "a"


def test_search_space_property():
    p = Parameter(name="x", kind=ParameterKind.PROMPT, value=1)
    agent = OptimizableAgent.from_callable(lambda x: x, parameters=[p])
    assert "x" in agent.search_space
    assert agent.search_space["x"].name == "x"


# -- add_tool_parameter -------------------------------------------------------


def test_add_tool_parameter_explicit_candidates():
    store = {"tools": ["search", "calc"]}
    agent = OptimizableAgent.from_callable(lambda x: x)
    param = agent.add_tool_parameter(
        "agent.tools",
        getter=lambda: store["tools"],
        setter=lambda v: store.__setitem__("tools", v),
        candidates=[["search", "calc"], ["search"]],
        component="agent",
    )
    assert param.kind is ParameterKind.TOOL
    assert param.optimizable is True
    assert param.component == "agent"
    assert param.value == ["search", "calc"]
    assert agent.search_space["agent.tools"] is param
    assert param.candidates == [["search", "calc"], ["search"]]


def test_add_tool_parameter_drop_one_candidates(monkeypatch):
    import adapt_agent.optimization.introspection as introspection

    def fake_tool_subset_candidates(tools, *, max_candidates=8):
        full = list(tools)
        out = [full]
        for i in range(len(full)):
            out.append([t for j, t in enumerate(full) if j != i])
        return out[:max_candidates]

    monkeypatch.setattr(
        introspection, "tool_subset_candidates", fake_tool_subset_candidates, raising=False
    )

    store = {"tools": ["a", "b", "c"]}
    agent = OptimizableAgent.from_callable(lambda x: x)
    param = agent.add_tool_parameter(
        "agent.tools",
        getter=lambda: store["tools"],
        setter=lambda v: store.__setitem__("tools", v),
        candidate_tools=["a", "b", "c"],
    )
    # Full set first, then each drop-one subset (ablation).
    assert param.candidates[0] == ["a", "b", "c"]
    assert ["b", "c"] in param.candidates
    assert ["a", "c"] in param.candidates
    assert ["a", "b"] in param.candidates
    assert len(param.candidates) == 4

    # And it is a real, searchable, optimizable TOOL parameter.
    agent.apply({"agent.tools": ["a", "c"]})
    assert store["tools"] == ["a", "c"]


def test_add_tool_parameter_skill_kind_and_fallback(monkeypatch):
    # When the helper is unavailable, fall back to the full set as a single
    # candidate; the parameter is still well-formed.
    import adapt_agent.optimization.introspection as introspection

    monkeypatch.delattr(introspection, "tool_subset_candidates", raising=False)

    store = {"skills": ["writer"]}
    agent = OptimizableAgent.from_callable(lambda x: x)
    param = agent.add_tool_parameter(
        "agent.skills",
        kind=ParameterKind.SKILL,
        getter=lambda: store["skills"],
        setter=lambda v: store.__setitem__("skills", v),
        candidate_tools=["writer"],
    )
    assert param.kind is ParameterKind.SKILL
    assert param.candidates == [["writer"]]


def test_add_tool_parameter_duplicate_name_raises():
    agent = OptimizableAgent.from_callable(lambda x: x)
    agent.add_tool_parameter(
        "agent.tools",
        getter=lambda: [],
        setter=lambda v: None,
        candidates=[[]],
    )
    with pytest.raises(ValueError):
        agent.add_tool_parameter(
            "agent.tools",
            getter=lambda: [],
            setter=lambda v: None,
            candidates=[[]],
        )


# -- wrap ---------------------------------------------------------------------


def test_wrap_passthrough_optimizable_agent():
    agent = OptimizableAgent.from_callable(lambda x: x)
    assert wrap(agent) is agent


def test_wrap_callable():
    target = wrap(lambda x: x * 3, name="cb")
    assert isinstance(target, OptimizableAgent)
    assert target.name == "cb"
    assert target.run(2) == 6


def test_wrap_single_object():
    class Agent:
        def run(self, x):
            return f"o:{x}"

    target = wrap(Agent())
    assert isinstance(target, OptimizableAgent)
    assert target.run("a") == "o:a"


def test_wrap_components():
    class R:
        def run(self, x):
            return f"r:{x}"

    target = wrap(None, components={"r": R()})
    assert isinstance(target, OptimizableAgent)
    assert target.run("y") == "r:y"
    assert list(target.components) == ["r"]


def test_wrap_components_with_callable_target_as_runner():
    # When components are given and target is callable but no runner, the
    # callable target becomes the runner.
    target = wrap(lambda x: f"fn:{x}", components={"data": {"k": "v"}})
    assert target.run("z") == "fn:z"


def test_wrap_object_with_explicit_runner():
    sentinel = object()
    target = wrap(sentinel, runner=lambda x: f"run:{x}")
    assert target.run("p") == "run:p"
    assert target.components["agent"] is sentinel
