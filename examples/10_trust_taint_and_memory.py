"""Example 10: Trust scores, taint tracking and bounded memory.

Three building blocks that record *what is known about the data and the agents
handling it*. None of them blocks anything alone -- they are bookkeeping -- so
this example also shows the wiring that turns each into an enforced control.

1.  ``TrustManager`` -- a per-agent score that moves with behaviour, never with
    what an agent says about itself, feeding a policy rule that blocks.
2.  ``TaintTracker`` -- follow untrusted data through a pipeline, see where a
    value came from, and refuse to act on it until it has been sanitised.
3.  ``MemorySystem`` -- bounded short- and long-term memory with search and
    consolidation.

No optional dependencies and no network. Run it with:

    python examples/10_trust_taint_and_memory.py
"""

from adapt_agent import (
    MemorySystem,
    PolicyEnforcer,
    TaintLevel,
    TaintTracker,
    TrustManager,
)


def trust_scores() -> None:
    print("== 1. TrustManager: a score that follows behaviour ==")
    trust = TrustManager(initial_trust=0.5)

    trust.update_trust_score("researcher", +0.2, reason="three clean runs")
    trust.update_trust_score("scraper", -0.3, reason="returned a blocked phrase")
    for agent_id in ("researcher", "scraper", "never-seen"):
        print(
            f"  {agent_id:11} score={trust.get_trust_score(agent_id):.2f} "
            f"trusted={trust.is_trusted(agent_id)}"
        )

    # An agent cannot talk its way up: a self-reported score is ignored, while
    # recorded policy violations count against it.
    claim = {"messages": [], "context": {}, "trust_score": 1.0, "policy_violations": ["a", "b"]}
    evaluated = trust.evaluate_agent_state("scraper", claim)
    print(f"  scraper claims 1.00; evaluated at {evaluated['score']:.2f} {evaluated['factors']}")

    # The score is only bookkeeping until something acts on it. Put it into the
    # state a policy rule reads, and it becomes a gate.
    policy = PolicyEnforcer()
    policy.add_rule(
        "low_trust", "Block low-trust agents", "state['trust_score'] < 0.4", action="block"
    )
    for agent_id in ("researcher", "scraper"):
        state = {"messages": [], "context": {}, "trust_score": trust.get_trust_score(agent_id)}
        print(f"  policy on {agent_id:10} -> {policy.check_state(state) or 'allowed'}")

    trust.update_trust_score("scraper", +0.1, reason="one clean run")
    history = [round(record["score"], 2) for record in trust.get_trust_history("scraper")]
    print(f"  scraper's score over time: {history}")


def taint_tracking() -> None:
    print("\n== 2. TaintTracker: where did this value come from? ==")
    taint = TaintTracker()
    taint.register_source("web", "external_api", TaintLevel.HIGH, {"url": "https://example.test"})
    taint.register_source("user", "user_input", TaintLevel.MEDIUM)
    taint.register_source("config", "internal", TaintLevel.UNTAINTED)

    # A fetched page and the user's question are combined into a prompt, which
    # produces a draft, which becomes the arguments of a tool call.
    taint.mark_tainted("page", ["web"])
    taint.mark_tainted("question", ["user"])
    taint.mark_tainted("system_prompt", ["config"])
    taint.propagate_taint("page", "prompt", operation="concatenate")
    taint.propagate_taint("question", "prompt", operation="concatenate")
    taint.propagate_taint("prompt", "draft", operation="llm_call")
    taint.propagate_taint("draft", "tool_args", operation="extract")

    for data_id in ("system_prompt", "question", "prompt", "tool_args"):
        sources = sorted(s.source_id for s in taint.get_taint_sources(data_id))
        print(f"  {data_id:13} level={taint.get_taint_level(data_id).name:9} from={sources}")
    # Walk the recorded hops backwards from the tool call to its origins.
    hops, frontier = [], ["tool_args"]
    while frontier:
        steps = taint.get_taint_flow(frontier.pop())
        hops.extend(f"{s['from']} -[{s['operation']}]-> {s['to']}" for s in steps)
        frontier.extend(s["from"] for s in steps)
    print("  how tool_args got tainted:")
    for hop in hops:
        print(f"    {hop}")

    # The enforcement point is yours to write: refuse a side effect on anything
    # still carrying untrusted input.
    def may_execute(data_id: str) -> bool:
        return taint.get_taint_level(data_id) in (TaintLevel.UNTAINTED, TaintLevel.LOW)

    print(f"  run the tool call?               {may_execute('tool_args')}")
    taint.sanitize("tool_args")  # after your own validation of the arguments
    print(f"  run it after validating the args? {may_execute('tool_args')}")

    # Fail-closed: data tainted by a source the tracker does not know is
    # treated as CRITICAL, never as clean.
    taint.mark_tainted("mystery", ["no-such-source"])
    print(f"  unknown source -> {taint.get_taint_level('mystery').name}")


def memory() -> None:
    print("\n== 3. MemorySystem: bounded, searchable, consolidating ==")
    mem = MemorySystem(short_term_capacity=3, long_term_capacity=2)

    for turn, text in enumerate(["likes tea", "lives in Leeds", "has a cat", "prefers email"]):
        mem.store_short_term(f"fact-{turn}", text, metadata={"turn": turn})
    stats = mem.get_stats()
    # Capacity 3: the oldest fact was evicted, not kept forever.
    print(f"  short-term holds {stats['short_term_count']} of 4 stored")
    print(f"  fact-0 (evicted): {mem.retrieve('fact-0')}")

    print(f"  search 'cat': {[hit['value'] for hit in mem.search('cat')]}")

    # Items read often enough are promoted to long-term on consolidation.
    for _ in range(3):
        mem.retrieve("fact-2")
    print(f"  consolidated {mem.consolidate()} item(s) into long-term")
    print(f"  from long-term: {mem.retrieve('fact-2', from_long_term=True)!r}")

    # Long-term is bounded too: the least-accessed entry goes first.
    for key in ("policy-a", "policy-b"):
        mem.store_long_term(key, f"value of {key}")
    print(f"  long-term holds {mem.get_stats()['long_term_count']} (capacity 2)")


def main() -> None:
    trust_scores()
    taint_tracking()
    memory()


if __name__ == "__main__":
    main()
