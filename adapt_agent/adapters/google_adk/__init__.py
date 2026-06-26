"""Google ADK adapter for ADAPT-Agent.

The `Google Agent Development Kit <https://google.github.io/adk-docs>`_ runs an
agent through a ``Runner``: ``runner.run_async(user_id=..., session_id=...,
new_message=...)`` (an async generator of ``Event`` objects) or the sync
``runner.run(...)``. There is no single result object -- you iterate events and
take the one where ``event.is_final_response()`` is true.

Because an ADK run needs a session plus ``user_id`` / ``session_id`` /
``new_message`` arguments, the idiomatic wrap target here is a **callable** you
provide that performs the run and returns (or yields) the events. The adapter
drains async/sync generators automatically and screens the text inside each
event's ``content.parts``. Importing this module never imports ``google.adk``.

Example
-------
>>> from google.adk.runners import InMemoryRunner   # doctest: +SKIP
>>> from google.genai import types                  # doctest: +SKIP
>>> runner = InMemoryRunner(agent=my_agent)          # doctest: +SKIP
>>> def run(payload):                                # doctest: +SKIP
...     msg = types.Content(role="user", parts=[types.Part(text=payload["prompt"])])
...     return runner.run(user_id="u", session_id="s", new_message=msg)
>>> from adapt_agent.adapters import GoogleADKAdapter
>>> from adapt_agent.security import Firewall
>>> adapter = GoogleADKAdapter(firewall=Firewall())
>>> guarded = adapter.wrap_agent(run)                # doctest: +SKIP
>>> guarded.execute({"prompt": "hi"})                # doctest: +SKIP
"""

from adapt_agent.adapters._governed import GovernedAdapter


class GoogleADKAdapter(GovernedAdapter):
    """Adapter for integrating ADAPT-Agent with Google ADK agents.

    Wrap a callable ``run(payload) -> events`` that drives an ADK ``Runner`` (the
    events may be a sync generator, an async generator, or a list). The full
    payload dict is passed through to your callable. See
    :class:`~adapt_agent.adapters._governed.GovernedAdapter` for the full
    constructor signature.
    """

    framework_name = "Google ADK"
    # An ADK run requires session/user arguments, so binding to a bare
    # ``runner.run`` would mismatch its signature; wrap a callable instead.
    run_method_names = ()
    operation = "google_adk.run"


__all__ = ["GoogleADKAdapter"]
