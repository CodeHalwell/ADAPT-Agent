# Security Policy

ADAPT-Agent (Adversarial Defense & Policy Training for LLM Agents) is itself a
security and governance library — it is used to enforce trust, policy, firewalling,
and taint tracking around LLM agents. We therefore take the security of the project,
and of the systems that depend on it, seriously. We are grateful to security
researchers and users who responsibly disclose issues.

## Supported versions

Security fixes are provided for the **latest minor release line** only; every
earlier line is unsupported.

| Version | Supported          |
| ------- | ------------------ |
| 0.5.x   | :white_check_mark: |
| < 0.5   | :x:                |

The row above names the current line (the minor of `adapt_agent.__version__`);
it moves forward with each new minor release, and a test fails if it is left
behind. We strongly recommend always running the latest released version. Once a
newer minor release is published, the previous minor series stops receiving
security updates.

## Reporting a vulnerability

**Please do not report security vulnerabilities through public GitHub issues,
discussions, or pull requests.**

Instead, report privately using either of the following:

1. **GitHub private security advisories (preferred):** open a report via the
   "Security" tab of the repository
   (<https://github.com/CodeHalwell/ADAPT-Agent/security/advisories/new>). This
   keeps the discussion private until a fix is ready and coordinated.
2. **Email:** send details to **danielhalwell@gmail.com**. If possible, use a
   subject line that begins with `[ADAPT-Agent Security]`.

Please include as much of the following as you can:

- A description of the vulnerability and its potential impact.
- The affected version(s) and environment (Python version, installed extras).
- Step-by-step reproduction instructions or a proof of concept.
- Any relevant logs, stack traces, or configuration (with secrets redacted).
- The component involved (for example, `core` trust/policy, `security` firewall or
  taint tracker, an adapter, or the CLI).

## Response process and timelines

- **Acknowledgement:** we aim to acknowledge your report within **3 business days**.
- **Assessment:** we will work to validate and assess severity within **10 business
  days** of acknowledgement, and will keep you updated on progress.
- **Resolution:** once a fix is ready, we will coordinate a release and, where
  appropriate, publish a GitHub Security Advisory with a CVE.
- **Credit:** with your permission, we are happy to credit you in the advisory and
  release notes.

Timelines are best-effort; this is a community-maintained, MIT-licensed open-source
project.

## Scope

In scope:

- Vulnerabilities in ADAPT-Agent's own code that allow security controls to be
  bypassed — for example, defeating policy enforcement, the firewall, taint
  tracking, or trust evaluation.
- Issues in the framework adapters (`adapt_agent.adapters`) or the native framework
  bindings (`adapt_agent.integrations`: middleware, hooks, plugins, guardrails and
  tool-call governance for LangGraph, Microsoft Agent Framework, Google ADK,
  Pydantic AI, CrewAI, OpenAI Agents SDK, Claude Agent SDK) where ADAPT-Agent fails
  to apply its protections -- for example a denied tool call that still runs, or a
  blocked result that still reaches the model. Gaps a binding documents as a known
  limitation of the framework's seams are not vulnerabilities, but a binding that
  behaves differently from its documentation is.
- Supply-chain or packaging issues in the distributed wheel/sdist.
- Code-execution, injection, or path-traversal issues reachable through the public
  API or the `adapt-agent` CLI. Note that `evaluate`, `optimize` and `train`
  import and run the `module:attribute` targets they are given, and a training
  config is code by design (see `docs/cli.md`); running a config or target you
  supplied is not a vulnerability, but a config key that leaks a credential or
  redirects graded output to another host would be.

Out of scope:

- Vulnerabilities in third-party frameworks (LangGraph, Microsoft Agent Framework,
  Google ADK, Pydantic AI, CrewAI, OpenAI Agents SDK, Claude Agent SDK) or the
  underlying LLMs themselves, unless ADAPT-Agent's mitigations fail to behave as
  documented.
- Misconfiguration in a user's own deployment that is not caused by an insecure
  default in ADAPT-Agent.
- Reports generated purely by automated scanners without a demonstrated, realistic
  impact.

## Disclosure policy

We follow a coordinated disclosure model. Please give us a reasonable opportunity to
release a fix before any public disclosure. We will work with you on disclosure
timing and will not pursue legal action against researchers who act in good faith
and in accordance with this policy.

Thank you for helping keep ADAPT-Agent and its users safe.
