# The bundled agent skill

ADAPT-Agent ships an **agent skill** inside its wheel. Installing the library
therefore also delivers the instructions a coding agent needs to use it — you
just have to put them where the agent looks:

```bash
uv add adapt-agent          # or: pip install adapt-agent
uv run adapt install skill  # or: adapt install skill
```

That copies the skill into `./.claude/skills/adapt-agent/`. An agent working in
the project discovers it automatically and loads it when a task matches its
description — writing evals, adding guardrails, or optimizing an agent.

## What a skill is

A skill is a folder containing a `SKILL.md` (YAML frontmatter plus markdown
instructions) and optional supporting files. Agents read the frontmatter
`description` to decide *when* the skill is relevant, then load the body — and
only the referenced files they actually need — to learn *how* to do the work.
This is why a skill beats pasting documentation into a prompt: the detail stays
out of context until it is needed.

The bundled skill is laid out as:

```
adapt-agent/
├── SKILL.md                      # when to use it + the core recipes
└── references/
    ├── evals.md                  # the full eval surface
    ├── guardrails.md             # firewall / policy / trust / taint
    └── optimization.md           # tunable knobs, optimizers, training config
```

## Installing

```bash
adapt install skill                       # every bundled skill -> ./.claude/skills
adapt install skill adapt-agent           # one named skill
adapt install skill --target user         # -> ~/.claude/skills (all your projects)
adapt install skill --dir path/to/skills  # any other directory
adapt install skill --force               # replace an existing installation
adapt install skill --json                # machine-readable result
```

| Target | Destination | Scope |
| --- | --- | --- |
| `project` (default) | `./.claude/skills/` | the current project; commit it to share with the team |
| `user` | `~/.claude/skills/` | every project you work on |
| `--dir PATH` | that directory | anything else that reads `SKILL.md` folders |

Installing is a copy, not a link, so the skill keeps working offline and can be
committed to version control. It never overwrites an existing installation
unless you pass `--force` — local edits are safe by default. `--force` replaces
the directory outright rather than merging, so files removed in a newer release
do not linger.

Upgrading the library does not move the copy already on disk. Re-run
`adapt install skill --force` after `pip install -U adapt-agent` to pick up the
new version.

## Listing what is bundled

```bash
adapt skills          # names, file counts, and the first line of each description
adapt skills --json
```

## From Python

The same registry is importable, which is useful in a postinstall hook, a
project bootstrap script, or a test:

```python
from adapt_agent.skills import available_skills, get_skill, install_skill, validate_skill

[s.name for s in available_skills()]        # -> ["adapt-agent"]

skill = get_skill("adapt-agent")
skill.description                            # what an agent matches on
skill.files                                  # ("SKILL.md", "references/evals.md", ...)
skill.read("references/evals.md")            # file contents, straight from the wheel

result = install_skill("adapt-agent", target="project", force=True)
result.path                                  # .../.claude/skills/adapt-agent
result.replaced                              # True when an install was overwritten

validate_skill(skill)                        # [] when the skill is well-formed
```

Files are read through `importlib.resources`, so this works from a wheel, an
editable checkout, or a zipped distribution. Nothing in `adapt_agent.skills`
imports an agent framework or an LLM SDK.

## Both console scripts

The CLI is published as `adapt-agent` and the shorter `adapt`; they are the same
program, so use whichever reads better:

```bash
adapt install skill
adapt-agent evaluate myapp.agents:agent --data golden.jsonl --metric checks
```

Under uv, prefix with `uv run` (`uv run adapt install skill`) or call it
directly once the environment is activated.

## Writing your own skill

The registry discovers any directory under `adapt_agent/skills/` that contains a
`SKILL.md`, so a fork can bundle house-specific skills the same way. Two rules
keep a skill portable:

1. **Frontmatter stays within the portable set** — `name`, `description`,
   `license`, `compatibility`, `metadata`, `allowed-tools`. Claude Code accepts
   more, but only these survive a claude.ai upload or the Skills API.
2. **`name` matches the directory name**, in lowercase-with-hyphens form.

`validate_skill()` checks both, plus description length, and the test suite runs
it over every bundled skill — along with a link check (every relative markdown
link resolves to a file that ships) and a packaging check (every skill file is
covered by a `package-data` glob, so it actually reaches the wheel).

A good `description` is the whole game for discovery: it must say what the skill
does *and* when to use it, in terms that match how someone would phrase the
request. Keep `SKILL.md` itself short and push depth into `references/` files
that the agent loads only when needed.
