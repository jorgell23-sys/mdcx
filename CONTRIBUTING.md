# Contributing

## What is most useful

A report that reproduces something is worth more than a patch that fixes it,
because the reproduction is the part that is expensive to reconstruct later.
Several of the corrections in this package came from reports that measured a
symptom rather than describing it, and in more than one case the measurement
showed the obvious fix was the wrong one.

What makes a report act on itself:

- the version, and whether it was reproduced against the published package
  rather than a working copy
- the smallest input that shows it
- what was expected and what happened, as two separate statements
- what was tried and ruled out, which stops the next person repeating it

## Running the tests

```
pip install -e ".[convert,mcp]" pytest
python -m pytest tests/ -q
```

Tests that need a model skip themselves when it is absent. That is a supported
way to install this package and the suite is meant to pass without those extras:

```
pip install -e ".[all]" pytest      # everything, including the models
```

The full suite with models takes a few minutes, most of it encoding.

## What a change needs

**A test that fails without it.** Not a test that passes with it -- one that has
been seen to fail against the unfixed code. A regression test that was never
observed failing is a test of nothing.

**A comment explaining why, where the why is not obvious.** This codebase leans
on that heavily: several constants here are the third value tried, and the two
that did not work are written down beside them so the next attempt starts
somewhere new. Prose that says what the code already says is noise; prose that
says what the code cannot is the point.

**A measurement, where the change is about cost or quality.** "Faster" and
"better" are claims. The number, the material it was measured on, and whether
anything else was running are what make them checkable.

## Conventions

Comments and docstrings are in English. Local variable names are frequently in
Spanish and that is deliberate and long-standing; matching the surrounding code
matters more than picking a side.

Continuous integration runs the suite on the oldest and newest supported
interpreters, on Linux and Windows. It has caught things a single machine
cannot -- including a declared Python version the package had never actually
worked on -- so a change that passes locally is not yet known to pass.

## Reporting something that affects security

Not through an issue. See [SECURITY.md](SECURITY.md).
