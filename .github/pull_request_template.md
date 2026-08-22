## What this changes

<!-- One or two sentences. -->

## Why

<!-- What was wrong, or what was missing. If a measurement motivated it, put
     the number here. -->

## The test that fails without it

<!-- Name it, and say that you saw it fail against the unfixed code. A
     regression test never observed failing is a test of nothing. -->

## Checklist

- [ ] The suite passes locally (`python -m pytest tests/ -q`)
- [ ] There is a test that fails without this change, and I have seen it fail
- [ ] Comments explain what the code cannot say for itself, where that applies
- [ ] If this changes what the package returns or how it behaves, the tool
      descriptions and the README say so — they are what a caller reads
