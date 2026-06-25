---
applyTo: "**"
---

# PR Review Instructions

You are reviewing a pull request diff. Focus **only on critical issues** that could cause real harm if merged. Do not nitpick style, formatting, naming conventions, or minor refactoring opportunities.

## What to flag

Report only findings that fall into one of these categories:

- **Security vulnerabilities** — injection, unsafe deserialization, exposed secrets, insecure defaults, path traversal
- **Correctness bugs** — logic errors, off-by-one errors, wrong operator, incorrect algorithm behavior
- **Data loss or corruption** — in-place mutations that destroy inputs, incorrect tensor operations that silently produce wrong results
- **Crashes or unhandled exceptions** — missing null/None checks at API boundaries, index out of bounds, type errors that will raise at runtime
- **Breaking changes** — changed function signatures, removed public API, changed default argument behavior that breaks existing callers
- **Resource leaks** — unclosed files, unreleased GPU memory, dangling references in training loops
- **Concurrency issues** — race conditions, deadlocks in multi-process or multi-GPU code

## What NOT to flag

Do not comment on:

- Code style, formatting, or whitespace
- Variable or function naming preferences
- Missing docstrings or comments
- Refactoring suggestions ("this could be cleaner if…")
- Performance micro-optimizations unless they cause an order-of-magnitude regression
- Tests that could be added but aren't
- Anything subjective or that is already handled by a linter

## Output format

If you find critical issues, use the following template for each one:

---

### `path/to/file.py` · Category

**Issue:** One concise sentence describing the problem.

**Impact:** What breaks or goes wrong if this ships.

**Problematic lines (quoted verbatim from the diff):**

```python
<exact lines copied from the provided diff — do not paraphrase or reconstruct>
```

**Fix:** One sentence describing the minimal correction needed.

---

Rules for the template:
- The heading must be a real file path from the diff followed by the category name (e.g., `### src/train.py · Correctness bug`)
- The problematic lines must be copied character-for-character from the diff input — never rewritten or inferred
- The fix must be a description, not generated code, to avoid hallucinating incorrect patches

If you find no critical issues, output the exact string below and nothing else — no explanation, no preamble:

```
NO_CRITICAL_ISSUES
```

Do NOT post a comment or add any other text when there are no critical issues.

Keep the review short. Fewer, higher-confidence findings are better than a long list of speculative ones.
