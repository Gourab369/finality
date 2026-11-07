---
name: self-improve
description: "Structured reasoning for non-trivial work (bugs, features, refactors, reviews, research, analysis, planning, ops): orient, recall past lessons, reason from evidence, verify, then record a reusable lesson so the next task goes better."
created_by: finality
status: active
---

# Self-improve: reason well, verify, get better each time

Use this for work with more than one step, an unclear cause, or a costly wrong answer. Skip it for trivial requests.

## 1. Orient before acting

1. **Goal and check.** Restate the goal in one sentence and decide how you will prove it is done (a passing test, a
   command's output, a cited source). Work backwards from that check.
2. **Classify the task** and follow its playbook in section 3. Mixed tasks: split them and use one playbook per part.
3. **Recall what was learned before.** Search `memory_recall` for similar past work and `lookup_table_query` for proven
   plans; use `knowledge_lookup` for facts and the skills list for procedures. Reuse what worked; avoid what failed.
4. **Knowns, unknowns, assumptions.** List them briefly. Resolve unknowns cheaply first: read the file, run the command,
   search the code. Ask the user (`ask_user`) only for what you cannot find out yourself.
5. **Choose the approach.** For a non-trivial decision, consider two or three options and pick by explicit criteria:
   correctness, risk, effort, reversibility, fit with existing conventions. Prefer the smallest change that fully solves it.

## 2. Reasoning discipline

- **Evidence over recall.** Check facts about this codebase or system by reading them; cite file and line or the URL.
  Keep facts, inferences and guesses apart, and never present a guess as a fact.
- **Hypothesis, test, update.** Change one variable at a time. Rank hypotheses by likelihood and cost to test.
- **Decompose** until every step has an observable result.
- **Sanity-check** numbers, units, limits and edge cases: empty, null, one, many, huge, unicode, paths with spaces,
  time zones, Windows vs Unix differences, concurrency.
- **Find the root cause.** Ask "why" until you reach a cause you can change; fix that, not the symptom.
- **Step back when stuck.** When two attempts failed for the same reason, re-examine your assumptions instead of trying a
  third variation of the same idea.
- **Be efficient.** Batch independent reads and searches, prefer targeted search to reading whole files, don't re-read
  what you already have, keep tool output small, and stop exploring as soon as you have enough to act.

## 3. Playbooks

**Bug or failing test**
1. Reproduce it and keep the exact command.
2. Read the whole error.
3. Locate the failure by tracing from the symptom to the code.
4. Rank the hypotheses and run the smallest experiment that tells them apart.
5. Fix the root cause.
6. Re-run the reproduction and the related tests.
7. Search for the same bug elsewhere.

**New feature**
1. Find the existing patterns and conventions.
2. Define the interface or contract first.
3. Build the smallest end-to-end slice.
4. Add tests for the behaviour and its edge cases.
5. Wire it in.
6. Update the config and docs.

**Refactor**
- Start with the tests green.
- Take small steps that keep the behaviour the same, running the tests after each one.
- Never mix feature changes into a refactor.

**Review or audit**
- Check correctness, security (injection, path traversal, auth, secrets, unsafe deserialization), error handling,
  concurrency, performance and tests.
- Report findings ranked by severity, each with file and line, the impact, and a concrete fix.

**Performance**
1. Measure a baseline.
2. Find the hot spot with data.
3. Change one thing and measure again.
4. Report before and after numbers.

**Research or question**
- Split it into sub-questions.
- Prefer primary sources: official docs, specs, source code.
- Cross-check anything important in two sources, and note dates and versions.
- Cite the URLs and say how certain you are.

**Analysis or data**
1. Pin down the question and the metric.
2. Inspect the data's shape and quality first.
3. Use simple methods before complex ones.
4. Check the results against intuition and known totals.
5. Show the method so it can be reproduced.

**Multi-step plan**
- Give every step a clear output and success criteria, and make dependencies explicit.
- Run independent steps in parallel, and put the riskiest unknown first.
- In Finality, use `plan_create` with self-contained instructions for sub-agents.

**Writing or docs**
- Know the reader and the purpose before writing.
- Lead with the answer and use concrete examples.
- Cut filler.
- Check the text against the code or facts it describes.

**Shell or ops**
1. Inspect the current state before changing it.
2. Prefer dry runs and reversible steps.
3. Make one change at a time and confirm the result.
4. Never widen the scope beyond the request.

## 4. Verify before finishing

- Does the result pass the check from step 1? Did you run the build, the tests or the command you said you would?
- Are the edge cases covered? Did you change anything beyond the request?
- Would a careful reviewer accept it?
- If something could not be verified, say exactly what and why.

## 5. Improve after every task

1. **Retrospective.** Think through two to four points: what worked, what was slow or wrong, the root cause of any
   failure, and what you would do differently.
2. **Record a lesson, when there is one.** End the final answer with one line, `Lesson: ...`, stating a concrete,
   reusable insight. For example: "In this repo, run tests with `cargo test -p <crate> --locked`; the full suite times out."
   The Memory agent files the session into events, and the Learner turns repeated successful patterns into skills and
   knowledgepedia entries, so a clear lesson here becomes an improvement next time.
3. **Apply it next time.** Step 1.3 (recall) is where earlier lessons come back; never skip it on non-trivial work.
4. **Keep lessons clean.** Never record secrets, one-off details, or guesses as lessons. One sharp lesson beats five
   vague ones; write none if nothing was learned.

## 6. Report

- Lead with the result, then the key evidence (what you ran or read), then caveats and next steps.
- Say clearly what is done and what is not.
- Keep it short.

These practices never override the permission mode, approvals, or the rules for untrusted content.
