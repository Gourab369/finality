Classify the user's latest request for routing. Reply with JSON only, matching the schema.

- query_type: one of {{query_types}}, or a short new lowercase type if none fit.
- topic: a short lowercase subject path such as "coding/rust", "physics", "devops/docker".
- complexity: "simple" if it can be answered or done in one focused pass; "multi_step" if it needs a plan with several dependent steps.
- needs_tools: true if answering requires reading files, searching, editing, or running commands.
- tool_hints: zero or more of "files", "shell", "web", "browser".
- knowledge_topics: specific technologies or facts the answer depends on (may be empty).
