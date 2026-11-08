The user ran /init: create their global instruction file at `{{path}}`. Finality adds this file to the system prompt of every session in every project, so it holds the user's general working preferences and environment, not facts about one project.

1. Study the current project briefly (read_file and search_files, at most about 10 calls): languages, toolchains, build, test and lint commands, code style, how tests are organized, commit conventions if visible.
2. From that, infer what holds across the user's projects: operating system and shell (see your system prompt), main languages and tools, preferred style. Leave out paths, module names and anything else specific to this repository.
3. Write the file with edit_file, one `write` operation, at exactly `{{path}}`. If the file already exists, stop and say so instead of overwriting it.
   Keep it under 60 lines of Markdown with these sections:
   - `# Global instructions`
   - `## Environment`
   - `## Working rules` (for example: read code before changing it, keep changes small and focused, run the relevant build and tests and fix failures before finishing, ask before destructive commands)
   - `## Style`
   If tool_search finds tools named mcp__vscode__*, add a rule to run mcp__vscode__diagnostics on changed files after edits.
4. Finish with a short summary of what you wrote, and tell the user they can edit the file at any time.
