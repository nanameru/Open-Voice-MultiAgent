# Agent Module Reorganization Plan

## Current Pain Points
- `main.py` mixes agent class definitions, supporting data models, and tool functions in a single monolithic file.
- A partially populated `src/agents/` tree exists for constants/models, but key agent classes remain in `main.py`.
- The `src/tools/` directory is empty; tool callables are embedded directly in agent classes.

## Target Layout
```
project-root/
├── main.py
├── src/
│   └── agents/
│       ├── __init__.py
│       ├── constants.py
│       ├── models.py
│       ├── shared.py            # shared dataclasses & utility helpers currently in main.py
│       ├── lead_editor_agent.py
│       └── specialist_editor_agent.py
```

- Remove the unused `src/tools/` package entirely. Tool implementations that remain bound to agent methods stay within the agent modules themselves. If free-standing tool functions are required later, colocate them with the owning agent module or create `src/agents/tools.py` rather than a top-level `src/tools/` package.
- Keep agent-facing constants (LLM instructions, model identifiers, etc.) inside `constants.py`. Expand it only with values consumed across multiple agents.
- Keep any shared Pydantic/dataclass schemas in `models.py`. If more than data structures are needed, use the proposed `shared.py` module for agent-specific helper logic.

## Migration Steps
1. **Create missing package scaffolding**
   - Ensure `src/agents/__init__.py` exports the public agent classes (e.g., `from .lead_editor_agent import LeadEditorAgent`).
   - Add a new `src/agents/shared.py` module that will host `CharacterData` and `StoryData` dataclasses, along with any other shared helper functions currently defined in `main.py`.

2. **Move agent classes out of `main.py`**
   - Cut the `LeadEditorAgent` class from `main.py` and paste it into `src/agents/lead_editor_agent.py`. Import shared dataclasses and utilities from `src.agents.shared` and constants from `src.agents.constants`.
   - Do the same for the `SpecialistEditorAgent` class, placing it in `src/agents/specialist_editor_agent.py` and referencing shared constructs as needed.

3. **Refactor tool callables**
   - Retain `@function_tool` methods within their respective agent classes; no need for a separate `tools` package unless functions are reused across multiple agents. If that happens, define them alongside shared helpers in `shared.py`.

4. **Update `main.py`**
   - Replace inline class definitions with imports: `from src.agents.lead_editor_agent import LeadEditorAgent` and `from src.agents.specialist_editor_agent import SpecialistEditorAgent`.
   - Import shared data structures via `from src.agents.shared import CharacterData, StoryData` (or whichever names you export).
   - Update references to constants/models to use the `src.agents` package imports, removing now-duplicate code in `main.py`.

5. **Prune obsolete directories**
   - Delete the `src/tools/` directory since it's unused. If version control history matters, remove the directory after confirming no references remain.

6. **Smoke-test imports**
   - Run `python -m compileall src main.py` or simply `python main.py --help` to confirm the module graph resolves without circular dependencies after the move.

## Notes
- Keep package-relative imports (e.g., `from src.agents.shared import StoryData`) to avoid ambiguity when the project is installed as a package.
- When moving code, retain logger names and function signatures to avoid breaking existing behaviour.
- Consider adding unit tests under a future `tests/agents/` package after refactoring to lock down agent interaction patterns.
