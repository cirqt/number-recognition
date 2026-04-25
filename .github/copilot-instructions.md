## Number Recognition — Copilot instructions

Keep suggestions tight and actionable. This repo is an educational handwritten-digit recognizer implemented from scratch using NumPy. The goal for any change is clarity and pedagogy first, performance second.

Key facts (quick):
- Language: Python 3.10+
- Minimal runtime deps: numpy, matplotlib. TensorFlow is used only for MNIST data loading in some phases (see README).
- Author intent: implement algorithms (forward, backprop, SGD, L2, etc.) by hand; avoid introducing heavy ML frameworks for model code.

Where to look (high value files):
- `README.md` — overarching project goals and quick-start commands.
- `ROADMAP.md` and `SKILLS.md` — project phases, domain vocabulary and notation (Nielsen's notation) used throughout.
- `src/` — primary source modules: data loading, model, training loop, evaluation. When a file in `src/` is empty or missing, prefer to follow the conventions described in README/ROADMAP instead of introducing new structural patterns.

Project patterns and conventions
- Pure-NumPy model code: prefer NumPy-only implementations for the `Network`/model logic. Do not replace model internals with Keras/PyTorch. Small helper scripts (data download, plotting) may use other libs.
- File responsibilities (follow README nomenclature):
  - `src/data.py` (or `src/loader.py`) — data loading and preprocessing (MNIST fetch, reshape, normalize, train/validation split).
  - `src/model.py` (or `src/network.py`) — Network class and core forward/backprop implementations.
  - `src/train.py` — training loop, CLI flags (learning rate, epochs, regularization), checkpointing (simple pickle/np.save).
  - `src/evaluate.py` — metrics, confusion matrix, matplotlib plots.
- CLI and quick-start: repository expects a venv and `python src/train.py` as the common workflow (see README). Keep any added CLI flags minimal and well-documented in README.

Examples from this repo
- The README expects the training entrypoint `python src/train.py` and a Windows venv activation snippet: `.venv\Scripts\activate`.

Development workflows (what Copilot should suggest)
- Small, well-scoped edits: implement a single function (forward pass, one BP equation), add a unit-test-like script, or add a small demo notebook. Provide short code + brief comment explaining the pedagogical intent.
- Use NumPy idioms (vectorized ops, broadcasting). If a suggestion relies on external libraries, explain why and keep it optional.
- When adding plotting or visualization, follow the project's Matplotlib usage pattern described in README (simple, static plots saved to `plots/` or shown interactively).

Testing and verification guidance
- Repository currently has no tests — prefer adding tiny runnable checks (e.g., a short script in `tests/` that constructs a tiny Network, runs one forward/backward pass, asserts shapes).
- Keep test dependencies minimal (use only stdlib + numpy). If adding pytest, document the new workflow in README.

What to avoid
- Don't replace educational implementations with calls to high-level ML APIs for model internals.
- Don't add heavy new dependencies without an explicit note in README and a short justification.

If you need more context
- Inspect `ROADMAP.md` and `SKILLS.md` for algorithmic notation and intended progression; new code should align with the roadmap phases.
- If files in `src/` are empty (current state), prefer to implement missing modules following the README structure and name choices (`data.py`/`model.py`/`train.py`/`evaluate.py`).

When updating this file
- Preserve the educational constraints (NumPy-first). If merging in new AI guidance, keep suggestions concrete and cite the file(s) changed.

Questions for maintainers
- Do you prefer `src/loader.py` vs `src/data.py` and `src/network.py` vs `src/model.py`? Keep naming consistent with README.

End of instructions.
