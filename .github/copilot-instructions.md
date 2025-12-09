# Copilot Instructions for Agent

Purpose: short, actionable guidance for AI coding agents working in this repository.

**Big Picture**
- **Entry point:** `run.py` — simple interactive CLI that instantiates `ChatBot` and loops on `input()`.
- **Model layer:** `model/model_class.py` defines an abstract `LLM` and concrete `ChatBot` and `Router` classes that wrap Hugging Face tokenizers/models (`AutoTokenizer`, `AutoModelForCausalLM`).
- **Config:** `model/model_config.yaml` holds model names. The code loads it via `utils/file_utils.read_yaml`.
- **Utilities:** `utils/file_utils.py` currently contains a small YAML loader used across the repo.

**Important Patterns & Conventions (project-specific)**
- **LLM wrapper:** `LLM` subclasses load tokenizer+model in `__init__` and expose a `generate(...)` method. When editing or adding LLMs follow this pattern.
- **Conversation history format:** history is a list of dicts: `{'role': 'user'|'assistant'|'system', 'content': '<text>'}`. `ChatBot.generate()` appends two entries per interaction (user then assistant).
- **Template usage:** code relies on `tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)` — the tokenizer is expected to supply chat templates. Prefer using the same API when creating prompts.
- **Special token:** code looks up `self.tokenizer("</think>").input_ids[0]` and uses that token id to slice generated output. Keep this when modifying decode logic.
- **Device handling:** `DEVICE` is set to `cuda` if available else `cpu`. Ensure tests and runs account for both.

**Key Files to Inspect for Changes**
- `run.py` — user-facing run loop.
- `model/model_class.py` — core logic for loading models and generation.
- `model/model_config.yaml` — where model names are set.
- `utils/file_utils.py` — YAML loader used to read config.

**Gotchas / Known Issues**
- Config key mismatch: `model_class.py` expects keys like `chat_model_name` and `router_model_name` but `model_config.yaml` currently provides `chat_model_name` and `Orchestration_model_name`. When changing model names, either update the YAML keys or `model_class.py` to match.
- `trust_remote_code=True` is used when loading models. That allows models with custom code to run — treat as a security consideration and pin model versions when possible.
- `max_new_tokens=2048` is set in `ChatBot.generate()`; be careful changing this for memory-constrained environments.

**Developer Workflows / Commands**
- Run interactively (local dev):

```powershell
python run.py
```

- Python deps (not provided in repo): install at minimum `torch`, `transformers`, `pyyaml`. Example:

```powershell
python -m pip install torch transformers pyyaml
```

- GPU: if you want CUDA acceleration ensure a CUDA-enabled `torch` wheel is installed and that `DEVICE` becomes `cuda`.

**When editing or extending**
- Add new LLMs by subclassing `LLM` and implementing `generate(self, **kwargs)` following the `ChatBot` example.
- Maintain the history list structure; changes to schema must be propagated to `run.py` and any other consumers.
- If you change tokenization/prompt templates, update all callers of `tokenizer.apply_chat_template()`.

**Example snippets**
- How `history` is built and used (from `model_class.py`):

```
messages = self.history[max(0, len(self.history)-6):] + [{'role': 'user', 'content': user_input}]
text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = self.tokenizer(text, return_tensors='pt')
outputs = self.model.generate(**inputs.to(DEVICE), eos_token_id=self.tokenizer.eos_token_id, max_new_tokens=2048)
```

**Questions the agent should ask before making changes**
- Should the YAML key names be normalized? (e.g., rename `Orchestration_model_name` → `router_model_name`)
- Are we allowed to change `trust_remote_code` usage or must we pin specific models?
- Is the interactive CLI the canonical runner, or will we add an API layer/serving component?

If anything here is unclear or you'd like more detail (examples of tests, CI rules, or an updated `requirements.txt`), tell me which part to expand and I'll iterate.
