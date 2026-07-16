*This project has been created as part of the 42 curriculum by lmatthes.*

# Call Me Maybe

![CI](https://github.com/eepylaurie/42-call_me_maybe/actions/workflows/ci.yml/badge.svg)

Introduction to function calling in LLMs.

## Description

This project implements **function calling** for a small language model: it
turns a natural-language request (e.g. *"What is the sum of 2 and 3?"*) into a
structured, machine-executable function call (`fn_add_numbers` with
`{"a": 2, "b": 3}`) rather than a natural-language answer.

The core challenge is **reliability**. A 0.6B-parameter model, if simply
prompted to emit JSON, produces valid output only some of the time. This
project reaches **100% valid, schema-compliant JSON** by using **constrained
decoding**: instead of hoping the model produces good JSON, generation is
guided token-by-token so that only characters keeping the output on a valid
path are ever allowed. The model still decides *which* function to call and
*which* arguments to supply; only the JSON structure is guaranteed.

The project runs locally using the provided `llm_sdk` package, which loads
Qwen3-0.6B by default.

## Instructions

The reviewer only needs [`uv`](https://docs.astral.sh/uv/) installed. The SDK
is vendored in `llm_sdk/` and consumed as a local path dependency.

```bash
# install dependencies (numpy, pydantic, and the local llm_sdk) into a venv
make install        # runs: uv sync

# run with the default input files in data/input/
make run            # runs: uv run python -m src

# lint (flake8 + mypy)
make lint
```

Custom paths can be given explicitly:

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

By default the program reads from `data/input/` and writes to
`data/output/function_calling_results.json` (the output directory is created
automatically and is not committed).

Pass `--verbose` to print the constrained-decoding trace to `stderr` (see
[Watching it work](#watching-it-work) below). The generated output file is
unchanged.

> **Note for 42 machines:** the CUDA build of `torch` is large, and the home
> quota is limited. The `Makefile` therefore routes uv's cache, the model
> cache, and the virtual environment to `goinfre` automatically when it is
> present. On any other machine this is a no-op and a plain `uv sync` works.

The first run downloads the Qwen3-0.6B weights (~1.5 GB) from the Hugging Face
Hub and caches them; subsequent runs load from cache.

## Algorithm explanation

The pipeline turns a prompt into valid JSON one token at a time:

```
        User prompt
            │
            ▼
     Build steering prompt        (request + available functions)
            │
            ▼
        LLM logits                (a score for every next token)
            │
            ▼
     Constraint machine  ──▶  Mask illegal tokens (-∞ logits)
            │
            ▼
     Select highest-scoring legal token
            │
            ▼
     Advance the machine, repeat until complete
            │
            ▼
        Valid JSON
```

Generation is a loop over a single primitive from the SDK,
`get_logits_from_input_ids(ids)`, which returns the model's score for every
possible next token given the context so far. Normally you would pick the
highest-scoring token and append it. Constrained decoding intervenes **before**
that choice: at each step, the logits of every token that would break the
required structure are set to negative infinity, so only valid tokens can be
selected.

The key realisation is that the target output is not free-form JSON but a
**fixed template with two kinds of holes**:

```
{"name": "<FUNCTION NAME>", "parameters": {"<key>": <VALUE>, ...}}
```

Everything except the function name and the argument values is fixed structure
(braces, quotes, keys, colons, commas).

So the problem is not "parse arbitrary JSON" but "emit a known stencil, and at
the free slots restrict to what is legal." At most positions, only a single
character is valid. As a result, the decoder leaves the model with almost no opportunity to produce invalid JSON.

The implementation has three layers:

1. **`vocabulary.py` — token ↔ text.** The constraint logic reasons about
   characters, but the model works in token IDs. This maps every token ID to
   the real string it represents. Qwen uses byte-level BPE, so tokens are
   stored in `vocab.json` under a byte-to-unicode encoding (a leading space is
   stored as `Ġ`, a newline as `Ċ`, etc.). The module rebuilds GPT-2's
   `bytes_to_unicode` table and reverses it to recover real text.

2. **`constraints.py` — the state machine.** A character-level, stateful
   machine that walks the template. It exposes `allowed_next()` (which
   characters are legal now), `advance(char)` (commit one and update state),
   and `is_complete()`. It moves through phases: emit the fixed prefix, choose
   the function name (a prefix walk over the allowed names), then, for the
   chosen function, emit each `"key": ` and constrain the value by type
   (float, integer, string, or boolean) with the correct separators, and
   finally close the object. It also exposes `accepts(text)`, which simulates advancing through a
   whole multi-character token and then restores its state, so the decoder can
   test tokens without committing to them.

3. **`decoder.py` — the loop.** For each step it gets the logits, computes the
   set of legal tokens (a fast first-character filter, then the full `accepts`
   check), masks every illegal logit to −∞, picks the highest-scoring survivor,
   advances the constraint by that token's characters, and repeats until the
   machine is complete. Generation stops when the structure is complete — it
   does not rely on the model emitting an end-of-sequence token.

### Worked example
 
**Input prompt:**
 
```
What is the sum of 265 and 345?
```
 
**Step by step**, the decoder builds the call one token at a time:
 
1. It forces `{"name": "` — only one legal character at each step.
2. At the name slot, the legal characters are those that continue an allowed
   function name. The model's logits pick `fn_add_numbers` over the
   alternatives.
3. It forces `", "parameters": {"a": ` — the first key is fixed, and the value
   is a number slot.
4. In the number slot only digits (and a leading sign or decimal point) are
   legal; the model emits `265`, then the separator.
5. It forces `"b": `, the model emits `345`, and the machine closes with `}}`.

**Final output:**
 
```json
{
  "name": "fn_add_numbers",
  "parameters": {
    "a": 265,
    "b": 345
  }
}
```

### Watching it work

Run any prompt with `--verbose` to watch the constraint machine in action. At
each generation step, the program prints the current decoding phase, how many
of the ~151k vocabulary tokens remain legal, and which token the model
ultimately selects:

```
step  1 | PREFIX       | legal    2/151643 | chose '{"'
step  6 | NAME         | legal   16/151643 | chose '_add'
step 10 | AFTER_NAME   | legal    7/151643 | chose 'parameters'
step 16 | VALUE_NUMBER | legal   11/151643 | chose '2'
step 17 | VALUE_NUMBER | legal   11/151643 | chose '.'
step 18 | VALUE_NUMBER | legal   10/151643 | chose '0'
step 19 | SEPARATOR    | legal   11/151643 | chose ','
step 27 | DONE         | legal   12/151643 | chose '}}'
```

At most positions, only a handful of tokens (out of 151,643) keep the output
on a valid path **(often just one or two)**. That dramatic reduction is the essence
of constrained decoding: the model still decides which function to call and
which values to generate, but every available choice is guaranteed to preserve
a valid JSON structure.

The trace also highlights two implementation details. First, numeric values are
emitted according to the schema one token at a time (for example, `2`, then
`.`, then `0` for the float `2.0`), allowing the constraint machine to validate
each character as it is produced. Second, generation stops as soon as the JSON
object is complete rather than waiting for an end-of-sequence token.

## Design decisions

- **Character-level rules, token-level masking.** Validity is expressed as
  "which characters may come next," which is simple to reason about and test.
  The decoder translates that into "which tokens are allowed" via the
  vocabulary. Keeping the two concerns separate keeps each side small.
- **Stateful machine built fresh per prompt.** The machine tracks its current
  phase and buffers; a new one is constructed for each prompt, so there is no
  state carried between prompts.
- **Injected model access.** The decoder receives its logits and encode
  functions as arguments instead of importing the SDK. This lets the entire
  masking-and-selection logic be tested with a fake model, and keeps the SDK
  dependency confined to `pipeline.py`.
- **Prompt steers, decoding guarantees.** The prompt lists the available
  functions and the request only to influence *which* valid choice the model
  prefers. The structure is enforced by the constraint machine, never by the
  prompt — the project does not rely on the model spontaneously producing
  correct JSON.
- **Single I/O boundary with one error type.** All file and JSON errors are
  funnelled into one `InputError`, so the program reports a clear message and
  never crashes on missing or malformed input.
- **Pydantic everywhere.** Input files and output are validated through pydantic
  models, so malformed data is rejected early with a readable message.

## Performance analysis

Measured on a held-out set of 11 prompts (Qwen3-0.6B, CPU) — a private
evaluation set of functions and prompts the implementation was not developed
against, including `integer` parameters not present in the public examples:

- **Validity: 100%.** Every output is valid JSON with exactly the required keys
  (`prompt`, `name`, `parameters`) and correct types. This is guaranteed by
  construction, not by chance.
- **Accuracy: ~91% (10 / 11).** On unseen functions the model selects the
  correct one and extracts correct arguments on 10 of 11 prompts, including
  generating working regular expressions for the substitution tasks. Reaching
  this on a held-out set shows the approach generalises rather than fitting the
  examples.
- **Speed.** All prompts are processed well within the 5-minute budget on
  standard hardware, after the one-time model load.

The single miss is the clearest illustration of what this project does and does
not control. On a template-formatting prompt (`Say "hello" to {name}`), the
model produced the argument without the inner quotes (`Say hello to {name}`).
The output was still structurally perfect: correct function, correct keys,
correct types, parseable JSON. In other words, constrained decoding guarantees
**structural and schema correctness on 100% of prompts**, while semantic
correctness of free-text values is bounded by the reasoning ability of the
underlying model. The two are decoupled by design, and only the second
remains — exactly where a 0.6B model is expected to be weakest.

## Challenges faced

- **Byte-level BPE.** Recovering real text from `vocab.json` required
  rebuilding GPT-2's byte-to-unicode mapping rather than reading tokens as
  literal strings. Tokens that split a multi-byte character are decoded with
  `errors="replace"` so they are simply never selected.
- **Ending an unbounded value.** A JSON number has no closing delimiter, so the
  machine offers the terminator (`,` or `}`) as a legal next character once the
  number is valid; when the model picks it, that character is re-dispatched
  into the following structural literal.
- **String escaping.** A `"` closes a string unless it follows an unescaped
  backslash, which the machine tracks explicitly.
- **Environment/tooling.** Resolving the vendored SDK import, and a severe
  slowdown caused by the project and its virtual environment living on an
  iCloud-synced folder, fixed by moving to a plain local path. On 42 machines,
  disk-quota limits required routing caches and the venv to `goinfre`.

## Testing strategy

Because the model is injected into the decoder, the whole pipeline can be
verified without loading the LLM:

- The **constraint machine** is driven character-by-character with plain
  strings and asserted to accept exactly the valid function calls (single and
  multiple parameters, numbers with signs and decimals, strings with escaped
  quotes) and to reject illegal characters.
- The **decoder** is run against a **fake vocabulary and fake logits
  function**, confirming that (a) every output is valid JSON regardless of the
  logits, and (b) biasing the logits steers the function choice — demonstrating
  that the model, not a heuristic, selects the function.
- **End-to-end**, the program is run against the real model on the provided
  test set and the output file is validated.

## Resources

- [uv documentation](https://docs.astral.sh/uv/) — dependency management and
  the project runner used throughout.
- [Hugging Face Transformers](https://huggingface.co/docs/transformers) — the
  library the SDK wraps to load and run the model.
- [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) — the default model, a
  small causal LM well suited to showing that structure can come from decoding
  rather than model size.
- [Tokenizer summary](https://huggingface.co/docs/transformers/en/tokenizer_summary)
  — background on byte-level BPE, the encoding used by Qwen's `vocab.json` and
  reversed in `vocabulary.py`.

### Use of AI

AI was used as a development assistant during this project. It helped with understanding concepts related to LLMs and function calling, discussing implementation approaches, reviewing code, and improving documentation.

All design decisions, implementation, debugging, and testing were done by me. AI-generated suggestions were reviewed, adapted when needed, and validated before being used.