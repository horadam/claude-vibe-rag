# How the features work

A plain-language guide to what's been built on top of the original RAG
service, written to be studied and explained. Each feature has the same parts:

- **What it is** and **why it matters**
- **How it works**, with links to the code
- **Say it in 30 seconds**: a version you could give in an interview
- **Questions you might get**, with short answers
- **Try it**: something to run or change to check your understanding

General concepts are also in the learning notebook
(`/Users/punchcode/Projects/learning-notes.md`, entries L017–L020; the provider work extends L005 and L009).

---

## Part 1 — Eval foundation (2026-11-06)

### The big picture

A RAG answer can go wrong in two different places:

1. **Retrieval** didn't find the right passages, so the model never saw the
   answer.
2. **Generation** had the right passages but answered wrongly, left something
   out, invented a detail, or refused when it shouldn't have.

The original harness only looked at the final answer, so a failure couldn't be
traced to either stage. The new harness scores both stages separately for
every question:

```
question ──► retrieve ──► [retrieval scores: hit, recall, MRR, precision]
                │
                ▼
             generate ──► [answer checks: keywords + refusal behaviour]
                │
                ▼
              judge   ──► [correct? grounded?]
                │
                ▼
         run record (config, git commit, timings, tokens) ──► results.md, --compare
```

---

### 1. A labelled eval set

**What it is.** [evals/questions.json](../evals/questions.json) holds 17 cases.
Each answerable case now lists `expected_sources`: the handbook files a correct
answer needs. Refusal cases list none.

**Why it matters.** Without labels you can only grade the final answer. Labels
are what make retrieval measurable: "did we fetch `pto_policy.md` for the PTO
question?" is a yes/no fact you can count.

**How it works.** [src/evaluation/dataset.py](../src/evaluation/dataset.py)
loads the file and refuses to run if it's wrong:

- every `expected_sources` file must exist in `corpus/`
- answerable cases must have sources; refusal cases must not
- ids must be unique; categories must be one of the known set
- a `must_contain` entry can be a list of alternatives:
  `["2 business days", "two business days"]` passes if either appears

Categories and what each one tests:

| Category | Tests | Example |
|---|---|---|
| `single_doc` | the basic loop works | "How long does a PIP run?" |
| `multi_doc` | two documents must both be retrieved and combined | PTO payout = PTO policy cap + termination timing |
| `ambiguous` | a word ("leave") means different things in different docs | "How much notice for leave?" |
| `out_of_scope` | the model refuses instead of guessing | "What's the 401(k) match?" |
| `false_premise` | the question assumes something untrue | "How many weeks of sabbatical after five years?" |

`false_premise` is new and harder than plain out-of-scope: the question
*sounds* answerable, and nearby leave policies tempt the model to repurpose
one as a sabbatical. `out_of_scope_401k_match` is a "near miss": the
compensation doc is about pay, so it will be retrieved, but it says nothing
about retirement.

**Say it in 30 seconds.** "Each eval question is labelled with the documents a
correct answer needs. That lets me score retrieval on its own, before the
model writes anything. The loader validates the labels against the corpus, so
a renamed file breaks the eval loudly instead of silently scoring zero."

**Questions you might get.**
- *Why label sources, not chunks?* Chunk ids change every time chunking
  changes. Sources are stable, so the same labels work across chunking
  experiments.
- *Isn't 17 cases too few?* Yes, for statistics: one case is ~6 percentage
  points. It's enough to catch regressions and show the method; the next step
  is a larger corpus and question set.

**Try it.** Add a case with `"expected_sources": ["sabbatical.md"]` and run
`uv run python -m unittest tests.test_dataset`. Read the error, then remove it.

---

### 2. Retrieval metrics: hit rate, recall, MRR, context precision

**What it is.** Four numbers that describe how well retrieval did, computed in
[src/evaluation/scoring.py](../src/evaluation/scoring.py) (`score_retrieval`).

**Why it matters.** If the answer is wrong *and* the right document wasn't
retrieved, fixing the prompt won't help. These numbers tell you which stage
to work on.

**How it works.** Take the sources of the top-k chunks, in rank order.
Example: expected `{a.md, b.md}`, retrieved `[x.md, b.md, b.md, y.md, z.md]`.

| Metric | Question it answers | Example value |
|---|---|---|
| Hit | Was *any* expected source retrieved? | yes |
| Recall@k | What share of the expected sources were retrieved? | 1 of 2 = 0.5 |
| Reciprocal rank | How high was the first relevant chunk? 1/rank | first relevant at rank 2 → 0.5 |
| Context precision@k | What share of retrieved chunks were relevant? | 2 of 5 = 0.4 |

Averaging reciprocal rank over all cases gives **MRR** (mean reciprocal rank).
Recall matters most for `multi_doc`: getting one of two documents gives a
half-answer. Precision is about noise: irrelevant chunks cost tokens and can
mislead the model.

**Important caveat for this project.** The handbook currently splits into only
14 chunks and k = 5, so each question retrieves about a third of the whole
index. Expect these numbers near 1.0. That says the corpus is too easy, not
that retrieval is excellent (notebook L017). The metrics are built now so the
upcoming larger corpus can be measured from day one.

**Say it in 30 seconds.** "I score retrieval separately from answers. Recall
tells me whether all the needed documents came back, MRR how high the first
good one ranked, and context precision how much noise went into the prompt.
When an answer is wrong I can see straight away whether retrieval missed or
the model misread."

**Questions you might get.**
- *Recall vs precision trade-off?* Raising k usually raises recall and lowers
  precision. The eval lets you choose k with numbers.
- *Why source-level instead of chunk-level?* See part 1: labels stay valid
  across chunking changes. The cost is that "right document, wrong section"
  still counts as a hit.

**Try it.** Once keys are set:
`uv run python eval_cli.py --retrieval-only --k 2 --label k2` and then
`--k 5 --compare latest`. Which multi-doc cases lose recall at k = 2?

---

### 3. Answer checks: normalisation, keywords and refusal behaviour

**What it is.** The cheap, deterministic grade of each answer, in
[scoring.py](../src/evaluation/scoring.py). A case passes when all required
phrases are present, no forbidden phrase is, and the model refused exactly
when it should have.

**Why it matters.** The old check failed on harmless differences: the model
writing `don’t` (curly apostrophe) instead of `don't`, `**1.5 days**` in
bold, or "two" instead of "2". A check that fails on formatting trains you to
ignore it.

**How it works.**
- `normalize()` lowercases, maps curly quotes and dashes to plain ones, strips
  markdown emphasis and collapses whitespace, on both the answer and the
  phrase.
- `is_refusal()` looks for the system prompt's exact refusal sentence.
  `REFUSAL_MESSAGE` in [src/generate.py](../src/generate.py) is now one
  constant used by both the prompt and the eval, so they can't drift apart.
- Refusal is scored both ways: **correct refusals** (refused when it should)
  and **false refusals** (refused an answerable question). A model that
  refuses everything would ace the first and fail the second.

**Say it in 30 seconds.** "Keyword checks are fast and free, but brittle, so I
normalise typography and allow alternative phrasings. Refusals are tracked in
both directions, because over-refusing is a failure too: an assistant that
says 'I don't know' to everything looks safe on half the metrics."

**Questions you might get.**
- *Why keep keyword checks if you have a judge?* They're free, deterministic
  and run on every change. The judge costs money and can itself be wrong.
- *What's the weakness?* A keyword can be present in a wrong answer ("not 26
  weeks but 10"). That's why `secondary_caregiver_max` also uses
  `must_not_contain`, and why the judge exists.

**Try it.** In `tests/test_scoring.py`, add a test that `"I do not have that
information in the handbook"` is *not* detected as a refusal. Is that the
behaviour you want? Why might a strict match be the right call here?

---

### 4. The judge: "correct" and "grounded", graded separately

**What it is.** An optional second model call
([src/evaluation/judge.py](../src/evaluation/judge.py)) that grades each
answer on two independent criteria. Run with `--judge`.

**Why it matters.**
- **Correct**: does the answer meet the human-written rubric? This catches
  answers that contain the keyword but miss a required distinction.
- **Grounded** (also called *faithfulness*): is every claim supported by the
  retrieved context? This catches hallucination, including true facts the
  model knew from training but wasn't given.

The two disagree in useful ways. *Correct but ungrounded*: the model answered
from memory. *Grounded but incorrect*: it faithfully summarised the wrong
chunks, which points to retrieval.

**How it works.**
- **The judge sees the context.** It gets the same chunks the generator saw.
  Groundedness can't be judged without them.
- **A different, larger model.** The judge defaults to `claude-opus-5-5`; the
  generator is `claude-sonnet-4-6`. On OpenAI it's `gpt-5` judging
  `gpt-5-mini`. Models tend to rate their own style of answer favourably
  (self-preference bias). Override with `ANTHROPIC_JUDGE_MODEL` /
  `OPENAI_JUDGE_MODEL` in `.env` (see feature 12).
- **A forced tool call.** The judge must answer by calling a `record_grade`
  tool whose schema allows only `PASS` or `FAIL`. The result arrives as
  structured data, so no JSON is pulled out of free text. The old version
  returned FAIL whenever parsing failed.
- **ERROR is not FAIL.** If the judge returns something invalid, the case is
  marked `ERROR` and counted separately. A broken judge then shows up as
  errors instead of quietly lowering the pass rate.

**Say it in 30 seconds.** "I use an LLM judge for what keywords can't check,
with two separate verdicts: correct against a human rubric, and grounded in
the retrieved context. The judge is a different model from the generator, it
sees the same context, and it answers through a forced tool call, so the
output is structured. Judge failures are counted as errors, not failures."

**Questions you might get.**
- *How do you know the judge is right?* Not yet measured, and saying so is
  the honest answer. The next step is to hand-label ~30 answers and report
  how often the judge agrees with a human. Until then, treat judge numbers as
  indicative.
- *Why not one score?* One score hides *why* an answer failed. Two verdicts
  point at retrieval vs generation.

**Try it.** Pick a case, write a deliberately wrong answer, and call
`build_judge_message()` in a Python shell to read exactly what the judge would
see. What would you add to the rubric to make the grade less subjective?

---

### 5. Run records, results.md and comparing runs

**What it is.** Every run saves a JSON record to `evals/runs/` (gitignored).
A full, unfiltered run also rewrites [evals/results.md](../evals/results.md).
`--compare latest` (or a file path) prints what changed since an earlier run.
Built in [src/evaluation/report.py](../src/evaluation/report.py).

**Why it matters.** A pass rate is meaningless without its context: which
model, what k, which version of the questions, which commit. For a portfolio
"proof" number, that context is what makes it believable and repeatable.

**How it works.**
- **Config.** The record stores the models, k, chunk settings, the git commit
  (marked if there were uncommitted changes) and a hash of
  `questions.json`, so two runs on different question sets aren't confused.
- **Per case.** Everything is kept: retrieved chunks with distances, every
  score, the answer, retrieve/generate latency and tokens.
- **Summary.** Latency is reported as p50 and p95. The median shows the
  typical case; p95 shows the slow tail users actually notice.
- **Compare.** Prints metric deltas and the cases that flipped
  (`pip_duration: PASS → FAIL`, `retrieval hit → miss`).
- **Exit code.** Non-zero on any failure or error, so CI can block a
  regression later.

**Say it in 30 seconds.** "Every eval run is a saved record with its config,
commit and dataset hash, so any number I quote can be traced to exactly what
produced it. To test a change I run it with a label and compare against the
previous run: it lists which metrics moved and which cases flipped."

**Questions you might get.**
- *Why gitignore the runs?* They're noisy and machine-specific. The committed
  `results.md` is the reviewed headline; the JSON is the evidence behind it.
- *Why p95 instead of the mean?* One slow call skews a mean; percentiles
  describe what users experience.

**Try it.** Run the eval twice without changing anything and compare. Do any
cases flip? That's run-to-run noise. Remember it when judging a "+1 case"
improvement.

---

### 6. Testable without API keys

**What it is.** API keys are now read when a client is first used
([src/config.py](../src/config.py) `require_env`), not at import. The runner
receives the pipeline stages as functions
([src/evaluation/runner.py](../src/evaluation/runner.py)).

**Why it matters.** Previously, importing any module crashed without both
keys, so nothing could be tested offline. Now 32 unit tests run in under a
second with no keys and no cost: `uv run python -m unittest`.

**How it works.**
- Clients are created inside small `@cache`d functions (`_openai()`,
  `_client()`), once, on first call.
- `run_eval(cases, retrieve_fn, generate_fn, judge_fn, k)` accepts any
  functions with the right shape. Tests pass fakes; `--retrieval-only` passes
  `None` for generation. This is dependency injection with plain functions,
  no framework.
- One failing API call doesn't end the run: the error is logged, stored on
  that case, and counted. The CLI checks keys before starting, so a missing
  key fails once instead of once per case.

**Say it in 30 seconds.** "The pipeline stages are passed into the eval runner
as functions, so the scoring and reporting logic is unit-tested with fakes,
offline and free. Real API calls only happen in real runs."

**Try it.** Write a fake `generate_fn` that always refuses, run it over the
real question set in a Python shell, and look at `summarize()`. Which metric
exposes it?

---

### Known limits of the eval foundation

- The corpus is too small for retrieval metrics to discriminate (L017).
- The judge isn't yet calibrated against human labels.
- 17 cases: one case ≈ 6 percentage points, so small differences are noise.
- No real run has been made yet. The first `--judge` run sets the baseline.

---

## Part 2 — UI: showing the evidence (2026-11-06)

The theme of this part: **a RAG answer is only as trustworthy as your ability
to check it.** Every UI change makes the evidence behind an answer, or behind
an eval number, one click away.

### 7. "How this answer was found": the answer inspector

**What it is.** A collapsible panel under every answer in the chat
([static/index.html](../static/index.html), `buildInspector`). Closed, it
summarises: "5 passages · 3 cited · 2.55 s". Open, it shows retrieval and
generation time, tokens in and out, the model, and every retrieved chunk in
rank order with its distance, whether it was cited, its full text, and a
link to it in the handbook.

**Why it matters.** It answers the user's real question: *why should I
believe this?* It's also a debugging tool: when an answer is wrong you can
see at once whether the right passage was retrieved (a retrieval problem) or
was there but misread (a generation problem). That is the same split the
eval makes (§1–4), applied to a single live answer.

**How it works.**
- `/ask` ([src/server.py](../src/server.py)) now returns `retrieved` (source,
  title, chunk index, distance, text, cited), `timing`, `usage` and
  `refused`, alongside the original `answer` and `citations`.
- Timing is measured around each stage with `time.perf_counter()`, a
  monotonic clock suited to measuring durations.
- "Cited" means the answer names that document's filename, the same rule as
  before (it's document-level, not sentence-level; see limits below).
- A refusal gets a "Not in the handbook" label, using the same
  `is_refusal()` check as the eval.

**A detail worth knowing.** The collection uses Chroma's default distance,
squared L2, not cosine as the README once said. OpenAI embeddings are
normalised to length 1, so the ranking is identical to cosine (for unit
vectors, L2² = 2 − 2·cos). Raw values are shown, labelled "lower is closer",
and no "similarity %" is invented from them.

**Say it in 30 seconds.** "Every answer can be opened to show its evidence:
the chunks that were retrieved, ranked by distance, which ones the answer
cited, the full text that was sent to the model, and how long retrieval and
generation took. A user can verify an answer, and I can tell straight away
whether a bad answer was a retrieval miss or a generation mistake."

**Questions you might get.**
- *Isn't showing all this too much for end users?* It's collapsed by
  default. The closed summary is one line; the detail is for people who want
  to check.
- *Why return the chunk text in the response?* It's at most k × ~500 tokens,
  and it's exactly what the model saw. Fetching it again later could show
  different text if the index was rebuilt in between.

**Try it.** Ask "How much notice do I need to give for leave?" and open the
inspector. Are both the PTO and parental leave policies among the passages?
Did the answer cite both?

---

### 8. Citations that open the exact passage

**What it is.** Filenames the model writes, like `(pto_policy.md)`, become
links titled with the document name ("PTO and Vacation Policy"). They open
the handbook page with the retrieved passage highlighted and scrolled into
view, with a banner explaining it and a "Clear highlight" button. The
handbook page also gained a list of all documents.

**Why it matters.** "Source: pto_policy.md" makes the user hunt through a
whole document. Jumping to the passage makes checking a claim take seconds.

**How it works.**
- In the chat, `linkCitations()` rewrites each filename the model cited into
  a markdown link to `/docs/<file>?chunk=<n>`, using the highest-ranked chunk
  retrieved from that file.
- The handbook page reads `?chunk=n` and fetches that chunk's stored text from
  `GET /api/passages/<file>/<n>` (served by `get_chunk()` in
  [src/retrieve.py](../src/retrieve.py)).
- **Matching** ([static/doc.html](../static/doc.html), `highlightChunk`):
  chunks are whole markdown paragraphs joined together, so rendering the
  chunk's markdown gives the same blocks (paragraphs, headings, list items,
  table rows) as the page. Blocks with identical normalised text are
  highlighted. There's no fuzzy matching and no character offsets, and it
  can't drift as long as chunking keeps splitting on paragraphs. If nothing
  matches, the page says so instead of failing silently.

**Say it in 30 seconds.** "Citations link to the exact passage the model was
given, highlighted in the source document. Because chunks are built from
whole paragraphs, I can match them back to the rendered page block by block
without storing offsets."

**Questions you might get.**
- *What if chunking changes to fixed token windows that cut mid-paragraph?*
  Block matching would break. You'd store character offsets per chunk at
  ingest and highlight by range. This design relies on paragraph chunking,
  and that's a deliberate coupling to know about.
- *Is this sentence-level citation?* No. It's passage-level, and "cited" is
  inferred from filenames in the text. Claude's citations feature can return
  the exact quoted span per claim; that's the natural next step.

**Try it.** Open `/docs/termination_offboarding.md?chunk=1`. Which section is
highlighted, and why does it begin with a heading?

---

### 9. The evals page

**What it is.** `/evals` ([static/evals.html](../static/evals.html)) turns the
saved run records (§5) into a page:
- pick a run, and optionally a run to compare with (by default the newest is
  compared with the one before it)
- metric tiles: each shows its value, the counts behind it, a change from
  the compared run marked better or worse, and a one-line explanation
- a per-category table
- a case table you can filter to "Problems only". Each row shows pass/fail
  chips for answer checks, retrieval and judge, and "Fixed / Broke since
  compared run". A row expands to show the question, the answer, what went
  wrong in plain words, which documents were needed, and what was retrieved,
  with the needed ones marked.

**Why it matters.** A results file proves a number; this page lets someone
*inspect* it, the way the inspector does for one answer. For a portfolio, it
turns "trust me, 16/17" into evidence a reviewer can click through.

**How it works.**
- `GET /api/evals/runs` lists runs, newest first, with headline metrics
  (`list_runs()` in [src/evaluation/report.py](../src/evaluation/report.py)).
  An unreadable file is logged and skipped instead of breaking the list.
- `GET /api/evals/runs/<id>` returns one full record. The id is looked up
  among the files that exist, so a crafted id can't read other paths.
- Deltas know which direction is good: for "false refusals", lower is better,
  so a drop shows as better.
- The page warns when the compared run used a different question set (the
  dataset hash differs), when the run was a subset, or when it ran on
  uncommitted code.
- Pass and fail are never shown by colour alone: each chip has an icon and a
  word (✓ Pass, ✕ Fail, ⚠ Judge error), which keeps it readable for
  colour-blind users.

**Say it in 30 seconds.** "Eval runs are browsable in the app. Any two runs
can be compared: the tiles show which metrics improved or regressed, and
each case shows whether it was fixed or broken by the change, with the
answer, the retrieved documents and the judge's reasons one click away."

**Try it.** After two real runs, compare them. Find one case that flipped and
explain from its detail whether retrieval or generation changed.

---

### 10. Errors you can act on

**What it is.** When something fails, the chat now shows what failed and
offers "Try again", instead of "Server error 500". The header shows how many
chunks are indexed, and an empty index produces a notice with the command to
fix it.

**How it works.**
- [src/server.py](../src/server.py) registers exception handlers for
  `openai.APIError` and `anthropic.APIError`. They return **502 Bad Gateway**
  (our server is fine; a service behind it failed) with the provider's own
  message.
- A missing index or missing key raises `RuntimeError`, returned as **503
  Service Unavailable** (not ready yet).
- `retrieve()` now opens the collection *before* calling the embedding API,
  so a missing index fails before any paid call.
- `/health` and the server no longer touch the index directly; they go through
  `index_size()` and `get_chunk()` in `retrieve.py`, as `CLAUDE.md` requires.
  (Since Part 4 that index is a JSON file, not Chroma.)

**Say it in 30 seconds.** "Errors are mapped to status codes that mean
something: 502 when an upstream model provider fails, 503 when the index or
a key is missing. The provider's message reaches the user, so 'invalid API
key' reads as exactly that."

---

### 11. Smaller changes worth being able to explain

- **Safe document embedding.** The handbook page used to paste markdown into
  a JavaScript template string, escaping only `` ` `` and `$`. A backslash or
  `</script>` in a document could break the page or inject code. Now the
  server embeds JSON in `<script type="application/json">` with every `<`
  written as `\u003c`, so document text can't close the tag. It matters as
  soon as documents come from anyone other than you.
- **Only real handbook files are served.** `/docs/<name>` and the passages
  endpoint check the name against the files in `corpus/` instead of building
  a path from user input.
- **Dark mode and phone layout.** All colours are tokens in
  [static/app.css](../static/app.css), redefined for dark mode, so every page
  follows the system setting. Pages were checked at 375 px wide with no
  horizontal scrolling.
- **Reduced motion.** Animations switch off for users who ask their OS for
  reduced motion.

### Known limits of the UI work

- It was verified in the browser with a stubbed `/ask` response and fake eval
  runs, because there are no API keys on this machine yet. The real
  end-to-end path still needs one look once keys are added.
- "Cited" is document-level and inferred from filenames in the answer.


---

## Part 3 — Choosing the model provider (2026-11-06)

### 12. Anthropic or OpenAI, from `.env` or the UI

**What it is.** Answers and eval grading can run on Anthropic (Claude) or
OpenAI (GPT):
- `LLM_PROVIDER` and `JUDGE_PROVIDER` in `.env` set the default.
- The chat page's **Model settings** switches it per browser tab and accepts
  the user's own keys.

Search always uses OpenAI embeddings. The index was built from OpenAI vectors,
and Anthropic has no embeddings API, so the OpenAI key is needed in both modes.

**Why it matters.** The project shouldn't stop working because one provider
account has no credit. It also makes a portfolio point: the RAG pipeline and
the evals don't depend on the vendor. You can run the same eval with
`LLM_PROVIDER=openai` and compare runs, because each run record stores the
provider and model.

**How it works.**
- **One adapter module.** [src/llm.py](../src/llm.py) is the only code that
  knows the two APIs. `complete()` sends a system prompt and a user message
  and returns text plus token counts. `complete_with_tool()` forces a tool
  call. [generate.py](../src/generate.py) and
  [judge.py](../src/evaluation/judge.py) only name a provider and a model.
- **The same request, two shapes.**

  | | Anthropic | OpenAI |
  |---|---|---|
  | System prompt | `system=` parameter | first message, role `system` |
  | Output limit | `max_tokens` | `max_completion_tokens` (includes hidden reasoning tokens, so it gets headroom) |
  | Forced tool | `tool_choice={"type":"tool","name":...}` | `tool_choice={"type":"function","function":{"name":...}}` |
  | Tool arguments | already an object | a JSON string, decoded; bad JSON → judge ERROR |
  | Tokens | `input_tokens` / `output_tokens` | `prompt_tokens` / `completion_tokens` |

- **Empty output is an error, not an answer.** If a provider returns no text
  (cut off, or a refusal field), `ProviderOutputError` reaches the UI as a 502
  with the reason. A blank answer would otherwise be scored as a failed case.
- **Keys from the browser.** The UI keeps keys in `sessionStorage`, so they
  survive a reload but not closing the tab. It sends them as `X-OpenAI-Key` /
  `X-Anthropic-Key` headers on `/ask`, never in the URL or body. The server:
  - builds a client for that request only, so a user's key never sits in a
    cached module-level client;
  - falls back to the `.env` key when no header is sent;
  - if neither exists, returns a 503 that says which key is missing and
    where to add it.

  `/api/config` tells the UI only *whether* the server has each key, never
  the key itself.
- **Provider errors are masked, not hidden.** OpenAI's "wrong key" message
  quotes part of the key. The 502 handlers replace anything shaped like
  `sk-…` and pass the rest of the message through, so the cause stays visible.

**Say it in 30 seconds.** "All the provider-specific code is in one adapter.
Generation and the judge ask for 'a completion' or 'a forced tool call', and
the adapter maps that to Anthropic's or OpenAI's API. That covers the system
prompt, the token limit, the tool-choice syntax and the token counts. A
setting or a toggle in the UI picks the provider, and the eval records which
one ran, so you can compare Claude and GPT on the same questions."

**Questions you might get.**
- *Why not use LangChain for this?* Two providers and two call shapes is about
  150 lines. Writing the adapter yourself keeps the differences visible, such
  as reasoning tokens counting against the limit, and adds no dependency.
- *Is it safe to send API keys from the browser?* Over HTTPS, to your own
  server, used per request and never stored or logged, it's the usual
  "bring your own key" pattern. The remaining risk is the browser: any script
  on the page can read `sessionStorage`. That's why the page loads no
  third-party scripts except two vendored libraries, and keys go when the
  tab closes.
- *Why does Anthropic mode still need an OpenAI key?* The question has to be
  embedded with the same model as the index. Switching embeddings would mean
  a second index and new retrieval numbers.

**Try it.** Run `uv run python eval_cli.py --judge --label openai` with
`LLM_PROVIDER=openai`, then later run the same command on Anthropic with
`--compare latest`. Which metrics move? Retrieval shouldn't, because both
runs use the same embeddings. Then explain why.

**Known limits.**
- The OpenAI default model names (`gpt-5-mini`, `gpt-5`) weren't checked
  against a live account here. Override them in `.env` if your account lists
  others.
- The eval CLI uses `.env` settings only; the UI toggle affects the chat.
- The refusal check looks for one exact sentence. GPT may paraphrase it more
  often than Claude; the out-of-scope cases will show that.

---

## Part 4 — Deploying without Chroma at query time (2026-11-10)

### 13. A JSON index searched in plain Python

**What it is.** Ingest still builds the Chroma collection, then writes every
chunk and its embedding to `index/hr_handbook.json`. The server searches that
file and never imports Chroma.

**Why it matters.** Serverless hosts such as Vercel have a read-only disk, a
bundle size limit and cold starts. Chroma at query time meant ~308 MB of
dependencies, a SQLite file opened read-write, and a slow import. With 14
chunks, an exact search is 14 distance calculations.

**How it works.**
- `export_index()` in [src/ingest.py](../src/ingest.py) reads the collection
  with its embeddings and writes them as JSON, sorted by source and chunk.
  `ingest_cli.py --export-only` reruns just this step, with no API calls.
- `load_index()` in [src/retrieve.py](../src/retrieve.py) parses the file once
  and re-reads it only when its modification time changes. A missing file is
  the same `RuntimeError` (so the same 503) as a missing collection was.
- `search()` sorts the chunks by `squared_l2()`, Chroma's default measure, so
  distances match earlier eval runs. This was checked against Chroma's own
  query results before Chroma was removed from the server.
- `chromadb` moved to an opt-in `ingest` dependency group, so Vercel doesn't
  install it (`uv sync --group ingest` does, locally), and
  [tests/test_retrieve.py](../tests/test_retrieve.py) checks that importing
  the server doesn't load it.

**Say it in 30 seconds.** "A vector database earns its place with an
approximate index over many vectors. This corpus has 14 chunks, so the server
does an exact search over a committed JSON file. Chroma is still used to build
the index, and the distances match what it reported."

**Questions you might get.**
- *When would you bring a vector database back?* When an exact scan gets too
  slow, which happens at many thousands of chunks in plain Python, or when the
  index has to change without a redeploy.
- *Why squared L2, not cosine?* For unit-length vectors they rank identically,
  and keeping Chroma's measure keeps old and new eval runs comparable.

**Try it.** Run `uv run --group ingest python ingest_cli.py --export-only`, then open the JSON
file. How many numbers does each embedding have, and why does the file take
about 440 KB?

