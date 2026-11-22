# Sentiment Classification + Aspect Extraction (SemEval-2014 Restaurants)

A system that uses a **local LLM** (Qwen3 1.7B via Ollama) to generate **structured JSON** from English restaurant reviews. For each review, the model performs two tasks:

- **Classification:** Determine whether the sentiment toward the evaluated target is `positive` or `negative`.
- **Information extraction:** Identify the evaluated target (the *aspect*) by copying its exact span from the review.

```text
"The pizza was delicious."  →  {"sentiment": "positive", "aspect": "pizza"}
```

The final system also includes a short reasoning field in the JSON output (see [Final system](#final-system)).

## Results (120 test samples)

| System | Setting | Raw JSON (`json.loads`) | After cleaning | Sentiment accuracy | Aspect exact match | Joint accuracy | Time / sample |
|---|---|---|---|---|---|---|---|
| v2 (initial) | Prompt-only | 100% | 100% | 94.2% | 90.0% | 84.2% | 0.72 s |
| **v4 Reason (final)** | JSON schema | **100%** | **100%** | **99.2%** | **93.3%** | **92.5%** | 1.40 s |
| v4 Reason | Prompt-only | 100% | 100% | 99.2% | 93.3% | 92.5% | 1.63 s |
| v5 CoT (reasoning before JSON) | Prompt-only | **0%** | 100% | 99.2% | 92.5% | 91.7% | 1.89 s |

*Joint accuracy* is the percentage of samples for which both the sentiment and aspect predictions are correct. Detailed results and plots for all experiments are provided in [analysis.ipynb](analysis.ipynb).

## Prompt-only vs. schema-constrained decoding

The assignment requires two settings:

- **Prompt-only:** The prompt requests JSON output, but the Ollama `format` parameter is not used.
- **Schema-constrained:** A JSON schema is supplied through the Ollama `format` parameter.

| System | Setting | Raw JSON | After cleaning | Schema validity | Sentiment | Aspect exact |
|---|---|---|---|---|---|---|
| v2 (detailed prompt) | Prompt-only | 100% | 100% | 100% | 0.942 | 0.900 |
| v2 | Schema | 100% | 100% | 100% | 0.942 | 0.900 |
| v4 Reason | Prompt-only | 100% | 100% | 100% | 0.992 | 0.933 |
| v4 Reason | Schema | 100% | 100% | 100% | 0.992 | 0.933 |
| weak (three-line prompt) | Prompt-only | **35%** | 100% | 100% | 0.992 | 0.525 |
| weak | Schema | 100% | 100% | 100% | 0.992 | 0.525 |

- **Detailed prompts (`v2`, `v4_reason`):** All 120 raw outputs were identical between the two settings. The model already produced clean JSON, so the schema added structural constraints without changing the predictions.
- **Weak prompt (`weak`):** The model frequently wrapped its responses in Markdown JSON code fences. Of the 120 outputs, 78 (65%) were recovered during cleaning, while 42 were directly parseable. Schema-constrained decoding eliminated this formatting issue.
- **Task accuracy:** Using a schema did not change task accuracy in these comparisons, including with the weak prompt. The constraint excludes invalid output tokens, but in these experiments it did not change the model's chosen content.

## Dataset and gold labels

**Source:** [tomaarsen/setfit-absa-semeval-restaurants](https://huggingface.co/datasets/tomaarsen/setfit-absa-semeval-restaurants) (SemEval-2014 Task 4; Pontiki et al., 2014). The dataset's test split does not include labels, so all samples were selected from its **train** split.

The **gold labels** (aspect spans and sentiment labels) come from the original SemEval expert annotations; the labels were not manually created. The manual work involved **selecting samples**, not annotating them.

Selection procedure ([prepare_data.py](prepare_data.py)):

1. Group rows belonging to the same sentence; sentences with multiple aspects appear in multiple rows.
2. Retain only sentences with **one aspect** and a **positive or negative** label (599 positive and 214 negative candidates).
3. Review candidates manually. Exclude 170 sentences: implicit sentiment (61), multiple targets (45), mixed sentiment (27), label mismatch (14), incomplete text (9), sarcasm (8), and invalid or meaningless aspects (6). Each exclusion and its reason are recorded in [data/excluded_samples.json](data/excluded_samples.json).
4. Shuffle the pool with a fixed random seed (42) and construct two separate datasets:

| Split | File | Samples | Purpose |
|---|---|---|---|
| Test | `data/absa_120.json` | 60 positive + 60 negative | Experiments and final reported results |
| Development | `data/absa_dev_80.json` | 40 positive + 40 negative | Selecting improvement methods; no overlapping sentences with the test set |

### Matching rules

Aspect comparison in [evaluate.py](evaluate.py) converts strings to lowercase with `str.lower()` and collapses repeated whitespace into a single space.

- **Exact match (primary metric):** The normalized predicted and gold aspect spans are identical.
- **Partial match:** One normalized span contains the other (e.g., `pizza` vs. `mushroom pizza`).

Each sentence has one gold aspect, and the model predicts one aspect. Consequently, exact entity F1 equals the aspect exact-match rate in this setting. Since the dataset is in English, Turkish-specific case handling for `İ` and `I` is not required.

## Project structure

```text
.
├── prepare_data.py          Download and filter data; create test and development sets
├── model.py                 Ollama API calls, JSON schemas, thinking mode, and parameters
├── parser.py                JSON parsing: direct → Markdown/<think> cleaning → extract { ... }
├── run.py                   Run a model on all samples and save predictions
├── run_pipeline.py          Two-stage approach (development-set experiment only)
├── evaluate.py              Metrics and error analysis
├── baseline_tfidf.py        Classical BoW / TF-IDF + logistic regression (sentiment only)
├── learning_curve_tfidf.py  Classical baseline learning curve (10–272 labeled examples)
├── exp_prompt_length.py     Few-shot prompt length and latency experiment
├── plot_prompt_length.py    Plot the prompt-length experiment
├── exp_tokenization.py      Hugging Face tokenizer and Ollama token-count experiment
├── demo.py                  Live test demonstration
├── parser_ablation.py       Parser extraction ablation experiment
├── analysis.ipynb           Main experimental plots and discussion
├── data/                    Test, development, live samples, and excluded samples
├── modelfiles/              Ollama Modelfiles (prompts and generation parameters)
└── results/
    ├── test/                Test-set results (120 samples)
    ├── dev/                 Development-set results (80 samples)
    └── *.txt, *.json, *.png Baselines, learning curve, latency, and tokenizer results
```

## Setup

**Requirements:** Python 3.12 and [Ollama](https://ollama.com) (tested with Ollama 0.40.16 on Apple M4). No cloud APIs are used; all inference runs locally.

```bash
# Python packages (the project used a .venv in the repository root)
pip install requests datasets pandas numpy matplotlib ipykernel scikit-learn transformers

# Base models
ollama pull qwen3:1.7b-q4_K_M
ollama pull qwen3:1.7b-q8_0     # Only for the Q8 experiment
ollama pull qwen3:4b            # Only for the model-size experiment
ollama pull gemma3:4b           # Only for the model-size experiment

# Create the final system's Ollama model (run from the project root)
ollama create absa-model-v4-reason -f modelfiles/Modelfile_v4_reason
```

Other experimental models are created in the same way:

| Ollama model | Modelfile | Description |
|---|---|---|
| `absa-model-v1` | `Modelfile_v1` | Detailed prompt: sentiment before aspect |
| `absa-model-v2` | `Modelfile_v2` | Detailed prompt: aspect before sentiment (main baseline) |
| `absa-model-v2-q8` | `Modelfile_v2_q8` | v2 with `qwen3:1.7b-q8_0` |
| `absa-model-weak` | `Modelfile_weak` | Short three-line prompt (parser stress test) |
| `absa-model-v3` | `Modelfile_v3` | Short prompt plus aspect extraction rules |
| `absa-model-v4-fewshot` | `Modelfile_v4_fewshot` | v2 plus five examples |
| `absa-model-v4-reason` | `Modelfile_v4_reason` | v2 plus a reasoning field inside JSON (**final**) |
| `absa-model-v5-cot` | `Modelfile_v5_cot` | v2 plus free-text reasoning before JSON (without schema) |
| `absa-stage-aspect` | `Modelfile_stage_aspect` | Stage 1 of the two-stage approach |
| `absa-stage-sentiment` | `Modelfile_stage_sentiment` | Stage 2 of the two-stage approach |
| `absa-model-v2-qwen3-4b`, `absa-model-v4-reason-qwen3-4b` | `Modelfile_v2_qwen3-4b`, `Modelfile_v4_reason_qwen3-4b` | v2 and v4 Reason with `qwen3:4b` |
| `absa-model-v2-gemma3-4b`, `absa-model-v4-reason-gemma3-4b` | `Modelfile_v2_gemma3-4b`, `Modelfile_v4_reason_gemma3-4b` | v2 and v4 Reason with `gemma3:4b` |

Modelfiles configure an existing model with a system prompt and generation parameters. **No model was trained or fine-tuned.**

## Reproducibility

| Parameter | Value |
|---|---|
| Model | `qwen3:1.7b-q4_K_M` (Q4_K_M quantization, verifiable with `ollama show`) |
| Ollama version | 0.40.16 |
| Hardware | Apple M4 |
| Temperature / seed | 0 / 42 |
| Context length (`num_ctx`) | 4096 (8192 for the latency experiment) |
| `num_predict` | 128 (most systems), 256 (v4 Reason), 384 (v5 CoT), 2048 (thinking mode) |

With temperature 0, two identical runs produced exactly the same raw outputs for all 120 samples. The same was observed with `--seed 7`. This is an observation for the tested machine and Ollama version, **not a universal guarantee**.

## Running the experiments

Run all commands from the repository root with Ollama running.

```bash
# 1. Generate datasets (optional; prepared files are included in data/)
python prepare_data.py

# 2. Run the final system on the test set: schema-constrained and prompt-only
python run.py absa-model-v4-reason results/test/results_v4_reason.json --schema --reason
python run.py absa-model-v4-reason results/test/results_v4_reason_noschema.json --reason

# 3. Evaluate (multiple result files can be compared side by side)
python evaluate.py results/test/results_v2_schema.json results/test/results_v4_reason.json

# 4. Reasoning before JSON (no schema)
python run.py absa-model-v5-cot results/test/results_v5_cot.json

# 5. Classical baseline and learning curve
python baseline_tfidf.py
python learning_curve_tfidf.py

# 6. Few-shot prompt length / latency experiment and plot
python exp_prompt_length.py
python plot_prompt_length.py

# 7. Tokenizer experiment (downloads tokenizer from Hugging Face on first run)
python exp_tokenization.py

# 8. Open analysis.ipynb and select "Run All" to reproduce plots
#    (Section 9 calls the model, so Ollama must be running)
```

`run.py` command-line options:

| Option | Description |
|---|---|
| `--data <file>` | Dataset file (default: `data/absa_120.json`) |
| `--schema` | Send a JSON schema to Ollama (constrained decoding) |
| `--reason` | Expect a `reason` field; with `--schema`, use the reasoning schema |
| `--think` | Enable Qwen3 thinking mode (use with `--num-predict 2048`) |
| `--temperature`, `--seed`, `--num-predict` | Override the corresponding Modelfile parameters for this run |

Each run sends a warm-up request before timing, so model-loading time is excluded from the inference-duration measurements.

## Final system

**v4 Reason + Qwen3 1.7B Q4 + temperature 0 + JSON schema.** The model produces a one-sentence reason before selecting sentiment. The JSON schema specifies the expected fields in the order aspect → reason → sentiment:

```json
{
  "aspect": "meal",
  "reason": "The reviewer says they have never had a disappointing meal here.",
  "sentiment": "positive"
}
```

The `reason` field is not scored. Evaluation compares only the `sentiment` and `aspect` fields against the gold labels.

**Selection process:** Four improvement methods (Q8, few-shot prompting, a reasoning field, and a two-stage call) were compared on the development set. The reasoning-field and two-stage methods tied on joint accuracy. The reasoning-field approach was selected because it needs only one model call and produces an interpretable explanation. The selected method was run on the test set **only once** in its schema-constrained setting. The prompt-only comparison was added later and produced the same outputs.

## Experiments performed

| Experiment | Research question | Main finding |
|---|---|---|
| v1 vs. v2 | Does task order matter? | Difference of one sample; no meaningful effect |
| Weak prompt | Does the parser help? | Direct JSON success dropped to 35% (78 responses in Markdown JSON fences); the parser recovered 100% |
| JSON schema | Can output format be constrained? | 100% valid JSON, without changing content or accuracy |
| Prompt-only vs. schema, v4 Reason | Does schema affect accuracy? | All 120 outputs were identical; only runtime differed (1.63 s vs. 1.40 s) |
| weak / v3 | What does prompt detail affect? | Detailed aspect rules improve extraction but introduce a negative sentiment bias |
| Q4 vs. Q8 | Effect of quantization? | Four net additional correct sentiment predictions on the test set (McNemar p = 0.22); no improvement on the development set. Q8 used 2.1 GB instead of 1.5 GB (+40%) and was 43% slower |
| Temperature 0–1.0 | Effect of sampling randomness? | Differences within seed variability (sentiment accuracy 0.931–0.942); JSON success remained 100% |
| Temperature 0 reproducibility | Does an identical prompt give the same result twice? | Raw outputs were identical on 120/120 samples in a repeated run and with a different seed |
| Thinking mode | Does extra reasoning help? | Sentiment 0.958, aspect 0.842, three empty outputs out of 120, and 19× slower |
| Reasoning field and v5 CoT | Does reasoning improve accuracy? | Sentiment 0.942 → 0.992; joint 0.842 → 0.925 (reasoning inside JSON) or 0.917 (reasoning before JSON). v5 CoT raw JSON success was 0%, but extraction recovered 100%. Sentiment improvement was borderline significant (McNemar p = 0.07) |
| Model size | Does a larger model help? | Qwen3 4B joint 0.925 (v2) and 0.950 (Reason), with 1.7 s and 3.0 s latency. Gemma3 4B joint 0.858, with 4.2 s and 5.7 s latency. Reasoning helped at 1.7B; gains at 4B were small or absent |
| Classical baseline | LLM vs. TF-IDF + logistic regression? | TF-IDF + LR achieved roughly 67–72% with 80 examples and 56% with 10; LLM achieved 94–99% without task-specific training. LR prediction time was 0.008 ms/sample |
| Learning curve | How much labeled data would a classical model need? | At 272 examples, accuracy was 64% and did not approach the LLM. Additional training data had a different distribution (previously excluded sentences), so this does not isolate the effect of data volume |
| Few-shot examples and latency | How does prompt length affect latency? | k = 0–40 (428–1703 tokens): cold prefill increased 1.4 → 5.9 s (~3.5 ms/token, nearly linear); decode increased 35 → 39 ms/token. Warm, cached requests remained around 0.7 s regardless of k. Plot: `results/prompt_length_experiment.png` |
| Tokenizer | Why does Turkish require more tokens? | `kitaplarımızdan` → `kit \| ap \| larımız \| dan` (four tokens); `books` is one token. Token count ratio across three Turkish/English pairs was 1.34. Hugging Face counts matched Ollama `prompt_eval_count` |
| Development-set methods | Which improvement performed best? | Reasoning field (98.8% sentiment accuracy) |

For error analysis (confusion matrices, negation, review length, aspect error types, hallucination, and sarcasm), see [analysis.ipynb](analysis.ipynb).

## Limitations

- **Curated dataset:** Only clear, single-aspect sentences were selected. Mixed sentiment and most sarcastic sentences were excluded, making this task easier than the original SemEval benchmark. One person performed the manual exclusions.
- **Small sample sizes:** Some analyses rely on few observations (five negation examples, eight sarcasm examples, 30 latency-test inputs with only one cold measurement per k, and three tokenizer sentence pairs). These results are descriptive; many sentiment differences are borderline or statistically insignificant.
- **Exploration on the test set:** weak, v3, Q8, temperature, thinking mode, v5 CoT, and model-size experiments were conducted on the test set. The final method was selected on development data, but the idea for it came from test-set error analysis.
- **Single model family in the main system:** The main experiments used Qwen3 1.7B; model-size experiments additionally tested Qwen3 4B and Gemma3 4B.
- **Classical baseline:** The baseline only predicts sentiment; it does not extract aspects. Additional learning-curve data had a different distribution from the test set.
- **Latency interpretation:** Attention cost and KV cache explanations are based on calculations from the model's dimensions, not on profiling. The O(n²) attention component could not be separated from measurement noise at these prompt lengths. An FP16 KV cache was assumed, not directly verified.
- **Empty outputs in thinking mode:** Exhaustion of the `num_predict` budget is a consistent explanation, but `done_reason` was not recorded, so it could not be confirmed.
