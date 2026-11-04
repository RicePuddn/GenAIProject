# DeepEval scenario set

`cases.csv` contains the ten seeded tickets and setup-specific expected tool traces. The runner executes one `/chat` request at a time, preserves the ordered API trace, and appends each result to JSONL. The same ticket ID is reused across variants to exercise write idempotency; each row records replay where reported. It never seeds, resets, or deletes database records. Each row has a `focus` label; routine single-tool lookups come first, followed by the two longer scenarios.

| Case | Focus | Expected route |
|---|---|---|
| EVAL-02 | Order contents | `get_order_items` |
| EVAL-03 | Processing order status | `get_order` |
| EVAL-04 | Captured payment | `get_payments` |
| EVAL-05 | Pending return status | `get_returns_and_refunds` |
| EVAL-06 | Delivered shipment tracking | `get_tracking_events` |
| EVAL-08 | Standard return window | `policy` |
| EVAL-09 | Find processing order | `get_customer_orders` |
| EVAL-10 | Shipment carrier and tracking number | `get_shipments` |
| EVAL-01 | Approved SGD 20 refund | Multi-step lookup and refund write |
| EVAL-07 | Leaking battery safety escalation | Multi-step lookup and human review |

## Install and prepare

From the repository root:

```sh
pip install 'deepeval>=3.0' requests
```

The application dependencies must also be installed. Start the API against the already seeded development database and configure `SUPPORT_OPS_DATABASE_URL`. Configure the Ollama OpenAI-compatible endpoint as for the application; for judging, set `DEEPEVAL_JUDGE_MODEL` (default `ollama/lfm3:latest`) and the corresponding DeepEval custom model credentials/configuration. `--judge-model` is recorded in result rows for reproducibility.

## Validate selection without calling the API

```sh
python deepeval/run.py --dry-run
python deepeval/run.py --dry-run --case TKT-EVAL-01 --setup s1-a
```

## Execute

```sh
python deepeval/run.py
python deepeval/run.py --case TKT-EVAL-01 --setup s1-a --output /tmp/deepeval-smoke.jsonl
```

The default is 10 cases × 24 setup variants, sequentially. `--case` and `--setup` may each be repeated. Successful write outcomes persist in the configured evaluation database. The preflight reads the customer list through the API and verifies all ten evaluation tickets belong to `CUS-EVAL-001` with a read-only PostgreSQL query.

## Metrics

The JSONL record supplies the `LLMTestCase` inputs: final reply for G-Eval against `Ground Truth`, tool calls and setup-specific expected tools/arguments for Tool Correctness and Argument Correctness, and ordered events for trajectory scoring. Score Task Completion, Step Efficiency and the trajectory criteria from that ordered trace. Score Plan Adherence and Plan Quality only when the selected setup's resolved strategy includes `plan_execute`; omit them for all-react variants. Graph write attempts appear in the API trace as `backend_write` and are represented that way in the expected-tool mapping.

The runner deliberately separates application execution from judge evaluation so raw outputs and traces remain inspectable and failures do not silently disappear. Configure DeepEval's judge with the same Ollama endpoint/model used for the study before scoring. See [DeepEval trajectory metrics](https://deepeval.com/docs/evaluation-trajectory-based-llm-evals) and [Tool Correctness](https://deepeval.com/docs/metrics-tool-correctness).
