# Build Instructions: Federated Learning IDS Simulation (Phase 1 of larger project)

## Objective
**Current goal (this build): get a working Federated Learning simulation running**
using Flower, trained on the user's own IDS datasets, entirely on a single
Ubuntu machine — no Mininet, no Ryu, no networking layer yet. Those come in a
later phase once FL itself is working and validated.

Larger project context (for awareness only — do not build these parts now):
eventually this FL pipeline will run over an emulated Mininet network with a
Ryu SDN controller, to study FL-based intrusion detection under non-IID and
adversarial-client conditions in an SDN-IoT setting.

Environment: Ubuntu (already installed in a UTM VM on macOS). Work should be
done entirely inside this Ubuntu environment via terminal commands.

Datasets: two prepared CSVs already sit in the project root
(`~/major-project/fl/`):
- `bccc_cleaned.csv`
- `cic_cleaned.csv`

Do not assume their schema. Inspect both first (columns, dtypes, label column
name/values, class balance, row counts) before writing any preprocessing code.

Run every phase to completion, verify success criteria before moving to the next
phase, and log/print clear status at each step. If a step fails, attempt one
reasonable fix (e.g., missing dependency, permissions) before reporting the error
back rather than silently skipping it.

---

## Phase 0 — Environment Verification & Setup

1. Check Python version: `python3 --version`.
2. `sudo apt update && sudo apt upgrade -y`
3. Install core packages: `sudo apt install -y python3-pip python3-venv`
4. Create a project directory: `~/major-project/fl/` with subfolders:
   ```
   major-project/fl/        (project root — you are already here)
     data/                  # copied datasets + per-client partitions
     src/                   # server.py, client.py, models.py
     scripts/               # orchestration + run scripts
     results/               # logs, metrics, plots
   ```
   Note: the project root is already called `fl` (`~/major-project/fl/`).
   The subfolder for FL source code is named `src/`, not `fl/`, to avoid a
   confusing `fl/fl/` nested path.
5. Set up a Python virtual environment: `python3 -m venv ~/major-project/fl/venv && source ~/major-project/fl/venv/bin/activate`
6. Install dependencies: `pip install flwr flwr-datasets torch torchvision
   scikit-learn pandas numpy matplotlib`

**Success criteria for Phase 0:** `python3 -c "import flwr, torch, pandas,
sklearn"` runs without error inside the venv.

---

## Phase 1 — Dataset Inspection & Preparation

1. Move both datasets into `data/`: `mv bccc_cleaned.csv cic_cleaned.csv data/`
   (run this from the project root, `~/major-project/fl/`)
2. Write and run `data/inspect_data.py` first, on both files, printing:
   - Shape (rows, columns), column names and dtypes
   - Candidate label column (look for something like `label`, `Label`,
     `class`, `attack_cat`, `Attack`, etc.) and its unique values / value counts
   - Basic null/missing value check
   - A few sample rows
   Do not proceed to preprocessing until this has been reviewed — the exact
   label column name and class encoding must come from what's actually in
   the files, not assumed.
3. Decide the setup based on what Phase 1.2 reveals — pick whichever is more
   natural given the actual data, and note the choice in a comment at the
   top of `prepare_data.py`:
   - **Option A (likely default):** treat `bccc_cleaned.csv` and
     `cic_cleaned.csv` as two different data sources/domains, each split
     across multiple simulated clients — this gives natural non-IID
     structure (client 0-2 from bccc, client 3-5 from cic, etc.), which is
     a realistic and easy-to-justify setup for an FL-IDS paper.
   - **Option B:** if the two files share an identical schema and look like
     they're meant to be combined, concatenate them first, then partition
     non-IID across clients using a Dirichlet distribution over the label
     classes (alpha default 0.5).
4. Write `data/prepare_data.py` that:
   - Loads the dataset(s) per the decision above.
   - Encodes the label column to numeric (binary normal-vs-attack at
     minimum; keep original multiclass label as a secondary column if present).
   - Encodes any categorical feature columns, normalizes numeric features
     (StandardScaler).
   - Splits into `N_CLIENTS` (default 5, configurable via CLI arg) partitions
     per the chosen non-IID strategy.
   - Saves each partition as `data/client_{i}.csv` plus a shared
     `data/test.csv` (held-out global test set, stratified sample across all
     sources so evaluation isn't biased toward one dataset).
5. Print per-client class distribution (and per-client source dataset, if
   Option A) to confirm the non-IID skew is real and visible.

**Success criteria:** `data/client_0.csv` ... `data/client_{N-1}.csv` and
`data/test.csv` exist and are non-empty; printed class distributions visibly
differ across clients.

---

## Phase 2 — FL Model and Client/Server Code

1. `src/models.py`: Define a simple, fast-training model suitable for tabular
   IDS data — an MLP classifier (a few dense layers) is the default choice.
   Keep it small (this needs to train many rounds quickly in simulation).
2. `src/client.py`: A Flower `NumPyClient` that:
   - Loads its assigned partition (`data/client_{ID}.csv`) based on a
     `--client-id` CLI arg.
   - Implements `get_parameters`, `fit`, `evaluate`.
   - Supports an optional `--malicious` flag that, when set, poisons its
     updates before sending (start with label-flipping poisoning: flip a
     configurable fraction of training labels before local training).
3. `src/server.py`: A Flower server that:
   - Supports switchable aggregation strategy via `--strategy` CLI arg:
     `fedavg` (default/baseline) and at least one robust alternative,
     `trimmed_mean` (implement via Flower's `Strategy` API, or fall back to
     a custom aggregation function if the built-in isn't available).
   - Evaluates the global model each round on `data/test.csv` and logs
     accuracy/F1/loss to `results/metrics_{strategy}.csv`.
   - Configurable number of rounds (default 20) and clients-per-round.
4. Use Flower's **simulation mode** (`flwr.simulation.start_simulation` /
   `flwr.simulation.run_simulation`, whichever matches the installed Flower
   version) to run the server and all `N_CLIENTS` clients as one local
   process rather than launching separate terminals — this is the standard,
   simplest way to run an FL simulation on a single machine and is exactly
   what's needed for this phase (no real networking involved).
5. `scripts/run_simulation.py`: entry point that wires together data
   loading, client function, strategy selection (`fedavg` or
   `trimmed_mean`), and number of rounds, then calls the Flower simulation
   runner. Should accept CLI args for strategy, rounds, and whether to
   include malicious clients (and how many).
6. Run it twice: once with `fedavg` + a couple of malicious clients active,
   once with `trimmed_mean` + the same malicious clients. Save both metrics
   CSVs to `results/`.

**Success criteria:** Both simulation runs complete end-to-end via
`python3 scripts/run_simulation.py --strategy fedavg ...` and
`--strategy trimmed_mean ...`, producing two metrics CSVs in `results/`
with accuracy generally increasing over rounds.

---

## Phase 3 — Results & Sanity Check

1. `scripts/plot_results.py`: Load the metrics CSVs and produce a simple
   comparison plot (accuracy/F1 vs. round, `fedavg` vs. `trimmed_mean`,
   both under the same poisoning condition) saved to `results/comparison.png`.
2. Print a short text summary: final accuracy for each strategy, and whether
   the robust strategy outperformed FedAvg under poisoning (this is the
   expected/hoped-for result — if it doesn't hold, don't force it, just
   report the actual numbers).

**Final deliverable (this phase):** A working, reproducible FL simulation in
`~/major-project/fl/` that runs via `python3 scripts/run_simulation.py`, plus
`results/comparison.png` and the underlying metrics CSVs, demonstrating FL-IDS
training on the user's real datasets under both benign and adversarial-client
conditions, with non-IID data partitioning.

---

## Notes / Known Risk Areas (flag these clearly if hit, don't silently paper over)
- The two datasets' actual schemas are unknown ahead of time — Phase 1's
  inspection step must genuinely be reviewed/reasoned about, not skipped in
  favor of assumed column names.
- Flower's simulation API has changed across versions (`start_simulation` in
  older versions vs. `run_simulation` / `ClientApp`+`ServerApp` in newer
  ones) — check the installed `flwr` version and use the matching API rather
  than assuming one.
- Class imbalance is common in IDS datasets (attacks often far outnumber or
  are far outnumbered by normal traffic) — report F1/precision/recall
  alongside accuracy, since accuracy alone can be misleading here.

---

## Future Phases (not part of this build — for context only)
Once this FL simulation is working and validated, the next phases will add:
- **Mininet** as the network substrate (virtual hosts = FL clients) so FL
  traffic traverses an emulated network instead of running as local processes.
- **Ryu** as the SDN controller, first for basic L2 forwarding, later
  extended to monitor traffic and act on IDS output (e.g., quarantining a
  host flagged as malicious).
Do not build these now — they depend on the FL simulation from this phase
being solid first.
