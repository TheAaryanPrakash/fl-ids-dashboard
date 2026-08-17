# Project context for Claude Code sessions

This file is auto-loaded at the start of future sessions in this directory. It exists so a new
session doesn't have to re-derive history that's expensive to reconstruct from the code alone —
mainly *why* things are built the way they are, and what's still in flux.

## Current status: everything below is done and working

Full history (`instructions.md` = original Phase 0-3 spec; a later message extended it with
Phase A-D networking; a further message added the dashboard + GitHub Pages deployment).

- **Phase 0-3** (`instructions.md`): venv + deps, dataset inspection/partitioning, Flower FL
  simulation (FedAvg + FedTrimmedAvg), poisoning experiments, plots. All complete.
- **Phase A-D** (Mininet/SDN, given as a follow-up in chat, not in `instructions.md`): Ryu turned
  out to be completely unrunnable (see below) and was replaced with a from-scratch OpenFlow 1.3
  controller. Mininet topology, networked client/server, full non-interactive orchestration
  script, and a local-vs-networked comparison — all complete and verified.
- **Dashboard**: `visualization/pipeline_summary.html`, F1 red/black themed, self-contained
  (Chart.js + Google Fonts via CDN, everything else inline), deployed to GitHub Pages.
- **GitHub**: public repo `TheAaryanPrakash/fl-ids-dashboard`, pushed with the whole project
  (venv excluded via `.gitignore`). Pages deploys via `.github/workflows/pages.yml` on every
  push to `main` that touches `visualization/**` (copies `pipeline_summary.html` → `index.html`
  and publishes via `actions/upload-pages-artifact` + `actions/deploy-pages`). Live at
  https://theaaryanprakash.github.io/fl-ids-dashboard/

If asked to keep working on this project, **don't re-run Phase 0-D from scratch** — the data
partitions, metrics CSVs, and networked results already exist and are real (not placeholders).
Only regenerate something if you're deliberately changing its inputs.

## Environment specifics

- Ubuntu 26.04, **arm64**, Python 3.14 as the system interpreter (very new — watch for package
  compat issues; PyTorch/Flower/Ray all installed fine on it, but plenty of other packages
  won't — see the Ryu saga below).
- Main venv: `venv/` at project root (flwr 1.33.0, torch 2.13.0, ray 2.57.0, pandas, sklearn,
  matplotlib, python-openflow). This is what `data/`, `src/`, `scripts/`, and `network/*.py`
  (controller/client/server) all run under.
- `sudo` in this environment **requires interactive password auth** — cannot be done from a
  Bash tool call, not even via a `!`-prefixed user command (same non-interactive shell). Any
  apt install or genuinely-needs-root action requires asking the user to run it in their own
  terminal.
- Scoped passwordless sudo is configured in `/etc/sudoers.d/fl-mininet` for: `mn`, `ovs-vsctl`,
  and `/usr/bin/python3 /home/aaryan/major-project/fl/scripts/network/run_networked_experiment.py`
  (system python3, **not** venv python3 — `mininet` the package only exists in system
  dist-packages, no working pip-installable equivalent). Sudoers matching needs the
  fully-qualified path; bare `python3` resolved via PATH does not reliably match.
- `gh` CLI is installed and authenticated as `TheAaryanPrakash` (repo + workflow scopes).

## Key design decisions (don't relitigate these without reason)

- **bccc/cic treated as separate domains, not concatenated** — they share zero feature columns
  (only `label`), confirmed by inspection, not assumed.
- **154-col union feature space, zero-padded per domain** — lets one MLP/FedAvg parameter vector
  serve clients from either domain.
- **pos_weight in the loss** (`src/client.py`) — without it the model collapses to predicting
  the majority class (~93% accuracy trivially, 0 detection value). Computed per-client from
  local label distribution, not globally — realistic for FL (each client only knows its own
  data).
- **`torch.manual_seed()` before building initial parameters** (`scripts/run_simulation.py`,
  `network/net_server.py`) — without this, fedavg vs trimmed_mean comparisons started from
  *different* random initial weights, confounding the comparison. Round-0 accuracy should be
  bit-identical across strategy runs; if it isn't, this seeding broke.
- **`beta=0.4` for FedTrimmedAvg**, not Flower's own default 0.2 — chosen to match the actual
  known corruption ratio (2/5 = 40% malicious clients). Flower's `beta` is the trim fraction
  *per side*, so 0.4 drops 2 of 5 client values per coordinate. Default 0.2 only drops 1, which
  isn't guaranteed to exclude both malicious clients.
- **Local training minibatch order is unseeded** — this is *why* re-running the same script
  twice gives slightly different numbers each time (confirmed: two non-networked runs of the
  identical unmodified script differed by ~0.009 accuracy, same magnitude as the
  networked-vs-local diff). Don't mistake this run-to-run noise for a bug.
- **Ryu is a dead end, not a config problem.** Full diagnosis lives in `network/controller.py`'s
  docstring: broken on Python 3.14 (removed `setuptools.easy_install` API, removed stdlib
  `distutils`), broken on a from-source Python 3.9 rebuild too (`oslo.config` namespace
  breakage → `collections.Mapping` removal → `ryu==2.2` literally being Python 2 source →
  `ryu==4.34` missing `eventlet.wsgi.ALREADY_HANDLED` in every installable eventlet, including
  the exact pin the original build instructions anticipated as the fix). Replaced with a
  hand-built OF1.3 controller on `python-openflow` + stdlib sockets. **Do not attempt to
  reinstall Ryu** without a genuinely new reason to believe the ecosystem has changed.

## Known loose ends / cleanup items

- `results/networked/` and `network/__pycache__/` are **root-owned** locally (from the
  Mininet orchestration script running under sudo). Asked the user to
  `sudo chown -R aaryan:aaryan results/networked network/__pycache__` twice; as of the last
  check it hadn't been done. Harmless (world-readable, git doesn't care) but worth doing for
  local editing convenience. Don't re-ask repeatedly if it keeps not happening — it's cosmetic.
- No `results/networked/metrics_trimmed_mean.csv` exists — Phase C only ran fedavg over the
  network per the original scope. The dashboard correctly flags this as missing rather than
  fabricating it; if a networked trimmed_mean run is ever wanted,
  `network/net_server.py --strategy trimmed_mean` already supports it, just needs
  `scripts/network/run_networked_experiment.py --strategy trimmed_mean` run under sudo.
- No license file in the repo — wasn't asked for, don't add one speculatively.

## Where to look for what

- `instructions.md` — original Phase 0-3 spec (the FL simulation itself).
- `network/controller.py` module docstring — the full Ryu post-mortem.
- `scripts/network/compare_networked.py` — the local-vs-networked correctness/overhead analysis.
- `visualization/pipeline_summary.html` — everything above, visualized; also the best single
  place to re-check actual final numbers (it reads them live from the CSVs at page load, and
  the embedded JSON in the file is a ready-made snapshot of every real number in the project as
  of its last regeneration).

## Plausible next steps (not started, not promised — just what the original spec pointed at)

`instructions.md`'s own "Future Phases" section describes closing the loop: an IDS-aware SDN
controller that acts on flagged traffic (e.g. quarantining a host), once the FL+networking
integration is solid — which it now is. That would mean extending `network/controller.py` with
actual flow-based monitoring/decision logic instead of pure L2 forwarding. Nothing here commits
to that being the next task — just noting it's the direction the project was originally aimed.
