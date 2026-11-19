# `xcad_code/scripts/`

97 scripts (31 shell drivers, 66 Python tools) that produced the numbers in the
manuscript, its figures, and the additional measurements reported in the
reviewer response. The reconstruction pipeline alone does not make the reported
results checkable, so the arm drivers, gates and diagnostics are versioned here.

## Directory layout assumed by these scripts

```
~/xcad/                       working root used on the experiment machine
  xcad_code/                  this repository's code tree
    .venv/                    virtualenv (created locally; not in git)
    scripts/                  this directory
    src/                      xCAD data pipeline (data/export_to_regcn_format.py,
                              analysis/, models/, utils/)
    xcad_model/
      src/                    model and evaluation entry points (main.py, rrgcn.py,
                              decoder.py) plus the r1_* / diagnose_* measurements
      rgcn/                   backbone utilities
      data/                   graph directories (xcad/, xcad_v2/, ...)
      models/, stats/         checkpoints and cached statistics, written at run time
    outputs/reports/          artifacts written by the scripts below
  baselines/                  RE-GCN / CEN / RETIA copies
  datasets/                   raw traces and cleaned variants (not redistributed)
  outputs/reports/            log/report root used by some of the queues
```

Paths are fixed by a few constants near the top of each script (`ROOT`, `OUT`,
`PY`, `D`, ...); edit them there, or symlink `~/xcad` to your checkout, if your
tree lives elsewhere. Whether a script reports into `xcad_code/outputs/reports/`
or the working-root `outputs/reports/` is visible in its own `OUT` constant.
Neither the reconstructed graph nor the raw traces are redistributed here — see
the top-level README.

## Naming convention

| Prefix | Role |
| --- | --- |
| `run_*`, `night_*`, `cw*`, `t5_*`, `mem_probe*`, `b7_*`, `rerun_*`, `retry_*`, `smoke_*`, `ablate_*`, `gru_*`, `diag_*`, `test_*`, `gpu_probe_*`, `rebuttal_probe.sh`, `watch_*` | GPU arm drivers and serial queues. Each arm writes its own log under the queue's report directory; arms must run serially on a single 24 GB GPU. |
| `probe_*` | Read-only diagnostics: parse logs, checkpoints or exported edges, print a verdict, optionally write one json under the report directory. They do not train and do not modify data. |
| `analyze_*`, `summarize_*`, `build_*`, `render_*`, `make_fig_*`, `gate_*` | Turn those logs into tables, figures and pass/fail gates. |
| `screen_*`, `probe_mit_*`, `probe_philly_*`, `fetch_maas_shards.py`, `measure_maas_scale.py`, `screen_second_dataset.sh` | Screening of additional public traces (Alibaba `cluster-trace-v2026-maas`, MIT Supercloud, Microsoft Philly). Their readings appear in the reviewer response, not in the manuscript. |
| `export_featdir_traincaliber.py` | Builds the train-only normalization variant directory used by the `--feat-dir` control. |
| `hybrid_profile_graph.py`, `param_dump_r26.py` | Learnable graph/profile score mixer; per-tensor parameter and checkpoint-size accounting. |

## Self-checks before a number is used

20 scripts print an explicit gate block (search for `门禁` / `GATES`) instead of
trusting an unverified reading: the expected checkpoint or file name, the
per-class test-edge counts implied by a log, or the flagship reading reproduced
bit-for-bit in test-only mode before any variant is compared against it. On a
mismatch they exit nonzero — `gate_cw_2026_11_18.py`, `gate_t5.py`,
`param_dump_r26.py` and `probe_r26_minority_paired.py` are the strictest
examples, the last one refusing to print anything unless all five GPU classes,
their sample counts and the macro average line up.

## Network access

Three scripts download public trace shards — `fetch_maas_shards.py`,
`probe_maas_zip.py`, `screen_maas.py`. Each reads an optional egress proxy from
the `XCAD_PROXY` environment variable and runs without it. The remaining
screening scripts call `curl`, which inherits `https_proxy`/`http_proxy` from
the environment if you use one.

## Comments that cite internal notes

Source comments name review points (`R1-1` … `R2-6`) because most probes exist to
test one specific objection, and some cite `docs/reviews/…` work notes that are
not part of this repository. Those references explain how a decision was
reached; every measurement quoted in the manuscript or the response is
reproducible from the scripts and the artifacts they write.
