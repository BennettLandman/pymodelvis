# Contributing

Issues and pull requests are welcome at https://github.com/BennettLandman/pymodelvis.

## Development setup

```bash
git clone https://github.com/BennettLandman/pymodelvis.git
cd pymodelvis
python -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"
pytest -q
```

## Guidelines

* **Keep the model untouched.** Every hook or torch-function mode must be removed in a `finally`
  block, and inputs and parameters must never be modified. `tests/test_capture.py` checks this.
* **Reduce on device.** Anything proportional to activation size must be computed where the tensor
  lives and bounded by `max_capture_mb`.
* **Fail soft.** A visualization problem must never break the user's forward pass. Fall back and
  add a note instead.
* **Deterministic.** No random sampling in reductions or layouts.
* **Tests.** Add a CPU-only test with a tiny model (see `tests/conftest.py`) for every new feature.
* **Figures.** If you change rendering, re-run the affected examples (`examples/run_all.sh`) and
  look at the outputs. The committed gallery should stay in sync with the code.

## Documentation and website

The docs are Markdown in `docs/`; the website (https://bennettlandman.github.io/pymodelvis/) is
built from them plus `examples/outputs/` and is published automatically on every push to `main`.
Preview it with `pip install -r website/requirements.txt && python website/build.py --serve`.
See [website/README.md](website/README.md).

## Adding support for an architecture family

Subclass `neural_flow.adapters.Adapter` (`match`, `defaults`, `concepts`) and register it with
`neural_flow.adapters.register_adapter`. See `neural_flow/adapters/` for the CNN, U-Net,
transformer and 3-D adapters.

## Reporting a problem

Please include the `neural-flow inspect <model> -i <input> --all-modules` output (or
`fig.flow.summary_table()`), your PyTorch version, and the error with `NEURAL_FLOW_DEBUG=1`.
