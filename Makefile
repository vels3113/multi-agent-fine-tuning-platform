ENV_FILE ?= ../project/.env
CONFIG ?= configs/p1a-thinking-gate.yaml
-include $(ENV_FILE)
export

.PHONY: install-comlrl smoke inspect train help

help:
	@echo "Targets:"
	@echo "  install-comlrl  Clone CoMLRL repo (run once per fresh server)"
	@echo "  smoke           Run P1a thinking-mode smoke test"
	@echo "  inspect         Capture and log every completion (no weight updates)"
	@echo "  train           Run full training via run.py (set CONFIG=path/to/yaml)"

install-comlrl:
	bash scripts/install-comlrl.sh

smoke: install-comlrl
	bash scripts/smoke.sh

inspect: install-comlrl
	bash scripts/docker-run.sh "pip install pyyaml datasets -q && python scripts/inspect_completions.py --config $(CONFIG)"

train: install-comlrl
	bash scripts/docker-run.sh "pip install pyyaml datasets -q && python run.py --config $(CONFIG)"
