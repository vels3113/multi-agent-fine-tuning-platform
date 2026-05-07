ENV_FILE ?= ../project/.env
-include $(ENV_FILE)
export

.PHONY: install-comlrl smoke train help

help:
	@echo "Targets:"
	@echo "  install-comlrl  Clone CoMLRL repo (run once per fresh server)"
	@echo "  smoke           Run P1a thinking-mode smoke test"
	@echo "  train           Run full training via run.py (set CONFIG=path/to/yaml)"

install-comlrl:
	bash scripts/install-comlrl.sh

smoke: install-comlrl
	bash scripts/smoke.sh

train: install-comlrl
	bash scripts/docker-run.sh "pip install pyyaml datasets -q && python run.py --config $(CONFIG)"
