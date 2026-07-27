# Equivalente de setup.ps1 para Linux/macOS.
#
# En Windows usa `.\setup.ps1`, que es el camino soportado: el entorno de
# referencia del proyecto es Windows 11 + Python 3.12.0.

VENV    ?= .venv
PY      := $(VENV)/bin/python
DATASET ?= data/processed/lsa64_dominante.npz

.PHONY: setup venv deps modelo test demo metricas limpiar ayuda

ayuda:
	@echo "make setup     - venv + dependencias + modelo de MediaPipe + pruebas"
	@echo "make test      - pytest -q"
	@echo "make demo      - demo en vivo con la webcam 0"
	@echo "make metricas  - reproduce las tablas del informe (TARDA: entrena 20 modelos)"
	@echo "make limpiar   - borra el venv y los __pycache__"

setup: deps modelo test
	@echo ""
	@echo "ENTORNO LISTO."
	@echo "  Corre la demo:  $(PY) scripts/demo_vivo.py --fuente 0"
	@echo "  El modelo entrenado ya viene en el repo: no hace falta reentrenar."

venv:
	@test -d $(VENV) || python3 -m venv $(VENV)

deps: venv
	$(PY) -m pip install --quiet --upgrade pip
	$(PY) -m pip install -r requirements.txt

# No aborta el setup: el pipeline offline funciona sin el .task, solo la
# captura lo necesita.
modelo: venv
	-$(PY) scripts/descargar_modelo.py

test: venv
	$(PY) -m pytest -q

demo: venv
	$(PY) scripts/demo_vivo.py --fuente 0

metricas: venv
	$(PY) scripts/reproducir_metricas.py --dataset-dominante $(DATASET)

limpiar:
	rm -rf $(VENV) .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
