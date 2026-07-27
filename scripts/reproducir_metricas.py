"""
reproducir_metricas.py — regenera las tablas de métricas del informe.

Lanza las cuatro evaluaciones de validación cruzada (2 modos de manos x 2
esquemas de fold) y las evaluaciones del modelo exportado, y escribe las tablas
exactas que aparecen en el informe, cada una con el archivo de evidencia del
que sale cada número.

Las cuatro evaluaciones se lanzan **juntas y en una sola corrida** a propósito:
el entrenamiento de Keras no es bit-determinista aunque se fije la semilla
(~±0.01 de corrida a corrida), así que mezclar cifras de corridas distintas da
una tabla que no es consistente consigo misma. Ya pasó una vez.

Uso:
    python scripts/reproducir_metricas.py                # corrida completa (TARDA)
    python scripts/reproducir_metricas.py --dry-run      # solo imprime los comandos
    python scripts/reproducir_metricas.py --solo-tablas  # tablas desde los JSON en disco

ATENCIÓN: la corrida completa entrena 20 modelos (4 evaluaciones x 5 folds).
Usa --solo-tablas si solo quieres regenerar las tablas desde la evidencia que
ya está en outputs/reports/.
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

import _raiz  # noqa: F401  (raíz del repo en sys.path; debe ir antes que lsch_mr)
from lsch_mr import config
from lsch_mr.consola import configurar_utf8

configurar_utf8()

UMBRALES = (0.60, 0.90, 0.95, 0.99)


# --------------------------------------------------------------------------- #
# Ejecución
# --------------------------------------------------------------------------- #
def _comandos(ds_dominante: Path, ds_ambas: Path, campo_senante: str) -> list[tuple[str, list[str]]]:
    """Los comandos de la corrida canónica, en orden, con su etiqueta."""
    py = [sys.executable]
    # Rutas relativas a la raíz del repo: los subprocesos se lanzan con
    # cwd=config.RAIZ, así que el comando que se imprime es el mismo que el
    # usuario puede copiar y pegar desde la raíz.
    entrenar = "scripts/entrenar.py"
    evaluar = "scripts/evaluar_modelo.py"
    return [
        ("dominante / estratificado",
         py + [entrenar, "--dataset", str(ds_dominante), "--cv", "--solo-cv"]),
        ("dominante / por señante",
         py + [entrenar, "--dataset", str(ds_dominante),
               "--cv-grupos", campo_senante, "--solo-cv"]),
        ("ambas / estratificado",
         py + [entrenar, "--dataset", str(ds_ambas), "--cv", "--solo-cv"]),
        ("ambas / por señante",
         py + [entrenar, "--dataset", str(ds_ambas),
               "--cv-grupos", campo_senante, "--solo-cv"]),
        ("métrica 1 sobre las predicciones out-of-fold",
         py + [evaluar, "--fuente", "cv"]),
        ("métrica 1 sobre el ONNX exportado (20% retenido)",
         py + [evaluar, "--fuente", "modelo",
               "--dataset", str(ds_dominante)]),
    ]


def _ejecutar(comandos, dry_run: bool) -> int:
    fallos = 0
    for i, (etiqueta, cmd) in enumerate(comandos, start=1):
        print(f"\n{'=' * 75}")
        print(f"[{i}/{len(comandos)}] {etiqueta}")
        print(f"  $ {' '.join(cmd[1:])}")
        print("=" * 75)
        if dry_run:
            continue
        r = subprocess.run(cmd, cwd=config.RAIZ)
        # evaluar_modelo.py devuelve 2 cuando la métrica no alcanza su umbral.
        # Es un veredicto, no un fallo de ejecución: la corrida sigue.
        if r.returncode == 2:
            print(f"  [aviso] '{etiqueta}' terminó con código 2: "
                  "la métrica no alcanza su umbral.")
        elif r.returncode != 0:
            print(f"  [ERROR] '{etiqueta}' falló con código {r.returncode}.")
            fallos += 1
    return fallos


# --------------------------------------------------------------------------- #
# Tablas
# --------------------------------------------------------------------------- #
def _cargar(reports: Path, nombre: str) -> dict | None:
    ruta = reports / nombre
    if not ruta.exists():
        return None
    return json.loads(ruta.read_text(encoding="utf-8"))


def _tabla_accuracy(reports: Path) -> list[str]:
    filas = [
        "| Modo | k-fold estratificado | **k-fold por señante** | Evidencia |",
        "|---|---|---|---|",
    ]
    for modo, etiqueta in (("dominante", "`dominante` (63 dim)"),
                           ("ambas", "`ambas` (126 dim)")):
        celdas = []
        for sufijo in ("", "_senante"):
            d = _cargar(reports, f"cv_metrics_{modo}{sufijo}.json")
            celdas.append(f"{d['mean_accuracy']:.3f} ± {d['std_accuracy']:.3f}"
                          if d else "—")
        # Solo se destaca la cifra reportable: el modo del modelo exportado
        # evaluado por señante. Marcar las cuatro no destacaría ninguna.
        senante = (f"**{celdas[1]}**" if modo == config.MODO_MANOS else celdas[1])
        filas.append(f"| {etiqueta} | {celdas[0]} | {senante} | "
                     f"`cv_metrics_{modo}[_senante].json` |")
    return filas


def _tabla_umbral(reports: Path, modo: str = "dominante") -> tuple[list[str], dict]:
    """Cobertura y precisión por umbral, desde las predicciones out-of-fold.

    Se recalcula desde el JSON en vez de citarla: así la tabla no puede
    desincronizarse de la corrida que la respalda.
    """
    d = _cargar(reports, f"cv_metrics_{modo}_senante.json")
    if not d or "out_of_fold" not in d:
        return ["_(falta cv_metrics_%s_senante.json)_" % modo], {}

    oof = d["out_of_fold"]
    yt, yp, cf = oof["y_true"], oof["y_pred"], oof["confianzas"]
    n = len(yt)
    filas = ["| Umbral | Cobertura | Precisión | Glosas erróneas mostradas |",
             "|---|---|---|---|"]
    for u in UMBRALES:
        mostradas = [(t, p) for t, p, c in zip(yt, yp, cf) if c >= u]
        if not mostradas:
            continue
        aciertos = sum(1 for t, p in mostradas if t == p)
        err = len(mostradas) - aciertos
        marca = "**" if abs(u - config.CONF_THRESHOLD) < 1e-9 else ""
        filas.append(f"| {marca}{u:.2f}{marca} | {marca}{100 * len(mostradas) / n:.1f}%{marca} "
                     f"| {marca}{100 * aciertos / len(mostradas):.1f}%{marca} "
                     f"| {marca}{err}{marca} |")

    conf_err = sorted(c for t, p, c in zip(yt, yp, cf) if t != p)
    calib = {}
    if conf_err:
        calib = {"n_errores": len(conf_err),
                 "mediana": statistics.median(conf_err),
                 "p95": conf_err[max(0, round(0.95 * (len(conf_err) - 1)))],
                 "max": max(conf_err)}
    return filas, calib


def _tabla_protocolos(reports: Path) -> list[str]:
    filas = ["| Protocolo | Accuracy | Qué mide |", "|---|---|---|"]
    modelo = _cargar(reports, "evaluacion_modelo.json")
    if modelo:
        # La accuracy del ONNX sobre su 20% retenido vive dentro del veredicto.
        acc = (modelo.get("veredicto") or {}).get("accuracy")
        if acc is not None:
            n = modelo.get("n_muestras", "?")
            filas.append(f"| Split 80/20 aleatorio ({n} muestras) | {acc:.3f} | "
                         "Nada defendible: repeticiones del mismo señante caen a "
                         "ambos lados. |")
    for sufijo, nombre, nota in (
            ("", "5-fold estratificado", "Mismo problema, repartido entre folds."),
            ("_senante", "**5-fold por señante**",
             "**Generalización a una persona nueva. Esta es la cifra.**")):
        d = _cargar(reports, f"cv_metrics_dominante{sufijo}.json")
        if d:
            filas.append(f"| {nombre} | {d['mean_accuracy']:.3f} ± "
                         f"{d['std_accuracy']:.3f} | {nota} |")
    return filas


def _escribir_tablas(reports: Path, salida: Path) -> None:
    acc = _tabla_accuracy(reports)
    umbral, calib = _tabla_umbral(reports)
    protocolos = _tabla_protocolos(reports)

    senante = _cargar(reports, "cv_metrics_dominante_senante.json")
    n_muestras = len(senante["out_of_fold"]["y_true"]) if senante else 0
    n_grupos = senante.get("n_grupos", 0) if senante else 0

    lineas = [
        "# Tablas de métricas del informe",
        "",
        "> Generado por `reproducir_metricas.py`. **No editar a mano**: cada",
        "> número se recalcula desde los JSON de `outputs/reports/`, que son la",
        "> evidencia en disco de la corrida.",
        "",
        f"Corpus: {n_muestras} muestras, {n_grupos} señantes. "
        f"Objetivo del MVP: accuracy ≥ {config.ACCURACY_OBJETIVO:.2f}.",
        "",
        "## 1. Accuracy por modo de manos y esquema de validación",
        "",
        *acc,
        "",
        "**La columna que vale es la de por señante.** Un k-fold al azar reparte",
        "las repeticiones de un mismo señante entre entrenamiento y validación,",
        "así que el modelo puede reconocer a la persona en vez de la seña.",
        "",
        "Reporta la media **con su desviación**, nunca un valor puntual: el",
        "entrenamiento de Keras no es bit-determinista aunque se fije la semilla.",
        "",
        "## 2. Los tres protocolos, sobre el modo `dominante`",
        "",
        *protocolos,
        "",
        f"## 3. Efecto del umbral de confianza (CONF_THRESHOLD = {config.CONF_THRESHOLD:.2f})",
        "",
        *umbral,
        "",
    ]
    if calib:
        lineas += [
            f"Calibración: de las {calib['n_errores']} predicciones erróneas, la",
            f"confianza mediana es {calib['mediana']:.2f} y la p95 llega a "
            f"{calib['p95']:.2f} (máximo {calib['max']:.2f}).",
            "",
            "**El umbral no compra precisión**: hay fallos con confianza máxima que",
            "ningún umbral filtra. La vía para más precisión es calibrar el modelo",
            "(temperature scaling), no mover el umbral.",
            "",
        ]
    lineas += [
        "## Evidencia",
        "",
        "| Cifra | Archivo |",
        "|---|---|",
        "| Accuracy `dominante`, estratificado | `outputs/reports/cv_metrics_dominante.json` |",
        "| Accuracy `dominante`, por señante | `outputs/reports/cv_metrics_dominante_senante.json` |",
        "| Accuracy `ambas`, estratificado | `outputs/reports/cv_metrics_ambas.json` |",
        "| Accuracy `ambas`, por señante | `outputs/reports/cv_metrics_ambas_senante.json` |",
        "| Split 80/20 sobre el ONNX | `outputs/reports/evaluacion_modelo.json` |",
        "| Matriz de confusión out-of-fold | `outputs/reports/evaluacion_cv_matriz.csv` |",
        "",
    ]
    texto = "\n".join(lineas)
    salida.parent.mkdir(parents=True, exist_ok=True)
    salida.write_text(texto, encoding="utf-8")
    print("\n" + texto)
    print(f"\n[tablas] escritas en {salida}")


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(
        description="Reproduce las tablas de métricas del informe.")
    ap.add_argument("--dataset-dominante",
                    default=str(config.DATA_PROCESSED_DIR / "lsa64_dominante.npz"))
    ap.add_argument("--dataset-ambas",
                    default=str(config.DATA_PROCESSED_DIR / "lsa64_ambas.npz"))
    ap.add_argument("--cv-grupos", default="2", metavar="CAMPO",
                    help="campo del sample_id que identifica al señante "
                         "(default 2, que es el de LSA64)")
    ap.add_argument("--reports", default=str(config.OUTPUTS_REPORTS_DIR))
    ap.add_argument("--salida", default=None,
                    help="archivo .md de las tablas (default: "
                         "<reports>/tablas_informe.md)")
    ap.add_argument("--dry-run", action="store_true",
                    help="imprime los comandos sin ejecutarlos")
    ap.add_argument("--solo-tablas", action="store_true",
                    help="no reentrena: regenera las tablas desde los JSON que "
                         "ya están en outputs/reports/")
    args = ap.parse_args()

    reports = Path(args.reports)
    salida = Path(args.salida) if args.salida else reports / "tablas_informe.md"
    # Absolutas: los subprocesos corren con cwd=config.RAIZ, así que una ruta
    # relativa se resolvería contra la raíz y no contra el directorio del usuario.
    ds_dom = Path(args.dataset_dominante).resolve()
    ds_amb = Path(args.dataset_ambas).resolve()

    if not args.solo_tablas:
        faltan = [str(d) for d in (ds_dom, ds_amb) if not d.exists()]
        if faltan:
            print("ERROR: no existe el dataset " + ", ".join(faltan) + ".\n"
                  "Constrúyelo con scripts/build_dataset.py (ver el README, "
                  "'Reproducir el entrenamiento completo') o usa --solo-tablas "
                  "para regenerar las tablas desde los JSON ya existentes.")
            return 1

        comandos = _comandos(ds_dom, ds_amb, args.cv_grupos)
        if not args.dry_run:
            print("Se van a entrenar 20 modelos (4 evaluaciones x 5 folds). "
                  "Esto tarda.\n")
        fallos = _ejecutar(comandos, args.dry_run)
        if args.dry_run:
            print("\n[dry-run] no se ejecutó nada.")
            return 0
        if fallos:
            print(f"\nERROR: {fallos} comando(s) fallaron; las tablas de abajo "
                  "pueden no corresponder a esta corrida.")

    if not (reports / "cv_metrics_dominante_senante.json").exists():
        print(f"ERROR: no hay evidencia en {reports}. Corre el script sin "
              "--solo-tablas para generarla.")
        return 1

    _escribir_tablas(reports, salida)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
