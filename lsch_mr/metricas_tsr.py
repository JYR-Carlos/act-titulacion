"""
Métricas del Task Success Rate (TSR) — métrica 4 del MVP (Sección 11).

TSR = pruebas exitosas / pruebas totales, con éxito definido como "el
participante interpretó correctamente la secuencia de señas **sin comunicación
adicional**". Umbral del MVP: ≥ 80% sobre un mínimo de 10 pruebas con
participantes distintos.

Funciones puras: no leen archivos ni imprimen. El CLI es `calcular_tsr.py`.

---------------------------------------------------------------------------
Por qué se reporta un intervalo de confianza y no solo el porcentaje
---------------------------------------------------------------------------
Con n=10, el TSR es un porcentaje sobre muy pocas observaciones y su
incertidumbre es enorme: 8 aciertos de 10 dan un 80% cuya horquilla al 95% va
aproximadamente de 49% a 94%. Es decir, **un resultado de exactamente 8/10 no
permite afirmar que el sistema supera el 80% real** — solo que lo alcanzó en esa
muestra. Reportar el punto sin el intervalo daría al umbral una precisión que el
tamaño muestral no respalda, y es justo el tipo de cifra que una comisión pide
defender.

Se usa el intervalo de **Wilson**, no el normal (Wald): con n pequeño y
proporciones cercanas a 1, Wald produce límites por encima de 1 y se estrecha
absurdamente cuando todos los intentos son éxitos (10/10 daría ±0).
"""
from __future__ import annotations

import math

# Umbral de la métrica 4 y tamaño muestral mínimo, ambos de la Sección 11.
TSR_OBJETIVO = 0.80
MIN_PRUEBAS = 10


def wilson(exitos: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Intervalo de confianza de Wilson para una proporción binomial.

    `z=1.96` -> 95%. Devuelve (inferior, superior), ambos dentro de [0, 1].
    Sin observaciones devuelve (0.0, 1.0): ignorancia total, que es lo correcto.
    """
    if n <= 0:
        return (0.0, 1.0)
    p = exitos / n
    denom = 1.0 + z * z / n
    centro = (p + z * z / (2 * n)) / denom
    margen = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, centro - margen), min(1.0, centro + margen))


def calcular_tsr(registros: list[dict],
                 objetivo: float = TSR_OBJETIVO,
                 min_pruebas: int = MIN_PRUEBAS) -> dict:
    """Calcula el TSR y valida que el protocolo se haya respetado.

    Cada registro es un dict con al menos:
        participante   identificador (anonimizado) de la persona
        exito          bool — interpretación correcta sin comunicación adicional

    y opcionalmente `secuencia_id`, `senas_ejecutadas`, `interpretacion`,
    `comentarios`.

    Las **alertas** son tan importantes como el porcentaje: un 100% sobre 4
    pruebas, o sobre 10 pruebas hechas a la misma persona, no cumple la métrica
    aunque el número salga bonito. El diseño pide 10 pruebas con participantes
    **distintos**, y eso se verifica aquí en vez de confiar en que quien llenó
    la planilla lo recordara.
    """
    n = len(registros)
    exitos = sum(1 for r in registros if r.get("exito"))
    tsr = exitos / n if n else 0.0
    participantes = [str(r.get("participante", "")).strip() for r in registros]
    unicos = sorted({p for p in participantes if p})

    alertas: list[str] = []
    if n < min_pruebas:
        alertas.append(
            f"Solo hay {n} prueba(s); el diseño exige un mínimo de {min_pruebas}.")
    if len(unicos) < min_pruebas:
        alertas.append(
            f"Solo hay {len(unicos)} participante(s) distinto(s); el diseño exige "
            f"{min_pruebas} personas diferentes, no {min_pruebas} repeticiones.")
    if any(not p for p in participantes):
        alertas.append("Hay registros sin identificador de participante: no se "
                       "puede verificar que fueran personas distintas.")
    repetidos = sorted({p for p in unicos if participantes.count(p) > 1})
    if repetidos:
        alertas.append(
            f"Participante(s) con más de una prueba: {', '.join(repetidos)}. "
            "Cada persona debe aportar una sola prueba, o el TSR pondera más a "
            "quien repitió.")

    ci_bajo, ci_alto = wilson(exitos, n)
    return {
        "n_pruebas": n,
        "n_exitos": exitos,
        "n_fallos": n - exitos,
        "n_participantes": len(unicos),
        "participantes": unicos,
        "tsr": round(tsr, 4),
        "objetivo": float(objetivo),
        "cumple": bool(tsr >= objetivo and n >= min_pruebas
                       and len(unicos) >= min_pruebas),
        "cumple_umbral_sin_validar_protocolo": bool(tsr >= objetivo),
        "ic95": [round(ci_bajo, 4), round(ci_alto, 4)],
        # Si el límite inferior del IC queda por debajo del objetivo, la muestra
        # es compatible con un sistema que NO cumple. Hay que decirlo.
        "ic95_respalda_objetivo": bool(ci_bajo >= objetivo),
        "alertas": alertas,
    }


def desglose_por_secuencia(registros: list[dict]) -> list[dict]:
    """TSR por guion de secuencia, para ver si una falla más que las otras.

    Si una de las secuencias concentra los fallos, el problema es de esas señas
    concretas (o de su encadenamiento), no del sistema en bloque — y eso cambia
    qué se arregla después.
    """
    por: dict[str, list[dict]] = {}
    for r in registros:
        por.setdefault(str(r.get("secuencia_id", "") or "(sin id)"), []).append(r)

    filas = []
    for sec, rs in sorted(por.items()):
        exitos = sum(1 for r in rs if r.get("exito"))
        filas.append({
            "secuencia_id": sec,
            "n": len(rs),
            "exitos": exitos,
            "tsr": round(exitos / len(rs), 4) if rs else 0.0,
        })
    return filas


def comentarios_cualitativos(registros: list[dict]) -> list[dict]:
    """Feedback cualitativo no vacío, con su contexto.

    El diseño pide registrar percepción de legibilidad/utilidad además del
    porcentaje. Se extraen aparte para que no se pierdan en el CSV: son lo que
    explica *por qué* falló una prueba.
    """
    salida = []
    for r in registros:
        texto = str(r.get("comentarios", "") or "").strip()
        if not texto:
            continue
        salida.append({
            "participante": r.get("participante", ""),
            "secuencia_id": r.get("secuencia_id", ""),
            "exito": bool(r.get("exito")),
            "comentarios": texto,
        })
    return salida
