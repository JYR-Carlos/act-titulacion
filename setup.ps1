<#
.SYNOPSIS
    Deja el repo listo para ejecutar, en un comando.

.DESCRIPTION
    Crea el entorno virtual, instala requirements.txt con las versiones
    pineadas, descarga el modelo hand_landmarker.task de MediaPipe (que no
    viene con pip install mediapipe) y corre las pruebas. Termina diciendo si
    el entorno quedó listo y cuál es el siguiente comando.

    Es idempotente: se puede volver a correr sobre un entorno ya montado.

.PARAMETER SinPruebas
    Salta la batería de pytest al final. Útil si solo quieres reinstalar
    dependencias.

.PARAMETER Recrear
    Borra .venv y lo rehace desde cero. Úsalo si el entorno quedó a medias o
    si cambiaste de versión de Python.

.EXAMPLE
    .\setup.ps1
    .\setup.ps1 -SinPruebas
    .\setup.ps1 -Recrear
#>
[CmdletBinding()]
param(
    [switch]$SinPruebas,
    [switch]$Recrear
)

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

$PYTHON_ESPERADO = "3.12"
$VENV = Join-Path $PSScriptRoot ".venv"
$VENV_PY = Join-Path $VENV "Scripts\python.exe"

$pasos = [System.Collections.Generic.List[string]]::new()
function Paso($texto) {
    Write-Host ""
    Write-Host "==> $texto" -ForegroundColor Cyan
}
function Ok($texto) {
    $script:pasos.Add("OK    $texto")
    Write-Host "    OK: $texto" -ForegroundColor Green
}
function Aviso($texto) {
    $script:pasos.Add("AVISO $texto")
    Write-Host "    AVISO: $texto" -ForegroundColor Yellow
}

# --------------------------------------------------------------------------- #
# 1. Intérprete de Python
# --------------------------------------------------------------------------- #
Paso "Comprobando Python"

$pythonCmd = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCmd) {
    Write-Host "No se encontró 'python' en el PATH." -ForegroundColor Red
    Write-Host "Instala Python $PYTHON_ESPERADO (la versión exacta está en .python-version)."
    exit 1
}

$version = (& python --version 2>&1) -replace '^Python\s+', ''
Write-Host "    Encontrado: Python $version  ($($pythonCmd.Source))"
if (-not $version.StartsWith($PYTHON_ESPERADO)) {
    # No es fatal, pero conviene saberlo: las versiones de requirements.txt se
    # fijaron contra 3.12 y mediapipe/tensorflow no publican rueda para todas.
    Aviso "se esperaba Python $PYTHON_ESPERADO.x y hay $version; si pip falla, esta es la causa"
} else {
    Ok "Python $version"
}

# --------------------------------------------------------------------------- #
# 2. Entorno virtual
# --------------------------------------------------------------------------- #
Paso "Preparando el entorno virtual (.venv)"

if ($Recrear -and (Test-Path $VENV)) {
    Write-Host "    -Recrear: borrando el .venv anterior"
    Remove-Item -Recurse -Force $VENV
}

if (Test-Path $VENV_PY) {
    Write-Host "    Ya existe .venv, se reutiliza (usa -Recrear para rehacerlo)"
} else {
    & python -m venv $VENV
    if ($LASTEXITCODE -ne 0) { Write-Host "Falló 'python -m venv'." -ForegroundColor Red; exit 1 }
}
if (-not (Test-Path $VENV_PY)) {
    Write-Host "No se creó $VENV_PY." -ForegroundColor Red
    exit 1
}
Ok "entorno virtual en .venv"

# --------------------------------------------------------------------------- #
# 3. Dependencias
# --------------------------------------------------------------------------- #
Paso "Instalando dependencias (requirements.txt)"
Write-Host "    Son ~2 GB con TensorFlow incluido; la primera vez tarda."

& $VENV_PY -m pip install --quiet --upgrade pip
& $VENV_PY -m pip install -r (Join-Path $PSScriptRoot "requirements.txt")
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "Falló la instalación de dependencias." -ForegroundColor Red
    Write-Host "Las versiones están pineadas con '=='; si tu Python no es $PYTHON_ESPERADO.x"
    Write-Host "es probable que no exista rueda para alguna de ellas."
    exit 1
}
Ok "dependencias instaladas"

# --------------------------------------------------------------------------- #
# 4. Modelo de MediaPipe
# --------------------------------------------------------------------------- #
Paso "Descargando el modelo HandLandmarker"

& $VENV_PY (Join-Path $PSScriptRoot "scripts\descargar_modelo.py")
if ($LASTEXITCODE -ne 0) {
    # No aborta: el pipeline offline (entrenar, exportar, métricas) funciona sin
    # el .task. Solo la captura lo necesita.
    Aviso "no se pudo descargar hand_landmarker.task; la captura no funcionará hasta tenerlo"
} else {
    Ok "models/hand_landmarker.task"
}

# --------------------------------------------------------------------------- #
# 5. Pruebas
# --------------------------------------------------------------------------- #
$pruebasOk = $true
if ($SinPruebas) {
    Paso "Pruebas: saltadas (-SinPruebas)"
    Aviso "no se corrieron las pruebas"
} else {
    Paso "Corriendo las pruebas"
    & $VENV_PY -m pytest -q
    if ($LASTEXITCODE -ne 0) {
        $pruebasOk = $false
        Aviso "hay pruebas que fallan (ver la salida de arriba)"
    } else {
        Ok "la batería de pytest pasa"
    }
}

# --------------------------------------------------------------------------- #
# Resumen
# --------------------------------------------------------------------------- #
Write-Host ""
Write-Host "-------------------------------------------------------------------"
foreach ($p in $pasos) { Write-Host "  $p" }
Write-Host "-------------------------------------------------------------------"

if ($pruebasOk) {
    Write-Host "ENTORNO LISTO." -ForegroundColor Green
    Write-Host ""
    Write-Host "  Activa el entorno:   .\.venv\Scripts\Activate.ps1"
    Write-Host "  Corre la demo:       python scripts/demo_vivo.py --fuente 0"
    Write-Host ""
    Write-Host "  El modelo entrenado ya viene en el repo: no hace falta reentrenar."
    exit 0
} else {
    Write-Host "ENTORNO INCOMPLETO: las dependencias están, pero las pruebas fallan." -ForegroundColor Yellow
    exit 1
}
