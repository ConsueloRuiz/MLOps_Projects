@echo off

rem This file is UTF-8 encoded, so we need to update the current code page while executing it
:: Este script activa un entorno virtual de Python en Windows ajustando correctamente las variables
:: de entorno necesarias. Primero detecta y almacena la página de códigos actual del sistema para
:: poder cambiar temporalmente a UTF-8 (requerido debido a que el archivo está codificado en UTF-8).
:: Luego establece la ruta del entorno virtual, actualiza el indicador de la consola (PROMPT),
:: deshabilita PYTHONHOME para evitar conflictos, y modifica la variable PATH para que utilice los
:: ejecutables del entorno virtual. También guarda los valores originales de cada variable para poder
:: restaurarlos al desactivar el entorno. Al finalizar, restablece la página de códigos original del
:: sistema.
for /f "tokens=2 delims=:." %%a in ('"%SystemRoot%\System32\chcp.com"') do (
    set _OLD_CODEPAGE=%%a
)
if defined _OLD_CODEPAGE (
    "%SystemRoot%\System32\chcp.com" 65001 > nul
)

set VIRTUAL_ENV=C:\Users\tonym\Downloads\EQUIPO47\MLOps_Projects\seol_bike_project\.venv

if not defined PROMPT set PROMPT=$P$G

if defined _OLD_VIRTUAL_PROMPT set PROMPT=%_OLD_VIRTUAL_PROMPT%
if defined _OLD_VIRTUAL_PYTHONHOME set PYTHONHOME=%_OLD_VIRTUAL_PYTHONHOME%

set _OLD_VIRTUAL_PROMPT=%PROMPT%
set PROMPT=(.venv) %PROMPT%

if defined PYTHONHOME set _OLD_VIRTUAL_PYTHONHOME=%PYTHONHOME%
set PYTHONHOME=

if defined _OLD_VIRTUAL_PATH set PATH=%_OLD_VIRTUAL_PATH%
if not defined _OLD_VIRTUAL_PATH set _OLD_VIRTUAL_PATH=%PATH%

set PATH=%VIRTUAL_ENV%\Scripts;%PATH%
set VIRTUAL_ENV_PROMPT=(.venv) 

:END
if defined _OLD_CODEPAGE (
    "%SystemRoot%\System32\chcp.com" %_OLD_CODEPAGE% > nul
    set _OLD_CODEPAGE=
)
