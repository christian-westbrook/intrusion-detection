echo off

REM Set system requirements here.

SET required_python_version_major=3
SET required_python_version_minor=6

SET required_pip_version_major=19
SET required_pip_version_minor=0

REM Retrieve this machine's Python and pip versions.
FOR /f "tokens=*" %%a IN ('python --version') DO SET python_version_string=%%a
FOR /f "tokens=*" %%b IN ('pip --version') DO SET pip_version_string=%%b

REM Preprocess the version strings for compatability with validation logic.

FOR /f "tokens=2" %%c IN ("%python_version_string%") DO SET python_version_token=%%c
FOR /f "tokens=2" %%d IN ("%pip_version_string%") DO SET pip_version_token=%%d

FOR /f "tokens=1 delims=." %%e IN ("%python_version_token%") DO SET python_version_major=%%e
FOR /f "tokens=2 delims=." %%f IN ("%python_version_token%") DO SET python_version_minor=%%f
FOR /f "tokens=3 delims=." %%g IN ("%python_version_token%") DO SET python_version_patch=%%g

FOR /f "tokens=1 delims=." %%h IN ("%pip_version_token%") DO SET pip_version_major=%%h
FOR /f "tokens=2 delims=." %%i IN ("%pip_version_token%") DO SET pip_version_minor=%%i
FOR /f "tokens=3 delims=." %%j IN ("%pip_version_token%") DO SET pip_version_patch=%%j

REM Validate that this machine's Python installation meets system requirements.

IF NOT %python_version_major% GEQ %required_python_version_major% (
	echo Python %required_python_version_major%.%required_python_version_minor% is a system requirement.
	echo Please navigate to https://www.python.org/ and install Python %required_python_version_major%.%required_python_version_minor% or greater to continue.
	EXIT /b 1
)

IF NOT %python_version_minor% GEQ %required_python_version_minor% (
	echo Python %required_python_version_major%.%required_python_version_minor% is a system requirement.
	echo Please navigate to https://www.python.org/ and install Python %required_python_version_major%.%required_python_version_minor% or greater to continue.
	EXIT /b 1
)

REM Validate that this machine's pip installation meets system requirements.

IF NOT %pip_version_major% GEQ %required_pip_version_major% (
	echo pip %required_pip_version_major%.%required_pip_version_minor% is a system requirement.
	echo Please navigate to https://pip.pypa.io/en/stable/ and install pip %required_pip_version_major%.%required_pip_version_minor% or greater to continue.
	EXIT /b 1
)

IF NOT %pip_version_minor% GEQ %required_pip_version_minor% (
	echo pip %required_pip_version_major%.%required_pip_version_minor% is a system requirement.
	echo Please navigate to https://pip.pypa.io/en/stable/ and install pip %required_python_version_major%.%required_pip_version_minor% or greater to continue.
	EXIT /b 1
)

REM Install required packages

pip install notebook
pip install --user kaggle
pip install pandas
pip install numpy
pip install sklearn
pip install scipy

echo Installation process complete
echo Scan installation output for error messages before proceeding
