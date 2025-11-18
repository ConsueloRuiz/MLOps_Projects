# Este script verifica que el entorno de ejecución utilice la versión correcta de Python requerida
# por el proyecto. Para ello, compara la versión mayor del intérprete actual con la versión esperada
# (por defecto, Python 3). Si el intérprete no coincide con la versión necesaria, el script detiene
# la ejecución y lanza un error explicando el problema. En caso de que la versión sea correcta,
# muestra un mensaje indicando que el entorno de desarrollo cumple los requisitos. Este tipo de
# validación es útil para evitar incompatibilidades y asegurar que el proyecto se ejecute bajo la
# versión de Python adecuada.
import sys
REQUIRED_PYTHON = "python3"


def main():
    system_major = sys.version_info.major
    if REQUIRED_PYTHON == "python":
        required_major = 2
    elif REQUIRED_PYTHON == "python3":
        required_major = 3
    else:
        raise ValueError("Unrecognized python interpreter: {}".format(
            REQUIRED_PYTHON))

    if system_major != required_major:
        raise TypeError(
            "This project requires Python {}. Found: Python {}".format(
                required_major, sys.version))
    else:
        print(">>> Development environment passes all tests!")


if __name__ == '__main__':
    main()
