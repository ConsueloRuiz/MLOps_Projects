# Este script configura los metadatos y la estructura del paquete del proyecto para que pueda ser
# instalado y utilizado como un módulo de Python. Utiliza setuptools para detectar automáticamente
# los paquetes dentro del directorio mediante find_packages(), asignar un nombre al paquete, definir
# su versión, descripción, autor y otros metadatos relevantes. Su propósito principal es permitir la
# correcta distribución, instalación y reutilización del código del proyecto dentro de entornos de
# desarrollo o despliegue, siguiendo las buenas prácticas de empaquetado en MLOps.
from setuptools import find_packages, setup

setup(
    name='src',
    packages=find_packages(),
    version='0.1.0',
    description='proyecto de MLops',
    author='consuelo_ruiz',
    license='',
)
