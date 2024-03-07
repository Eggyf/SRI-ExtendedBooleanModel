# Proyecto de SRI

## Modelos Booleano y Booleano Extendido

## Integrantes:
-  Abdel Fregel Hernández C412
- Lázaro David Alba Ajete C411


## La solución desarrollada consta de varios componentes:

- Preprocesamiento de Consultas: Las consultas se procesan para tokenizarlas, eliminar ruido, eliminar palabras vacías y reducir la morfología.
- Modelo de Recuperación de Documentos: Se implementa un modelo de recuperación de documentos basado en el modelo booleano clásico y una versión extendida que considera la relevancia de los términos usando el tf-idf.

## TF-IDF

$$\mathrm{TF}(t,d) = \frac{f_{t,d}}{\sum_{t' \in d} f_{t',d}}$$
Donde:

- (t) representa un término (palabra) específico.
- (d) es el documento en el que se encuentra el término.
- (f_{t,d}) es la frecuencia del término (t) en el documento (d).
- $$(\sum_{t’ \in d} f_{t’,d})$$ es la suma de las frecuencias de todos los términos en el documento (d).

## Consulta

Una consulta en el contexto de este proyecto es una cadena de texto con la información que un usuario desea recuperar de un conjunto de documentos. Las consultas pueden contener términos de búsqueda simples o complejos, así como operadores lógicos como "and", "or" y "not".

## Ejecutar el proyecto
- Descargar el repositorio de github e instalar las dependencias necesarias
- correr el archivo
- Se vera una interfaz de usuario en el puerto que indica al correr el proyecto
