# SIA2022

Requisitos:
>- Python3

En el archivo tp2/config.json, se configura la forma en al que se ejecuta el programa.

>- "cot": Iteracion maxima para cortar.
>- "n": Tasa de aprendizaje.
>- "x": Elementos de entrada.
>- "y": Valores esperados. 
>- "perceptron_type": "step" | "linear" | "non-linear-tan" | "non-linear-logistic" | "multi-layer-xor" | "multi-layer-even" | "multi-layer-number"
>- "b": Parametro para funciones de activacion (tangente y logistica).
>- "normalization_type": Tipo de normalizacion.
>- "config_by_code": Solo se usa para hacer los graficos/test, dejarlo siempre en false.
>- "inner_layers": Capas internas.
>- "nodes_count": Cantidad de nodos por capa.
>- "momentum": True para optimizar con momentum, sino false.
>- "adaptative": True para optimizar con adaptative, sino false.
>- "adam": True para optimizar con adam, sino false.
>- "cross_validation": True para dividir el conjunto de entrenamiento.
>- "k": Cantidad de conjuntos del cross validation.
>- "delta": Rango en el que se genera el ruido.
>- "percentage_train": Cuando no se usa cross validatrion, se usa para sacar el porcentaje del set de con el que se va a entrenar.

Ejecutar el archivo tp2/main.py.
