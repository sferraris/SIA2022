# SIA2022

# Lado A

Requisitos:
>- pip install arcade
>- Python3

En el archivo tp1_ladoa/config.json, se configura la forma en al que se ejecuta el programa.

>- "matrix_size": Tamaño de la matriz del juego.
>- "color_number": Cantidad de colores (maximo 26).
>- "moves": Cantidad de movimientos maximos (con -1 la cantidad es infinito).
>- "heuristic": "cells_left" | "colors_left" | "shortest_path"
>- "algorythm": "a*" | "DFS" | "BFS" | "local_greedy" | "global_greedy"

Ejecutar el archivo tp1_ladoa/game.py.

# Lado B

Requisitos:
>- Python3

En el archivo tp1_ladob/config.json, se configura la forma en al que se ejecuta el programa.
>- "target_color": Color al que se quiere llegar. (array de 3 numeros formato rgb)
>- "color_palette": Array de colores rgb que se tiene como paleta inicial. (puede ser vacio)
>- "population_length": si es -1 toma la paleta de colores,
sino genera la cantidad de colores aleatorios que se pase.
>- "K": Cantidad de individuos que se seleccionan para ser padres.
>- "mutation_probability": La probabilidad que un hijo mute.
>- "mutation_delta": El rango en el que se altera la proporcion al mutar. (entre -delta y delta)
>- "mutation_type": "uniform" | "gen" | "multi_gen_limited" | "complete"
>- "selection_method": "elite" | "random" | "deterministic_tournament"
>- "mutation_statistics": true si se quiere ver las estadisticas de los metodos de mutacion.
>- "selection_statistics": true si se quiere ver las estadisticas de los metodos de seleccion.
>- "max_cycles": maxima cantidad de generaciones que se generan, en caso de ser -1 se toma como
si no hubiera un maximo

Ejecutar el archivo tp1_ladob/main.py.
