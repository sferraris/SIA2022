# SIA2022

- pip install arcade



-Armar un arbol con los caminos de las soluciones
-Armar una funcion que reciba una heuristica
-Armar una funcion que use un algortimo específico 
 de búsqueda para buscar las soluciones

Heuristicas -> Costo --> 1 movimiento en todas las posibilidades
>- Que color me garantiza mayor cantidad de celdas a pintar
>- Cantidad de colores restantes a eliminar en la tabla --> minimo 6 movimientos
>- Cantidad de celdas a eliminar --> maximo ROWxROW - 1 celdas
>- Cantidad de filas / columnas completadas (no admisible)

# TP1 - Lado B
>- Genotipo - Alelos - Locus (Cromosomas son ejemplos de Genotipos)
>- Genotipo -> R | G | B
>- Alelos:
>  - Para R -> [0,255] entero
>  - Para G -> [0,255] entero
>  - Para B -> [0,255] entero
>- Locus:
>  - 1º R
>  - 2º G
>  - 3º B

- Estructura / Arquitectura ~ Genotipo --> listo<br/>
- Población Inicial --> Listo <br/>
- Función de adaptabilidad - fitness o aptitud <br/>
- Método de selección de padres<br/>
- Método de cruza<br/>
- Método de mutación<br/>
- Método de selección de nueva generación<br/>
- Se pueden usar los mismos métodos de selección de padres<br/>
- Condición de corte ( opcional )<br/>