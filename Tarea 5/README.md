# Puzzle-8
Este proyecto implementa un solucionador para el **Puzzle 8** utilizando el algoritmo 
de búsqueda **A***. El Puzzle 8 es un rompecabezas deslizante que consiste en una 
cuadrícula de 3x3 con 8 piezas numeradas y un espacio vacío. El objetivo es 
reorganizar las piezas desde un estado inicial hasta un estado final deslizando las 
piezas adyacentes al espacio vacío.

## ¿Cómo resolvemos el Puzzle 8?
Para resolver el Puzzle 8, utilizamos el algoritmo **A***, que es un algoritmo de búsqueda informada. Este algoritmo combina:
1. **El costo acumulado** (`g(n)`): El número de movimientos realizados desde el estado inicial hasta el estado actual.
2. **La heurística** (`h(n)`): Una estimación del número de movimientos restantes para llegar al estado final.

En este proyecto, la heurística utilizada es la **distancia de Manhattan**, que calcula la suma de las distancias horizontales y verticales de cada pieza a su posición objetivo.

El algoritmo A* garantiza encontrar la solución óptima (el camino más corto) siempre que la heurística sea **admisible** (nunca sobrestima el costo real).

## Características del Proyecto
- **Algoritmo A***: Implementación del algoritmo de búsqueda informada A* para resolver el Puzzle 8.
- **Heurística**: Uso de la distancia de Manhattan para estimar el costo restante.
- **Dos versiones**:
  - Una versión utiliza `heapq` para manejar la cola de prioridad de manera eficiente.
  - La otra versión ordena manualmente la lista de nodos abiertos.
- **Visualización**: El programa muestra el estado del puzzle después de cada movimiento, permitiendo ver cómo se resuelve paso a paso.
- **Eficiencia**: La versión con `heapq` es significativamente más rápida.

### ¿Por qué `heapq` es más eficiente?

- **`heapq`** es una implementación de un **min-heap**, una estructura de datos optimizada para mantener el elemento con la prioridad más alta (en este caso, el menor costo total) en la parte superior.
- Un **min-heap** es un árbol binario balanceado donde cada nodo padre tiene un valor menor o igual que sus hijos.
- Las operaciones de inserción (`heappush`) y extracción (`heappop`) tienen una complejidad de **O(log n)**, lo que hace que el algoritmo sea más eficiente.

### Sin `heapq`:
- Si no se usa `heapq`, se debe utilizar una lista y ordenarla manualmente en cada iteración.
- Esto implica que la lista se reorganiza completamente en cada operación, lo que tiene una complejidad de **O(n log n)**.
- En problemas grandes, como el Puzzle 8, esto puede hacer que el algoritmo sea mucho más lento.


## Estructura del Código
El código está organizado de la siguiente manera:

1. **Clase `Puzzle`**:
   - Contiene los métodos para resolver el puzzle:
     - `encontrar_vacio`: Encuentra la posición del espacio vacío.
     - `mover`: Realiza un movimiento en una dirección dada.
     - `heuristica`: Calcula la distancia de Manhattan.
     - `resolver`: Implementa el algoritmo A*.

2. **Función `imprimir_estado`**:
   - Muestra el estado actual del puzzle en la consola.

3. **Función `main`**:
   - Define los estados inicial y final.
   - Crea una instancia de `Puzzle` y llama al método `resolver`.
   - Muestra la solución paso a paso.
   - 
## Autores

- **Juan Antonio Velázquez Alarcón**
- **Alexis Guillén Ruiz**
