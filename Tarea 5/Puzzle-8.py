import time

class Puzzle:
    def __init__(self, estado_inicial, estado_final):
        self.estado_inicial = estado_inicial
        self.estado_final = estado_final
        self.tamaño = 3

    def encontrar_vacio(self, estado):
        """Encuentra la posición del espacio vacío (0)."""
        for i in range(self.tamaño):
            for j in range(self.tamaño):
                if estado[i][j] == 0:
                    return (i, j)
        return None

    def mover(self, estado, direccion):
        """Realiza un movimiento en la dirección dada."""
        nuevo_estado = [fila[:] for fila in estado]  # Copia el estado actual
        i, j = self.encontrar_vacio(estado)
        if direccion == 'arriba' and i > 0:
            nuevo_estado[i][j], nuevo_estado[i-1][j] = nuevo_estado[i-1][j], nuevo_estado[i][j]
        elif direccion == 'abajo' and i < self.tamaño - 1:
            nuevo_estado[i][j], nuevo_estado[i+1][j] = nuevo_estado[i+1][j], nuevo_estado[i][j]
        elif direccion == 'izquierda' and j > 0:
            nuevo_estado[i][j], nuevo_estado[i][j-1] = nuevo_estado[i][j-1], nuevo_estado[i][j]
        elif direccion == 'derecha' and j < self.tamaño - 1:
            nuevo_estado[i][j], nuevo_estado[i][j+1] = nuevo_estado[i][j+1], nuevo_estado[i][j]
        else:
            return None
        return nuevo_estado

    def heuristica(self, estado):
        """Calcula la heurística (distancia de Manhattan) para un estado."""
        distancia = 0
        for i in range(self.tamaño):
            for j in range(self.tamaño):
                if estado[i][j] != 0:
                    fila_objetivo, col_objetivo = divmod(estado[i][j] - 1, self.tamaño)
                    distancia += abs(i - fila_objetivo) + abs(j - col_objetivo)
        return distancia

    def resolver(self):
        """Resuelve el puzzle utilizando el algoritmo A*."""
        nodos_abiertos = [(self.estado_inicial, [], 0)]  # (estado, camino, costo_acumulado)
        nodos_cerrados = set()

        while nodos_abiertos:
            # Ordenar nodos_abiertos por costo total (costo_acumulado + heurística)
            nodos_abiertos.sort(key=lambda x: x[2] + self.heuristica(x[0]))
            estado_actual, camino, costo_acumulado = nodos_abiertos.pop(0)

            if estado_actual == self.estado_final:
                return camino  # Solución encontrada

            if str(estado_actual) in nodos_cerrados:
                continue  # Saltar nodos ya visitados
            nodos_cerrados.add(str(estado_actual))

            # Generar nuevos estados
            for direccion in ['arriba', 'abajo', 'izquierda', 'derecha']:
                nuevo_estado = self.mover(estado_actual, direccion)
                if nuevo_estado:
                    nuevo_camino = camino + [direccion]
                    nuevo_costo_acumulado = costo_acumulado + 1
                    nodos_abiertos.append((nuevo_estado, nuevo_camino, nuevo_costo_acumulado))
        return None  # No se encontró solución

def imprimir_estado(estado, paso=None):
    """Imprime el estado actual del puzzle."""
    if paso is not None:
        print(f"Paso {paso}:")
    for fila in estado:
        print(fila)
    print()

def main():
    inicio = time.time()  # Guarda el tiempo inicial
    # Definir estados inicial y final
    estado_inicial = [[0, 8, 4], [5, 6, 2], [3, 1, 7]]
    estado_final = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]

    print("Estado inicial:")
    imprimir_estado(estado_inicial)

    print("Estado final:")
    imprimir_estado(estado_final)

    # Resolver el puzzle
    puzzle = Puzzle(estado_inicial, estado_final)
    solucion = puzzle.resolver()

    if solucion:
        print("Solución encontrada en", len(solucion), "movimientos:")
        estado_actual = estado_inicial
        for paso, movimiento in enumerate(solucion, start=1):
            print(f"Movimiento {paso}: {movimiento}")
            estado_actual = puzzle.mover(estado_actual, movimiento)
            imprimir_estado(estado_actual, paso)
    else:
        print("No se encontró solución.")

    fin = time.time()  # Guarda el tiempo final
    print(f"Tiempo de ejecución: {fin - inicio:.4f} segundos")

if __name__ == "__main__":
    main()
