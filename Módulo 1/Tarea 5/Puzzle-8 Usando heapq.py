from heapq import heappush, heappop
import time

class Puzzle:
    def __init__(self, estado_inicial, estado_final):
        self.estado_inicial = estado_inicial
        self.estado_final = estado_final
        self.tamaño = 3

    def encontrar_vacio(self, estado):
        for i in range(self.tamaño):
            for j in range(self.tamaño):
                if estado[i][j] == 0:
                    return (i, j)
        return None

    def mover(self, estado, direccion):
        nuevo_estado = [fila[:] for fila in estado]
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
        distancia = 0
        for i in range(self.tamaño):
            for j in range(self.tamaño):
                if estado[i][j] != 0:
                    fila_objetivo, col_objetivo = divmod(estado[i][j] - 1, self.tamaño)
                    distancia += abs(i - fila_objetivo) + abs(j - col_objetivo)
        return distancia

    def resolver(self):
        cola_prioridad = []
        heappush(cola_prioridad, (0, self.estado_inicial, []))
        visitados = set()

        while cola_prioridad:
            _, estado_actual, camino = heappop(cola_prioridad)
            if estado_actual == self.estado_final:
                return camino
            if str(estado_actual) in visitados:
                continue
            visitados.add(str(estado_actual))

            for direccion in ['arriba', 'abajo', 'izquierda', 'derecha']:
                nuevo_estado = self.mover(estado_actual, direccion)
                if nuevo_estado:
                    nuevo_camino = camino + [direccion]
                    costo = len(nuevo_camino) + self.heuristica(nuevo_estado)
                    heappush(cola_prioridad, (costo, nuevo_estado, nuevo_camino))
        return None

def imprimir_estado(estado, paso=None):
    """Imprime el estado actual del puzzle."""
    if paso is not None:
        print(f"Paso {paso}:")
    for fila in estado:
        print(fila)
    print()

def main():
    inicio = time.time()  # Guarda el tiempo inicial
    estado_inicial = [[0, 8, 4], [5, 6, 2], [3, 1, 7]]
    estado_final = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]

    print("Estado inicial:")
    imprimir_estado(estado_inicial)

    print("Estado final:")
    imprimir_estado(estado_final)

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