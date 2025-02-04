class Nodo: 
    def __init__(self, valor):
        self.valor = valor
        self.derecha = None
        self.izquierda = None

class Arbol:
    def __init__(self):
        self.raiz = None

    def insertar(self, valor):
        self.raiz = self._insertar_recursivo(self.raiz, valor)

    def _insertar_recursivo(self, nodo, valor):
        if nodo is None:
            return Nodo(valor)
        
        if valor < nodo.valor:
            nodo.izquierda = self._insertar_recursivo(nodo.izquierda, valor)
        else:
            nodo.derecha = self._insertar_recursivo(nodo.derecha, valor)
        return nodo
    
    def buscar(self, valor):
        return self._buscar_recursivo(self.raiz, valor)
    
    def _buscar_recursivo(self, nodo, valor):
        if nodo is None:
            return False
        
        if nodo.valor == valor:
            return True
        
        if valor < nodo.valor:
            return self._buscar_recursivo(nodo.izquierda, valor)
        else:
            return self._buscar_recursivo(nodo.derecha, valor)
        
    def imprimir_inorder(self, nodo):
        if nodo is not None:
            self.imprimir_inorder(nodo.izquierda)
            print(nodo.valor)
            self.imprimir_inorder(nodo.derecha)

arbol = Arbol()  # Creación de la instancia del árbol
arbol.insertar(10)
arbol.insertar(3)
arbol.insertar(15)
arbol.insertar(12)
arbol.insertar(18)
arbol.insertar(6)
arbol.imprimir_inorder(arbol.raiz)
print(arbol.buscar(15))
print(arbol.buscar(7))
