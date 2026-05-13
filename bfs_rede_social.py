from collections import deque

def bfs(g, rotulos, rotulo_inicial):
    n = len(g)                              # número de nós
    no_ini = rotulos.index(rotulo_inicial)  # número do nó inicial
    visitados = [False] * n                 # lista de visitados
    q = deque()                             # fila usada pelo BFS
    q.append(no_ini)
    hops = [0] * n   # lista de hops (número de arestas da origem ao nó)
    caminho = []     # resultado retornado pela função
    
    # esse laço executa o BFS
    while (q): # enquanto a fila existir...
        x = q.popleft()      # desenfileirar nó
        if not visitados[x]: # se nó ainda não foi visitado
            visitados[x] = True # 1. marcar como visitado
            caminho.append(x)   # 2. adicionar ao caminho
            for y in range(n): # 3. enfileirar os vizinhos não visitados
                if (g[x][y] == 1) and (not visitados[y]):
                    q.append(y)
                    if hops[y]==0: hops[y] = hops[x] + 1
        
    
    return [(rotulos[i], hops[i]) for i in caminho]
    

rotulos = ['Frank', 'Rakesh', 'Eiben', 'Kamber', 'Han']
grafo = [
    [0, 1, 0, 1, 0],
    [0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0],
    [0, 1, 0, 0, 1],
    [0, 0, 0, 1, 0]
    ]


no_inicial = 'Frank'
resultado = bfs(grafo, rotulos, no_inicial)
print('BFS do nó', no_inicial, ':', resultado)

"""    
GRAFO (Rede Social - arestas direcionadas)

(Frank)-->(Rakesh)<---(Eiben)
     |        ^  
     |        |           
     ---->(Kamber)<--->(Han)
"""

