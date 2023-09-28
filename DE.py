import numpy as np
import matplotlib.pyplot as plt

def ackley(xx, a=20, b=0.2, c=2*np.pi):
    d = len(xx)

    sum1 = 0
    sum2 = 0
    for xi in xx:
        sum1 += xi**2
        sum2 += np.cos(c * xi)

    term1 = -a * np.exp(-b * np.sqrt(sum1 / d))
    term2 = -np.exp(sum2 / d)

    y = term1 + term2 + a + np.exp(1)

    return y

def sphere(x):
    return np.sum(x**2)

# Parámetros del algoritmo DE
MaxIt = 1000  # Número máximo de iteraciones
nPop = 50     # Tamaño de la población

# Límites de las variables de decisión
VarMin = -5
VarMax = 5

# Solicitar al usuario ingresar los valores de CR y F
CR = float(input("Ingrese el valor de Probabilidad de Cruce (CR): "))
F = float(input("Ingrese el valor de Factor de Escala (F): "))

print(f"Probabilidad de cruce (CR): {CR}, Factor de escala (F): {F}")

# Inicialización de variables
BestSol = None
BestSolCost = float('inf')

# Inicialización de población
pop = [{'Position': np.random.uniform(VarMin, VarMax, 1), 'Cost': float('inf')} for _ in range(nPop)]

# Inicialización de lista para almacenar los mejores costos en cada iteración
BestCost = []

# Bucle principal de DE
for it in range(MaxIt):
    for i in range(nPop):
        x = pop[i]['Position'][0]  # Acceder al valor escalar

        # Selección de 3 índices distintos aleatorios
        A = np.random.permutation(nPop)
        A = A[A != i][:3]
        r1, r2, r3 = A

        # Mutación
        v = pop[r1]['Position'] + F * (pop[r2]['Position'] - pop[r3]['Position'])
        v = np.maximum(v, VarMin)
        v = np.minimum(v, VarMax)

        # Crossover
        j0 = np.random.randint(0, 1)  # Usar 1 como límite superior para obtener un valor aleatorio dentro del rango (0, 1)
        u = np.where(np.random.rand(1) <= CR, v, x)

        # Evaluación del costo de la nueva solución
        cost = ackley(u)  # Cambiar a "sphere(u)" para probar con la función Sphere

        # Selección
        if cost < pop[i]['Cost']:
            pop[i]['Position'] = u
            pop[i]['Cost'] = cost

            # Actualización del mejor costo global
            if cost < BestSolCost:
                BestSol = pop[i].copy()
                BestSolCost = cost

    # Almacenar el mejor costo de esta iteración
    BestCost.append(BestSolCost)

# Mostrar resultados
plt.figure()
plt.semilogy(BestCost, linewidth=2)
plt.xlabel('Iteración')
plt.ylabel('Mejor Costo')
plt.grid(True)
plt.title(f"CR={CR}, F={F}")
plt.show()
