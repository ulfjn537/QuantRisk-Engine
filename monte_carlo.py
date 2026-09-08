import yfinance as yf
import numpy as np
import pandas as pd

def simular_monte_carlo(precio_inicial:float , mu:float , sigma:float , dias:int , num_simulaciones:int):
    matriz_precios = np.zeros((dias + 1, num_simulaciones))
    dt=1
    matriz_precios[0,:]=precio_inicial
    for t in range (1 , dias+1):
        epsilon = np.random.standard_normal(num_simulaciones)
        #esto nos da un array de tamaño num_simulaciones con numeros aleatorios de una normal
        matriz_precios[t,:] = matriz_precios[t-1,:]*np.exp((mu - sigma**2 / 2)*dt + sigma*np.sqrt(dt)*epsilon) #estas haciendo operaciones entre dos arrays
    return matriz_precios

import numpy as np

def simular_montecarlo_portfolio(
    capital_total_inicial: float,
    pesos: np.ndarray,
    mu_vector: np.ndarray,
    matriz_cov: np.ndarray,
    dias: int,
    num_simulaciones: int
) -> np.ndarray:
    
    n = len(pesos)
    
    # Manejo de la Descomposición de Cholesky
    try:
        L = np.linalg.cholesky(matriz_cov)
    except np.linalg.LinAlgError:
        # Corrección por si la matriz no es estrictamente definida positiva
        L = np.linalg.cholesky(matriz_cov + np.eye(n) * 1e-8)
        
    varianzas = np.diag(matriz_cov) # Array con la diagonal (varianzas anuales)
    
    # Factor de escalado de tiempo (asumiendo datos anuales de entrada)
    dt = 1.0 / dias
    
    # 4. Generar ruido aleatorio
    ruido_independiente_3D = np.random.normal(size=(dias, num_simulaciones, n))
    
    # 5. Correlacionar el ruido con L.T (multiplicación matricial)
    ruido_correlacionado = ruido_independiente_3D @ L.T
    
    # 6. Calcular el MBG escalado a nivel diario
    drift_diario = (mu_vector - 0.5 * varianzas) * dt
    ruido_diario = ruido_correlacionado * np.sqrt(dt)

    # Retornos exponenciales diarios
    retornos_diarios = np.exp(drift_diario + ruido_diario)

    # Capital asignado en el día 0 a cada activo según sus pesos
    capitales_iniciales_activos = capital_total_inicial * pesos

    # Crecimiento acumulado a lo largo de los días (axis=0)
    factores_acumulados = np.cumprod(retornos_diarios, axis=0)

    # Valor en dólares de cada activo individual
    trayectorias_activos = capitales_iniciales_activos * factores_acumulados

    # Sumar la aportación de todos los activos (axis=2) para la cartera total
    valor_cartera_futuro = np.sum(trayectorias_activos, axis=2)

    # Matriz final de dimensiones (dias + 1, num_simulaciones)
    matriz_precios = np.zeros((dias + 1, num_simulaciones))
    matriz_precios[0, :] = capital_total_inicial
    matriz_precios[1:, :] = valor_cartera_futuro
    
    return matriz_precios




# =====================================================================
# BLOQUE DE PRUEBA (Solo se ejecuta si ejecutas este archivo directamente)
# =====================================================================
if __name__ == "__main__":
    print("🚀 Ejecutando test de simulación de Monte Carlo...\n")
    
    # 1. Datos simulados para 3 activos (ej: AAPL, MSFT, GOOGL)
    pesos_test = np.array([0.4, 0.35, 0.25])
    mu_test = np.array([0.12, 0.10, 0.15])  # Retornos esperados anuales (12%, 10%, 15%)
    
    # Matriz de covarianza simulada
    cov_test = np.array([
        [0.04,   0.018,  0.012],
        [0.018,  0.035,  0.015],
        [0.012,  0.015,  0.050]
    ])
    
    capital_test = 10000.0  # $10,000 USD
    dias_test = 252        # 1 año bursátil
    simulaciones_test = 10000

    # 2. Ejecutar la función
    matriz_resultados = simular_montecarlo_portfolio(
        capital_total_inicial=capital_test,
        pesos=pesos_test,
        mu_vector=mu_test,
        matriz_cov=cov_test,
        dias=dias_test,
        num_simulaciones=simulaciones_test
    )

    # 3. Imprimir verificaciones
    print("✅ Simulación completada con éxito.")
    print(f"📊 Forma de la matriz resultante: {matriz_resultados.shape}")
    print(f"💰 Capital Inicial Día 0 (Comprobación): ${matriz_resultados[0, 0]:,.2f}")
    
    precios_finales = matriz_resultados[-1, :]
    print(f"📈 Valor promedio final tras 1 año: ${np.mean(precios_finales):,.2f}")
    print(f"📉 Valor mínimo simulado: ${np.min(precios_finales):,.2f}")
    print(f"🚀 Valor máximo simulado: ${np.max(precios_finales):,.2f}")