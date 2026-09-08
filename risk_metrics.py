# buscamos una funcion para calcular el VaR y ES dada una simulacion de montecarlo de una accion especifica con unas fechas especificas de datos
# para poder calcularlo necesitaremos de entrada; la matriz de simulación, el precio inicial S0 y el nivel de confianza
# necesitamos transformar el vector de los precios finales, restandole el inicial y dividiendo por él
# para las métricas de percentil usamos la libreria NumPy
import numpy as np

def calcular_var_cvar(matriz_simulaciones  , precio_inicial:float , nivel_confianza=0.95):
    precios_finales = matriz_simulaciones[-1, :]
    rendimientos = (precios_finales - precio_inicial) / precio_inicial

    alpha = (1.0 - nivel_confianza) * 100
    var_porcentual = -np.percentile(rendimientos, alpha) #se pone - porque estas midiendo perdidas no ganancia negativa
    var_usd = var_porcentual * precio_inicial

    # ahora vamos a calcular el cvar
    peores_casos = rendimientos[rendimientos <= -var_porcentual]
    cvar_porcentual = -np.mean(peores_casos) #hay que tener en cuenta que mide la perdida en positivo
    cvar_usd = cvar_porcentual * precio_inicial

    return {
    "var_pct": var_porcentual,
    "var_usd": var_usd,
    "cvar_pct": cvar_porcentual,
    "cvar_usd": cvar_usd,
}

import numpy as np

def calcular_var_cvar_montecarlo(matriz_precios: np.ndarray, capital_inicial: float, nivel_confianza: float , tasa_libre_riesgo: float ) -> dict:
    """Calcula el VaR y CVaR histórico a partir de la matriz de Monte Carlo."""
    valores_finales = matriz_precios[-1, :]
    resultados = valores_finales - capital_inicial
    retornos_simulados = (valores_finales - capital_inicial) / capital_inicial

    alpha = 1.0 - nivel_confianza
    var_abs = -np.percentile(resultados, alpha * 100)
    var_pct = (var_abs / capital_inicial) * 100
    
    peores_escenarios = resultados[resultados <= -var_abs]
    cvar_abs = -np.mean(peores_escenarios)
    cvar_pct = (cvar_abs / capital_inicial) * 100
    # 2. Retorno medio y probabilidad de pérdida
    retorno_medio_pct = np.mean(retornos_simulados)
    prob_perdida = (np.sum(resultados < 0) / len(resultados)) * 100
    
    # 3. Métricas de Eficiencia
    volatilidad_simulada = np.std(retornos_simulados) #calcula la desviacion standar
    sharpe_ratio = (retorno_medio_pct - tasa_libre_riesgo) / volatilidad_simulada if volatilidad_simulada > 0 else 0
    starr_ratio = (retorno_medio_pct - tasa_libre_riesgo) / (cvar_pct / 100) if cvar_pct > 0 else 0

    return {
        "media_final": np.mean(valores_finales),
        "retorno_medio_pct": retorno_medio_pct * 100,
        "var_dolares": var_abs,
        "var_porcentaje": var_pct,
        "cvar_dolares": cvar_abs,
        "cvar_porcentaje": cvar_pct,
        "prob_perdida": prob_perdida,
        "sharpe_ratio": sharpe_ratio,
        "starr_ratio": starr_ratio
    }