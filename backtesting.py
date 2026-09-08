import numpy as np
import pandas as pd
import yfinance as yf

from data_engine import obtener_parametros_cartera
from monte_carlo import simular_montecarlo_portfolio
from risk_metrics import calcular_var_cvar_montecarlo


def obtener_precios_crisis(tickers: list[str], fecha_inicio: str, fecha_fin: str) -> pd.DataFrame:
    """Descarga los precios de cierre de los activos en el rango de fechas de la crisis."""
    datos = yf.download(tickers, start=fecha_inicio, end=fecha_fin)["Close"]
    if isinstance(datos, pd.Series): # si datos es una serie de pandas, que sucede cuando la cartera posee un sólo activo, cuando haces datos[tickers] puede lanzar un error
        datos = datos.to_frame() # esto fuerza a que, aunque haya un solo ticker, el resultado se transforme en un pd.DataFrame
    return datos[tickers].dropna()


def calcular_drawdown_historico(precios: pd.DataFrame, pesos: np.ndarray, capital_inicial: float) -> tuple[pd.Series, float]:
    """Calcula el valor diario de la cartera y el Maximum Drawdown (% de caída máxima)."""
    # Rendimiento acumulado diario de cada activo respecto al día 1 de la crisis
    retornos_acumulados = precios / precios.iloc[0] #Si una acción cotizaba a $100 el día 1 y pasa por $110 y $80 en los días siguientes, la serie se convierte en [1.0, 1.10, 0.80]. Esto representa el crecimiento o caída acumulada de cada activo de forma normalizada.
    
    # Evolución diaria del valor total de la cartera
    valor_cartera = (retornos_acumulados * pesos).sum(axis=1) * capital_inicial #.sum hace la suma de todos
                                                
    # Registro de picos máximos y caídas desde el pico (Drawdowns)
    picos_maximos = valor_cartera.cummax() #calcula el maximo guardado hasta la fecha, Ejemplo: Si la cartera vale [10k, 12k, 11k, 15k, 13k], cummax() genera [10k, 12k, 12k, 15k, 15k].
    drawdowns = (valor_cartera - picos_maximos) / picos_maximos #Mide la distancia porcentual entre el valor actual de la cartera y el pico más alto registrado hasta la fecha.
    
    mdd_pct = float(drawdowns.min()) * 100 #busca el drawdown minimo, es decir, la bajada más grande (en términos porcentuales)
    return valor_cartera, mdd_pct


def ejecutar_experimento_out_of_sample(
    tickers: list[str],
    pesos: np.ndarray,
    capital_inicial: float,
    fecha_inicio_entreno: str,
    fecha_fin_entreno: str,
    fecha_inicio_crisis: str,
    fecha_fin_crisis: str,
    simulaciones: int = 10000,
    nivel_confianza: float = 0.95
) -> dict:
    """Experimento Out-of-Sample:

    1. In-Sample: Entrena mu y covarianza con tu data_engine.py antes de la crisis.
    2. Predicción Monte Carlo: Genera el VaR 95% teórico.
    3. Out-of-Sample: Evalúa la caída real (Maximum Drawdown) durante la crisis.
    """
    # 1. In-Sample: usar tu función original de data_engine.py
    mu_diario, cov_diaria = obtener_parametros_cartera(tickers, fecha_inicio_entreno, fecha_fin_entreno)
    
    # Anualizar parámetros para Monte Carlo
    mu_anual = mu_diario * 252
    cov_anual = cov_diaria * 252
    
    # Precios reales durante la crisis
    precios_crisis = obtener_precios_crisis(tickers, fecha_inicio_crisis, fecha_fin_crisis)
    dias_crisis = len(precios_crisis)
    
    # 2. Predicción con Monte Carlo
    matriz_sim = simular_montecarlo_portfolio(
        capital_inicial, pesos, mu_anual, cov_anual, dias_crisis, simulaciones
    )
    metricas_predichas = calcular_var_cvar_montecarlo(
        matriz_sim, capital_inicial, nivel_confianza=nivel_confianza, tasa_libre_riesgo=0.02
    )
    
    # 3. Datos reales durante la crisis
    _, mdd_real_pct = calcular_drawdown_historico(precios_crisis, pesos, capital_inicial)
    
    var_predicho_pct = metricas_predichas['var_porcentaje']
    
    return {
        'dias_crisis': dias_crisis,
        'nivel_confianza': nivel_confianza, # <-- 3. Lo guardamos para el reporte
        'var_predicho_pct': metricas_predichas['var_porcentaje'],
        'var_predicho_usd': metricas_predichas['var_dolares'],
        'cvar_predicho_pct': metricas_predichas['cvar_porcentaje'],
        'cvar_predicho_usd': metricas_predichas['cvar_dolares'],
        'mdd_real_pct': mdd_real_pct,
        'mdd_real_usd': capital_inicial * (abs(mdd_real_pct) / 100),
        'modelo_fallo': abs(mdd_real_pct) > metricas_predichas['var_porcentaje']
    }




if __name__ == "__main__":
    tickers = ["AAPL", "MSFT", "GOOGL"]
    pesos = np.array([0.4, 0.3, 0.3])
    capital = 10000.0
    
    # Entrenamiento: 10 años normales antes del COVID
    # Crisis: Período del crash del COVID (Feb 2020 - Abr 2020)
    res = ejecutar_experimento_out_of_sample(
        tickers=tickers,
        pesos=pesos,
        capital_inicial=capital,
        fecha_inicio_entreno="2010-01-01",
        fecha_fin_entreno="2020-01-31",
        fecha_inicio_crisis="2020-02-01",
        fecha_fin_crisis="2020-04-30"
    )
    
    print(f"VaR 95% Predicho por Monte Carlo: {res['var_95_predicho_pct']:.2f}% (${res['var_95_predicho_usd']:,.2f})")
    print(f"Máxima Caída Real (MDD COVID): {res['mdd_real_pct']:.2f}% (${res['mdd_real_usd']:,.2f})")
    print(f"¿El modelo fue perforado por la realidad?: {'SÍ' if res['modelo_fallo'] else 'NO'}")