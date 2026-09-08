from data_engine import obtener_parametros_riesgo, descargar_datos
from monte_carlo import simular_monte_carlo
from risk_metrics import calcular_var_cvar


def test_pipeline_completo_riesgo():
    # 1. Configuración del escenario de prueba
    ticker = "AAPL"
    fecha_inicio = "2016-01-01"
    fecha_fin = "2026-01-01"
    dias_futuros = 365  # 1 año bursátil
    num_simulaciones = 5000
    capital_invertido = 10000  # Ejemplo: Si invertimos $10,000 USD

    print(f"\n==================================================")
    print(f" 🚀 TEST DE INTEGRACIÓN: MOTOR DE RIESGO DE MONTE CARLO")
    print(f"==================================================")
    print(f"Activo: {ticker} | Periodo Histórico: {fecha_inicio} a {fecha_fin}")

    # 2. Ingesta de datos
    precios = descargar_datos(ticker, fecha_inicio, fecha_fin)
    precio_actual = float(precios.iloc[-1]) #selecciona y devuelve el último elemento (la última fila) de la serie o tabla de datos precios.
    mu, sigma, _ = obtener_parametros_riesgo(ticker, fecha_inicio, fecha_fin)

    print(f"Precio Actual (S0): ${precio_actual:.2f}")
    print(f"Parámetros Estimados -> Mu (Drift): {mu:.6f} | Sigma (Volatilidad): {sigma:.6f}")

    # 3. Simulación de Monte Carlo
    matriz_sims = simular_monte_carlo(precio_actual, mu, sigma, dias_futuros, num_simulaciones)

    # 4. Cálculo de métricas de riesgo (al 95% de confianza)
    metricas = calcular_var_cvar( matriz_sims, precio_inicial=precio_actual, nivel_confianza=0.95 )

    # Escalamos el VaR/CVaR en USD a nuestro portafolio de $10,000
    var_mi_capital = metricas["var_pct"] * capital_invertido
    cvar_mi_capital = metricas["cvar_pct"] * capital_invertido

    # 5. Visualización clara de métricas
    print(f"\n--------------------------------------------------")
    print(f" 📊 INFORME DE RIESGO A 1 AÑO (Confianza 95%)")
    print(f"--------------------------------------------------")
    print(
        f"• VaR %    : {metricas['var_pct']*100:>6.2f}%  --> Caída máxima esperada el 95% de las veces."
    )
    print(
        f"• VaR USD  : ${metricas['var_usd']:>6.2f}  --> Pérdida por cada acción de ${precio_actual:.2f}."
    )
    print(
        f"• CVaR %   : {metricas['cvar_pct']*100:>6.2f}%  --> Pérdida PROMEDIO si entramos en el peor 5% de casos."
    )
    print(
        f"• CVaR USD : ${metricas['cvar_usd']:>6.2f}  --> Pérdida promedio por acción en la cola extrema."
    )

    print(f"\n💡 IMPACTO REAL EN TU PORTAFOLIO DE ${capital_invertido:,.2f} USD:")
    print(f"   - Umbral de Riesgo (VaR 95%)  : Te arriesgas a perder ${var_mi_capital:,.2f}")
    print(f"   - Escenario Catastrófico (CVaR): Perderías en promedio ${cvar_mi_capital:,.2f}")
    print(f"==================================================\n")


if __name__ == "__main__":
    test_pipeline_completo_riesgo()
