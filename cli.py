import numpy as np

# Importaciones de los módulos del proyecto
from data_engine import obtener_parametros_cartera
from monte_carlo import simular_montecarlo_portfolio
from risk_metrics import calcular_var_cvar_montecarlo
from visualizer import graficar_comparativa_carteras
from backtesting import ejecutar_experimento_out_of_sample


def solicitar_cartera_real(nombre_cartera: str) -> tuple[list[str], np.ndarray, float]:
    """Solicita los tickers y la inversión en dinero para cada activo de la cartera."""
    print(f"\n" + "=" * 60)
    print(f" 📌 CONFIGURACIÓN DE {nombre_cartera.upper()}")
    print("=" * 60)

    tickers_input = (
        input(" Introduce los tickers de las acciones (ej: AAPL MSFT NVDA): ")
        .strip()
        .upper()
    )
    tickers = tickers_input.split()

    montos = []
    print("\n Introduce el dinero invertido en cada activo:")
    for ticker in tickers:
        monto = float(input(f"   • Cantidad invertida en {ticker} ($/€): "))
        montos.append(monto)

    capital_total = sum(montos)
    pesos = np.array(montos) / capital_total

    print(f"\n Resumen {nombre_cartera}: Capital Total = ${capital_total:,.2f}")
    for t, p, m in zip(tickers, pesos, montos):
        print(f"   - {t:<6}: ${m:>10,.2f} ({p * 100:>5.1f}%)")

    return tickers, pesos, capital_total


def imprimir_reporte_comparativo(m_a: dict, m_b: dict, cap_a: float, cap_b: float, dias: int, sims: int):
    """Muestra el panel formateado comparando ambas carteras."""
    print("\n" + "=" * 78)
    print(" 📊 INFORME COMPARATIVO DE INVERSIÓN & RIESGO DE COLA")
    print("=" * 78)
    print(f" Horizonte: {dias} días | Simulaciones: {sims:,}")
    print(f" Capital Cartera A: ${cap_a:,.2f} | Capital Cartera B: ${cap_b:,.2f}")
    print("-" * 78)

    print(f"{'Métrica':<34} | {'Cartera A':<18} | {'Cartera B':<18}")
    print("-" * 78)

    # Expectativa
    print(" [1] EXPECTATIVA DE RETORNO")
    print(f"   • Capital Medio Final ($)        | ${m_a['media_final']:>16,.2f} | ${m_b['media_final']:>16,.2f}")
    print(f"   • Retorno Esperado (%)           | {m_a['retorno_medio_pct']:>16.2f}% | {m_b['retorno_medio_pct']:>16.2f}%")
    print("-" * 78)

    # Riesgo Extremo
    print(" [2] RIESGO DE COLA (PEORES ESCENARIOS)")
    print(f"   • VaR 95% ($ Máx Pérdida)        | ${m_a['var_dolares']:>16,.2f} | ${m_b['var_dolares']:>16,.2f}")
    print(f"   • VaR 95% (% Pérdida)            | {m_a['var_porcentaje']:>16.2f}% | {m_b['var_porcentaje']:>16.2f}%")
    print(f"   • CVaR 95% ($ Pérdida Media)     | ${m_a['cvar_dolares']:>16,.2f} | ${m_b['cvar_dolares']:>16,.2f}")
    print(f"   • CVaR 95% (% Pérdida Media)     | {m_a['cvar_porcentaje']:>16.2f}% | {m_b['cvar_porcentaje']:>16.2f}%")
    print(f"   • Probabilidad de Pérdida        | {m_a['prob_perdida']:>16.2f}% | {m_b['prob_perdida']:>16.2f}%")
    print("-" * 78)

    # Eficiencia
    print(" [3] EFICIENCIA (RETORNO / RIESGO)")
    print(f"   • Ratio de Sharpe (Volatilidad)  | {m_a['sharpe_ratio']:>16.2f} | {m_b['sharpe_ratio']:>16.2f}")
    print(f"   • Ratio STARR (Riesgo CVaR)      | {m_a['starr_ratio']:>16.2f} | {m_b['starr_ratio']:>16.2f}")
    print("=" * 78)


def imprimir_reporte_backtesting(res: dict):
    """Muestra el panel de resultados de la prueba Out-of-Sample."""
    conf_pct = res.get('nivel_confianza', 0.95) * 100

    print("\n" + "=" * 78)
    print(" 📉 INFORME DE PRUEBA DE ESTRÉS OUT-OF-SAMPLE (BACKTESTING DE CRISIS)")
    print("=" * 78)
    print(f" Duración de la ventana de crisis analizada: {res['dias_crisis']} días bursátiles")
    print("-" * 78)
    print(f"{'Métrica de Riesgo':<42} | {'Resultado Calculado':<30}")
    print("-" * 78)

    print(f" [1] PREDICCIÓN MONTE CARLO (IN-SAMPLE)")
    print(f"   • VaR {conf_pct:.0f}% (% Máx Pérdida Estimada)    | {res['var_predicho_pct']:>28.2f}%")
    print(f"   • VaR {conf_pct:.0f}% ($ Máx Pérdida Estimada)    | ${res['var_predicho_usd']:>27,.2f}")
    print(f"   • CVaR {conf_pct:.0f}% (% Pérdida Media Cola)     | {res['cvar_predicho_pct']:>28.2f}%")
    print(f"   • CVaR {conf_pct:.0f}% ($ Pérdida Media Cola)     | ${res['cvar_predicho_usd']:>27,.2f}")
    print("-" * 78)

    print(" [2] IMPACTO REAL EN CRISIS (OUT-OF-SAMPLE)")
    print(f"   • Maximum Drawdown Real (% Caída Máx) | {res['mdd_real_pct']:>28.2f}%")
    print(f"   • Maximum Drawdown Real ($ Caída Máx) | ${res['mdd_real_usd']:>27,.2f}")
    print("-" * 78)

    print(" [3] DIAGNÓSTICO DE FIABILIDAD DEL MODELO")
    estado = "❌ FALLO / MODELO PERFORADO" if res['modelo_fallo'] else "✅ DENTRO DEL MARGEN ESTIMADO"
    print(f"   • Evaluación del VaR Teórico         | {estado:>30}")
    print("=" * 78)

    if res['modelo_fallo']:
        print(" ⚠️  CONCLUSIÓN: La caída real durante la crisis superó la pérdida máxima")
        print("     estimada por el VaR 95%. Esto demuestra empíricamente que la distribución")
        print("     Normal subestima el riesgo extremo en momentos de pánico bursátil.")
    else:
        print(" ℹ️  CONCLUSIÓN: La pérdida real se mantuvo contenida dentro del límite")
        print("     teórico estimado por el modelo de Monte Carlo.")
    print("=" * 78)


def ejecutar_comparador_carteras():
    """Ejecuta el análisis comparativo Monte Carlo entre dos carteras."""
    dias = int(input("\n 📅 Días a simular [Default 252]: ") or 252)
    sims = int(input(" 🎲 Número de simulaciones [Default 10000]: ") or 10000)

    fecha_inicio = input(" 📅 Fecha inicio datos históricos (YYYY-MM-DD) [Default: 2022-01-01]: ").strip() or "2022-01-01"
    fecha_fin = input(" 📅 Fecha fin datos históricos (YYYY-MM-DD) [Default: 2024-01-01]: ").strip() or "2024-01-01"

    tickers_a, pesos_a, capital_a = solicitar_cartera_real("Cartera A")
    tickers_b, pesos_b, capital_b = solicitar_cartera_real("Cartera B")

    print("\n⏳ Descargando datos con data_engine.py y calibrando parámetros...")
    mu_diario_a, cov_diaria_a = obtener_parametros_cartera(tickers_a, fecha_inicio, fecha_fin)
    mu_diario_b, cov_diaria_b = obtener_parametros_cartera(tickers_b, fecha_inicio, fecha_fin)

    mu_a, cov_a = mu_diario_a * 252, cov_diaria_a * 252
    mu_b, cov_b = mu_diario_b * 252, cov_diaria_b * 252

    print("⏳ Ejecutando simulaciones estocásticas de Monte Carlo...")
    matriz_a = simular_montecarlo_portfolio(capital_a, pesos_a, mu_a, cov_a, dias, sims)
    metrics_a = calcular_var_cvar_montecarlo(matriz_a, capital_a, nivel_confianza=0.95, tasa_libre_riesgo=0.02)

    matriz_b = simular_montecarlo_portfolio(capital_b, pesos_b, mu_b, cov_b, dias, sims)
    metrics_b = calcular_var_cvar_montecarlo(matriz_b, capital_b, nivel_confianza=0.95, tasa_libre_riesgo=0.02)

    imprimir_reporte_comparativo(metrics_a, metrics_b, capital_a, capital_b, dias, sims)

    graficar = input("\n📊 ¿Deseas desplegar las gráficas comparativas? (s/n) [Default: s]: ").strip().lower() or "s"
    if graficar == "s":
        graficar_comparativa_carteras(
            matriz_a, matriz_b, capital_a, metrics_a["var_dolares"], metrics_b["var_dolares"]
        )


def ejecutar_menu_backtesting():
    """Ejecuta el experimento de estrés Out-of-Sample / Backtesting."""
    tickers, pesos, capital = solicitar_cartera_real("Cartera para Prueba de Crisis")

    # Solicitamos el nivel de confianza (convertimos de ej. 99 a 0.99)
    conf_input = input("\n 🎯 Nivel de confianza % [Default 95]: ").strip() or "95"
    nivel_confianza = float(conf_input) / 100.0

    print("\n--- ⚙️ VENTANAS TEMPORALES DEL EXPERIMENTO ---")
    print("1. Período In-Sample (Entrenamiento en tiempos normales)")
    fecha_in_inicio = input("   📅 Inicio [Default: 2010-01-01]: ").strip() or "2010-01-01"
    fecha_in_fin = input("   📅 Fin    [Default: 2020-01-31]: ").strip() or "2020-01-31"

    print("\n2. Período Out-of-Sample (Crisis Bursátil Real)")
    fecha_out_inicio = input("   📅 Inicio Crisis [Default: 2020-02-01]: ").strip() or "2020-02-01"
    fecha_out_fin = input("   📅 Fin Crisis    [Default: 2020-04-30]: ").strip() or "2020-04-30"

    sims = int(input("\n🎲 Número de simulaciones [Default 10000]: ") or 10000)

    print("\n⏳ Ejecutando experimento Out-of-Sample y descargando datos históricos...")
    res = ejecutar_experimento_out_of_sample(
        tickers=tickers,
        pesos=pesos,
        capital_inicial=capital,
        fecha_inicio_entreno=fecha_in_inicio,
        fecha_fin_entreno=fecha_in_fin,
        fecha_inicio_crisis=fecha_out_inicio,
        fecha_fin_crisis=fecha_out_fin,
        simulaciones=sims,
        nivel_confianza=nivel_confianza
    )

    imprimir_reporte_backtesting(res)


def main():
    while True:
        print("\n==================================================================")
        print("   JanusRisk CLI v2.0 - Sistema Integrado de Análisis de Riesgo")
        print("==================================================================")
        print(" 1. Comparar 2 Carteras Reales con Monte Carlo")
        print(" 2. Experimento Out-of-Sample / Backtesting (Estrés de Crisis)")
        print(" 3. Salir")
        
        opcion = input("\n Selecciona una opción (1-3): ").strip()
        
        if opcion == "1":
            ejecutar_comparador_carteras()
        elif opcion == "2":
            ejecutar_menu_backtesting()
        elif opcion == "3":
            print("\n Saliendo del sistema JanusRisk... ¡Buen trabajo con el TFG!\n")
            break
        else:
            print(" ❌ Opción inválida. Elige un número del 1 al 3.")


if __name__ == "__main__":
    main()