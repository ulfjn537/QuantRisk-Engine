import numpy as np
from inversion import Inversion
from portfolio_engine import Portfolio

def probar_cartera():
    print("--- 🧪 INICIANDO PRUEBAS DE PORTFOLIO ENGINE ---")

    # 1. Crear las inversiones individuales
    inv1 = Inversion(ticker="AAPL", monto=5000.0)
    inv2 = Inversion(ticker="MSFT", monto=3000.0)
    inv3 = Inversion(ticker="GOOGL", monto=2000.0)

    activos = [inv1, inv2, inv3]

    # 2. Instanciar la cartera
    cartera = Portfolio(activos)

    # 3. Comprobar atributos base
    print(f"\n[1] Capital Total Esperado: $10000.0 | Calculado: ${cartera.capital_total}")
    print(f"[2] Tickers en la cartera: {cartera.tickers}")
    print(f"[3] Pesos asignados (w): {cartera.pesos}")
    
    # Validar que la suma de pesos sea exactamente 100% (1.0)
    assert np.isclose(np.sum(cartera.pesos), 1.0), "⚠️ Error: La suma de pesos debe ser igual a 1.0"
    print("    └─ Suma de pesos verificada (100%).")

    # 4. Probar la integración con data_engine y cálculos financieros
    fecha_inicio = "2023-01-01"
    fecha_fin = "2024-01-01"

    print(f"\n[4] Consultando data_engine ({fecha_inicio} a {fecha_fin})...")
    
    try:
        mu_p = cartera.calcular_rentabilidad_esperada(fecha_inicio, fecha_fin)
        sigma_p = cartera.calcular_riesgo(fecha_inicio, fecha_fin)

        print("\n--- 📊 RESULTADOS OBTENIDOS ---")
        print(f"• Rentabilidad Esperada Diaria (μ_p): {mu_p:.6f} ({mu_p * 100:.4f}%)")
        print(f"• Riesgo / Volatilidad Diaria (σ_p): {sigma_p:.6f} ({sigma_p * 100:.4f}%)")
        print(f"• Volatilidad Anualizada Aprox (σ_p * √252): {sigma_p * np.sqrt(252) * 100:.2f}%")
        print("\n✅ ¡TODAS LAS PRUEBAS PASARON CON ÉXITO!")

    except Exception as e:
        print(f"\n❌ ERROR DURANTE LA EJECUCIÓN: {e}")

if __name__ == "__main__":
    probar_cartera()
