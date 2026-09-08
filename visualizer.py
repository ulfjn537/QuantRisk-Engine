import numpy as np
import matplotlib.pyplot as plt

def graficar_comparativa_carteras(
    matriz_a: np.ndarray, 
    matriz_b: np.ndarray, 
    capital_inicial: float,
    var_a: float,
    var_b: float
):
    """
    Genera una figura con dos subgráficos:
    1. Trayectorias simuladas a lo largo del tiempo.
    2. Histogramas comparativos del valor final de las carteras con sus lineas de VaR.
    """
    plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    dias = matriz_a.shape[0] - 1
    eje_tiempo = np.arange(dias + 1)
    
    # --- SUBGRÁFICO 1: Trayectorias de las Simulaciones ---
    # Dibujamos las primeras 100 rutas para no saturar la gráfica
    num_rutas = min(100, matriz_a.shape[1])
    
    ax1.plot(eje_tiempo, matriz_a[:, :num_rutas], color='navy', alpha=0.05)
    ax1.plot(eje_tiempo, matriz_b[:, :num_rutas], color='crimson', alpha=0.05)
    
    # Trayectorias medias (líneas gruesas)
    ax1.plot(eje_tiempo, np.mean(matriz_a, axis=1), color='blue', linewidth=2.5, label='Cartera A (Media)')
    ax1.plot(eje_tiempo, np.mean(matriz_b, axis=1), color='red', linewidth=2.5, label='Cartera B (Media)')
    ax1.axhline(capital_inicial, color='black', linestyle='--', label='Capital Inicial')
    
    ax1.set_title('Trayectorias de Monte Carlo (Muestra de 100 rutas)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Días de Operación')
    ax1.set_ylabel('Valor de la Cartera ($)')
    ax1.legend(loc='upper left')
    
    # --- SUBGRÁFICO 2: Distribución Final y VaR ---
    precios_a = matriz_a[-1, :]
    precios_b = matriz_b[-1, :]
    
    ax2.hist(precios_a, bins=50, alpha=0.5, color='blue', label='Cartera A', density=True)
    ax2.hist(precios_b, bins=50, alpha=0.5, color='red', label='Cartera B', density=True)
    
    # Líneas verticales indicando el umbral de VaR 95%
    corte_var_a = capital_inicial - var_a
    corte_var_b = capital_inicial - var_b
    
    ax2.axvline(corte_var_a, color='navy', linestyle=':', linewidth=2, label=f'VaR A: ${var_a:,.0f}')
    ax2.axvline(corte_var_b, color='darkred', linestyle=':', linewidth=2, label=f'VaR B: ${var_b:,.0f}')
    
    ax2.set_title('Distribución de Valor Final & Umbrales de VaR (95%)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Valor Final ($)')
    ax2.set_ylabel('Densidad de Probabilidad')
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    plt.show()
