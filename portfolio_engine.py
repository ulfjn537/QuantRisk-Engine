import numpy as np
from inversion import Inversion
import data_engine
# en este código tengo que crear un portfolio obtener su rentambilidad y riesgo y hacer las simulaciones de montecarlo
# en primer llugar haremos una funcion que defina una cartera
#en segundo lugar dada una cartera hay que obtener la matriz de covarianza el retorno esperado y el riesgo
# y en tercer lugar hacer la factorizacion cholesky y la generacion de ruido correlacionado
# no haremos aquí la simulación montecarlo para dividir responsabilidades 

class Portfolio:
    # los atributos de un portfolio son self.activos, self.tickers, self.capital.total, self.pesos
    def __init__ (self, activos: list[Inversion]):
        if len(activos) == 0: 
            raise ValueError("la cartera debe tener elementos")
        self.activos = activos
        self.tickers = [] # creo una lista vacía
        self.capital_total = 0
        montos = []
        for inv in activos:
            self.tickers.append(inv.ticker)
            self.capital_total += inv.monto
            montos.append(inv.monto)

        self.pesos = np.array(montos) / self.capital_total

    def calcular_rentabilidad_esperada(self, fecha_inicio: str, fecha_fin: str):
        # Le pasas las fechas directamente a data_engine
        mu_vector, _ = data_engine.obtener_parametros_cartera(self.tickers, fecha_inicio, fecha_fin)  # Producto escalar: sum(w_i * mu_i)
        return float(np.dot(self.pesos, mu_vector))

    def calcular_riesgo(self, fecha_inicio: str, fecha_fin: str):
        _, matriz_cov = data_engine.obtener_parametros_cartera(self.tickers, fecha_inicio, fecha_fin)
        # Varianza = w^T * Sigma * w
        varianza = np.dot(self.pesos.T, np.dot(matriz_cov, self.pesos))
        # Desviación estándar (Volatilidad)
        return float(np.sqrt(varianza))

  