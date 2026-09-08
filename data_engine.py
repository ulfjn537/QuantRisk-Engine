import yfinance as yf
import numpy as np
import pandas as pd

def descargar_datos(ticker: str, fecha_inicio: str, fecha_fin: str):
    datos = yf.download(ticker, start=fecha_inicio, end=fecha_fin)
    return datos["Close"].squeeze() #nos devuelve solo la columna con los precios de cierre, con squeeze obligamos a convertirlo en una serie simple de dim1
# nos devuelve un array de precios en el que cada uno viene asociado a su fecha

def calcular_estadisticas(precios: pd.Series): 
    retornos = np.log(precios / precios.shift(1)).dropna() # .dropna() elimina los valores faltantes en el primer elemento de la serie
    mu = float(retornos.mean())
    sigma = float(retornos.std())
    return mu, sigma

def obtener_parametros_riesgo(ticker:str, fecha_inicio:str , fecha_fin:str ):
  datos_precios = descargar_datos(ticker, fecha_inicio, fecha_fin)
  mu , sigma = calcular_estadisticas(datos_precios)
  if sigma != 0.0:
    sharpe_ratio = mu / sigma
  else: sharpe_ratio = 0.0
  return mu, sigma, sharpe_ratio

def obtener_parametros_cartera(tickers: list[str], fecha_inicio: str, fecha_fin: str):
    # Descarga todos los precios de la cartera en una sola consulta
    # es un dataframe de pandas, una tabla con los tickers en las columnas y con fechas en las filas
    precios = yf.download(tickers, start=fecha_inicio, end=fecha_fin)["Close"]
    
    # Calcula retornos logarítmicos continuos
    # es un dataframe de pandas con tickers como columnas y diás como filas (se elimina el primer dia) y la informacion son los retornos en porcentajes de cada día
    retornos = np.log(precios / precios.shift(1)).dropna()
    
    # Extrae el vector mu (N,) y la matriz de covarianza Sigma (N x N)
    mu_vector = retornos.mean().values
    matriz_cov = retornos.cov().values
    # .values pasa de tipo panda a tipo numpy, siendo esto arrays de numpy
    # .cov() es una funcion de pandas que da la matriz de covarianzas directamente
    
    return mu_vector, matriz_cov


      
      
      
