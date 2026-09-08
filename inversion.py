class Inversion:
    def __init__(self, ticker: str, monto: float):
        # Validaciones
        if monto <= 0:
            raise ValueError("El monto debe ser mayor a 0")
        
        self.ticker = ticker
        self.monto = monto

    def get_ticker(Inversion):
        return Inversion.ticker
    def get_monto(Inversion):
        return Inversion.monto

    