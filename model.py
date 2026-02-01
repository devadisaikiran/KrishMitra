# model.py

class RFXGEnsemble:
    def __init__(self, rf_model, xgb_model):
        self.rf = rf_model
        self.xgb = xgb_model

    def predict(self, X):
        rf_pred = self.rf.predict(X)
        xgb_pred = self.xgb.predict(X)
        return (rf_pred + xgb_pred) / 2
