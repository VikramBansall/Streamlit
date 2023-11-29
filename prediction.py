import joblib


def predict(df):
    model = joblib.load("rf_model.sav")
    return model.predict(df)
