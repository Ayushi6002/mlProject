import dill

with open("artifacts/preprocessor.pkl", "rb") as f:
    preprocessor = dill.load(f)

print(preprocessor)
