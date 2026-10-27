import joblib

model = joblib.load("models/random_forest.joblib")

print("Árvores:", len(model.estimators_))
print("Classes:", model.classes_)