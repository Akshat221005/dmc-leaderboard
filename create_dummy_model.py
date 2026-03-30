import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression

# fake training data
X = np.random.rand(10, 224*224*3)
y = np.random.randint(0, 2, 10)

model = LogisticRegression()
model.fit(X, y)

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Dummy model created")