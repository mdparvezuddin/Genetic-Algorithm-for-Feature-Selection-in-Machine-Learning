# Genetic-Algorithm-for-Feature-Selection-in-Machine-Learning
# Install necessary packages
!pip install deap scikit-learn numpy pandas matplotlib seaborn

# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from time import time
from deap import base, creator, tools, algorithms
import random

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
num_features = X.shape[1]

# Fitness function for GA
def evalFeatureSet(individual):
    if sum(individual) == 0:
        return 0,
    selected_indices = [i for i in range(len(individual)) if individual[i] == 1]
    X_train_sel = X_train[:, selected_indices]
    X_test_sel = X_test[:, selected_indices]
    clf = KNeighborsClassifier()
    clf.fit(X_train_sel, y_train)
    predictions = clf.predict(X_test_sel)
    accuracy = accuracy_score(y_test, predictions)
    return accuracy,

# GA setup
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)
toolbox = base.Toolbox()
toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=num_features)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evalFeatureSet)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

# Store results
results = []

# 1. No Feature Selection
start = time()
clf = KNeighborsClassifier()
clf.fit(X_train, y_train)
acc = accuracy_score(y_test, clf.predict(X_test))
end = time()
results.append(["No Feature Selection", 4, acc*100, round(end - start, 2)])

# 2. Correlation-Based
start = time()
df = pd.DataFrame(X, columns=feature_names)
cor_matrix = df.corr().abs()
upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
df_reduced = df.drop(to_drop, axis=1)
X_new = df_reduced.values
X_train_corr, X_test_corr, _, _ = train_test_split(X_new, y, test_size=0.3, random_state=42)
clf = KNeighborsClassifier()
clf.fit(X_train_corr, y_train)
acc = accuracy_score(y_test, clf.predict(X_test_corr))
end = time()
results.append(["Correlation-Based", X_new.shape[1], acc*100, round(end - start, 2)])

# 3. RFE
start = time()
model = LogisticRegression(max_iter=1000)
rfe = RFE(model, n_features_to_select=2)
rfe.fit(X, y)
X_rfe = X[:, rfe.support_]
X_train_rfe, X_test_rfe, _, _ = train_test_split(X_rfe, y, test_size=0.3, random_state=42)
clf = KNeighborsClassifier()
clf.fit(X_train_rfe, y_train)
acc = accuracy_score(y_test, clf.predict(X_test_rfe))
end = time()
results.append(["Recursive Feature Elimination", 2, acc*100, round(end - start, 2)])

# 4. Genetic Algorithm
start = time()
population = toolbox.population(n=20)
algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=30, verbose=False)
best_ind = tools.selBest(population, 1)[0]
selected_idx = [i for i in range(len(best_ind)) if best_ind[i] == 1]
X_train_ga = X_train[:, selected_idx]
X_test_ga = X_test[:, selected_idx]
clf = KNeighborsClassifier()
clf.fit(X_train_ga, y_train)
acc = accuracy_score(y_test, clf.predict(X_test_ga))
end = time()
results.append(["Genetic Algorithm", len(selected_idx), acc*100, round(end - start, 2)])

# Results DataFrame
df_results = pd.DataFrame(results, columns=["Method", "Selected Features", "Accuracy (%)", "Time (s)"])
print(df_results)

# === Visualization (Bar Charts Only) ===

# Accuracy Chart
plt.figure(figsize=(8, 5))
sns.barplot(x="Method", y="Accuracy (%)", data=df_results, palette="Blues_d")
plt.title("Comparison of Accuracy")
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()

# Time Chart
plt.figure(figsize=(8, 5))
sns.barplot(x="Method", y="Time (s)", data=df_results, palette="Oranges_d")
plt.title("Comparison of Computation Time")
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()
