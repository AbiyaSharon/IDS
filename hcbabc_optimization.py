import numpy as np
import random
from sklearn.metrics import accuracy_score
# -------------------------------
# Chaotic Maps Function
# -------------------------------
def chaotic_map(iteration, max_iter, map_type="logistic"):
    if map_type == "logistic":
        x = 0.7  # Initial value
        r = 4    # Parameter for the logistic map
        for _ in range(iteration):
            x = r * x * (1 - x)
        return x
    # Add other chaotic maps if needed
# -------------------------------
# Fitness Function
# -------------------------------
def fitness_function(solution, X_train, y_train, X_test, y_test, model_builder):
    # This function evaluates the solution by training the model with given hyperparameters
    # model_builder should be a function that builds the model using the hyperparameters in 'solution'
    # Example: solution = [learning_rate, num_units, batch_size]
    model = model_builder(solution)
    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=int(solution[2]), verbose=0)
    # Predict and calculate accuracy
    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred, axis=1)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy
# -------------------------------
# Initialization
# -------------------------------
def initialize_population(pop_size, bounds):
    population = []
    for _ in range(pop_size):
        individual = [random.uniform(bounds[i][0], bounds[i][1]) for i in range(len(bounds))]
        population.append(individual)
    return population
# -------------------------------
# Hybrid Chaotic Bat ABC Optimization
# -------------------------------
def HCBABC(X_train, y_train, X_test, y_test, model_builder, bounds, pop_size=20, max_iter=100):
    # Initialize parameters for Bat Algorithm
    f_min, f_max = 0, 2  # Frequency range for Bat Algorithm
    loudness = np.ones(pop_size) * 0.5
    pulse_rate = np.ones(pop_size) * 0.5
    velocities = np.zeros((pop_size, len(bounds)))
    # Initialize population
    population = initialize_population(pop_size, bounds)
    fitness = [fitness_function(ind, X_train, y_train, X_test, y_test, model_builder) for ind in population]
    best_solution = population[np.argmax(fitness)]
    best_fitness = max(fitness)
    # Main optimization loop
    for iteration in range(max_iter):
        for i in range(pop_size):
            # Bat Algorithm Phase
            freq = f_min + (f_max - f_min) * chaotic_map(iteration, max_iter)
            velocities[i] += (population[i] - best_solution) * freq
            new_solution = population[i] + velocities[i]
            # Apply boundaries
            new_solution = np.clip(new_solution, [b[0] for b in bounds], [b[1] for b in bounds])
            # Evaluate new solution
            if random.random() > pulse_rate[i]:
                new_fitness = fitness_function(new_solution, X_train, y_train, X_test, y_test, model_builder)
                # Update if the new solution is better
                if new_fitness > fitness[i] and random.random() < loudness[i]:
                    population[i] = new_solution
                    fitness[i] = new_fitness
            # Artificial Bee Colony Phase
            # Employeed bee phase
            for bee in range(pop_size):
                phi = chaotic_map(iteration, max_iter)  # Random search factor using chaotic maps
                k = random.choice(range(pop_size))  # Random index
                candidate_solution = population[bee] + phi * (population[bee] - population[k])
                candidate_solution = np.clip(candidate_solution, [b[0] for b in bounds], [b[1] for b in bounds])
                candidate_fitness = fitness_function(candidate_solution, X_train, y_train, X_test, y_test, model_builder)
                if candidate_fitness > fitness[bee]:
                    population[bee] = candidate_solution
                    fitness[bee] = candidate_fitness
            # Onlooker bee phase
            total_fitness = sum(fitness)
            for bee in range(pop_size):
                prob = fitness[bee] / total_fitness
                if random.random() < prob:
                    k = random.choice(range(pop_size))
                    phi = chaotic_map(iteration, max_iter)
                    candidate_solution = population[bee] + phi * (population[bee] - population[k])
                    candidate_solution = np.clip(candidate_solution, [b[0] for b in bounds], [b[1] for b in bounds])
                    candidate_fitness = fitness_function(candidate_solution, X_train, y_train, X_test, y_test, model_builder)
                    if candidate_fitness > fitness[bee]:
                        population[bee] = candidate_solution
                        fitness[bee] = candidate_fitness
            # Scout bee phase
            for bee in range(pop_size):
                if random.random() < chaotic_map(iteration, max_iter) * 0.05:
                    population[bee] = [random.uniform(bounds[i][0], bounds[i][1]) for i in range(len(bounds))]
                    fitness[bee] = fitness_function(population[bee], X_train, y_train, X_test, y_test, model_builder)
        # Update the global best solution
        best_idx = np.argmax(fitness)
        if fitness[best_idx] > best_fitness:
            best_fitness = fitness[best_idx]
            best_solution = population[best_idx]
        print(f"Iteration {iteration + 1}/{max_iter}, Best Fitness: {best_fitness}")
    return best_solution, best_fitness
# -------------------------------
# Example of Usage
# -------------------------------
# Define bounds for hyperparameters (e.g., learning rate, number of units, batch size)
bounds = [(0.0001, 0.001), (16, 128), (64, 128)]  # Adjust based on the model's hyperparameters
# Example function to build the model (to be passed to the optimizer)
def model_builder(hyperparameters,X_train):
    # Example: create a simple neural network model using hyperparameters
    from tensorflow.keras import models, layers
    learning_rate, num_units, batch_size = hyperparameters
    model = models.Sequential()
    model.add(layers.Dense(int(num_units), activation='relu', input_shape=(X_train.shape[1],)))
    model.add(layers.Dense(10, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model
# # Run the HCBABC optimization
# best_solution, best_fitness = HCBABC(X_train, y_train, X_test, y_test, model_builder, bounds)
# print("Best Hyperparameters:", best_solution)
# print("Best Fitness (Accuracy):", best_fitness)