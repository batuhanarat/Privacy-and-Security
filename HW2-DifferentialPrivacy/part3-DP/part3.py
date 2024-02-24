import matplotlib.pyplot as plt
import numpy as np
import statistics
import pandas as pd
import math
import random
import csv

from numpy import sqrt, exp



''' Functions to implement '''

# TODO: Implement this function!
def read_dataset(file_path):
    dataset = pd.read_csv(file_path)
    return dataset


# TODO: Implement this function!
def get_histogram(dataset, state='TX', year='2020'):
    filtered_data = dataset.query("state == @state and date.str.contains(@year)")
    positives = filtered_data["positive"].to_list()
    #plt.bar(filtered_data["date"], positives)
    #plt.xticks(rotation=60)
    #plt.title('Positive Test Case for State ' + state + ' in year ' + year)
    #plt.show()
    return positives


# TODO: Implement this function!
def get_dp_histogram(dataset, state, year, epsilon, N):
    filtered_data = dataset.query(f"state == '{state}' and date.str.contains('{year}')")
    positives = filtered_data["positive"].values
    sensitivity = N
    laplace_noise = np.random.laplace(loc=0, scale=sensitivity / epsilon, size=len(positives))
    noisy_data = positives + laplace_noise
    #plt.bar(filtered_data["date"], noisy_data)
    #plt.xticks(rotation=60)
    #plt.title('Positive Test Case for State ' + state + ' in year ' + year)
    #plt.show()
    return noisy_data



# TODO: Implement this function!
def calculate_average_error(actual_hist, noisy_hist):
    hist_size = len(actual_hist)
    total_difference = 0
    for i in range(hist_size):
        difference = abs(actual_hist[i] - noisy_hist[i])
        total_difference += difference
    average_error = float(total_difference/ hist_size)
    return average_error

# TODO: Implement this function!
def epsilon_experiment(dataset, state, year, eps_values, N):
    total_epsilons = len(eps_values)
    positives = get_histogram(dataset, state, year)
    average_error = np.zeros(total_epsilons)

    for (i, epsilon) in enumerate(eps_values):
        cumulative_error = 0
        for j in range(10):
            noised = get_dp_histogram(dataset, state, year, epsilon, N)
            cumulative_error += calculate_average_error(positives, noised)
        average_error[i] = cumulative_error / 10

    return average_error



# TODO: Implement this function!
def N_experiment(dataset, state, year, epsilon, N_values):

    positives = get_histogram(dataset, state, year)
    average_error = np.zeros(len(N_values))

    for (i, N) in enumerate(N_values):
        cumulative_error = 0
        for j in range(10):
            noised = get_dp_histogram(dataset, state, year, epsilon, N)
            cumulative_error += calculate_average_error(positives, noised)
        average_error[i] = cumulative_error / 10
    return average_error


# FUNCTIONS FOR LAPLACE END #
# FUNCTIONS FOR EXPONENTIAL START #


# TODO: Implement this function!
def max_deaths_exponential(dataset, state, year, epsilon):
    sensitivity = 1 #not valid for çaycı hüseyin
    filtered_data = dataset.query("state == @state and date.str.contains(@year)")
    deaths = filtered_data["death"].to_numpy()
    exp_probs = []

    for count in deaths:
        prob = math.exp((epsilon * count) / (2 * sensitivity))
        exp_probs.append(prob)

    sum_probs = sum(exp_probs)
    normalized_probs = []
    for p in exp_probs:
        normalized_prob = p / sum_probs
        normalized_probs.append(normalized_prob)

    max_deaths = np.random.choice(len(deaths), p=normalized_probs)
    return max_deaths


# TODO: Implement this function!
def exponential_experiment(dataset, state, year, epsilon_list):
    accuracy = np.zeros(len(epsilon_list))
    filtered_data = dataset.query("state == @state and date.str.contains(@year)")
    deaths = filtered_data["death"].to_numpy()
    max_deaths_month = deaths.argmax(axis=0)

    for (i, epsilon) in enumerate(epsilon_list):
        for j in range(1000):
            random_month = max_deaths_exponential(dataset, state, year, epsilon)
            if max_deaths_month == random_month:
                accuracy[i] += 1

    accuracy = accuracy * 100 / 1000
    return accuracy



# FUNCTIONS TO IMPLEMENT END #


def main():
    filename = "covid19-states-history.csv"
    dataset = read_dataset(filename)

    state = "TX"
    year = "2020"
    positives = get_histogram(dataset, state, year)

    print("**** LAPLACE EXPERIMENT RESULTS ****")
    eps_values = [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 1.0]
    error_avg = epsilon_experiment(dataset, state, year, eps_values, 2)
    for i in range(len(eps_values)):
        print("eps = ", eps_values[i], " error = ", error_avg[i])


    print("**** N EXPERIMENT RESULTS ****")
    N_values = [1, 2, 4, 8]
    error_avg = N_experiment(dataset, state, year, 0.5, N_values)
    for i in range(len(N_values)):
        print("N = ", N_values[i], " error = ", error_avg[i])

    state = "WY"
    year = "2020"

    print("**** EXPONENTIAL EXPERIMENT RESULTS ****")
    eps_values = [0.0001, 0.001, 0.01, 0.05, 0.1, 1.0]
    exponential_experiment_result = exponential_experiment(dataset, state, year, eps_values)
    for i in range(len(eps_values)):
        print("eps = ", eps_values[i], " accuracy = ", exponential_experiment_result[i])




if __name__ == "__main__":
    main()
