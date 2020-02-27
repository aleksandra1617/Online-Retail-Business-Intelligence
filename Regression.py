import statistics as stat
import matplotlib.pyplot as plt
import numpy as np


def get_column_data(column_index, dataset):
    column_data = []

    # For each entry in the "student_data" list
    for entry in dataset:
        # Append all data from column with given column index into this list
        column_data.append(float(entry[column_index]))

    # Return the sum of the data and a list containing it
    return column_data


def calc_covariance(x, y, x_label="X mean", y_label="Y mean"):
    summed_xy_diff = 0

    # Calculate the mean for both x and y
    x_mean = sum(x) / len(x)
    print(x_label, ": ", sum(x) / len(x))

    # Mean calc using numpy
    y_mean = sum(y) / len(y)
    print(y_label, ": ", sum(y) / len(y))

    for count in range(len(x)):
        # Calculate the difference between the current value and the attribute mean
        x_diff = x[count] - x_mean
        y_diff = y[count] - y_mean

        # Multiply both differences
        xy_diff = x_diff * y_diff

        # Sum the xy difference for all pairs in the data set
        summed_xy_diff += xy_diff

    # Divide summed difference by n-1
    covariance = summed_xy_diff/(len(x)-1)
    return covariance


# LINEAR REGRESSION
# Examples: Trying to predict home prices or number of T-shirts to produce given some input set of features.
def calc_lin_regression(x_data, y_data, given_x_point, xy_corr, graph_desc):
    x_mean = sum(x_data) / len(x_data)
    y_mean = sum(y_data) / len(y_data)

    # Find A and B with the respective formulas
    a = (xy_corr * stat.stdev(y_data)) / stat.stdev(x_data)
    b = y_mean - (a * x_mean)

    print("a: ", a)
    print("b: ", b)

    # Predict Y with linear regression formula
    y_point = a * given_x_point + b
    print("Y point: ", y_point)

    # Cast all data from string to float
    x_data = [float(element) for element in x_data]

    plt.scatter(x_data, y_data)
    y_pre = np.multiply(a, x_data) + b
    plt.plot(x_data, y_pre, label="Line", color="#a3435d")

    # Plot the point onto the regression line
    plt.plot(given_x_point, y_point, 'r*')
    plt.xlabel(graph_desc[0])
    plt.ylabel(graph_desc[1])
    plt.title(graph_desc[2])

    plt.show()
