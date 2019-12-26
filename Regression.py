import os
import statistics as stat
import matplotlib.pyplot as plt

os.getcwd()
print(os.getcwd())

# Prepare data
file_object = open(os.getcwd() + "\gpa.txt", "r")
student_data = [line.strip().split(' ') for line in file_object.readlines()]

print("student data: ", student_data)


def get_column_data(column_index):
    column_data = []

    # For each entry in the "student_data" list
    for entry in student_data:
        # Append all data from column with given column index into this list
        column_data.append(float(entry[column_index]))

    # Return the sum of the data and a list containing it
    return column_data


print("num records: ", len(student_data))

# Fetch all the data from a column of a specific index
hs_col_data = get_column_data(0)
math_col_data = get_column_data(1)
verb_col_data = get_column_data(2)
cs_col_data = get_column_data(3)
uni_col_data = get_column_data(4)

# Sum all the GPA data for math and verbal GPA
math_sum = sum(math_col_data)
verb_sum = sum(verb_col_data)

# Results
print("Sum Math Grade: ", int(math_sum))
print("Sum Verbal Grade: ", int(verb_sum))
print("Math Mean: ", round(math_sum / len(student_data)))
print("Verbal Mean: ", round(verb_sum / len(student_data)))
print("Standard Deviation - Math: ", round(stat.stdev(math_col_data), 4))
print("Standard Deviation - Verb: ", round(stat.stdev(verb_col_data), 4))

# TODO: Draw a Scatter plot comparing the student's high school gpas to their overall GPAs.
plt.scatter(hs_col_data, uni_col_data)
#plt.plot(x_data, y_data, label="Second Line", color="#a3435d")

# Adding Labels
plt.xlabel("High School GPA")
plt.ylabel("Overall University GPA")

# Render buffer
plt.show()


# TODO: Find the correlation between high school GPA and overall university GPA.
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

    # STEP 6: Divide summed difference by n-1
    covariance = summed_xy_diff/(len(x)-1)
    return covariance


hs_and_uni_cov = calc_covariance(hs_col_data, uni_col_data, "HS Mean", "Overall Mean")
hs_uni_r = hs_and_uni_cov / (stat.stdev(hs_col_data) * stat.stdev(uni_col_data))
print("Correlation coefficient: ", round(hs_uni_r, 4))

# What would you expect the correlation between math and verbal SAT scores to be?
math_and_verb_cov = calc_covariance(math_col_data, verb_col_data, "Math Mean", "Verbal Mean")
math_and_verb_r = math_and_verb_cov / (stat.stdev(math_col_data) * stat.stdev(verb_col_data))
print("Correlation coefficient: ", round(math_and_verb_r, 4))