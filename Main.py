# NEXT STEPS
# TODO Create a 3D plot to apply clustering including all 3 customer features - MonetaryValue, Frequency and Recency
#      A VIP customer would be a part of the VIP cluster, best values for the customer features.
# TODO: Create an infographic and host it on a web page using Django.
# TODO: Gather live data from news articles (Can using NLTK, Urllib and BeautifulSoup4).
# TODO: Use Natural Language Processing to automate some of the data cleaning/integration.

#############################################################################################################
# Online Retail Analysis - MAIN FILE                                                                        #
#                                                                                                           #
# 1) RESOURCES                                                                                              #
#    DATA SETS: https://archive.ics.uci.edu/ml/datasets/Online+Retail                                       #
#    Python 3.8 DOC: https://docs.python.org/3/                                                             #
#                                                                                                           #
# 2) PROJECT OVERVIEW                                                                                       #
#    This is the version used in this project but any version of python 3 should be compatible.             #
#    Run a Cleanup Program Once before starting the Analysis:                                               #
#       - Put all the data together(in one file/database).                                                  #
#       - Remove unneeded data, example data for a sales company from more than 30 years ago;               #
#       - Eliminate repetition of data;                                                                     #
#       - Extract the data in a set format. To do that generic column tags can be created for the user.     #
#       Example: the column breach_start can be tagged with date-time tag. The script can then be           #
#       trained to check then be trained to check if there are any of these symbols '.', '/', '\'           #
#       and split the data on those symbols as long as it is day, month, year it will format correctly.     #
#       More importantly it should have a way to figure out the order of the date, i.e. if the data is      #
#       year, month, day instead of the set format of the database which can be for example day, month,     #
#       year. If all the data used for the analysis is in the same format it can be set to match that       #
#       format.                                                                                             #
#                                                                                                           #
#############################################################################################################
"""
CASE: Analyse the data to give insights to an online retail business. (TODO: GIVE MORE INFORMATION ABOUT THE BUSINESS)
      Interesting questions to answer:
      1) Identify VIP customers. (Rank Customers)
         - Transaction frequency for each customer?
      2) Identify most profitable items. (Rank Items)
         - What types of items are often bought?

      General Questions:
         - Most common customer location?
         - During holidays is there an increase of how many items are bought and which items are the ones that do increase?
"""

import csv
import os
import copy
from datetime import datetime, timedelta
import numpy as np
import statistics as stat
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# My Modules
import ID3_Classification
import Regression

# Run this function to test if everything is functioning correctly in the ID3 module! It uses the Play Tennis dataset
# to build the tree and run classification. A successful run will have a print saying "The answer is: No"
# ID3_Algorithm.test_run_algorithm()

# region LOAD DATA
# This structure is a list containing a dictionary with details about a data set for each new data set loaded.
# The reason why the top-level container is a list is because I want to handle the data in order of loading,
# and the dictionary does not contain ordered data.
# TODO: Modify the code to support working with multiple datasets at once.
dataset_details = []


# Returns None if the data set fails to load.
# @param ds_name - the file name of the given data set.
# @param num_attrib - the number of attributes or columns of the data set that is being loaded.
# @param num_entries - the number of entries in the data set (not counting the first row as it contains column names).
def load_data_set(file_ext, ds_name, num_attrib, num_entries):
    # On load gather data information to help with integration
    new_ds_details = {"Name": ds_name, "NumAttrib": num_attrib, "NumEntries": num_entries}
    dataset_details.append(new_ds_details)

    file_path = os.getcwd() + "\Data\."    # The . is at the end of the string so that it does noy create a break.
    file_path = file_path[:len(file_path)-1] + ds_name + "." + file_ext

    # If the file fails to load display the current working directory.
    try:
        file_object = open(file_path, "r")
        reader = csv.reader(file_object, delimiter=",")
    except FileNotFoundError:
        print("\n"+117*"-"+"\n"
              "\tNO SUCH FILE OR DIRECTORY: Current Working Directory  - ", os.getcwd(), " \n"
              "\tPlease make sure your data set is stored within the 'Data' directory"
              " in your current working directory!"
              "\n"+117*"-"+"\n")
        return None

    ds_list = list(reader)
    return ds_list


# Using a dictionary because it works faster than the alternative - list and is also cleaner than creating
# 8 new variables used to store a 1 digit number. TODO: Check if using an enum is more efficient.
col_ids = {'InvoiceNo': 0, 'StockCode': 1, 'Description': 2, 'Quantity': 3,
           'InvoicePeriod': 4, 'UnitPrice': 5, 'CustomerID': 6, 'Country': 7}
# endregion

# 541910 is the number of row including the column names.
data_set1 = load_data_set("csv", "Online Retail", 8, 541910)


# region DATA CLEANING
# Looks through the whole dataset and returns a list of compromised records for each column.
def count_compromised_entries(compromised_rows, row):

    for count in range(len(row)):
        value = row[count]
        if value == '':
            compromised_rows[count].append(row)
            break

    """ After running the code above we can see that small portion of the dataset has missing item description.
        Those entries can be removed safely as removing a few entries does not meddle with the results too much. 
        Even so, this data can be easily filled as the item description will always correlate to the stock code. 
        So it can be used to lookup the item description. The other column which has identified missing values 
        is the customer ID column. There is 133 626 missing values from that column. That is about a fifth of 
        the whole dataset so it would be best to try to fill at least some of them."""
    # TODO: Handle missing values.

# TODO: Create a cleaner function that asks the user what the acceptable values for each column are to catch any values
#       that do not fit. This would help a lot when working with a large dataset.
#       Can use RegX to check for specific structure of the data in the column.
# endregion

# region QUESTION 1 - VIP CUSTOMER
#    1.1) Create a customer profile by gathering all the data about each customer ID.
#         Required fields:
#           -> Monetary Value - How much a customer contributed to the business?
#           -> Frequency - How often each customer makes a purchase?
#              It takes into account only the period from the first to the last purchase!
#           -> Recency - How recently did this customer make his/her last purchase?
#           -> Status - VIP if the customer is among the top 20%
#           # TODO: Add a few more attributes like favourite item, the item the customer buys most frequently.
#            TODO: Add other customer statuses like 'new', 'big-spender', 'consistent'

# Dictionary structure: {CustomerID: {MonetaryValue: number, Frequency: number, Recency: Number}, ...}
# TODO: When applying classification on this data to make sure to find the min and the max and create an algorithm to
#       appropriately group the values.


customer_profiles = {}
num_of_customers_per_month = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


# Calculates the average frequency between the dates given. IMPORTANT: Make sure to sort the dates in ascending or
# descending order before passing it to this function as it may obscure the results.
def calc_frequency(dates):
    # Convert all the dates to python time (strptime)
    pythoned_dates = [datetime.strptime(d, '%d/%m/%Y') for d in dates]

    # Data needed for the frequency calculation
    time_difference = timedelta(0)
    difference_count = 0

    # Skipping 0 here as there is no date before the one at index 0.
    for i in range(1, len(pythoned_dates)):
        difference_count += 1
        time_difference += pythoned_dates[i] - pythoned_dates[i - 1]

    # print("Dates: ", dates)
    # print("Date count: ", difference_count+1)

    # If this statement triggers the information is insufficient
    if difference_count == 0:
        # There is a record of 1 purchase date recorded for this customer.
        # Most if not all customers buy items in bulk so despite making purchases only in one day for the whole year
        # does not mean the monetary value of the customer is low and that they should not be considered a VIP customer.
        return -1

    return (time_difference / difference_count).days


def generate_customer_profiles(frequency_data, row):
    global customer_profiles

    invoice_no = row[col_ids['InvoiceNo']]
    customer_ID = row[col_ids['CustomerID']]
    unit_price = row[col_ids['UnitPrice']]
    quantity = row[col_ids['Quantity']]
    invoice_period = row[col_ids['InvoicePeriod']].split()

    # To easier preform the calculations for customer purchase frequency it is useful to split the date and time
    # into separate columns. Due to time constraint I will not be modifying the original dataset as this could
    # cause more issues which will therefore slow the progress of the project.
    # A list containing the date and the time of the purchase
    date, time = invoice_period[0], invoice_period[1]

    # This is a temporary guard to filter out any entries with missing customer ID or wrong customer ID.
    if customer_ID.isdigit():
        # Check if the transaction was cancelled
        if invoice_no[0] != 'C':

            # This if statement is nested in case there is a customer which had cancelled all their transactions.
            # In this way their ID won't be added to the customer profiles.
            if customer_ID not in customer_profiles:
                customer_profiles.update({customer_ID: {"MonetaryValue": 0, "Frequency": 0, "Recency": 0, "Status": None}})
                frequency_data.update({customer_ID: []})

            # -------- <<< MONETARY VALUE >>>  --------
            # Calculate the overall monetary value and frequency for this transaction
            # and add it to the existing amount.
            customer_profiles[customer_ID]["MonetaryValue"] += float(unit_price) * float(quantity)

            # -------- <<< FREQUENCY & RECENCY >>>  --------
            # Frequency - On average how often does this customer make a purchase?
            # Count all the purchases
            if date not in frequency_data[customer_ID]:
                frequency_data[customer_ID].append(date)

            # Gather data needed for visualisation
            py_date = datetime.strptime(date, '%d/%m/%Y')
            num_of_customers_per_month[py_date.month-1] += 1

# endregion


months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
num_of_purchases = []


def k_means_clustering(x_data, y_data, x_name, y_name, num_of_clusters):
    monetary_value_arr = x_data
    frequency_arr = y_data

    # TODO: replace this with my data and test
    data_points = np.array([])

    for count in range(len(monetary_value_arr)):

        new_point = np.array([frequency_arr[count], monetary_value_arr[count]])
        if count == 0:
            data_points = np.hstack((data_points, new_point))
        else:
            data_points = np.vstack((data_points, new_point))

    kmeans = KMeans(n_clusters=num_of_clusters)
    kmeans.fit(data_points)

    # Note: for ideal data plot clustering is done based on equal variance.
    centroids = kmeans.cluster_centers_
    colous = ["g*", "r."]
    # label = kmeans.labels_

    for i in range(len(data_points)):
        plt.plot(data_points[i][0], data_points[i][1], colous[1], markersize=5)

    plt.scatter(centroids[:, 0], centroids[:, 1], marker="x", s=200, linewidths=10, zorder=10)
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.show()


def main(data_set):
    # Contains a list of compromised entries for each column
    compromised_rows = {0: [], 1: [], 2: [], 3: [],
                        4: [], 5: [], 6: [], 7: []}

    # Will contain all the data needed to calculate the transaction frequency for each customer.
    # Dictionary structure: {CustomerID: list of transaction dates, ...}
    purchase_activity = {}

    # PRE-PROCESSING SEARCH THROUGH DATASET
    for count in range(len(data_set)):
        row = data_set[count]

        # This call will display a log of all the entries that may require cleaning, currently only checks for entries
        # with missing values can also use regular expressions to make sure the data in the columns is appropriate.
        # For example/TODO: checking if the date & time are formatted the same way for all entries.
        count_compromised_entries(compromised_rows, row)

        if count != 0:
            # This function cannot handle the calculations of purchase frequency because it needs to first gather all
            # the purchase dates. The frequency will be calculated right after the loop.
            generate_customer_profiles(purchase_activity, row)

    # region Console Output
    print("\nCompromised rows after scan: ")

    # Display the row with missing values.
    print("1) Missing Value Count")
    for column_id in compromised_rows:
        print("Column:", column_id, ",  Num Rows:", len(compromised_rows[column_id]))

    # print("\n Frequency & Recency Calculations for the Period (01/12/2010-09/12/2011): ")

    for customer_ID in purchase_activity:
        purchase_dates = purchase_activity[customer_ID]
        pythoned_dates = [datetime.strptime(d, '%d/%m/%Y') for d in purchase_dates]

        num_of_purchases.append(len(purchase_dates))

        # FREQUENCY
        frequency = calc_frequency(purchase_dates)

        # RECENCY
        last_purchase_date = max(pythoned_dates)

        # Get the last recorded date in the dataset and convert it to python format
        current_dt = data_set[len(data_set)-1:][0][col_ids['InvoicePeriod']]
        current_date = datetime.strptime(current_dt.split()[0], '%d/%m/%Y')
        recency = (current_date - last_purchase_date).days

        # print("Customer ", customer_ID, ": On average there is <<", frequency, ">> days between each purchase and the recency <<", recency, ">>")

        customer_profiles[customer_ID]["Frequency"] = frequency
        customer_profiles[customer_ID]["Recency"] = recency

    print("\nCustomer Profiles: ", customer_profiles)
    print("Num customers: ", len(customer_profiles.keys()))

    # region Question 1.2 - CLASSIFICATION
    # Reformat data
    print("\n\n ------------- << Pre-classification Calculations >> -------------")
    customers_ref = {"MonetaryValue": [], "Frequency": [], "Recency": []}
    data_ranges = {"MonetaryValue": [], "Frequency": [], "Recency": []}

    for customer_ID in customer_profiles:
        customers_ref["MonetaryValue"].append(customer_profiles[customer_ID]["MonetaryValue"])
        customers_ref["Frequency"].append(customer_profiles[customer_ID]["Frequency"])
        customers_ref["Recency"].append(customer_profiles[customer_ID]["Recency"])

    print("IMPORTANT NOTE! << ", customers_ref["Frequency"].count(-1), " >> customers have only made one or two "
          "purchases and therefore the data is insufficient to calculate frequency. ")

    # Identify the minimum criteria for being a VIP customer.
    # What is 80% of the revenue?
    overall_revenue = sum(customers_ref["MonetaryValue"])
    eighty_percent = overall_revenue * 0.8
    print("Overall Revenue:  << ", overall_revenue, " >>")
    print("80% of the Revenue:  << ", eighty_percent, " >>")

    # Do the top 20(how many people) make up 80 % of the revenue following the 80/20 rule.
    sorted_monetary = copy.deepcopy(customers_ref["MonetaryValue"])
    sorted_monetary.sort(reverse=True)
    twenty_percent = int(len(sorted_monetary) * 0.2)

    # The revenue of the top 20% of customers
    twenty_percent_revenue = sum(sorted_monetary[:twenty_percent])
    print("The Revenue of Top << 20% >> of Customers:  << ", twenty_percent_revenue, " >>")

    # Determine the ranges for the outcomes
    for attrib_key in data_ranges:
        r1 = "0  ... " + str(int(max(customers_ref[attrib_key])/2))
        r2 = str(int(max(customers_ref[attrib_key])/2)) + " ... " + str(int(max(customers_ref[attrib_key])))

        data_ranges[attrib_key].append(r1)
        data_ranges[attrib_key].append(r2)

    # print(customers_ref["Frequency"][:twenty_percent])
    print("Minimum monetary value to be considered VIP: £", min(sorted_monetary[:twenty_percent]))

    # Number of customers that have frequency between 35 & 45
    customer_count = 0
    acc_mon_val = 0

    # Set the status of top 20% customers to VIP.
    for customer_ID in customer_profiles:
        if customer_profiles[customer_ID]["MonetaryValue"] in sorted_monetary[:twenty_percent]:
            customer_profiles[customer_ID]["Status"] = "VIP"

        # Accumulate data to find if it is accurate to say a customer with frequency 40
        # should have a monetary value of 50 000
        if (customer_profiles[customer_ID]["Frequency"] > 35) and (customer_profiles[customer_ID]["Frequency"] < 45):
            customer_count += 1
            acc_mon_val += customer_profiles[customer_ID]["MonetaryValue"]

    # Build training dataset
    customer_status = []    # Create a target_attribute list
    customer_attrib_data = {"MonetaryValue": [], "Frequency": [], "Recency": []}   # The rest of the customer attributes

    for customer_ID in customer_profiles:
        customer_status.append(customer_profiles[customer_ID]["Status"])

        # Interpret value ranges.
        for attribute in data_ranges:
            range_tags = data_ranges[attribute]

            for range_tag in range_tags:
                range_values = range_tag.split(" ... ")

    # Train algorithm.
    tree = ID3_Algorithm.ID3DecisionTree()
    tree.build_tree(customers_ref, customer_status)

    # APPLY CLASSIFICATION
    # The index of the entry in the dataset.
    entry_index = 0
    tree.classify(entry_index, customers_ref)

    # print(customers_ref["Status"])
    # endregion

    # region Question 1.3 - REGRESSION
    # Fetch the X and Y data
    print("\n\n ------------- << Regression >> ------------- ")
    monetary_value_arr = customers_ref["MonetaryValue"]
    frequency_arr = customers_ref["Frequency"]

    # TASK: Find the correlation between high school GPA and overall university GPA.
    cov = Regression.calc_covariance(monetary_value_arr, frequency_arr, "Monetary Value", "Frequency")
    print("Cov: ", cov)
    corr_coeff = cov / (stat.stdev(monetary_value_arr) * stat.stdev(frequency_arr))
    print("Correlation coefficient: ", round(corr_coeff, 4))

    # Linear Regression
    # Looking at the scatter plot it is obvious that linear regression is not the best choice for this data
    # Due to the time constraint I had to use the method I am familiar with.
    Regression.calc_lin_regression(monetary_value_arr, frequency_arr, 50000, corr_coeff, ("Monetary Value", "Frequency", "Linear Regression"))

    # Find out the average monetary value for customers with frequency between 35 and 45.
    print("AVG Monetary value for customers with frequency between 35 and 45: ", acc_mon_val / customer_count)
    # endregion

    # region Question 2 - Running Clustering
    recency_arr = customers_ref["Recency"]

    # Clustering for new customers
    k_means_clustering(num_of_purchases, recency_arr, "Recency", "Num Of Purchases", 3)

    # Clustering VIP Customers
    k_means_clustering(monetary_value_arr, frequency_arr, "Frequency", "Monetary Value", 6)
    # endregion

    # region QUESTION 3 - Bar Chart of the Customer Activity per Months
    plt.bar(months, num_of_customers_per_month)
    plt.xlabel("Months")
    plt.ylabel("Num Of Customers")
    plt.title("Active Customers per Month")
    plt.show()
    # endregion


# Start the program
sample_size = 100000
main(data_set1)  # [:sample_size]
