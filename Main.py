# NEXT STEPS
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
#       - Remove unneeded data, example data for a sales company from over 10 years ago;                    #
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
    print(file_path)

    # If the file fails to load display the current working directory.
    try:
        file_object = open(file_path, "r")
        reader = csv.reader(file_object, delimiter=",")
    except FileNotFoundError:
        print("\n"+117*"-"+"\n"
              "\tNO SUCH FILE OR DIRECTORY: Current Working Directory  - ", os.getcwd(), " \n"
              "\tPlease make sure your data set is stored within the 'Data Sets' directory"
              " in your current working directory!"
              "\n"+117*"-"+"\n")
        return None

    ds_list = list(reader)
    return ds_list


# 541910 is the number of row including the column names.
data_set1 = load_data_set("csv", "Online Retail", 8, 541910)

# Using a dictionary because it works faster than the alternative - list and is also cleaner than creating
# 8 new variables ust to store a 1 digit number. TODO: Check if using an enum is more efficient.
col_ids = {'InvoiceNo': 0, 'StockCode': 1, 'Description': 2, 'Quantity': 3,
           'InvoiceDate': 4, 'UnitPrice': 5, 'CustomerID': 6, 'Country': 7}
# endregion

def display_data(data_set, num_entries=None):
    print("Print of first <<", num_entries, ">> rows of the dataset.")

    if num_entries is None:
        for row in data_set:
            print(row)
    else:
        # Going until num_entries +1 because the first row contains the column names
        for row_count in range(num_entries+1):
            row = data_set[row_count]
            print(row)


# region DATA CLEANING
# Looks through the whole dataset and returns
def count_compromised_entries(data_set):
    compromised_rows = {0: [], 1: [], 2: [], 3: [],
                        4: [], 5: [], 6: [], 7: []}

    for row in data_set:
        for count in range(len(row)):
            value = row[count]
            if value == '':
                compromised_rows[count].append(row)
                break

    print("\n\nCompromised rows after scan: ")

    # Display the row with missing values.
    print("1) Missing Value Count")
    for column_id in compromised_rows:
        print("Column:", column_id, ",  Num Rows:", len(compromised_rows[column_id]))

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


count_compromised_entries(data_set1)
# endregion

# region QUESTION 1 - VIP CUSTOMER
#    1.1) Create a customer profile by gathering all the data about each customer ID.
#         Required fields:
#           -> Monetary Value - How much a customer contributed to the business?
#           -> Frequency - How often each customer makes an order?
#           -> Recency - How recently did this customer make his/her last purchase?
#           # TODO: Add a few more attributes like favourite item, the item the customer buys most frequently.

# Dictionary structure: {CustomerID: {MonetaryValue: number, Frequency: number, Recency: Number}, ...}
# TODO: When applying classification on this data to make sure to find the min and the max and create an algorithm to
# appropriately group the values.
customer_profiles = {}


def generate_customer_profiles(data_set):
    # Will contain all the data needed to calculate the transaction frequency for each customer.
    # Dictionary structure: {CustomerID: list of transaction dates, ...}
    frequency_data = {}

    for row in data_set:

        invoice_no = row[col_ids['InvoiceNo']]
        customer_ID = row[col_ids['CustomerID']]
        unit_price = row[col_ids['UnitPrice']]
        quantity = row[col_ids['Quantity']]
        invoice_date = row[col_ids['InvoiceDate']]

        # This is a temporary guard to filter out any entries with missing customer ID or wrong customer ID.
        if customer_ID.isdigit():
            # Check if the transaction was cancelled
            if invoice_no[0] != 'C':

                # This if statement is nested in case there is a customer which had cancelled all their transactions.
                # In this way their ID won't be added to the customer profiles.
                if customer_ID not in customer_profiles:
                    customer_profiles.update({customer_ID: {"MonetaryValue": 0, "Frequency": 0, "Recency": 0}})
                    frequency_data.update({customer_ID: []})

                # Calculate the overall monetary value and frequency for this transaction
                # and add it to the existing amount.
                customer_profiles[customer_ID]["MonetaryValue"] += float(unit_price) * float(quantity)

                # Frequency - On average how often does this customer make a purchase?
                # Count all the purchases
                frequency_data[customer_ID].append(invoice_date)

                # Calculate frequency between each transaction and get the average
                # https://www.mrexcel.com/board/threads/calculating-how-often-something-happens-excel-2000.188528/
                # the min is always the date at index 0 and the max is always the date at the end of the list

    print(customer_profiles)


generate_customer_profiles(data_set1)
# endregion

