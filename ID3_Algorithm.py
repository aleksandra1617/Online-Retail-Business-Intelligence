# TODO mention in the report that with every level of the tree the data gets smaller and smaller.
# NEXT STEPS
# TODO: Create an infographic and host it on a web page.
# TODO: Gather live data from news articles (Can try using NLTK & urllib).
# TODO: Use Natural Language Processing to automate some of the data cleaning/integration.

###################################################################################################################
# Online Retail Analysis - ID3 CLASSIFICATION                                                                     #
#    NOTE! Concepts will be explained with examples from the Street data set, which can be found below.           #
#    The reason for this is because that data set is very small and easy to follow.                               #
#                                                                                                                 #
# 1) RESOURCES                                                                                                    #
#    ID3 TUTORIALS:                                                                                               #
#      1) https://sefiks.com/2017/11/20/a-step-by-step-id3-decision-tree-example/                                 #
#      2) https://medium.com/coinmonks/what-is-entropy-and-why-information-gain-is-matter-4e85d46d2f01            #
#                                                                                                                 #
#    DECISION TREE TUTORIAL: https://www.lucidchart.com/pages/decision-tree                                       #
#    ENTROPY (MORE DETAILS): https://en.wikipedia.org/wiki/Entropy_(information_theory)                           #
#                                                                                                                 #
# 2) DATA SETS                                                                                                    #
#    TEST DATA SET: This data set can be found by navigating to the STREET DATA SET region in this file.          #
#    It is a part of the ID3 file because I believe it would be useful to have an example of how the ID3 code     #
#    works with a data set and also provides an opportunity to better understand what the code is doing.          #
#    To have a look at ID3 applied to a small data set just make a call the test_run() function at the            #
#    end of the file.                                                                                             #
#                                                                                                                 #
# 3) ALGORITHM OVERVIEW                                                                                           #
#    Used to generate a decision tree from a given data set. It works by evaluating each attribute                #
#    in the data set to place the nodes in an order that will return an accurate result.                          #
#                                                                                                                 #
# 4) USES                                                                                                         #
#    A) Classify labeled data generally to do with NLP, approving loans and credit cards, etc.                    #
#    B) Another non-standard use of this algorithm is to use it to fill a missing value in the data set           #
#    during the pre-processing stage.                                                                             #
#                                                                                                                 #
###################################################################################################################

import math
import copy

# region PERFORMANCE IMPROVEMENTS (for Python 3.8)
""" 
Applied: (TO DOCUMENT)

TODO: 
   1) Remove ever dict.keys() used and replace it with dict because dict.keys() creates a list of keys in memory. 
      (More costly than looking through the dictionary itself! Further information below.)  
      https://stackoverflow.com/questions/4730993/python-key-in-dict-keys-performance-for-large-dictionaries
"""
# endregion

# region PLAY TENNIS DATA SET
DATASET_BY_ATTRIB_DICT = {"outlook": ["Sunny", "Sunny", "Overcast", "Rain", "Rain", "Rain", "Overcast",
                                      "Sunny", "Sunny", "Rain", "Sunny", "Overcast", "Overcast", "Rain"],
                          "temperature": ["Hot", "Hot", "Hot", "Mild", "Cool", "Cool", "Cool",
                                          "Mild", "Cool", "Mild", "Mild", "Mild", "Hot", "Mild"],
                          "humidity": ["High", "High", "High", "High", "Normal", "Normal", "Normal",
                                       "High", "Normal", "Normal", "Normal", "High", "Normal", "High"],
                          "wind": ["Weak", "Strong", "Weak", "Weak", "Weak", "Strong", "Strong",
                                   "Weak", "Weak", "Weak", "Strong", "Strong", "Weak", "Strong"]}


# Answer as to whether or not it is a good time to play tennis.
TARGET_ATTRIB_LIST = ["No", "No", "Yes", "Yes", "Yes", "No", "Yes", "No", "Yes", "Yes", "Yes", "Yes", "Yes", "No"]

# CONSTANT VARIABLES  # TODO: Optimise these variables by making them immutable (specifying they are const with Python)
TARGET_ATTRIB_NAME = "play tennis"
TRAIN_DATA_SIZE = len(TARGET_ATTRIB_LIST)
# endregion


# Represents a tree node and links to derived nodes.
class Node:

    def __init__(self, node_name, derived_nodes=[]):
        self.node_name = node_name
        self.derived_nodes = derived_nodes


class ID3DecisionTree:
    def __init__(self):
        self.root_node = None

        # Keeps track of all the nodes at the end of the branches that are available to link to.
        # In this way, no code needs to be ran to find the next available space for a new node.
        # The node at index 0 is always the one to add to first, once the new node is linked to it, it gets popped off
        # and the new node gets appended to the end of this list.
        self.active_branch_nodes = []

        # TODO: Merge this list with the active_branch_nodes to be in dictionary format like so
        # {attrib1: [outcome1, outcome2], attrib2: [outcome1, outcome2, outcome3]}
        self.linked_attributes = []

        # IMPORTANT NOTE:
        # Key to understanding how the DecisionTree class works is understanding the dataset_occurrence_dict
        # structure, as that is what is used for most calculations. This structure contains only the data from the
        # dataset required to construct the tree. Any repetition of attribute data has been removed to reduce load.
        # The 'dataset_occurrence_dict' structure is an unordered dictionary, where the structure itself gives more
        # information about the dataset. For example, every attribute of the data set is a key, which contains
        # a dictionary of its outcomes/possible values, and for each outcome, there is a dictionary showing the
        # distribution of the outcomes for the selected target attribute.
        # Example of dictionary structure below.
        """ Example structure: (where 'AN'-attribute name; 'ON'-outcome name; 'TON'-target outcome name) 
            dataset_occurrence_dict = {"AN 1": {"ON 1": {"TON 1": 1, "TON 2": 2},
                                                "ON 2": {"TON 1": 0, "TON 2": 1},
                                                "ON 3": {"TON 1": 0, "TON 2": 1}
                                                },
                                       "AN 2": {"ON 1": {"TON 1": 4, "TON 2": 0},
                                                "ON 2": {"TON 1": 1, "TON 2": 0}
                                                }
                                       }
                                                
            The example above can be read, for attribute 1 - AN1, there are 3 outcomes - ON1, ON2, ON3. 
            The target has 2 possible outcomes TON1 and TON2. Those values are being tracked/accounted for, 
            for each possible outcome of each attribute. For AN1, ON1 there is 1 occurrence of TON1 and 2 occurrences of 
            TON2. For AN1, ON2 there are 0 occurrences of TON1, and 1 occurrence of TON2 therefore the answer for this 
            branch is TON2. Same for AN1, ON3 - answer TON2. If all the occurrences of TON1 and TON2 for attrib 1 (AN1)
            are summed, we get the number of entries in the given data set. 
        """
        self.dataset_occurrence_dict = {}

    # region BUILD TREE UTILITIES
    """ Construct dataset distribution/occurrence dictionary - "dataset_occurrence_dict".
    PARAMETERS
      :param (dict) dataset_by_attrib_dict
      :param (list) target_list """
    def generate_occurrences(self, dataset_by_attrib_dict, target_list):
        # TODO: assert that all attribute lists have the same length

        # Update the dictionary with each attribute
        for attrib_name in dataset_by_attrib_dict.keys():
            # STEP 1: ADD the current attribute to the 'dataset_occurrence_dict' structure
            self.dataset_occurrence_dict.update({attrib_name: {}})

            # STEP 2: Fetch a list containing only the unique data from attribute_list and target_list.
            attribute_list = dataset_by_attrib_dict[attrib_name]
            unique_attrib_outcomes = list(set(attribute_list))
            unique_answers = list(set(target_list))

            # For each unique outcome of the current attribute
            for attrib_outcome in unique_attrib_outcomes:
                #   2.1) Update dictionary to store the next attribute outcome
                self.dataset_occurrence_dict[attrib_name].update({attrib_outcome: {}})
                # print(self.dataset_occurrence_dict)

                #   2.2) For the current attribute, look at each of its outcomes and add them onto the dictionary
                for outcome in unique_answers:
                    self.dataset_occurrence_dict[attrib_name][attrib_outcome].update({outcome: 0})
                    # print(self.dataset_occurrence_dict)

            # STEP 3: Goes through the dataset and counts the target outcome occurrences for each attribute occurrence
            for itter in range(len(attribute_list)):
                #   3.1) Fetch the current attribute outcome and the current target outcome from the dataset.
                curr_attrib_occ = attribute_list[itter]
                curr_target_occ = target_list[itter]

                #   3.2) Update the count for the current target outcome in the current attribute outcome by 1
                self.dataset_occurrence_dict[attrib_name][curr_attrib_occ][curr_target_occ] += 1

    """ After a node is added to the tree the "dataset_occurrence_dict" dictionary should be updated.
       PARAMETERS
         :param (list) attrib_list - the raw attrib data from the dataset.
         :param (list) target_list - the raw target data from the dataset. """
    def get_next_branch_occurrences(self, dataset_by_attrib_dict, target_list):
        # This is the outcome to update the dataset_occurrence_dict by

        # A completely separate dictionary from the original, this dictionary will only hold a subdictionary
        # of the original
        subdict = copy.deepcopy(dataset_by_attrib_dict)
        subtar = copy.deepcopy(target_list)

        indices_to_remove = []
        attrib_to_remove = None

        # Looking through every possible attribute in the dictionary
        for attrib_key in subdict:
            attrib_found = False
            # Count through each list of outcomes for the given attribute.
            for count in range(len(subdict[attrib_key])):
                # If the active outcome name is equal to the current outcome value in the list
                if dataset_by_attrib_dict[attrib_key][count] == self.active_branch_nodes[0].node_name:
                    attrib_found = True
                    # According to the algorithm, the attribute containing the currently active outcome
                    # should be removed
                    if attrib_key in subdict:
                        attrib_to_remove = attrib_key
                else:
                    indices_to_remove.append(count)
                    # print(subdict[attrib_key][count])
                    # subdict[attrib_key].pop(count)
                    # TODO: assert that there is only one 0 in the list otherwise it is trying to remove the wrong values

            if attrib_found:
                break

        # Processing the subdict data
        #print("Subdict: ", subdict)
        del subdict[attrib_to_remove]

        for attrib in subdict:
            #print("Discarding data in ", attrib)
            complete_list = subdict[attrib]

            sublist = [value for index, value in enumerate(complete_list) if index not in indices_to_remove]
            subdict[attrib] = sublist

        #print("After processing the data: ", subdict)

        # Processing the subtar data
        #print("Discarding data in target list")
        #print("Target data before processing: ", subtar)
        # print(indices_to_remove)
        subtar = [value for index, value in enumerate(subtar) if index not in indices_to_remove]
        #print("Target data after processing: ", subtar)

        # TODO: Call this function recursively on each branch, pass in the shrinked dictionary
        # TODO: test the base case thoroughly
        # TODO: Build a new dataset_by_attrib_dict for the current outcome
        # TODO: REMOVE outlook from the dataset dict when all its outcomes have children nodes assigned
        # (How to know if an attribute is complete???)

        return subdict, subtar

    """ Checks if a branch is complete, i.e. the target outcome was found. 
    PARAMETERS
      :param  (dict) target_val_dist_for_attrib 
      :returns (list) comp_branches - contains all the target outcomes reached for the given attribute."""
    def track_target_outcomes(self, target_val_dist_for_attrib):
        comp_branches = []

        # Looks through each attribute outcome
        for attrib_outcome_key in target_val_dist_for_attrib.keys():

            # Tracks how many non-zero occurrences of a target outcome there are for this attribute outcome.
            non_zero_outcome_count = 0

            # This variable is set to the target outcome if the branch outcome is (100%) certain.
            branch_answer = None

            # Checks what the distribution of target outcomes is for the current attribute outcome.
            # Ex: question - how sdo people drive based on the terrain, if the terrain is flat do they drive slow
            # or fast, and what is it if the terrain is steep.
            # Target outcomes - fast and slow; attrib outcomes - flat and steep.
            # Distribution dictionary looks like this ->{'fast': {'slow': 0, 'fast': 1}, 'steep':{'slow': 2, 'fast': 1}}
            for target_outcome_key in target_val_dist_for_attrib[attrib_outcome_key].keys():

                # Fetch the number of occurrences for each target outcome for the current attribute
                """"Another Example: if the target is can_buy_computer(possible values/outcomes: Yes or No) and the current 
                attribute is age (possible values/outcomes:  <=30, 31..40 and >40) this will return how many of the entries 
                where age is <=30 are no, then how many of the entries where age is <=30 are yes, then how many 
                of the entries where age is 31..40 are yes and so on, until all cases are looked at. """
                outcome_occurrences = target_val_dist_for_attrib[attrib_outcome_key][target_outcome_key]

                # Check if the answer is certain and end the branch, i.e. count how many branches have
                # certain target outcome
                if outcome_occurrences > 0:
                    non_zero_outcome_count += 1

                    if non_zero_outcome_count == 1:
                        branch_answer = target_outcome_key

            if non_zero_outcome_count == 0:
                print("INVALID RESULT!")
            elif non_zero_outcome_count == 1:
                print("THE ANSWER FOR <<", attrib_outcome_key, ">> is <<", branch_answer, ">>")
                comp_branches.append({attrib_outcome_key: branch_answer})
            elif non_zero_outcome_count > 1:
                print("THE BRANCH <<", attrib_outcome_key, ">> IS STILL ACTIVE!")

        return comp_branches

    # Counts the occurrences of each value for a given attribute.
    def count_value_occ(self, unique_values, attrib_data):
        attrib_val_occ = {}

        # Construct dictionary
        for value in unique_values:
            attrib_val_occ.update({value: 0})

        # Initialise Dictionary
        for u_value in unique_values:
            attrib_val_occ[u_value] = attrib_data.count(u_value)

        return attrib_val_occ

    def calc_entropy(self, attrib_uv_count, overall):
        entropy = 0
        # print("UV: ", attrib_uv_count)

        for key in attrib_uv_count.keys():

            # if there is some occurrence of the value calculate entropy,
            # otherwise ignore it (when there is 0 occurrences of the value)
            if attrib_uv_count[key] != 0:
                fraction = attrib_uv_count[key] / overall
                target_attrib_calc = fraction * math.log2(fraction)

                entropy += target_attrib_calc

        return abs(entropy)

    def calc_attrib_entropy(self, attrib_occurrences):
        entropy_list = {}

        for attrib_val_key in attrib_occurrences.keys():
            attrib_val = attrib_occurrences[attrib_val_key]
            overall = 0
            for target_values in attrib_val.values():
                overall += target_values

            print("CALC TARGET ENTROPY FOR EACH ATTRIB OUTCOME: ", attrib_val)
            attrib_entropy = self.calc_entropy(attrib_val, overall)
            entropy_list.update({attrib_val_key: attrib_entropy})

        print("Entropy list: ", entropy_list)

        return entropy_list

    # WEIGHTED AVERAGE ENTROPY for the children
    def calc_entropy_weigh_avg(self, target_val_dist_attrib, overall, attrib_entropy):
        weighted_entropy_avg = 0
        for key in target_val_dist_attrib.keys():
            curr_value = 0

            for value in target_val_dist_attrib[key].values():
                curr_value += value
            weighted_entropy_avg += curr_value / overall * attrib_entropy[key]
            # overall += curr_value

        return weighted_entropy_avg

    def calc_info_gain(self, target_entropy, target_dist_for_attrib):

        # CALCULATE ENTROPY OF Attribute
        attrib_entropy = self.calc_attrib_entropy(target_dist_for_attrib)
        # print("Attrib Entropy: ", attrib_entropy)

        weighted_avg_e = self.calc_entropy_weigh_avg(target_dist_for_attrib, TRAIN_DATA_SIZE, attrib_entropy)
        # print("Attrib Weighted AVG: ", weighted_avg_e)

        attrib_info_gain = target_entropy - weighted_avg_e

        return attrib_info_gain

    # IMPORTANT NOTE: An attribute node should always be made together with its outcomes, never an outcome alone
    # as it is not how this function was setup.
    # :param (str) name - should always be the name of an attribute.
    def build_node(self, name, completed_branches):
        attrib_node = Node(name)
        derived_nodes = []

        completed_outcomes = []
        for branch in completed_branches:
            completed_outcomes.append(list(branch.keys())[0])

        # if all outcome branches for thi attribute are completed, then the attribute is complete and its outcomes
        # should be popped off the active_branch_nodes list
        # print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> CHECK COMPLETE ATTRIB: ", completed_branches)

        # print(self.dataset_occurrence_dict[name].keys())
        for outcome_name in self.dataset_occurrence_dict[name]:
            new_outcome_node = Node(outcome_name)
            # print("STATUS: NEW OUTCOME NODE CREATED")

            # Check if the branch for the current outcome is complete (Target answer is 100% certain).
            for branch in completed_branches:
                if outcome_name in branch:
                    # print("FOUND OUTCOME <<", outcome_name, ">> in ", branch)

                    if len(new_outcome_node.derived_nodes) == 0:
                        # Formally end the node
                        endpoint_node = Node(branch[outcome_name], None)
                        new_outcome_node.derived_nodes.append(endpoint_node)
                        # print("STATUS: NEW OUTCOME ENDPOINT NODE CREATED & LINKED")

            # The temp_outcome node is created so that the outcome node stored in the tree and the outcome node stored
            # in the active_branch_nodes list are the same. This is important because I never append directly onto the
            # tree but to a reference of the active branch of the tree. This allows to append to any depth of the tree
            # without needing to do any traversal to find the next available node.
            temp_outcome = copy.deepcopy(new_outcome_node)
            derived_nodes.append(temp_outcome)

            # If the branch is still active/available to add to
            if outcome_name not in completed_outcomes:
                # Add the new node to the active branch list
                self.active_branch_nodes.append(temp_outcome)
            """print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Completed Nodes:", acc_completed)
        acc_completed[name]["completed"] = True
        all_outcomes_list = list(self.dataset_occurrence_dict[name].keys())

        for outcome in all_outcomes_list:
                if outcome in acc_completed[name]["outcomes"]:
                    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> ", outcome, " TRUE")
                else:
                    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>> ", outcome, " FALSE")
                    acc_completed[name]["completed"] = False

            print(all_outcomes_list)"""

            new_outcome_node.derived_nodes.clear()

        # print("STATUS: NEW NODE CREATED")
        attrib_node.derived_nodes = derived_nodes
        return attrib_node

    # IMPORTANT NODE: active_branch_nodes is only updated when build_node function is called, therefore
    # the link will not be appropriate unless the node was created through the build_node function.
    def link_node(self, new_node):
        """
        print("  <<< CHECKING IF THE TREE SEGMENT IS BUILT RIGHT! >>>    ")
        # TEMP
        print("ATTRIBUTE/PARENT NODE: ", new_node.node_name)
        print("DERIVED NODES LIST: ", new_node.derived_nodes)

        print("FOR EACH NODE IN DERIVED NODES.")
        for node in new_node.derived_nodes:
            print("\t OUTCOME NODE FOR ATTRIB: ", node.node_name)
            for other in node.derived_nodes:
                print("\t\t TARGET OUTCOME REACHED: ", other.node_name)"""
        if self.root_node is None:
            self.root_node = new_node

        else:
            # Add the new node to the tree
            # I hard coded 0 as the active node index because index 0 is always the next available node to link to.
            self.active_branch_nodes[0].derived_nodes.append(new_node)

            # Update the available nodes!
            # The node at index 0 is already taken so that node should be popped off
            self.active_branch_nodes.pop(0)

    # Builds a part of the tree (attribute node with setup derived nodes/outcome nodes) and links it to the tree.
    def build_tree_chunk(self, dataset_by_attrib_dict, target_attrib_list):
        self.generate_occurrences(dataset_by_attrib_dict, target_attrib_list)
        # print("Main DICTIONARY", self.dataset_occurrence_dict)

        # TARGET ATTRIBUTE CALCULATIONS - Required for the calculation of info_gain for the rest of the attributes.
        target_uv_data = list(set(target_attrib_list))  # TODO: POSSIBLE EFFICIENCY DECREASE
        target_uv_count = self.count_value_occ(target_uv_data, target_attrib_list)
        # print("Target Unique Value Count: ", target_uv_count)

        target_entropy = self.calc_entropy(target_uv_count, TRAIN_DATA_SIZE)
        # print("TARGET ENTROPY: ", target_entropy)

        # Build each node(calc its entropy and info_gain, and assigning each attributes outcomes as children)
        # store the node in the node list and sort the nodes by info_gain to build the tree with them.
        next_node_data = {"name": None, "info gain": 0, "completed": None}

        for attrib_name in self.dataset_occurrence_dict.keys():
            print("\n", "-" * 50)

            # ATTRIB CALCULATIONS
            print("attrib_name: ", attrib_name)

            # Contains a data structure representing the target attribute's value distribution
            # with regard to another attribute
            target_dist_for_attrib = self.dataset_occurrence_dict[attrib_name]
            # print("Target occurrences: ", target_dist_for_attrib)

            # Check if any of the branches is completed
            completed_branches = self.track_target_outcomes(target_dist_for_attrib)
            print("COMPLETED BRANCHES: ", completed_branches)

            attrib_info_gain = self.calc_info_gain(target_entropy, target_dist_for_attrib)
            # print("The INFO GAIN for <<", attrib_name, ">> is ", attrib_info_gain)

            if next_node_data["info gain"] < attrib_info_gain:
                next_node_data["name"] = attrib_name
                next_node_data["info gain"] = attrib_info_gain
                next_node_data["completed"] = completed_branches

        print("------> The next new node is: ", next_node_data["name"], "\n\n")
        new_node = self.build_node(next_node_data["name"], next_node_data["completed"])
        self.link_node(new_node)

    # endregion

    def build_tree(self, dataset_by_attrib_dict, target_attrib_list):

        self.build_tree_chunk(dataset_by_attrib_dict, target_attrib_list)
        print("\n\n")

        while len(self.active_branch_nodes) != 0:
            print(">>>>>>>>>>>>>>>>>>> Current active node: ", self.active_branch_nodes[0].node_name)
            # self.linked_attrib_names
            sub_attrib_dict, sub_tar_list = self.get_next_branch_occurrences(dataset_by_attrib_dict, target_attrib_list)
            self.build_tree_chunk(sub_attrib_dict, sub_tar_list)
            print("\n\n>>>>>>>>>>>>>>>>>>> List of active nodes: ", self.active_branch_nodes)

        print("\n\n", "<"*5, "THE TREE IS COMPLETE!", ">"*5, "\n\n")

    def visualise_tree(self):
        current_node = self.root_node
        while current_node is not None:
            print(current_node.node_name)

            # TODO this recursively, base case -> len(node.derived_nodes) == 0
            # EXTRA TODO pass in a variable called branch_track that will start off as "",
            # each time a recursion is spawned add a "\t", that way the print will have a sort of a hiearchy

    # This function runs classification on one entry and returns the answer.
    # Should only be called after the tree model was built.
    def classify(self, entry_index, dataset_by_attrib_dict):
        answer = None

        # TODO: assert that root node is not none
        current_node = self.root_node

        while current_node.derived_nodes is not None:
            print("\n  <<< TRAVERSING TREE >>>  ")
            print("Current Attrib: ", current_node.node_name)

            # Ask the tree which attribute/column to look for first
            column_name = current_node.node_name

            # Fetch the value for the given entry (entry_index) from the column identified by the tree.
            current_outcome_name = dataset_by_attrib_dict[column_name][entry_index]
            print("\tCurrent outcome name: ", current_outcome_name)

            # Get that node from the derived nodes list
            for outcome_node in current_node.derived_nodes:
                if outcome_node.node_name == current_outcome_name:
                    # print("\n  <<< TRAVERSING TREE >>>  ")
                    # print("FOUND VALUE FOR ENTRY <<", entry_index, ">>  ->  <<", outcome_node.node_name, ">>")
                    current_node = outcome_node.derived_nodes[0]
                    # print("Current Attrib: ", current_node.node_name)
                    answer = current_node.node_name

        print("    <<< FOUND VALUE >>>  ")
        print("    The answer is: ", answer)

        return answer


def test_run_algorithm():
    print(" "*10, " << ID3 CLASSIFICATION ALGORITHM >> ", " "*10)

    tree = ID3DecisionTree()
    tree.build_tree(DATASET_BY_ATTRIB_DICT, TARGET_ATTRIB_LIST)

    # APPLY CLASSIFICATION
    # The index of the entry in the dataset.
    entry_index = 0
    tree.classify(entry_index, DATASET_BY_ATTRIB_DICT)


test_run_algorithm()

"""
# Remove the completed branches
for branch in completed_branches:
    for key in branch.keys():
        target_val_dist_for_grade.pop(key)

print("After removing completed branches: ", target_val_dist_for_grade)
"""

# region Build Decision Tree

# endregion

""" 
What is "Training Data"? 
    Building the tree is done with training data which already has the answer to whatever question is being asked. 
    the example given with the data on the slides that asks if someone can buy a laptop is training data
    because it already knows the answer.
"""
"""
Apply information gain function to each attribute calculate_gain(attr_out)
Should that be applied to the target as well? No
Example:
    - G(train_data, O) = 0.246
    - G(train_data, H) = 0.151
    - G(train_data, W) = 0.048

Once the root node is known, look at how many unique values are there.
If there are 4 possible values and they are not numbers, 
for example "Sunny", "Rainy", etc. there should be 4 nodes. 
"""

# region Apply Classification
"""
What is "Test Data"?
    Test data is when we get a new entry and we want to classify it. 
    For example: In the bank they may use an already trained ID3 algorithm to check if you should get a credit card or not.
    They will have different attributes like - number of times you have gone bankrupt; what is your current net worth; 
    are you a student;  what is your credit score; etc.
    Then the target attribute will be EligibleForCreditCard(True or False)
"""

# Use the built decision tree to look through a row of data from the data set. This is done using test data.
# (How to evaluate if the classification has an error?)
""" 
Steps: 
    1. Find which is the current attribute to look through (To start with ask the tree which attribute is the root node)
        1.1 (When building the tree need to make sure the attributes have the exact same name as the Node data)
        1.2 Search trough all possible attributes
        1.3 Check if the attribute name == the node name
        
    2. Find the attribute value for the current row
        2.1 Ask the data set which value is given for this attribute
        2.2 Find the which of the children nodes in the tree are equivalent to the given value
        
    Repeat these steps recursively until an answer is found. 
"""
# endregion
