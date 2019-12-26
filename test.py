class Node:
    def __init__(self, n):
        self.n = n
        self.branches = []

class Tree:

    def __init__(self):
        self.root = None


tree = Tree()
"""
node1 = Node("0")

node2 = Node("0.1")
node2.branches = None

node3 = Node("0.2")

node1.branches = [node2]
node1.branches.append(node3)

tree.root = node1

node3.n = "0.3"

node4 = Node("0.4")
node5 = Node("0.5")

node3.branches.append(node4)
node4.branches = [node5]

print(tree.root.n)
print(tree.root.branches[1].n)
print(tree.root.branches[1].branches[0].n)
print(tree.root.branches[1].branches[0].branches[0].n)"""
"""
tree.root = Node("Root")
parent = tree.root
for count in range(4):
    child = Node(count)
        parent.branches.append(child)
    parent = child

# [2,4,6,2]
data = {"A":[2,2,4,7,2,6,2,8,2], "B":[1,3,5,7]}
import copy
b = copy.deepcopy(data)
b["B"][0] = 2

ind_to_remove = [0,3,4,7,8]
complete_list = [2,2,4,7,2,6,2,8,2]

sub_list = [value for index, value in enumerate(complete_list) if index not in ind_to_remove]

val_to_remove = [2,2,7,2,2]
#b["A"].pop(ind_to_remove)
print("original: ", data)
print("Copy: ", b)
print("formated: ", sub_list)

a = [2,2,4,6,8]
print(a.index(2))
print(list(data.values()))

for outcome_data in data.values():
    for count in range(len(outcome_data)):
        if outcome_data[count] == 2:
            print("TEST: ",count)

indices = [count for outcome_data in data.values() for count in range(len(outcome_data)) if outcome_data[count] == 2]

print(indices)
#print([index for outcome in outcome_data if outcome == 2])
"""

print("0".isdigit())
