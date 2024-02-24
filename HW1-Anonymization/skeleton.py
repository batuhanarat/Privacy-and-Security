 ste##############################################################################
# This skeleton was created by Efehan Guner (efehanguner21@ku.edu.tr)    #
# Note: requires Python 3.5+                                                 #
##############################################################################

import csv
import glob
import os
import sys

import  copy
import time
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt




if sys.version_info[0] < 3 or sys.version_info[1] < 5:
    sys.stdout.write("Requires Python 3.x.\n")
    sys.exit(1)

##############################################################################
# Helper Functions                                                           #
# These functions are provided to you as starting points. They may help your #
# code remain structured and organized. But you are not required to use      #
# them. You can modify them or implement your own helper functions.          #
##############################################################################

def read_dataset(dataset_file: str):
    """ Read a dataset into a list and return.

    Args:
        dataset_file (str): path to the dataset file.

    Returns:
        list[dict]: a list of dataset rows.
    """
    result = []
    with open(dataset_file) as f:
        records = csv.DictReader(f)
        for row in records:
            result.append(row)
    # print(result[0]['age']) # debug: testing.
    return result


def write_dataset(dataset, dataset_file: str) -> bool:
    """ Writes a dataset to a csv file.

    Args:
        dataset: the data in list[dict] format
        dataset_file: str, the path to the csv file

    Returns:
        bool: True if succeeds.
    """
    assert len(dataset)>0, "The anonymized dataset is empty."
    keys = dataset[0].keys()
    with open(dataset_file, 'w', newline='')  as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(dataset)
    return True



def read_DGH(DGH_file: str):
    """ Reads one DGH file and returns in desired format.

    Args:
        DGH_file (str): the path to DGH file.
    """
    #TODO: complete this code so that a DGH file is read and returned
    # in your own desired format.
    with open(DGH_file) as f:
        lines = f.readlines()
    # Create a root node. We assume the first line is the root.
    root_data = lines[0].strip()
    tree = Tree(root_data)
  # Dictionary to keep track of the last node at each depth
    depth_node_map = {0: tree.root}

    for line in lines[1:]:
        data = line.strip()
        depth = line.count('\t')

        # Create a new node
        new_node = Node(data,depth)

        # Find the parent node at the previous depth
        parent_node = depth_node_map[depth - 1]
        parent_node.add_child(new_node)

        # Update the last node at this depth
        depth_node_map[depth] = new_node

    return tree

def read_DGHs(DGH_folder: str) -> dict:
    """ Read all DGH files from a directory and put them into a dictionary.

    Args:
        DGH_folder (str): the path to the directory containing DGH files.

    Returns:
        dict: a dictionary where each key is attribute name and values
            are DGHs in your desired format.
    """


    DGHs = {}
    for DGH_file in glob.glob(DGH_folder + "/*.txt"):
        attribute_name = os.path.basename(DGH_file)[:-4]
        DGHs[attribute_name] = read_DGH(DGH_file);

    return DGHs

def costMDBetweenTwoRecords(record1, record2,DGHs,attribute_names) -> int:
    result = 0

    for attribute_name in attribute_names:
        DGH_tree = DGHs[attribute_name]  # Get the DGH tree for the attribute

        # Find the corresponding node for the raw and anonymized values
        raw_value = record1[attribute_name]
        anon_value = record2[attribute_name]
        raw_node = DGH_tree.search(raw_value)
        anon_node = DGH_tree.search(anon_value)

        # Calculate the cost as the absolute difference in node heights
        cost = abs(raw_node.height - anon_node.height)
        result += cost

    return result

def costLMOfRecord(anonymized_record,weight,DGHs,attribute_names) ->float:
    result = 0

    for attribute_name in attribute_names:
        DGH_tree = DGHs[attribute_name]  # Get the DGH tree for the attribute
        anon_value = anonymized_record[attribute_name]
        anon_node = DGH_tree.search(anon_value)
        lm = LMHelper(anon_node,DGH_tree)
        result += (weight * lm)

    return result
def LMHelper(Anon_Node, DGH_tree) -> float:
    denominator = DGH_tree.get_external_nodes_count() - 1
    newTree = Tree.from_node(Anon_Node)
    nominator = newTree.get_external_nodes_count() - 1
    return nominator/denominator

def generalize_ec(cluster, DGHs):
    generalized_cluster = copy.deepcopy(cluster)  # Make a deep copy to avoid modifying the original cluster
    for attribute in DGHs.keys():  # For each attribute
        tree = DGHs[attribute]
        generalized_node = None

        for record in generalized_cluster:
            # Debugging checks
            if record is None:
                print("Error: record is None")
                continue
            if attribute not in record:
                print(f"Error: Attribute '{attribute}' not found in record")
                continue

            current_node = tree.search(record[attribute])
            if current_node is None:
                print(f"Error: No node found in tree for attribute '{attribute}' with value '{record[attribute]}'")
                continue
            # End of debugging checks

            if generalized_node is None:
                generalized_node = current_node
            else:
                generalized_node = tree.generalize(generalized_node, current_node)

        # Apply the generalized value to each record in the cluster for this attribute
        if generalized_node is not None:
            generalized_value = generalized_node.data
            for record in generalized_cluster:
                record[attribute] = generalized_value

    return generalized_cluster



class CustomData:
    def __init__(self,index,score):
        self.index = index
        self.score = score

    def __lt__(self, other):
        if isinstance(other, CustomData):
            return (self.score < other.score())
        else:
            return NotImplemented
class Tree:
    def __init__(self, root_data):
        self.root = Node(root_data, 0)

    @classmethod
    def from_node(cls, root_node):
        tree = cls.__new__(cls)
        tree.root = root_node
        return tree

    def get_external_nodes_count(self):
        # Start the count from the root node
        return self._count_external_nodes(self.root)

    def print_tree(self, node=None, prefix=""):
        if node is None:
            node = self.root

        # This is the root node, so print the node data.
        if prefix == "":
            print(node.data)
        else:
            # Print the branch (prefix) and the current node data
            print(prefix + "└── " + node.data)

        # Prepare the prefix for the child nodes
        child_prefix = prefix + "    " if prefix else "|   "

        # Recursively print each child, updating the prefix
        for i, child in enumerate(node.children):
            # If this is the last child, don't extend the vertical bar (|)

            next_prefix = child_prefix
            if i < len(node.children) - 1:
                next_prefix += "|   "
            else:
                next_prefix += "    "

            self.print_tree(child, next_prefix)

    def _count_external_nodes(self, node):
        # Base case: If the node is None, return 0
        if node is None:
            return 0

        # Check if the node is external
        if node.is_external():
            return 1
        else:
            # If the node is not external, count the external nodes in its children
            count = 0
            for child in node.children:
                count += self._count_external_nodes(child)
            return count

    def search(self, target, node=None):

        if node is None:
            node = self.root

        # Check if the current node contains the target value
        if node.data == target:
            return node

        # Recursively search in each child
        for child in node.children:
            result = self.search(target, child)
            if result is not None:
                return result
        return None

    def generalize(self, NodeA, NodeB):

        if NodeA.data == NodeB.data:
            return NodeA

        heightA = NodeA.height
        heightB = NodeB.height

        if heightA == heightB:
            C = NodeA.parent
            D = NodeB.parent
            return self.generalize(C, D)

        elif heightA < heightB:
            C = NodeB.parent
            return self.generalize(NodeA, C)

        else:
            D = NodeA.parent
            return self.generalize(D, NodeB)
class Node:
    def __init__(self,data,height,parent = None):
        self.data= data
        self.height = height
        self.parent = parent
        self.children = []

    def add_child(self, child_node):
        child_node.parent = self
        self.children.append(child_node)

    def is_external(self):
        return len(self.children) == 0



##############################################################################
# Mandatory Functions                                                        #
# You need to complete these functions without changing their parameters.    #
##############################################################################


#search for both nodes
#find the difference of their height
#do that for every entry



def cost_MD(raw_dataset_file: str, anonymized_dataset_file: str,
    DGH_folder: str) -> float:

    raw_dataset = read_dataset(raw_dataset_file)
    anonymized_dataset = read_dataset(anonymized_dataset_file)
    assert(len(raw_dataset)>0 and len(raw_dataset) == len(anonymized_dataset)
        and len(raw_dataset[0]) == len(anonymized_dataset[0]))
    DGHs = read_DGHs(DGH_folder)

    #TODO: complete this function.
    total_cost = 0
    attribute_names = list(raw_dataset[0].keys())[:-1]  # Exclude the last entry as it's not a Qi

    # Iterate through each row in the datasets (excluding the header row)
    for i in range(0, len(raw_dataset)):
        raw_entry = raw_dataset[i]
        anon_entry = anonymized_dataset[i]
        cost = costMDBetweenTwoRecords(raw_entry,anon_entry,DGHs,attribute_names)
        total_cost += cost


    return total_cost

# take the anonymized one, construct another tree with that with making it root
# take the external nodes count of that tree
# at the denominator write the normal external count of the all tree

def cost_LM(raw_dataset_file: str, anonymized_dataset_file: str,
    DGH_folder: str) -> float:
    """Calculate Loss Metric (LM) cost between two datasets.

    Args:
        raw_dataset_file (str): the path to the raw dataset file.
        anonymized_dataset_file (str): the path to the anonymized dataset file.
        DGH_folder (str): the path to the DGH directory.

    Returns:
        float: the calculated cost.
    """
    raw_dataset = read_dataset(raw_dataset_file)
    anonymized_dataset = read_dataset(anonymized_dataset_file)
    assert(len(raw_dataset)>0 and len(raw_dataset) == len(anonymized_dataset)
        and len(raw_dataset[0]) == len(anonymized_dataset[0]))
    DGHs = read_DGHs(DGH_folder)

    #TODO: complete this function.
    total_lm_cost = 0
    attribute_names = list(raw_dataset[0].keys())[:-1]  # Exclude the last entry as it's not a Qi

    total_qi= len(attribute_names )
    weight = 1/total_qi
    lm_costs = 0

    # Iterate through each row in the datasets (excluding the header row)

    for i in range(0, len(raw_dataset)):
        raw_entry = raw_dataset[i]
        anon_entry = anonymized_dataset[i]
        total_lm_cost += costLMOfRecord(anon_entry,weight,DGHs,attribute_names)
        # Iterate through each attribute in the entry


    return total_lm_cost


def random_anonymizer(raw_dataset_file: str, DGH_folder: str, k: int,
    output_file: str, s: int):
    """ K-anonymization a dataset, given a set of DGHs and a k-anonymity param.

    Args:
        raw_dataset_file (str): the path to the raw dataset file.
        DGH_folder (str): the path to the DGH directory.
        k (int): k-anonymity parameter.
        output_file (str): the path to the output dataset file.
        s (int): seed of the randomization function
    """
    raw_dataset = read_dataset(raw_dataset_file)
    DGHs = read_DGHs(DGH_folder)    

    for i in range(len(raw_dataset)): ##set indexing to not lose original places of records
        raw_dataset[i]['index'] = i

    raw_dataset = np.array(raw_dataset)
    np.random.seed(s) ## to ensure consistency between runs
    np.random.shuffle(raw_dataset)  ##shuffle the dataset to randomize

    clusters = []

    D = len(raw_dataset)
    
    #TODO: START WRITING YOUR CODE HERE. Do not modify code in this function above this line.
    # Store your results in the list named "clusters". 
    # Order of the clusters is important. First cluster should be the first EC, second cluster second EC, ...

    for i in range(0, len(raw_dataset), k):
        cluster = raw_dataset[i:i+k]


        if(len(cluster)<k):
            old_cluster = clusters[-1]
            clusters.pop()
            cluster = np.concatenate((old_cluster, cluster))

        clusters.append(cluster)

    for cluster in clusters:
        for attribute in DGHs.keys():  # For each attribute
            tree = DGHs[attribute]  # Get the DGH tree for the attribute
            generalized_node = None
            for record in cluster:
                current_node = tree.search(record[attribute])
                if generalized_node is None:
                    generalized_node = current_node
                else:
                    generalized_node = tree.generalize(generalized_node, current_node)
            # Apply the generalized value to each record in the cluster for this attribute
            for record in cluster:
                record[attribute] = generalized_node.data

    # END OF STUDENT'S CODE. Do not modify code in this function below this line.

    anonymized_dataset = [None] * D

    for cluster in clusters:        #restructure according to previous indexes
        for item in cluster:
            anonymized_dataset[item['index']] = item
            del item['index']

    write_dataset(anonymized_dataset, output_file)



def clustering_anonymizer(raw_dataset_file: str, DGH_folder: str, k: int, output_file: str):
    raw_dataset = read_dataset(raw_dataset_file)
    DGHs = read_DGHs(DGH_folder)
    ec_records_indexes = []

    anonymized_dataset = [None] * len(raw_dataset)
    used_records = [0] * len(raw_dataset)  # 0 means unused

    while used_records.count(0)>=k:
        index = used_records.index(0)
        used_records[index] = 1  # Mark as used
        current_record = raw_dataset[index]
        list_of_all_customDatas = []

        for i in range(len(raw_dataset)):
            if i != index and used_records[i] == 0:
                hypothetical_cost = 0
                for attribute in DGHs.keys():
                    tree = DGHs[attribute]
                    generalized_node = tree.generalize(tree.search(current_record[attribute]), tree.search(raw_dataset[i][attribute]))
                    hypothetical_cost += LMHelper(generalized_node, tree)
                list_of_all_customDatas.append(CustomData(i, hypothetical_cost))

        list_of_all_customDatas.sort(key=lambda x: x.score)
        closest_records_indexes = [data.index for data in list_of_all_customDatas[:k-1]]

        for record_index in closest_records_indexes:
            used_records[record_index] = 1

        ec_records_indexes = [index] + closest_records_indexes
        ec_cluster = [raw_dataset[i] for i in ec_records_indexes]

        generalized_ec = generalize_ec(ec_cluster, DGHs)

        for record_index, gen_record in zip(ec_records_indexes, generalized_ec):
            anonymized_dataset[record_index] = gen_record

        # Handle the remaining records if the dataset size is not a perfect multiple of k

        # Handle the remaining records if the dataset size is not a perfect multiple of k
    if(used_records.count(0) > 0):

        remaining_indices =[i for i, used in enumerate(used_records) if used == 0]

        for rem_i in remaining_indices:
            ec_records_indexes.append(rem_i)

        ec_cluster = [raw_dataset[i] for i in ec_records_indexes]
        generalized_ec = generalize_ec(ec_cluster, DGHs)

        for record_index, gen_record in zip(ec_records_indexes, generalized_ec):
            anonymized_dataset[record_index] = gen_record



    write_dataset(anonymized_dataset, output_file)



def topdown_anonymizer(raw_dataset_file: str, DGH_folder: str, k: int,
    output_file: str):
    """ Top-down anonymization a dataset, given a set of DGHs.

    Args:
        raw_dataset_file (str): the path to the raw dataset file.
        DGH_folder (str): the path to the DGH directory.
        k (int): k-anonymity parameter.
        output_file (str): the path to the output dataset file.
    """
    raw_dataset = read_dataset(raw_dataset_file)
    DGHs = read_DGHs(DGH_folder)

    anonymized_dataset = []
    len(raw_dataset)
    #TODO: complete this function.

    write_dataset(anonymized_dataset, output_file)


#Uncomment below code to see all dghs
"""
dgh_folder_path = 'DGHs/'
dghs = read_DGHs(dgh_folder_path)


# Print each tree
for attribute_name, tree in dghs.items():
    print(f"\nTree for {attribute_name}:")
    tree.print_tree()

    print(f"\n external nodes:  { tree.get_external_nodes_count()}")

"""
#Uncomment below code to see plots and performance

"""

k = [4, 8, 16, 32, 64, 128, 256]
k.reverse()

time_costs_cluster = []  # To store time costs
time_costs_random = []  # To store time costs
md_costs_cluster = []    # To store MD costs
md_costs_random = []    # To store MD costs
lmcosts_cluster = []     # To store LM costs
lmcosts_random = []     # To store LM costs


for _ in k:
    t1 = time.time()
    clustering_anonymizer("mini-adult1.csv", "DGHs/", _, "result100.csv")
    elapsed_t1 = time.time() - t1
    time_costs_cluster.append(elapsed_t1)

    t2 = time.time()
    random_anonymizer("mini-adult1.csv", "DGHs/", _, "result101.csv", 0)
    elapsed_t2 = time.time() - t2
    time_costs_random.append(elapsed_t2)

    cost_md1 = cost_MD("mini-adult1.csv", "result100.csv", "DGHs/")
    md_costs_cluster.append(cost_md1)
    cost_lm1 = cost_LM("mini-adult1.csv", "result100.csv", "DGHs/")
    lmcosts_cluster.append(cost_lm1)

    cost_md2 = cost_MD("mini-adult1.csv", "result101.csv", "DGHs/")
    md_costs_random.append(cost_md2)
    cost_lm2 = cost_LM("mini-adult1.csv", "result101.csv", "DGHs/")
    lmcosts_random.append(cost_lm2)
    

    print(f"k={_}, Clustering Anonymizer-Time Cost: {elapsed_t1} seconds, Cost MD: {cost_md1}, Cost LM: {cost_lm1}")
    print(f"k={_}, Random Anonymizer-Time Cost: {elapsed_t2} seconds, Cost MD: {cost_md2}, Cost LM: {cost_lm2}")

plt.figure(figsize=(18, 6))

# Time vs. k
plt.subplot(1, 3, 1)
plt.plot(k, time_costs_cluster, marker='o', label='Clustering', color='black')
plt.plot(k, time_costs_random, marker='o', label='Random', color='red')
plt.title('Time vs. k')
plt.xlabel('k')
plt.ylabel('Time (seconds)')
plt.legend()

# Cost MD vs. k
plt.subplot(1, 3, 2)
plt.plot(k, md_costs_cluster, marker='o', label='Clustering', color='black')
plt.plot(k, md_costs_random, marker='o', label='Random', color='red')
plt.title('Cost MD vs. k')
plt.xlabel('k')
plt.ylabel('MD Cost')
plt.legend()

# Cost LM vs. k
plt.subplot(1, 3, 3)
plt.plot(k, lmcosts_cluster, marker='o', label='Clustering', color='black')
plt.plot(k, lmcosts_random, marker='o', label='Random', color='red')
plt.title('Cost LM vs. k')
plt.xlabel('k')
plt.ylabel('LM Cost')
plt.legend()

plt.tight_layout()  # To avoid overlapping labels
plt.show()
"""

# Command line argument handling and calling of respective anonymizer:
if len(sys.argv) < 6:
  print(f"Usage: python3 {sys.argv[0]} algorithm DGH-folder raw-dataset.csv anonymized.csv k")
  print(f"\tWhere algorithm is one of [clustering, random, topdown]")
  sys.exit(1)

algorithm = sys.argv[1]
if algorithm not in ['clustering', 'random', 'topdown']:
  print("Invalid algorithm.")
  sys.exit(2)

dgh_path = sys.argv[2]
raw_file = sys.argv[3]
anonymized_file = sys.argv[4]
k = int(sys.argv[5])

function = eval(f"{algorithm}_anonymizer")
if function == random_anonymizer:
  if len(sys.argv) < 7:
      print(f"Usage: python3 {sys.argv[0]} algorithm DGH-folder raw-dataset.csv anonymized.csv k seed(for random only)")
      print(f"\tWhere algorithm is one of [clustering, random, topdown]")
      sys.exit(1)
      
  seed = int(sys.argv[6])
  t1=time.time()
  function(raw_file, dgh_path, k, anonymized_file, seed)
  elapsed_t1 =time.time()-t1
else:
  t1=time.time()
  function(raw_file, dgh_path, k, anonymized_file)
  elapsed_t1 =time.time()-t1


print(f"{algorithm} -Time Cost: {elapsed_t1} seconds")

cost_md = cost_MD(raw_file, anonymized_file, dgh_path)
print(f"Cost MD:  {cost_md} ")

cost_lm = cost_LM(raw_file, anonymized_file, dgh_path)
print(f"Cost LM:  {cost_lm} ")

#print (f"Results of {k}-anonimity:\n\tCost_MD: {cost_md}\n\tCost_LM: {cost_lm}\n")


# Sample usage:
# python3 code.py clustering DGHs/ adult-hw1.csv result.csv 300
