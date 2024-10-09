import pickle

import numpy as np

with open("../../Graph/ShenZhen/SZ_nodes_extended.txt", "r") as file:
    my_node_list = eval(file.readline())

my_node_list = my_node_list[0]

cs = np.array([1,1,0],dtype=float)
print(cs)

print(my_node_list)
existPlan = [[my_node_list,cs,{}]]

print(existPlan)

pickle.dump(existPlan, open("../../Graph/ShenZhen/SZ_existingplan.pkl", "wb"))