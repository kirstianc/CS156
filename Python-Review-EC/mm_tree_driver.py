# coding: utf-8
#NAME:  MM_tree_driver.py
#DESCRIPTION: contains main driver function for mm_class_defs.py

"""
AUTHOR: Ian Chavez

   Unpublished-rights reserved under the copyright laws of the United States.

   This data and information is proprietary to, and a valuable trade secret
   of, Leonard P. Wesley and Ian Chavez. It is given in confidence by Leonard
   P. Wesley and Ian Chavez. Its use, duplication, or disclosure is subject to
   the restrictions set forth in the License Agreement under which it has been
   distributed.

      Unpublished Copyright © 2022  Leonard P. Wesley and Ian Chavez
      All Rights Reserved
"""
# import the MM_node and MM_tree classes from mm_class_defs module
from mm_class_defs import MM_node, MM_tree

def main():
    # prepare the console
    print("=================\nMin-Max Tree Driver:\n=================")

    # create a Min-Max Tree
    tree = MM_tree("MyTree")
    print("Tree created:", tree)

    # create and add nodes to the tree
    print("=================\nCreating nodes:\n=================")
    node1 = MM_node("Node1", 10)
    tree.insert_node("MyTree", node1, True) 
    print("Node created:\n", node1)
    print("-----------------")

    node2 = MM_node("Node2", 20)
    tree.insert_node("Node1", node2, True) 
    print("Node created:\n", node2)
    print("-----------------")

    node3 = MM_node("Node3", 15)
    tree.insert_node("Node1", node3, True)  
    print("Node created:\n", node3)
    print("-----------------")

    node4 = MM_node("Node4", 30)
    tree.insert_node("Node2", node4, True)  
    print("Node created:\n", node4)
    print("-----------------")

    node5 = MM_node("Node5", 5)
    tree.insert_node("Node3", node5, True) 
    print("Node created:\n", node5)
    print("-----------------")

    node6 = MM_node("Node6", 25)
    tree.insert_node("Node3", node6, True) 
    print("Node created:\n", node6)

    # print the tree structure
    print("=================\nMin-Max Tree Structure:")
    print(tree)

    # delete nodes
    tree.delete_node("Node4")
    print("=================\nNode deleted:", node4)

    tree.delete_node("Node5")
    print("Node deleted:", node5)

    # print the updated tree structure
    print("=================\nUpdated Min-Max Tree Structure:")
    print(tree)
    print("=================\nEnd of Min-Max Tree Driver")

if __name__ == "__main__":
    main()
