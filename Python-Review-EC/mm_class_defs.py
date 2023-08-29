# coding: utf-8
#NAME:  mm_class_defs.py
#DESCRIPTION: contain class definitions that will be imported and used to create and
#             manipulate Min-Max Tree objects 

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

========================== MODIFICATION HISTORY ==============================
08/29/23:
    MOD:     Creation of file and initial organization
    AUTHOR:  Ian Chavez
    COMMENT: n/a
====================== END OF MODIFICATION HISTORY ============================
"""

class MM_Node:
    """
    slots/attributes --------------------------------
    """
    name
    value
    left_child
    right_child 
    parent
    """
    getters --------------------------------
    """
    def get_name(self):
        return self.name
    
    def get_value(self):
        return self.value
    
    def get_parent(self):
        return self.parent
    
    def get_left_child(self):
        return self.left_child
    
    def get_right_child(self):
        return self.right_child
    
    def get_children(self):
        return (self.left_child, self.right_child)
    """
    setter functions --------------------------------
    """
    def set_name(self, name):
        self.name = name

    def set_value(self, value):
        self.value = value

    def set_parent(self, parent):
        self.parent = parent
    
    def set_left_child(self, left_child):
        self.left_child = left_child
    
    def set_right_child(self, right_child):
        self.right_child = right_child

    def set_children(self, left_child, right_child):
        self.left_child = left_child
        self.right_child = right_child
    """
    other functions --------------------------------
    """
    def __str__ (self):
        return "NAME: " + str(self.name) + "\nVALUE: " + str(self.value) + "\nPARENT: " + str(self.parent) + "\nLEFT CHILD: " + str(self.left_child) + "\nRIGHT CHILD: " + str(self.right_child)

    def __init__(self):
        self.name = None
        self.value = None
        self.left_child = None
        self.right_child = None
        self.parent = None

    def __init__(self, name):
        self.name = name
        self.value = None
        self.left_child = None
        self.right_child = None
        self.parent = None

    def __init__(self, name, value):
        self.name = name
        self.value = value
        self.left_child = None
        self.right_child = None
        self.parent = None

class MM_tree: 
    """
    slots/attributes --------------------------------
    """
    name 
    nodes
    number_of_nodes
    """
    getter functions --------------------------------
    """
    def get_name(self):
        return self.name
    
    def get_nodes(self):
        return self.nodes
    
    def get_count(self):
        return self.number_of_nodes
    """
    setter functions --------------------------------
    """
    def set_name(self, name):
        self.name = name
    
    def set_nodes(self, nodes):
        self.nodes = nodes
    
    def set_count(self, number_of_nodes):
        self.number_of_nodes = number_of_nodes
    """
    other functions --------------------------------
    """
    def __init__(self):
        self.name = None
        self.nodes = {}
        self.number_of_nodes = 0
    
    def __init__(self, name):
        self.name = name
        self.nodes = {}
        self.number_of_nodes = 0
    
    def __str__(self):
        return "NAME: " + str(self.name) + "\nNODES: " + str(self.nodes) + "\nNUMBER OF NODES: " + str(self.number_of_nodes)
    
def insert_node(self, name, node, overwrite):
    if name in self.nodes:
        if overwrite:
            existing_node = self.nodes[name]
            existing_parent = existing_node.parent

            existing_node.name = node.name
            existing_node.value = node.value

            if existing_parent is not None:
                if node.value < existing_parent.value:
                    existing_parent.left_child = existing_node
                else:
                    existing_parent.right_child = existing_node

            if node.left_child and node.left_child.value > node.value:
                node.right_child = node.left_child
                node.left_child = None

            if node.right_child and node.right_child.value < node.value:
                node.left_child = node.right_child
                node.right_child = None

            node.parent = existing_parent

            return True
        else:
            return False
    else:
        self.nodes[name] = node
        return True

    
    def delete_node(self, name):
        if name in self.nodes:
            del self.nodes[name]
            return True
        else:
            return False
    

"""
    other functions
    • insert_node - "takes 3 arguments -         
        This function 
            - checks the values of the “parent”, “left_child”, and “right_child” slots of the new inserted node
            - update any AND all existing nodes in the tree as appropriate.
        
        Error messages should be printed if found. 
            e.g. if the new inserted node specifies a left or right child node that already has a parent node 
                    then an error message should be printed. 
        
        If the new inserted node specifies a parent node that does not yet exist in the tree, 
            an error message should be printed. 
        
        If the new inserted node specifies a parent node that already has left-child and 
            right-child slot values that are not None, then an error message should be printed.

    • delete_node - "Takes one argument which is the name of the mm_node to remove from the tree.
            If the specified node to delete does not exist in the tree --> nothing is done and False is returned. 
            If the node exists, the instance and associated key in the “nodes” dictionary --> removed.
            
            The remaining nodes, if any, in the tree must be updated appropriately. 
            
            e.g. if the deleted node is the parent of one or two nodes, 
            then those child nodes must have their parent slot values updated and set to None 
            
            Similar checks must be made for child node information.
"""
