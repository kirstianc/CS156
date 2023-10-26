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
"""

class MM_node:
    """
    slots/attributes --------------------------------
    """

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

class MM_tree: 
    """
    slots/attributes --------------------------------
    """
    def __init__(self):
        self.name = None
        self.nodes = {}
        self.number_of_nodes = 0
    
    def __init__(self, name):
        self.name = name
        self.nodes = {}
        self.number_of_nodes = 0
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
    def __str__(self):
        return "NAME: " + str(self.name) + "\nNODES: " + str(self.nodes) + "\nNUMBER OF NODES: " + str(self.number_of_nodes)
    
    def insert_node(self, name, node, overwrite):
        if name in self.nodes:
            if overwrite:
                
                # error checking #
                # check if parent node of inserting node is same as overwriting node
                if node.get_parent() != self.nodes[name].get_parent():
                    print("ERROR: parent node mismatch")
                    return False
                
                # check if inserting node's parent node exists
                if name not in self.nodes:
                    print("ERROR: parent node does not exist")
                    return False
                
                # check if inserting node's parent node already has two children & if inserting node is parent node's child
                if node.get_parent() != None and node.get_parent().get_left_child() != None and node.get_parent().get_right_child() != None and node.get_parent().get_left_child() != name and node.get_parent().get_right_child() != name:
                    print("ERROR: parent node already has two children")
                    return False

                # overwrite #
                self.nodes[name] = node
                return True
            
            else:
                # do not overwrite thus nothing done #
                return False
            
        else:
            # insert node #
            self.nodes[name] = node
            return True

        
    def delete_node(self, name):
        # check if node exists #
        if name in self.nodes:
            del self.nodes[name]

            # update tree to reflect deletion #
            for node in self.nodes:
                # check if node is parent of deleted node #
                if self.nodes[node].get_left_child() == name:
                    self.nodes[node].set_left_child(None)
                if self.nodes[node].get_right_child() == name:
                    self.nodes[node].set_right_child(None)
                # check if node is child of deleted node #
                if self.nodes[node].get_parent() == name:
                    self.nodes[node].set_parent(None)
        
            return True
        else:
            # node does not exist, do nothing #
            return False
