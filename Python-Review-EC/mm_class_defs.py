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
