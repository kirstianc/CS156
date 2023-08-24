# coding: utf-8     # <- This is an encoding declaration  REQUIRED
"""
In Python 3.0+ the default encoding of source files is already UTF-8 so you can safely delete that line,
but because unless it says something other than some variation of "utf-8", it has no effect.
See Should I use encoding declaration in Python 3?
"""

#NAME:  cs156_pr3.py
"""
Python review of lists.
"""

"""
Imports
"""
import os


# Main program
def main():
    """This is the main program of the Python examples of lists."""

    # Create a list
    l1 = [1, 2, 'a', 'uy65', 8]
    print("Creating list l1 = [1, 2, 'a', 'uy65', 8] =", l1)

    # Adding an item to a list
    l1.append('d')
    print("Appending 'd' to l1 via  l1.append('d') = ", l1)

    # Removing an item from a list
    l1.remove(8)
    print("Removing 8 from l1 via  l1.remove(8) = ", l1)

    # Referencing/indexing a location in a list
    print("The third item in list l1 {} is {}".format(l1, l1[2]))

    # Looping through a list
    for indx in range(len(l1)):
        print("Item # {} in l1 is {}".format(indx, l1[indx]))

    for itm in l1:
        print("The item is ", itm)

    # Sort a list
    l2 = ['h', 'f', 'e', 's']
    print("Sorting a list =", l2)
    l2.sort()
    print("Sorted list =", l2)

    print()
    # List comprehenmsion
    # example 1
    import numpy as np
    a = [4, 6, 7, 3, 2]
    print("a =", a)
    print("Executing b = [x for x in a if x > 5]")
    b = [x for x in a if x > 5]
    print("b = ", b)
    print()

    a = [4, 6, 7, 3, 2]
    print("Executing b = [x * 2 for x in a if x > 5]")
    b = [x * 2 for x in a if x > 5]
    print("b = ", b)
    print()

    # Iterating over a matrix  Finding the max number in each row
    A = np.random.randint(10, size=(4, 4))
    print("The array matrix A = \n", A)
    max_element = [max(i) for i in A]
    print("Finding max number in each row via max_element = [max(i) for i in A]", max_element)
    print()


"""
Module execute/load check.   REQUIRED
"""
if __name__ == '__main__':
    print("cs156_pr2.py:  Module is executed.")
    main()
else:
    print("cs156_pr2.py:  Module is imported.")