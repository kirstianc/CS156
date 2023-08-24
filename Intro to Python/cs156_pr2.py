# coding: utf-8     # <- This is an encoding declaration  REQUIRED
"""
In Python 3.0+ the default encoding of source files is already UTF-8 so you can safely delete that line,
but because unless it says something other than some variation of "utf-8", it has no effect.
See Should I use encoding declaration in Python 3?
"""

#NAME:  cs156_pr2.py
"""
Python review of strings.
"""

"""
Imports
"""
import string


# Main program
def main():
    """This is the main program of the Python examples of strings."""

    print('This is a full sentence.')
    print("This is also a full sentence.")

    # Concatenation
    greeting = 'Good morning'
    print('Please enter your name:')
    #userName = input()
    #print(greeting + ', ' + userName + '.')

    # Escape Characters
    """ \n 	Prints a new line
        \t 	Prints a tab
        \' 	Prints a single quotation mark
        \'' 	Prints a double quotation mark
    """
    print('This is line 1\n\nThis is line 3')


    # Slicing
    s1 = 'lkasjdklswiperuihfdjfhdfh'
    print("String s1 =", s1)
    print("Length of s1 = ", len(s1))
    print("Get the first five characters via s1[:5] =", s1[:5])
    print("Get characters 6 to 10 via s1[6:11] =", s1[6:11])
    print("Get the last charcter via s1[-1] =", s1[-1])
    print("Get the last three characters cia s1[-3:] =", s1[-3:])
    print("Reverse a string via s1[::-1] =", s1[::-1])


    # Making strings upper, lower, title, and capitalized case
    print("s1 uppercase = ", s1.upper())
    print("Making 'ABC' lower case =", 'ABC'.lower())
    print("Making s1 Capitalized = ", s1.capitalize())
    print("Making 'hello there' title case =", 'hello there'.title())

    # Spliting strings at specified character
    print("Spliting {} at the character {} = {}".format(s1, 'f', s1.split('f')))


"""
Module execute/load check.   REQUIRED
"""
if __name__ == '__main__':
    print("cs156_pr2.py:  Module is executed.")
    main()
else:
    print("cs156_pr2.py:  Module is imported.")