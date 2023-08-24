# coding: utf-8
# NAME:  cs156_pr4.py

"""
Python review of classes
"""

class MyPass:
    pass

# Create a class named MyClass, with a property named x:
class MyClass:
  x = 5

# Create a class named Person
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

class Person_With_str:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def __str__(self):
        return f"{self.name}({self.age})"

class Person_With_method:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def myfunc(self):
        print("Hello my name is " + self.name)

class Person_With_self:
  def __init__(mysillyobject, name, age):
    mysillyobject.name = name
    mysillyobject.age = age

  def myfunc(abc):
    print("Hello my name is " + abc.name)




# MAIN function
def main():
    """Demonstrastion of Python Class objects."""

    """
    Create an instance of a class Object
    Now we can use the class named MyClass to create objects:
    Create an object named p1, and print the value of x:
    """
    p1 = MyClass()
    print("Printing p1.x:", p1.x)
    print()

    """
    The __init__() Function
    Create a class named Person, use the __init__() function to assign values for name and age:
    All classes have a function called __init__(), which is always executed when the class is being initiated.
    Use the __init__() function to assign values to object properties, or other operations that are 
    necessary to do when the object is being created:
    """
    print("Creating an instance of class Person.")
    p1 = Person("John", 36)
    print("Printing the name and age of the created instance.")
    print(p1.name)
    print(p1.age)
    print("Note: The __init__() function is called automatically every time the class is being used to create a new object.")
    print()

    """
    The __str__() Function
    The __str__() function controls what should be returned when the class object is represented as a string.
    If the __str__() function is not set, the string representation of the object is returned:
    """
    print("The string representation of an object WITHOUT the __str__() function:")
    p1 = Person("John", 36)
    print(p1)
    print()
    print("The string representation of an object WITH the __str__() function:")
    p1 = Person_With_str("John", 36)
    print(p1)
    print()

    """
    Object Methods
    Objects can also contain methods. Methods in objects are functions that belong to the object.
    Let us create a method in a new Person_With_method class:
    """
    print("Print a greeting, and execute it on the p1 object:")
    p1 = Person_With_method("John", 36)
    p1.myfunc()
    print()

    """
    The self Parameter
    The self parameter is a reference to the current instance of the class, and is used to access 
    variables that belongs to the class.
    It does not have to be named self , you can call it whatever you like, but it has to be the 
    first parameter of any function in the class:
    """
    print("Use the words mysillyobject and abc instead of self:")
    p1 = Person_With_self("John", 36)
    p1.myfunc()
    print()

    """
    Modify Object Properties
    You can modify properties on objects:
    """
    print("Modify the age of the person instance to set age to 40.")
    print("p1 age before changing = ", p1.age)
    p1.age = 40
    print("p1 age after changing = ", p1.age)
    print()

    """
    Delete Object Properties
    You can delete properties on objects by using the del keyword:
    """
    print("Delete object properties using the 'del' keyword.")
    print("For Example:  del  p1.age")
    print()

    """
    The pass Statement
    class definitions cannot be empty, but if you for some reason have a 
    class definition with no content, put in the pass statement to avoid 
    getting an error.
    """
    print("The pass Statement\n class definitions cannot be empty, but if you for some reason have a \n class definition with no content, put in the pass statement to avoid\n getting an error.")
    print("\nFor Example:\nclass Person:\n   pass")
    print()

    # Return from main function
    return



"""
Module execute/load check.   
"""
if __name__ == '__main__':
    main()
else:
    print("cs156_pr4.py module is imported but is intended to be executed.")