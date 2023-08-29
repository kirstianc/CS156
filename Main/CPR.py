# coding: utf-8
#NAME:  CPR.py
#DESCRIPTION: This will be a program to display all credit cards and due dates + recommended payment dates. 
#             This is used to help me keep track of my credit cards and when I should pay them.
#             This will also be used to review Python.

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
08/28/23:
    MOD:     Creation of file and initial organization
    AUTHOR:  Ian Chavez
    COMMENT: n/a
====================== END OF MODIFICATION HISTORY ============================
"""
import os

credit_card_companies = {}

def print_dict (input):
    os.system('cls')
    print("=====================================================================")
    print("Companies :: Due Dates")
    for key, value in input.items():
        print(key, ":",value)
    print("=====================================================================")

def print_clr (input):
    os.system('cls')
    print("=====================================================================") 
    print(input)
    print("=====================================================================") 

"""cannot Overload functions in Python..? -change to print_clr2"""
def print_clr2 (input, input2): 
    os.system('cls')
    print("=====================================================================") 
    print(input)
    print(input2)
    print("=====================================================================") 


def add_cc (user_input):
    if(user_input == "Y" or user_input == "y"):
        print_clr("Please enter the name of the credit card company. (Example: Chase, Discover, etc.)")
        credit_card_company = input()

        print_clr("Please enter the due date of the credit card. (Example: 1, 15, 27, etc.)")
        credit_card_due_date = input()

        credit_card_companies[credit_card_company] = credit_card_due_date
        print_clr("Credit card company and due date added. Do you have another credit card to add? (Y/N)")
        user_input = input()
        if(user_input == "Y" or user_input == "y"):
            add_cc("y")
        else:
            print_clr("Thank you for using CPR.py. Goodbye.")

def main ():
    print_clr2("Hello welcome to CPR.py.", "Would you like to input a credit card and it's due date? (Y/N)") 
    
    user_input = input()
    add_cc(user_input)

    print_clr("Would you like to see what credit cards you have and their due dates? (Y/N)")
    user_input = input()
    if(user_input == "Y" or user_input == "y"):
        print_dict(credit_card_companies)
    else:
        print_clr("Thank you for using CPR.py. Goodbye.")    

    print("CPR.py:  Main function is executed.")

"""
Module execute/load check
"""
if __name__ == '__main__':
    main()
    print("CPR.py:  Module is executed.")

else:
    print("CPR.py:  Module is imported.")

