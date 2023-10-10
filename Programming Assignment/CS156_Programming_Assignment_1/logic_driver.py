#NAME:  logic_driver.py

from logic import *
from agents import *
from utils import *


################################################################################
def main():
    """ Main driver program for Logic Based Agents module."""

    print("STARTING TEST OF AIMA 3rd Ed LOGIC FUNCTIONS (CHAPS 7 TO 10)")
    print()

    ### PropKB
    kb = PropKB()
    print ("1: kb.clauses = ", kb.clauses)

    print ("2: kb.tell(A & B)")
    kb.tell(A & B)
    print ("3: kb.clauses = ", kb.clauses)
    kb.ask(A)
    print("For '.ask', the result {} means true, with no substitutions.")
    print("3.1: kb.ask(A) = ", kb.ask(A))
    print()

    print ("4: kb.tell(B => C)")
    kb.tell(B >> C)
    print ("5: kb.clauses = ", kb.clauses)

    kb.ask(C) ## The result {} means true, with no substitutions
    print()
    print ("For '.ask', the result {} means true, with no substitutions.")
    print ("6: kb.ask(C) = ", kb.ask(C))

    kb.ask(P)
    print ("7: kb.ask(P) = ", kb.ask(P))

    kb.retract(B)
    print ("8: kb.retract(B) = ", kb.clauses)
    kb.ask(C)
    print ("9: kb.ask(C) = ", kb.ask(C))

    print()
    print ("10: pl_true(P, {}) = ", pl_true(P, {}))
    pl_true(P, {})
    print ("11: pl_true(P | Q, {P: True}) = ", pl_true(P | Q, {P: True}))
    pl_true(P | Q, {P: True})


    # Notice that the function pl_true cannot reason by cases:
    print ("12: pl_true(P | ~P) = ", pl_true(P | ~P) )
    pl_true(P | ~P)

    # However, tt_true can:
    print ("13: tt_true(P | ~P) = ", tt_true(P | ~P))
    tt_true(P | ~P)


    # The following are tautologies from [Fig. 7.11]:
    print()
    print("TESTING TAUTOLOGIES")
    print('14: tt_true("(A & B) <=> (B & A)") = ', tt_true("(A & B) <=> (B & A)"))


    print('15: tt_true("(A | B) <=> (B | A)") = ', tt_true("(A | B) <=> (B | A)") )

    print('16: tt_true("((A & B) & C) <=> (A & (B & C))") = ', tt_true("((A & B) & C) <=> (A & (B & C))") )


    print('17: tt_true("((A | B) | C) <=> (A | (B | C))") = ', tt_true("((A | B) | C) <=> (A | (B | C))") )

    print('18: tt_true("~~A <=> A") = ', tt_true("~~A <=> A") )

    print('22: tt_true("~(A & B) <=> (~A | ~B)") = ', tt_true("~(A & B) <=> (~A | ~B)") )

    print('23: tt_true("~(A | B) <=> (~A & ~B)") = ', tt_true("~(A | B) <=> (~A & ~B)") )

    print('24: tt_true("(A & (B | C)) <=> ((A & B) | (A & C))") = ', tt_true("(A & (B | C)) <=> ((A & B) | (A & C))") )

    print('25: tt_true("(A | (B & C)) <=> ((A | B) & (A | C))") = ', tt_true("(A | (B & C)) <=> ((A | B) & (A | C))") )

    # The following are not tautologies:
    print()
    print('The following are not tautologies')
    print('26: tt_true(A & ~A) = ', tt_true(A & ~A))

    print('27: tt_true(A & B) = ', tt_true(A & B) )



    ### Unification:
    print()
    print("CHECKING UNIFICATION")
    print('33: unify(x, x, {}) = ', unify(x, x, {}) )

    print('34: unify(x, 3, {}) = ', unify(x, 3, {}) )

    print()
    print('35: to_cnf((P&Q) | (~P & ~Q)) = ', to_cnf((P&Q) | (~P & ~Q)) )

    print()
    print("DONE")

    print("Saving to file: logic_driver_test.txt")
    f = open("logic_driver_test.txt", "w")
    f.write("STARTING TEST OF AIMA 3rd Ed LOGIC FUNCTIONS (CHAPS 7 TO 10)\n")
    f.write("\n")
    f.write("1: kb.clauses = " + str(kb.clauses) + "\n")
    f.write("2: kb.tell(A & B)\n")
    f.write("3: kb.clauses = " + str(kb.clauses) + "\n")
    f.write("3.1: kb.ask(A) = " + str(kb.ask(A)) + "\n")
    f.write

    print("Done.")
###
if __name__ == '__main__':
    main()
else:
    print("logic_driver.py: This module is intended to be executed and not imported.")