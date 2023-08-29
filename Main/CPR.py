# coding: utf-8
#NAME:  CPR.py
"""
Brief description of what the module does or purpose. REQUIRED
"""

"""
Modification history and info.   NOT REQUIRED 
"""
"""
AUTHOR: Leonard P. Wesley

   Unpublished-rights reserved under the copyright laws of the United States.

   This data and information is proprietary to, and a valuable trade secret
   of, Leonard P. Wesley and < add your name(s) >. It is given in confidence by Leonard
   P. Wesley and < add your name(s)>. Its use, duplication, or disclosure is subject to
   the restrictions set forth in the License Agreement under which it has been
   distributed.

      Unpublished Copyright © 2022  Leonard P. Wesley and< add your name(s) >
      All Rights Reserved

  ============== VARIABLE, FUNCTION, etc. NAMING CONVENTIONS ==================
<ALL CAPITOL LETTERS>:  Indicates a symbol defined by a
        #define statement or a Macro.

   <Capitalized Word>:  Indicates a user defined global var, fun, or typedef.

   <all small letters>:  A variable or built in functions.


========================== MODIFICATION HISTORY ==============================
The format for each modification entry is:

MM/DD/YY:
    MOD:     <a description of what was done>
    AUTHOR:  <who made the mod>
    COMMENT: <any special notes to make about the mod (e.g., what other
              modules/code depends on the mod) >

    Each entry is separated by the following line:

====================== END OF MODIFICATION HISTORY ============================
========================== MODIFICATION HISTORY ==============================
10/23/22:
    MOD:    Added gencode_fasta_dictionary_GRCh38 and msbi_fasta_data
            to the set of arguments in the function call for the
            epigen_pipeline_chromosome_snp_modification module
    AUTHOR:  <NAME>
    COMMENT: Appended gencode_fasta_dictionary_GRCh38 and msbi_fasta_data
             to the arguments for epigen_pipeline_chromosome_snp_modification
====================== END OF MODIFICATION HISTORY ============================
"""


"""
Imports
"""
import os                   # This is just an example import statement
from copy import deepcopy   # This is another import example


"""
Module execute/load check.   REQUIRED
"""
if __name__ == '__main__':
    print("cs156_pr1.py:  Module is executed.")

else:
    print("cs156_pr1.py:  Module is imported.")

