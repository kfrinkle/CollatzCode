# CollatzCode
This is where all code related to the Collatz Conjecture will be located

There are three main programs here which are the three versions of the Collatz Code found in the article `Computing Streaks of Consecutive Numbers with the Same Collatz Height'.

CollatzMPIOneStreakCheckV4b.cpp -- the original Collatz streak computing code, no bells and whistles.  This is Version 1
FastestOne.cpp -- the modified code that introduces lookup tables and other shortcuts.  This is Version 2
LinearArrayExtra2DCleanBB.cp -- even more modified to increase performance.  This is Version 3

v4_Async -- better performance after desyncing the work nodes and giving them smaller intervals to check, marching along the numberline
v5_Table_rebuild -- even better performance after rebuilding the table occasionally with modes sampled from a set of ranges