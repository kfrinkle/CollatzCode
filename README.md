# CollatzCode
This is where all code related to the Collatz Conjecture will be located, what follows are the descriptions.

v1-v3_Legacy has old base 2 code in it: CollatzMPIOneStreakCheckV4b.cpp -- the original Collatz streak computing code, no bells and whistles. FastestOne.cpp -- the modified code that introduces lookup tables and other shortcuts. LinearArrayExtra2DCleanBB.cpp -- even more modified to increase performance.

v4_Async -- better performance after desyncing the work nodes and giving them smaller intervals to check, marching along the numberline

v5_Table_rebuild -- even better performance after rebuilding the table occasionally with modes sampled from a set of ranges

v6_rebase -- changed from base 2 stored in 32bit integer types to base 64 stored in 64bit integer types, also merged all running versions into one file, different modes toggled by arguements, such as 'skip' for skipping evens. See comments at the start of the file for more about arguements.

v7_jumbler -- implemented a runningSample to improve coalesence
