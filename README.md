# CollatzCode
This is where all code related to the Collatz Conjecture will be located, what follows are the descriptions.

CollatzMPIOneStreakCheckV4b.cpp -- the original Collatz streak computing code, no bells and whistles.

FastestOne.cpp -- the modified code that introduces lookup tables and other shortcuts.

LinearArrayExtra2DCleanBB.cpp -- even more modified to increase performance.

LinArrSRChunkCleanSkipEvens.cpp -- asynchronous chunk distribution code

LinArrSRChunkCleanSkipEvensAllData.cpp -- asynchronous chunk distribution code, skipping evens version

v4_Async -- better performance after desyncing the work nodes and giving them smaller intervals to check, marching along the numberline

v5_Table_rebuild -- even better performance after rebuilding the table occasionally with modes sampled from a set of ranges