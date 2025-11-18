In this folder, you will find all the cpp files used to generate the data found at https://doi.org/10.6084/m9.figshare.29892131.v1

The code used on each sheet of the above link is as follows:

1. CollatzMPIOneStreakCheckV4b.cpp -- oldest code, base 2
2. FastestOne.cpp -- second oldest code, base 2
3. LinearArrayExtra2DCleanBB.cpp --third oldest code, base 2
4, 23-24. LinArrSRChunkCleanAllData.cpp -- new MPI structure modification of LinearArrayExtra2DCleanBB.cpp, base 2 
5-7. CollatzSRNewTable.cpp -- modification of table build in LinArrSRChunkCleanAllData.cpp, base 2
8-10. CollatzSRNewTableMod1.cpp -- modification of CollatzSRNewTable.cpp, still do (3x+1)/2, but only one /2 at a time
11-13. CollatzSRNewTableMod2.cpp -- modification of CollatzSRNewTable.cpp, only do 3x+1 and /2 (no step skipping)
14-16. b64Unified.cpp -- base 2^64 version of CollatzSRNewTableMod2.cpp
17-19, 22. b64UnifiedJumblerv2.cpp -- modification of table build process of b64Unified.cpp, fastest version so far, base 2^64
20. b64UnifiedJumblerv2noOffsetTest.cpp -- b64UnifiedJumblerv2.cpp but with no random sampling, using even spaced samples only, base 2^64
21. CollatzSRNewBFAttemptAllFreqs.cpp + MurmurHash3.cpp + MurmurHash3.h -- bloom filter attempt, collect all samples with freq > 1, base 2

