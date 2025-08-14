// 2023.02.25 -- both divide by 2 and multply by three and add 1 functions appear to be working
// 2023.02.27 -- created the compare function for binary numbers to determine if we have a new max
// to run the slurm version, add a command line integer for number of digits to generate
// 2023.03.09 -- data aligns with Mike's data collection, commented out majority of print statements
// 2023.03.15 -- started adding in the MPI functionality.
// 2023.03.16 -- moved entire Collatz process to its own function!
// 2023.03.19 -- created timers for functions to figure out what is taking up what time
// 2023.03.22 -- fixed counting the 0th entry twice and not counting the last entry at all

// 2023.03.24 -- The program has been modified to check for longest runs over a range
// 2023.04.01 -- This program will allow a large number of processes to compute a single long streak

// 2023.06.21 -- Start new approach where we store all values in Collatz Sequence
// 2023.06.21 -- got away from storing ALL steps in sequence

// 2023.07.04 -- added depth of coalescence statistics to code
// 2023.09.22 -- added a frackton of comments
// 2023.09.27 -- fixed small issue with comparing in the CollatzCompare function
// 2023.10.13 -- Modified the code to compare the size of the numbers first before comparing bit-by-bit

// 2023.11.15 -- implemented linear array approach so that divide by 2 jut shifts lsd up, no circular array used
// a third argument is now required, which should be larger than the Collatz sequence height of base number

// TO RUN THIS CODE, here is an example.... there are three command line arguments
// sbatch jobCollatzLinear2Dclean2BB.mpi 4081 25 29490
// the above command submits the job where the run command in the job file would look like:
// time prun ./LinearArrayExtra2DCleanBB $1 $2 $3
// the example says to test 2^4081 where each process has a chunk of numbers 2^25 long, and add 29490 extra padding to
// the front of the array that holds the binary number.  We always set this close to the height of 2^k, in general.
// definitely overkill...

// 11/6/2024 -- Add a running tally for how many chunks each process has done and mins and maxes for times

// 11/11/2024

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cmath>
#include <math.h>
#include <string>
#include <sstream>
#include <ctime>
#include <cstddef>
#include <stdlib.h>
#include <iomanip>
#include <time.h>
#include "mpi.h"
#include <climits>
#include <float.h>

using namespace std;

void div2(int num[], int size[]); // divide by 2 function
void div2Bin(int num[], int size); // divide by 2 function in binary
int powerfulDiv2Bin(int num[], int size); // divide by 2 many times in binary (returns new size of binary number)
int mul3p1Bin(int num[], int size); // multiply by 3 and add 1 in binary  (returns new size of binary number)
int mul3p1d2Bin(int num[], int size, int lsd); // multiply by 3, add 1, then divide by 2 in binary  (returns new size of binary number)
int digitsum(int num[], int mod[], int size); // function to compute binary mod 9 calculation
int bincompare(int num1[], int num2[], int size, int lsd); // used to compare two binary numbers
void printnum(int num[], int size); // print out number in correct order
void add1Bin(int num[]); // just add 1 to a binary number
void shift1Bin(int num[], int pos); // add 2^pos to a binary number
int Collatz(int num[], int sizeNum); // generate Collatz sequence for binary number of length size
void CollatzSteps(int num[], int sizeNum, int **ColSeq, int ColSeqSizes[]); // generate Collatz sequence, save the values along the way in 2D array
int CollatzCompare(int num[], int sizeNum, int **ColSeq, int steps, int CoalData[], int BinExtra, int ColSeqSizes[]); // generate Collatz sequence, compare the values along the way

//Here add the global variable that you'll use to store the number of times you don't have to compare because sizes were different.
long long int timesNoCompare = 0;
long long int timesYesCompare = 0;
long long int skippedSteps = 0;
long long int shiftWasGreaterThan1 = 0;
int maxSkipSteps = 0;

int arraySize = 0;

int main(int argc, char *argv[])
{
	int i = 0; // loop variable only
	int j = 0; // loop variable only

	// do not change the value of size anywhere, we need that...
	int sizeNum = 0; //length of number, will be 2^sizeNum + 1 for starters
	//int runner = 0; // amount of sequential numbers to test
	int powa = 0; // power of 2 to add to number to break up the set of integers for streak check
	unsigned int powa2 = 1; // power of 2 value in base 10 (for each node to work on, should be less than INT MAX)

	// extra is argv[3] below
	int extra = 0; // this needs to be greater than the number of steps in the sequence for 2^k+1 for shifting purposes
	int extra2D = 1000; // just enough  padding on the end of the 2D array to hold extra 3x+1 digits for start of code
    	int namelength = 0; // used for printing out the name of the compute node
    	int rank, size; // for COMM_WORLD info
	int steps; // keep track of the number of steps in Collatz Sequence
	long long int currrun = 0; // current run of sequence variable
	int runnercheck = 0; // check the iterations, output every runnercheck amount (about every 1/8th of total run)
	int ColSteps = 0; // how many steps in initial number for streak
	unsigned long long int streakChunk = 0; // how many steps in each chunk of a streak there is

	double timeS, timeE;

    	sizeNum = atoi(argv[1]); // input of exponent k in 2^k+1
	powa = atoi(argv[2]); // power of 2 to add to base number for each proccess for the streak (shifted by process)
	extra = atoi(argv[3]); // how much padding to add to left end of number -- CHANGE IN COLLATZ FUNCTION TO MATCH

	// now we compute the value of 2^powa in base 10 so we can work out the shift of each process for their part of the work
	for (i = 1; i <= powa; i++)
	{
		powa2 = powa2*2;
	}

	// now that we know how many each process will do (in base 10) we set up variable to display results every 1/8th of the way through
	runnercheck = powa2 >> 3;

	cout << "powa = " << powa << ", and powa2 = " << powa2 << ", and runnercheck = " << runnercheck << endl;


	//clock_t start; // used for timing functions
	//clock_t end; // used for timing functions
	//double cpu_time_used; // used for computing length of function run

	//MPI:Status variable status is a class, and can be used for debugging purposes if need be
    	MPI::Status status;

    	MPI::Init(argc, argv);

    	char name[MPI_MAX_PROCESSOR_NAME];  // create character array for the names of the compute nodes

    	size = MPI::COMM_WORLD.Get_size(); // get total size of the world
    	rank = MPI::COMM_WORLD.Get_rank(); // each process gets its own rank

    	//we can use this for debugging, each process can now tell us what compute node they are on
    	MPI::Get_processor_name(name, namelength);

	//cout << "Rank " << rank << ": has a size value of " << sizeNum << endl;

	//to avoid that pesky add 1 everywhere, just do it now
	//remember, to put the number 2^k+1 into a container in binary, you need k+1 digits
	sizeNum = sizeNum + 1;

	//The ColData array holds all the important data for each Collatz computation
	//ColData[0] holds number of steps to coalesce, [1] and [2] hold min and max coalescence values
	int *ColData = new int[3] (); // Array to hold coalescence data

        long long int ii = 0; // for tallying up number of numbers checked
	long long int ColSum = 0; // Coalescence sum for average
	long long int ColSumReduce = 0; // for coalscence stats via MPI Reduce
	long long int ColSumReducei = 0; // actual numbers in sequence that were checked (might be less than estimate at runtime
	int ColMaxReduce = 0; // maximum depth a COllatz sequence had to run through before it matched with 2^k+1's sequence
	int ColMinReduce = 0; // minimum depth a COllatz sequence had to run through before it matched with 2^k+1's sequenc
	long long int timesNoCompareReduce = 0; //total number of times we didn't have to compare digit-by-digit for coalescence
	long long int timesYesCompareReduce = 0; //total number of times we did have to compare digit-by-digit for coalescence
	long long int skippedStepsReduce = 0; //total number of times that we skipped a step in computing the collatz sequence
	long long int shiftWasGreaterThan1Reduce = 0;
	int maxSkipStepsReduce = 0;

	//int CoalMin = 0; // Coalescence min value
	//int CoalMax = 0; // Coalescence max value

	//We need two copies of the binary number, remember that if you pass an array into a function and change it there, it changes everywhere
	arraySize = sizeNum + extra;
	int *binnumber = new int[sizeNum + extra] (); // will hold the binary number
	int *binnumberHold = new int[sizeNum + extra] (); // will hold the binary number, this wont change in Collatz function

	//These are normal for checking that we broke the problem up at the correct locations for each process
	int *firstnumber = new int[sizeNum + extra] (); // will hold the first binary number run
	int *lastnumber = new int[sizeNum + extra] (); // will hold the last binary number run through the sequence

	int *stepArrGather = new int[size] (); // will hold the number of steps for all processes via Gatherv

	int binextra = sizeNum + extra; // needs this for compare function

	cout << "Node " << name <<  " -- Rank " << rank << ": Large array will have " << (long long int)powa2 * (long long int)size << " entries." << endl;
	//binnumber[sizeNum-1] = 1;
	//Set the leading value in the array to 1 so we have 2^k
	binnumberHold[sizeNum-1] = 1;

	//binnumber[0] = 1;
	//Set the 0th place to 1 so we all have 2^k+1 now.
	binnumberHold[0] = 1;

	//this is for incremental testing. I you need to break up the problem into smaller pieces starting at a power of 2, do it here
	//This would give you starting values of 2^k+2^r+1 where r is any of the 21....30 below.  Uncomment more for even more options!
	/*binnumberHold[21] = 1;
	binnumberHold[22] = 1;
	binnumberHold[23] = 1;
	binnumberHold[24] = 1;
	binnumberHold[25] = 1;
	binnumberHold[26] = 1;
	binnumberHold[27] = 1;
	binnumberHold[28] = 1;
	binnumberHold[29] = 1;
	binnumberHold[30] = 1;*/

	//First ALL processes compute the Collatz sequence for 2^k+1 and store the number of steps in that sequence
	//set up the number that can be modified by copying from binnumberHold first
	for (j = 0; j < sizeNum + extra; j++)
	{
		binnumber[j] = binnumberHold[j];
	}
	//compute the number of steps
	ColSteps = Collatz(binnumber, sizeNum);

	//set  min value to number of steps so it will get updated on first run through  function
	ColData[1] = ColSteps;

	//reset the number since binnumber was modified in the ColSteps function
	for (j = 0; j < sizeNum + extra; j++)
	{
		binnumber[j] = binnumberHold[j];
	}


	//now initialize the 2D array to fill with the numbers in the Collatz sequence now that we know the number of steps
	//pay attention to how this is allocated.  This will be sequential memory for a 2D array!!!!
	int **ColSeq = (int**)malloc(ColSteps * sizeof(int*)); // will hold the values of the Collatz Sequence
	ColSeq[0] = (int*)malloc(ColSteps * (sizeNum + extra2D) * sizeof(int));
	for (j = 1; j < ColSteps; j++)
	{
		ColSeq[j] = ColSeq[j-1] +  (sizeNum + extra2D);
	}
	//let's make sure these are all set to zero first as malloc does not guarantee this (try calloc?)
	for (i = 0; i < ColSteps; i++)
	{
		for (j = 0; j < sizeNum + extra2D; j++)
		{
			ColSeq[i][j] = 0;
		}
	}

	//Here I initialize the array storing the size of each number in the collatz sequence.
	int *ColSeqSizes = new int[ColSteps];
	for (i = 0; i < ColSteps; i++)
	{
		ColSeqSizes[i] = 0;
	}

	// ok now let's fill the array!  This is done by each process calling the CollatzSteps function, ColSeq is where sequence is stored.
	CollatzSteps(binnumber, sizeNum, ColSeq, ColSeqSizes);

	int *chunks = new int[size]; // holds which process is working on which chunk, by index

	/// YOU NEED AN ARRAY "chunksizes" of length size for holding the number of steps a process took
	/// when working on its chunk, this will be used to determine if you have a break or not in that chunk

	int *chunksizes = new int[size];

	/// globalchunksize = powa2 in Saxon's schema

	/// EVERYTHING BEFORE THIS LINE DOES NOT CHANGE, YOU MAY NEED TO ADD SOME MORE VARIABLE DECLARATIONS ABOVE
	/// BUT THE CREATION OF THE 2D ARRAY and BINNUMBER, BUNUMBERHOLD MUST STAY

	/// AT THIS POINT, INSTEAD OF EACH PROCESS CREATING ITS OWN STARTING POINT, PROCESS 0 TAKES OVER
	/// THIS IS WHERE PROCESS 0 SENDS THE FIRST SIZE-1 BATCH OF CHUNKS OUT

	// THIS IS PROCESS 0 ALONE -- THIS PORTION OF PROCESS 0 SHOULD ONLY BE UP THROUGH A BREAK BEING FOUND
	// this is the initial sending
	int chunksSent = 0;
	int chunksProcessed = 0;
	double minTime = DBL_MAX;
	double maxTime = -0.01;

	if(rank==0)
	{

		//init
		for (i = 1; i < size; i++)
		{
			//shift1Bin(binnumberHold, powa);
			// YOU NOW NEED TO SEND THIS TO PROCESS i+1
			//shift1Bin(binnumberHold, powa);
			cout << "Rank[0]: Sending chunk " << chunksSent+1 << " to rank " << i << endl;
			MPI_Send(binnumberHold, binextra, MPI_INT, i, 0, MPI::COMM_WORLD);
			shift1Bin(binnumberHold, powa);
			// DATA ABOUT WHICH CHUNK IS SENT TO WHICH PROCESS SHOULD BE STARTED HERE
			chunks[i] = i; // First chunk is 1 just so that the first set of chunks matches the ranks
			chunksSent++;
		//printnum(binnumber, sizeNum + extra);
		}
		bool kg = true;
//		int chunksSent = 0;
		while(kg)
		{
			int recvChunkSize = 0;
//			cout << "Rank[0]: I am waiting to receive a chunk from anyone" << endl;
			MPI_Recv(&recvChunkSize, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI::COMM_WORLD, status);
			int recvRank = status.MPI::Status::Get_source();
			chunksizes[recvRank] = recvChunkSize;
//			cout << "Rank[0]: I have received chunksize " << recvChunkSize <<  " from Rank " << recvRank << ".  Time to check it." << endl;
			if(recvChunkSize < powa2)
			{
				cout << "Rank[0]: Rank[" << recvRank << "] found a break first" << endl;
				kg = false;
				binnumberHold[0] = 2;
				MPI_Send(binnumberHold, binextra, MPI_INT, recvRank, 0, MPI::COMM_WORLD);
			}
			else if(recvChunkSize==powa2)
			{
				//cout << "Rank[0]: Rank[" << recvRank << "] did not find a break, recvChunkSize was: " << recvChunkSize << endl;
				if (chunksSent%100 == 0)
				{
					cout << "Rank[0]: Sending chunk " << chunksSent+1 << " to rank " << recvRank << endl;
				}
				MPI_Send(binnumberHold, binextra, MPI_INT, recvRank, 0, MPI::COMM_WORLD);
				chunksSent++;
				chunks[recvRank] = chunksSent;
				shift1Bin(binnumberHold, powa);
			}
		}
	}

	// work
	if(rank > 0)
	{
		bool kg = true;
		while(kg)
		{
//			cout << "Rank[" << rank << "]: I am waiting for a new chunk from Rank 0" << endl;
			MPI_Recv(binnumberHold, binextra, MPI_INT, 0, 0, MPI::COMM_WORLD, status);
//			cout << "Rank[" << rank << "]: Received from Rank 0.  Starting work" << endl;
			timeS = MPI_Wtime();
			if(binnumberHold[0] == 2)
			{
//				cout << "Rank [" << rank << "]: Someone found a break.  I should stop." << endl;
				kg = false;
				break;
			}

			if(kg == false)
			{
			//	cout << "Rank [" << rank << "]: I shouldn't be here." << endl;
			}
			//do work to get steps
			streakChunk = 0;

			for (j = 0; j < sizeNum + extra; j++)
        		{
                		binnumber[j] = binnumberHold[j];
                		firstnumber[j] = binnumberHold[j];
		        }


			for (i = 0; i < powa2; i++)
        		{
        			// store the value of steps in the CollatzCompare function
        			// remember CollatzCompare compares the numbers at each step in the sequence to that of 2^k+1
                		// ColSeq is the array which holds the numbers in the sequence for 2^k+1
        	        	// ColData is the array of useful information (steps, max and min steps)
	                	steps = CollatzCompare(binnumber, sizeNum, ColSeq, ColSteps, ColData, binextra, ColSeqSizes);

                		// add number of steps to sum for average to be computed later
                		ColSum = ColSum + ColData[0];
                		ColData[0] = 0;

                		// if we have the same number of steps, the streak keeps going!
                		if (steps == ColSteps)
                		{
                        		streakChunk++;
					ii++;
                		}
                		// else we print out the break info! Keep checking the .out file for this statement!
                		else
                		{
                        		cout << "Node " << name <<  " -- Rank " << rank << ": Broke streak at " << streakChunk << endl;
                        		break;
                		}
                		// print out status every 1/8th of the way through
                		//if (i % runnercheck == 0)
                		//{
                        		//cout << "Node " << name <<  " -- Rank " << rank << ": We are on iteration " << i << "/" << powa2 << ", and this collatz sequence had " << steps << " steps." << endl;
                		//}
                		// if we are on the last number in our chunk to work on skip this, as we do not want to increment for another pass
                		// don't add 1 to last number tested so we have it!
				if (i != powa2-1)
                                {
                                        add1Bin(binnumberHold);

                                      /*if (binnumberHold[0]==0 && i != powa2-2)
                                        {
                                                add1Bin(binnumberHold);
                                                i++;
                                                streakChunk++;
                                        }*/
                                        for (j = 0; j < sizeNum + extra; j++)
                                        {
                                                binnumber[j] = binnumberHold[j];
                                        }


                                }
                	// else it is the last number worked on, so store that in lastnumber!

        		}

                        for (j = 0; j < sizeNum + extra; j++)
                        {
                                lastnumber[j] = binnumberHold[j];
                        }
			timeE = MPI_Wtime();
			timeE = timeE - timeS;
			if(timeE > maxTime)
			{
				maxTime = timeE;
			}
			else if (timeE < minTime)
			{
				minTime = timeE;
			}

			if (chunksProcessed % 1000 == 0)
			{
				cout << "Rank [" << rank << "]: Time elapsed on received chunk: " << timeE << " sending streakChunk of: " << streakChunk << endl;
			}
			chunksProcessed++;
			MPI_Send(&streakChunk, 1, MPI_INT, 0, 0, MPI::COMM_WORLD);
		}
	}
	int minRank = 0;

	// for end
	if(rank==0)
	{
		for(int i = 0; i<size-2; i++)
		{
			int recvChunkSize;
			MPI_Recv(&recvChunkSize, 1, MPI_INT, MPI_ANY_SOURCE, 0,MPI:: COMM_WORLD, status);
			int recvRank = status.MPI::Status::Get_source();
			chunksizes[recvRank] = recvChunkSize;

//			cout << "Rank[0]: Sending rank " << recvRank << " the kill signal" << endl;

			MPI_Send(binnumberHold, binextra, MPI_INT, recvRank, 0, MPI::COMM_WORLD);
                       // chunksSent++;
                        //chunks[recvRank] = chunksSent;
                        //shift1Bin(binnumberHold, powa);
		}

		unsigned long long int breakAtChunk = INT_MAX;
		int breakAtIndex = 0;
		minRank = 0;
		//int *breaksArray = new int[size]();
		//cout << "Rank[0]: Time to search for breaks" << endl;
		for(int i = 1; i<size; i++)
		{
			if(chunksizes[i] < powa2)
			{
				cout << "Rank[0]: " << chunks[i]-1 << ", " << powa2 << ", " << chunksizes[i] << endl;
				cout << "Rank[0]: Break found by rank " << i << " at chunk " << chunks[i]-1 << " at index " << (unsigned long long int) powa2 * (chunks[i]-1) + chunksizes[i] << endl;
				if(chunks[i] < breakAtChunk)
				{
					breakAtChunk = chunks[i];
					breakAtIndex = chunksizes[i];
					minRank = i;
				}
			}
		}
		streakChunk = (unsigned long long int) powa2 * (breakAtChunk-1) + breakAtIndex;

	}


	MPI::COMM_WORLD.Barrier();

//	if(rank>0)
//	{
//		cout << "Rank[" << rank << "]: Worked on " << chunksProcessed << " chunks" << endl;
//	}




	// ok, so let's let everyone catch up here!  If a process(es) got done early, it will wait here.
        cout << "Node " << name <<  " -- Rank " << rank << ": Waiting at the barrier after " << chunksProcessed << " chunks processed with a streak length of " << streakChunk << endl;

        MPI::COMM_WORLD.Barrier();

	double *minMin = new double[size];
	double *maxMax = new double[size];
	int *chunksWorked = new int[size];
	int avgChunks = 0;
	int minChunks = INT_MAX;
	int maxChunks = -1;
	double minOfMins;
	double maxOfMaxes;

	double minAvg = 0.0;
	double maxAvg = 0.0;

	if (rank == 0)
	{
		minTime = DBL_MAX;
		maxTime = -1.0;
	}

	MPI::COMM_WORLD.Gather(&chunksProcessed, 1, MPI::INT, chunksWorked, 1, MPI::INT, 0);
	MPI::COMM_WORLD.Gather(&minTime, 1, MPI::DOUBLE, minMin, 1, MPI::DOUBLE, 0);
	MPI::COMM_WORLD.Gather(&maxTime, 1, MPI::DOUBLE, maxMax, 1, MPI::DOUBLE, 0);

	MPI::COMM_WORLD.Reduce(&minTime, &minOfMins, 1, MPI::DOUBLE, MPI::MIN, 0);
	MPI::COMM_WORLD.Reduce(&maxTime, &maxOfMaxes, 1, MPI::DOUBLE, MPI::MAX, 0);

	// simple reduce functions to tally the information about max/min and total number of steps
	MPI::COMM_WORLD.Reduce(&ColSum, &ColSumReduce, 1, MPI::LONG_LONG, MPI::SUM, 0);
	MPI::COMM_WORLD.Reduce(&ii, &ColSumReducei, 1, MPI::LONG_LONG, MPI::SUM, 0);
	MPI::COMM_WORLD.Reduce(&ColData[1], &ColMinReduce, 1, MPI::INT, MPI::MIN, 0);
	MPI::COMM_WORLD.Reduce(&ColData[2], &ColMaxReduce, 1, MPI::INT, MPI::MAX, 0);
	MPI::COMM_WORLD.Reduce(&maxSkipSteps, &maxSkipStepsReduce, 1, MPI::INT, MPI::MAX, 0);

	MPI::COMM_WORLD.Reduce(&timesNoCompare, &timesNoCompareReduce, 1, MPI::LONG_LONG, MPI::SUM, 0);
	MPI::COMM_WORLD.Reduce(&timesYesCompare, &timesYesCompareReduce, 1, MPI::LONG_LONG, MPI::SUM, 0);
	MPI::COMM_WORLD.Reduce(&skippedSteps, &skippedStepsReduce, 1, MPI::LONG_LONG, MPI::SUM, 0);
	MPI::COMM_WORLD.Reduce(&shiftWasGreaterThan1, &shiftWasGreaterThan1Reduce, 1, MPI::LONG_LONG, MPI::SUM, 0);

//	THIS DOES NOT NEED TO HAPPEN ANYMORE SINCE PROCESS 0 WILL COMPUTE THE LENGTH BASED ON CHUNKS AND CHUNKSIZES
//	now we have to comptue the length of the streak!  So process 0 gathers everyone's streak values.
	//MPI::COMM_WORLD.Gather(&streakChunk, 1, MPI::INT, stepArrGather, 1, MPI::INT, 0);

	if (rank == 0)
	{
		for(int i=1; i<size;i++)
		{
			minAvg = minMin[i]+minAvg;
			maxAvg = maxMax[i]+maxAvg;
			if(chunksWorked[i] > maxChunks)
			{
				maxChunks = chunksWorked[i];
			}
			if(chunksWorked[i] < minChunks)
			{
				minChunks = chunksWorked[i];
			}
		}
		minAvg = minAvg/(size-1);
		maxAvg = maxAvg/(size-1);

		cout << endl << endl;
		cout << "Chunks/Min/Max chunk processing times for all processes:" << endl;
		for(int i = 1; i<size; i++)
		{
			cout << " [" << i << "]: " << chunksWorked[i] << ", " << minMin[i] << ",  " << maxMax[i] << endl;
			avgChunks = avgChunks + chunksWorked[i];
		}
//              cout << "Maximum times for all processes:" << endl;
//              for(int i = 1; i<size; i++)
//		{
//                      cout << " [" << i << "]: " << maxMax[i];
//              }
//		cout << endl << endl;
//		cout << "Total chunks worked on per process:";
//		for(int i = 1; i<size; i++)
//		{
//			cout << " [" << i << "]: " << chunksWorked[i] << endl;
//		}
//		cout << endl << endl;


		// check that we added 1 the correct amount of times by looking at the VERY LAST number checked
//		cout << "Rank 0: First number tested was : " << endl;
//		printnum(firstnumber, sizeNum + extra);
		// this last number may not look right if it was actually sequenced.... so this may not be useful
		// HAVE THE PROCESS THAT FOUND THE BREAK SEND process 0 ITS VALUE of lastnumber
//		MPI::COMM_WORLD.Recv(lastnumber, sizeNum + extra, MPI::INT, size-1, 0, status);
//		cout << "Rank 0: Last number tested was : " << endl;
//		printnum(lastnumber, sizeNum + extra);

		// now the fun stats!

		cout << "Node " << name <<  " -- Rank " << rank << ": Streak length of " << streakChunk << " found by Rank " << minRank << endl;
		cout << "Node " << name <<  " -- Rank " << rank << ": Total amount of chunks worked on was: " << chunksSent << endl;

		cout << "Node " << name <<  " -- Rank " << rank << ": Numbers start at (2^" << sizeNum - 1 << ") + 1 and has a run which is " << streakChunk << " long. Range spanned "  << (long long) chunksSent*powa2 << " numbers" << endl;
		cout << "Node " << name <<  " -- Rank " << rank << ": Shortest coalescence value is :  " << ColMinReduce << endl;
		cout << "Node " << name <<  " -- Rank " << rank << ": Longest coalescence value is  :  " << ColMaxReduce << endl;
		cout << "Node " << name <<  " -- Rank " << rank << ": Sum of coalescence values is  :  " << ColSumReduce << endl;
		cout << "Node " << name <<  " -- Rank " << rank << ": Average coalescence value is  :  " << ColSumReduce/ColSumReducei << endl;
		cout << "Node " << name <<  " -- Rank " << rank << ": Total number of values checked is " << ColSumReducei << endl;
		cout << "Node " << name <<  " -- Rank " << rank << ": Total times the sizes were unequal (didn't compare digit-by-digit) is :" << timesNoCompareReduce << endl;
		cout << "Node " << name <<  " -- Rank " << rank << ": Total times the sizes were equal (did compare digit-by-digit) is      :" << timesYesCompareReduce << endl;
		cout << "Node " << name <<  " -- Rank " << rank << ": Total times we skipped a step is: " << skippedStepsReduce << endl;
		cout << "Node " << name <<  " -- Rank " << rank << ": Total times shift was greater than 1 is: " << shiftWasGreaterThan1Reduce << endl;
		cout << "Node " << name <<  " -- Rank " << rank << ": Max number of sequential divide by 2's is: " << maxSkipStepsReduce << endl << endl;

		cout << "Node " << name <<  " -- Rank " << rank << ": Total Chunks Processed is: " <<  avgChunks << endl;
                cout << "Node " << name <<  " -- Rank " << rank << ": Minimum Chunks Processed is: " << minChunks << endl;
                cout << "Node " << name <<  " -- Rank " << rank << ": Maximum Chunks Processed is: " <<  maxChunks << endl;
                cout << "Node " << name <<  " -- Rank " << rank << ": AVG Chunks Processed is is : " << avgChunks/(size-1) << endl;
                cout << "Node " << name <<  " -- Rank " << rank << ": AVG Minimum Time (ms) is : " << minAvg*1000 << endl;
                cout << "Node " << name <<  " -- Rank " << rank << ": AVG Maximum Time (ms) is : " << maxAvg*1000 << endl;
                cout << "Node " << name <<  " -- Rank " << rank << ": ABS Minimum Times (ms) is : " << minOfMins*1000 << endl;
                cout << "Node " << name <<  " -- Rank " << rank << ": ABS Maximum Time (ms) is : " << maxOfMaxes*1000 << endl;



//		cout << "MinAvgTime/MaxAvgTime/AbsMinTime/AbsMaxTime/AvgChunksProcessed/MaxChunks/MinChunks:" << endl;
//		cout << minAvg*1000 << ", " << maxAvg*1000 << ", " << minOfMins*1000 << ", " << maxOfMaxes*1000 << ", " << avgChunks/(size-1) << ", " << maxChunks << ", " << minChunks << endl;
//		cout << endl << endl;


	}


	// last rank needs to send the last number it checked.  this may not be useful depending on where streak break was found
	// CHANGE THIS TO THE PROCESS THAT FOUND THE BREAK, IF YOU WANT JUST HAVE THE PROCESS PRINT OUT LASTNUMBER, YOU DONT HAVE TO SEND IT
//	if (rank == size-1)
//	{
//		MPI::COMM_WORLD.Send(lastnumber, sizeNum + extra, MPI::INT, 0, 0);
//	}

	// now time to delete the arrays!
	delete[] stepArrGather;
	delete[] binnumber;
	delete[] binnumberHold;
	delete[] firstnumber;
	delete[] lastnumber;
	delete[] ColData;
	delete[] ColSeqSizes;
	delete[] chunks;
	delete[] chunksizes;

	// now to free that pesky 2D array!
	free(ColSeq[0]);
	free(ColSeq);


        MPI::Finalize();

	return 0;
}


// generate Collatz sequence for binary number binnumber of length binsize, return number of steps
int Collatz(int binnumber[], int binsize)
{

	int steps = 0; // just keep track of the number of steps!

	// for binary, binsize = 1 means your number is 1, the end of the sequence
	while (binsize > 1)
	{
		// if its off, do 3*x+1, note binsize value is returned
		if(binnumber[0] == 1)
		{
			binsize = mul3p1Bin(binnumber, binsize);
		}
		// if its even, do x>>1, note binsize value is automatically adjusted
		else
		{
			div2Bin(binnumber, binsize);
			binsize--;
		}

		steps ++;
	}

	return steps;

}

// generate Collatz sequence, saving all numbers in the sequence as it goes
void CollatzSteps(int binnumber[], int binsize, int **ColSeq, int ColSeqSizes[])
{

	int j = 0; // loop variable only
	int steps = 0; // keep track of number of steps

	while (binsize > 1)
	{
		// once again, if odd, 3x+1
		if(binnumber[0] == 1)
		{
			binsize = mul3p1Bin(binnumber, binsize);
		}
		// else x>>1
		else
		{
			div2Bin(binnumber, binsize);
			binsize--;
		}
		// copy the number over to array after each step in the process
		// note we only go out to binsize, so the 2D array better be zeroed out first!
		for (j = 0; j < binsize; j++)
		{
			ColSeq[steps][j] = binnumber[j];
		}

		//Here we add in the binsize of this number in the sequence.
		ColSeqSizes[steps] = binsize;

		steps ++;

	}

//	return steps;
}


// generate Collatz sequence, stopping when it hits a number at same index in ColSeq array
int CollatzCompare(int binnumber[], int binsize, int **ColSeq, int ColSteps, int ColData[], int BinExtra, int ColSeqSizes[])
{
// ColData[0] holds number of steps to coalesce, [1] and [2] hold min and max coalescence values

	int steps = 0; // the step count variable
	int cv = 0; // compare value between current step and base step
	int shifts = 0;
	int lsd = 0;

	while (binsize > 1)
	{
		if(binnumber[lsd] == 1)
		{
			binsize = mul3p1d2Bin(binnumber, binsize, lsd);
			steps += 2;
		}
		else
		{
			shifts = 0;
			while (binnumber[lsd] == 0)
			{
				shifts++;
				lsd += 1;
			}

			steps += shifts;
			binsize -= shifts;
			skippedSteps += (shifts - 1);

			if (shifts > maxSkipSteps)
				maxSkipSteps = shifts;
			if (shifts > 1)
				shiftWasGreaterThan1++;
		}

		//First we compare the current size of the number to the size of the original sequence's number
		if (binsize == ColSeqSizes[steps - 1])
		{
			cv = bincompare(binnumber, ColSeq[steps - 1], binsize, lsd);
			timesYesCompare++;
		}
		else
		{
			//here you will update your global variable since you didn't have to compare digit-by-digit
			timesNoCompare++;
		}

		// if the current number and that in ColSeq[steps] are the same, gather data and stop!
		if (cv == -1)
		{
			// put number of steps taken into 0th entry
			ColData[0] = steps;
			// then check we have another max/min number of steps
			if (steps > ColData[2])
			{
				ColData[2] = steps;
			}
			else if (steps < ColData[1])
			{
				ColData[1] = steps;
			}
			// so put the number of steps into ColSteps since we know current number will have
			// same sequence length as one we are streak checking
			steps = ColSteps;
			break;
		}

                // if we have gotten all the way to number of steps for base number and  cv != -1, then we need to stop
                // as this is a break!
    else if (steps >= ColSteps)
    {
            steps = 0;
            break;
    }

	}

	return steps;

}



// used to compare two numbers, returns 0 if num0 is larger, 1 if num1 is larger, -1 if they are equal
//With the current changes, please only call this method when the sizes are equal; thanks!
int bincompare(int num0[], int num1[], int size, int lsd)
{
	int i = 0;
	int kg = 1; //Keep going
	int wb = 0; //Who is bigger
	int digit = lsd;

	//if size + lsd <= arraySize, there is not wraparound
//	if (size + lsd <= arraySize)
//	{
		while (kg && digit < size + lsd)
		{
			if (num0[digit] > num1[i])
			{
				wb = 0;
				kg = 0;
			}
			else if (num0[digit] < num1[i])
			{
				wb = 1;
				kg = 0;
			}

			i++;
			digit++;
		}

		if (kg == 1)
		{
			wb = -1;
		}
//	}
/*
	//otherwise, there is wraparound :(
	else
	{
		//first go from lsd to arraysize
		while (kg && digit < arraySize)
		{
			if (num0[digit] > num1[i])
			{
				wb = 0;
				kg = 0;
			}
			else if (num0[digit] < num1[i])
			{
				wb = 1;
				kg = 0;
			}

			i++;
			digit++;
		}

		digit = 0;
		//now do the rest of the comparisons
		//is this long equation really better than a mod? idk
		while (kg && digit < (size - (arraySize - lsd)))
		{
			if (num0[digit] > num1[i])
			{
				wb = 0;
				kg = 0;
			}
			else if (num0[digit] < num1[i])
			{
				wb = 1;
				kg = 0;
			}

			i++;
			digit++;
		}

		if (kg == 1)
		{
			wb = -1;
		}
	}
*/
	return wb;
}


 // function to compute binary mod 9 calculation
int digitsum(int num[], int mod[], int size)
{
	int i = 0;
	int digitsum = 0;

        for (i = 0; i < size; i++)
        {
                digitsum = digitsum + num[i]*mod[i];
        }

	return (digitsum%9);

}



// binary multiply by 3 and add 1 and return the new binary number size
int mul3p1Bin(int num[], int size)
{
	int i = 0;
	int carry = 0; // keep track of the carrys
	int tempCurr = 0;  // store b_i temporarily for computations after b_i is overwritten
	int tempNext = 0; // store b_i+1 temporarily for same reason

	// for the 0's place, we simply add 1 to the entry as  multiplying by two shifts number to the left and contributes nothing to 0th place
	// afterwards this takes the binary array and performs 3x+1 without creating a new array.  In order to accomplish this we do have to keep
	// the current value in the array in a temp variable to use it to add to the next value
	// basically, we start with an array X, shift to the left 1 place, which gives 2X, and add original X to 2X to get 3X.
	tempCurr = num[0];  // keep this value in temp because it has to get added to the 1st entry while computing X+2X
	num[0] = num[0] + 1; // add the 1 from the 3X+1
	carry = num[0]>>1;  // compute the carry
	num[0] = num[0]&1; // now ensure that the entry is 0 or 1.

//	cout << "Original number has " << size << " binary digits." << endl;

	for (i = 0; i < size; i++)
	{
		tempNext = num[i+1]; // store current position so it can be used for next entry in X+2X
		num[i + 1] = tempCurr + num[i+1] + carry; // add correct entries plus carry to current position
		carry = num[i+1]>>1; // compute the carry
		num[i+1] = num[i+1]&1; // get the entry to 0 or 1
		tempCurr = tempNext; // don't lose the new current position!
	}
	num[size+1] = carry;

	// now let's figure out how large our number is and return that value! We know that 3X+1 will be at least 1 digit larger than X
	if (carry == 0)
	{
//		cout << "3x+1 number has " << size + 1 << " binary digits." << endl;
		return (size + 1);
	}
	else
	{
//		cout << "3x+1 number has " << size + 2 << " binary digits." << endl;
		num[size+1] = carry;
		return (size + 2);
	}

}


// binary multiply by 3, add 1, divide by 2, and return the new binary number size
int mul3p1d2Bin(int num[], int size, int lsd)
{
	skippedSteps++;
	//if lsd + size < arraysize, then you can just go from lsd to size without any worries :)

//	if (lsd + size < arraySize)
//	{
		int i = lsd;
		int carry = 1; // keep track of the carrys; this is 1 because we're adding 1.
	        int tempCurr = 1;  // store b_i temporarily for computations after b_i is overwritten; it's 1 because this method should only be used for odd numbers.

		for (; i < lsd + size; i++)
		{
			num[i] = tempCurr + num[i+1] + carry; // add correct entries plus carry to current position
			carry = num[i]>>1; // compute the carry
			num[i] = num[i]&1; // get the entry to 0 or 1
			tempCurr = num[i + 1];
		}

		num[i] = carry;

		// now let's figure out how large our number is and return that value! We know that 3X+1 will be at least 1 digit larger than X
        	if (carry == 0)
        	{
            		return (size);
        	}
        	else
        	{
			return (size + 1);
        	}
//	}

/*
	else
	{
		int i = lsd;
		int carry = 1; // keep track of the carrys; this is 1 because we're adding 1.
        	int tempCurr = 1;  // store b_i temporarily for computations after b_i is overwritten; it's 1 because this method should only be used for odd numbers.

		int stepsWeHaveTaken = 0;
		//First we do from lsd to the end of the array
		for (;i < arraySize - 1; i++)
		{
			num[i] = tempCurr + num[i+1] + carry; // add correct entries plus carry to current position
			carry = num[i]>>1; // compute the carry
			num[i] = num[i]%2; // get the entry to 0 or 1
			tempCurr = num[i + 1];
			stepsWeHaveTaken++;
		}

		//Now we do the transition between beginning and end
		num[i] = tempCurr + num[0] + carry;
		carry = num[i]>>1; // compute the carry
		num[i] = num[i]%2; // get the entry to 0 or 1
		tempCurr = num[0];
		i = 0;
		stepsWeHaveTaken++;

		//Now we do the rest of the stuff
		while(stepsWeHaveTaken < size)
		{
			num[i] = tempCurr + num[i+1] + carry; // add correct entries plus carry to current position
			carry = num[i]>>1; // compute the carry
			num[i] = num[i]%2; // get the entry to 0 or 1
			tempCurr = num[i + 1];
			stepsWeHaveTaken++;
			i++;
		}

		num[i] = carry;

		// now let's figure out how large our number is and return that value! We know that 3X+1 will be at least 1 digit larger than X
        	if (carry == 0)
        	{
            		return (size);
        	}
        	else
        	{
			return (size + 1);
        	}
	}
*/
}




// binary divide by two simply shifts all the bits down 1
void div2Bin(int num[], int size)
{
	int i = 0;
	for (i = 0; i < size-1; i++)
	{
		num[i] = num[i+1];
	}
	num[size-1] = 0;

}

//will return the number of steps/how much the size shrinks (same number)
int powerfulDiv2Bin (int num[], int size)
{
	int i = 1;
	int shift = 1;

	while (num[i] == 0)
	{
		shift++;
		i++;
	}

	for (;i < size; i++)
	{
		num[i - shift] = num[i];
	}

	for (i = size - shift; i < size; i++)
	{
		num[i] = 0;
	}

	if (shift > 1)
	{
		shiftWasGreaterThan1++;
	}
	if (shift > maxSkipSteps)
	{
		maxSkipSteps = shift;
	}
	return shift;
}

// binary add 1, simply adds one at 0th entry and goes until carry is zero
// no stop for overflow here, so be warned!
void add1Bin(int num[])
{
	int i = 0;
	int carry = 1;
	int temp = 0;
	while (carry > 0)
	{
		temp = num[i];
		num[i] = (temp + carry)&1;
		carry = (temp + carry)>>1;
		i ++;
	}

}

//  this is not adding 1, but adding 2^pow to the number
void shift1Bin(int num[], int pow)
{
	int i = pow;
	int carry = 0;
 	int temp = 0;
	// add 1  to the pow + 1 position (starting at 0)
	temp = num[i]+1;
	num[i] = (temp + carry)&1;
	carry = (temp + carry)>>1;
	i++;
	while (carry > 0)
	{
		temp = num[i];
		num[i] = (temp + carry)&1;
		carry = (temp + carry)>>1;
		i++;
	}

}

// this is a "base 10 divide by 2" function which keeps track of the remainder and length of new number after division by 2
// rmsize -- zeroth entry is the remainder, first entry is size of remaining number after >>1
// note that rmsize is an array with two entries and is updated ALONG with thenumber
void div2(int num[], int rmsize[])
{
	int size = rmsize[1];
	int dvr = 0;
	int i = 0;
	for (i = size-1; i > 0; i--)
	{
		dvr = num[i]&1;
		num[i] = num[i]>>1;
		num[i-1] = num[i-1]+10*dvr;
	}
	dvr = num[0]&1;
	num[0] = num[0]>>1;
	while (0&&size>0 == num[size-1])
	{
		size = size - 1;
	}
	rmsize[0] = dvr;
	rmsize[1] = size;
//	return 0;

}

// a simple print function for large number arrays.  this assume the entry index corresponds to the base power value
// i.e. entry 0 is a^0,  entry 1 is a^1, entry 2 is a^2 etc...
void printnum(int num[], int size)
{
	int i = 0;
//	cout << endl << "the number is ";
	for (i = size-1; i >= 0; i--)
	{
		cout << num[i];
	}
	cout << endl;
}
