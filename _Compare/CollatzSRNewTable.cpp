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

// SR 2024.11.3 -- implemented a new job structure, having rank 0 dispense work to all the other ranks and keep track of data
//	right now it stops after the first break is found, but could be modified to keep going

// TO RUN THIS CODE, here is an example.... there are three command line arguments
// sbatch jobCollatzLinear2Dclean2BB.mpi 4081 25 29490
// the above command submits the job where the run command in the job file would look like:
// time prun ./LinearArrayExtra2DCleanBB $1 $2 $3
// the example says to test 2^4081 where each process has a chunk of numbers 2^25 long, and add 29490 extra padding to
// the front of the array that holds the binary number.  We always set this close to the height of 2^k, in general.
// definitely overkill...

#include <iostream>
#include <cstdlib>
#include <cmath>
#include <math.h>
#include <ctime>
#include <cstddef>
#include <stdlib.h>
#include <iomanip>
#include <time.h>
#include "mpi.h"
#include <climits>
#include <chrono>
#include <vector>

using namespace std;

//used to store info of found breaks
struct BreakInfo{
	int rank;
	int chunk;
	int chunkSize;
};

int addDecToBin(int bin[], int decimal, int binSize);
void div2(int num[], int size[]); // divide by 2 function
void div2Bin(int num[], int size); // divide by 2 function in binary
int powerfulDiv2Bin(int num[], int size); // divide by 2 many times in binary (returns new size of binary number)
int mul3p1Bin(int num[], int size); // multiply by 3 and add 1 in binary  (returns new size of binary number)
int mul3p1d2Bin(int num[], int size, int lsd); // multiply by 3, add 1, then divide by 2 in binary  (returns new size of binary number)
int digitsum(int num[], int mod[], int size); // function to compute binary mod 9 calculation
int bincompare(int num1[], int num2[], int size, int lsd); // used to compare two binary numbers
int bincompare2(int num1[], int num2[], int size, int lsd); // used to compare two binary numbers
void printnum(int num[], int size); // print out number in correct order
void add1Bin(int num[]); // just add 1 to a binary number
void shift1Bin(int num[], int pos); // add 2^pos to a binary number
int Collatz(int num[], int sizeNum); // generate Collatz sequence for binary number of length size
void CollatzSteps(int num[], int sizeNum, int **ColSeq, int ColSeqSizes[]); // generate Collatz sequence, save the values along the way in 2D array
int CollatzCompare(int num[], int sizeNum, int **ColSeq, int steps, int CoalData[], int BinExtra, int ColSeqSizes[]); // generate Collatz sequence, compare the values along the way
void updateTable(int** ColSeq, int* ColSeqSizes, int ColSteps, int binsize, int startPower, int amountOfSamples, int spacing);

//Here add the global variable that you'll use to store the number of times you don't have to compare because sizes were different.
long long int timesNoCompare = 0;
long long int timesYesCompare = 0;
long long int skippedSteps = 0;
long long int shiftWasGreaterThan1 = 0;
int maxSkipSteps = 0;

int arraySize = 0;

int main(int argc, char *argv[])
{
	//int i = 0; // loop variable only : MADE LOCAL TO LOOP SCOPES
	//int j = 0; // loop variable only : MADE LOCAL TO LOOP SCOPES

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
	//int streakChunk = 0; // how many steps in each chunk of a streak there is : MADE INTO LOCAL VARIABLE @ RANK > 0
	int iter = 0; //keeps track of chunks processed per node

	int breakCount = 0; //keeps track of the amount of breaks found
	BreakInfo* bInfos; //stores info of the breaks in BreakInfo structs


    sizeNum = atoi(argv[1]); // input of exponent k in 2^k+1
	powa = atoi(argv[2]); // power of 2 to add to base number for each proccess for the streak (shifted by process)
	extra = atoi(argv[3]); // how much padding to add to left end of number -- CHANGE IN COLLATZ FUNCTION TO MATCH
	int tableSampleAmount = atoi(argv[4]); //number of integers to build the table with
	int tableSampleSpacing = atoi(argv[5]); //distance between the numbers to build the table with

	// now we compute the value of 2^powa in base 10 so we can work out the shift of each process for their part of the work
	for (int i = 1; i <= powa; i++)
	{
		powa2 = powa2*2;
	}

	// now that we know how many each process will do (in base 10) we set up variable to display results every 1/8th of the way through
	runnercheck = powa2 >> 3;

	//cout << "powa = " << powa << ", and powa2 = " << powa2 << ", and runnercheck = " << runnercheck << endl;


	//clock_t start; // used for timing functions
	//clock_t end; // used for timing functions
	//double cpu_time_used; // used for computing length of function run

	//MPI:Status variable status is a class, and can be used for debugging purposes if need be
    	MPI_Status status;

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

	int chunkCount = 0; //keeps track of current chunk, starts at 0
	int chunks[size]; //stores chunk assignemnts, index = rank of work node assigned chunk
	int chunkSizes[size]; //chunk sizes returned by work nodes
	int lastNode; //keeps track of the last node to process
	int chunkCounts[size]; //gather for local chunk counts
	long minTimes[size]; //gather for local min times
	long maxTimes[size]; //gather for local max times

	long minTime = LONG_MAX; //keeps track of minimum times locally
	long maxTime = 0;	//keeps track of maximum times locally

	//int CoalMin = 0; // Coalescence min value
	//int CoalMax = 0; // Coalescence max value

	//We need two copies of the binary number, remember that if you pass an array into a function and change it there, it changes everywhere
	//size_t arraySize = sizeNum + extra;
	int* binnumber = new int[sizeNum + extra] (); // will hold the binary number
	int *binnumberHold = new int[sizeNum + extra] (); // will hold the binary number, this wont change in Collatz function

	//These are normal for checking that we broke the problem up at the correct locations for each process
	int *firstnumber = new int[sizeNum + extra] (); // will hold the first binary number run
	int *lastnumber = new int[sizeNum + extra] (); // will hold the last binary number run through the sequence

	int *stepArrGather = new int[size] (); // will hold the number of steps for all processes via Gatherv

	int binextra = sizeNum + extra; // needs this for compare function

	//cout << "Node " << name <<  " -- Rank " << rank << ": Large array will have " << (long long int)powa2 * (long long int)size << " entries." << endl;
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
	for (int j = 0; j < sizeNum + extra; j++)
	{
		binnumber[j] = binnumberHold[j];
	}
	//compute the number of steps
	ColSteps = Collatz(binnumber, sizeNum);

	//set  min value to number of steps so it will get updated on first run through  function
	ColData[1] = ColSteps;

	//reset the number since binnumber was modified in the ColSteps function
	for (int j = 0; j < sizeNum + extra; j++)
	{
		binnumber[j] = binnumberHold[j];
	}


	//now initialize the 2D array to fill with the numbers in the Collatz sequence now that we know the number of steps
	//pay attention to how this is allocated.  This will be sequential memory for a 2D array!!!!
	int **ColSeq = (int**)malloc(ColSteps * sizeof(int*)); // will hold the values of the Collatz Sequence
	ColSeq[0] = (int*)malloc(ColSteps * (sizeNum + extra2D) * sizeof(int));
	for (int j = 1; j < ColSteps; j++)
	{
		ColSeq[j] = ColSeq[j-1] +  (sizeNum + extra2D);
	}
	//let's make sure these are all set to zero first as malloc does not guarantee this (try calloc?)
	for (int i = 0; i < ColSteps; i++)
	{
		for (int j = 0; j < sizeNum + extra2D; j++)
		{
			ColSeq[i][j] = 0;
		}
	}

	//Here I initialize the array storing the size of each number in the collatz sequence.
	int *ColSeqSizes = new int[ColSteps];
	for (int i = 0; i < ColSteps; i++)
	{
		ColSeqSizes[i] = 0;
	}

	// ok now let's fill the array!  This is done by each process calling the CollatzSteps function, ColSeq is where sequence is stored.
	CollatzSteps(binnumber, sizeNum, ColSeq, ColSeqSizes);
	/*
	if(rank == 0){
		printf("Initial List:\n");

		for (int i = 0; i < ColSteps; i++){
			printnum(ColSeq[i], ColSeqSizes[i]);
		}
	}*/

	//MPI_Barrier(MPI_COMM_WORLD);
	updateTable(ColSeq, ColSeqSizes, ColSteps, sizeNum + extra2D, sizeNum, tableSampleAmount, tableSampleSpacing);
	/*
	if(rank == 0){
		printf("After List:\n");

		for (int i = 0; i < ColSteps; i++){
			printnum(ColSeq[i], ColSeqSizes[i]);
		}
	} */

	
	cout << rank << " finished initialization " << endl;

	//record first number
	for (int i = 0; i < sizeNum + extra; i++){
		firstnumber[i] = binnumberHold[i];
	}

	//RANK 0 AS CONTROL NODE
	//First Initializes the work nodes
	//then loops till a chunksize is found that != powa2
	// then collects remaining work and sends kill command
	//processes and gathers information about the breaks in bInfos
	if (rank == 0){
		
		bool kg = true; //keep going boolean for work

		//initial assignments
		for (int i = 1; i < size; i++)
		{	
			//record chunk assignment
			chunks[i] = chunkCount++;
			//printf("Rank %i receives chunk %i\n", i, chunkCount - 1);
			
			//send chunk assignment
			MPI_Send(binnumberHold, sizeNum+extra, MPI::INT, i, 0, MPI_COMM_WORLD);
			
			//shiftBin afterwards for next chunk
			shift1Bin(binnumberHold, powa);
			
		}

		//work loop
		while(kg){
			int chunkSize; //collection variable received
			MPI_Recv(&chunkSize, 1, MPI::INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
			int sender = status.MPI_SOURCE; //rank received from

			chunkSizes[sender] = chunkSize; //store the received value in the appropiate position in chunkSizes

			//condition to find a break
			if (chunkSize < powa2){
				breakCount++;
				kg = false;
				binnumberHold[0] = 2; //binHold is flipped to two as a stop signal
				printf("Rank %i breaks with a chunk size of %i in chunk %i\n", sender, chunkSizes[sender], chunks[sender]);
			}

			//keep going if a break isnt found, assigning the sender more work
			else{
				chunks[sender] = chunkCount++;
				//printf("Rank %i receives chunk %i\n", sender, chunkCount - 1);
			}

			//send either more work or kill command to the sender
			MPI_Send(binnumberHold, sizeNum+extra, MPI::INT, sender, 0, MPI_COMM_WORLD);

			//if more work was sent, shiftBin again
			if (kg){
				shift1Bin(binnumberHold, powa);
				if (chunkCount % 10000 == 0){
					printf("chunk %i has been sent\n", chunkCount);
				}
			}
		}

		//cleanup when break is found, continuing to record any further breaks in lt round
		for(int i = 2; i < size; i++){

			//same as work loop, gathering the remaining chunksizes
			int chunkSize;
			MPI_Recv(&chunkSize, 1, MPI::INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
			int sender = status.MPI_SOURCE;

			//record received chunk
			chunkSizes[sender] = chunkSize;

			//keeps track if any more breaks are found
			if (chunkSize < powa2){
				breakCount++;
				printf("Rank %i breaks with a chunk size of %i in chunk %i\n", sender, chunkSizes[sender], chunks[sender]);
			}

			//send kill command
			MPI_Send(binnumberHold, sizeNum, MPI::INT, sender, 0, MPI_COMM_WORLD);
		}

		bInfos = new BreakInfo[breakCount]; //initialize bInfos with number of breaks found

		//search for and record information about the breaks in ascending order in bInfos
		for (int i = 0; i < breakCount; i++){

			int minChunk = INT_MAX; // to keep track of the earliest assigned chunk
			int minRank = 0; //keeps track of the rank of said chunk

			//goes through and finds the earliest found break, recoring the rank and the chunk number
			for (int j = 1; j < size; j++){
				if (chunkSizes[j] != powa2 && chunks[j] < minChunk){
						minRank = j;
						minChunk = chunks[j];
				}
			}

			//stores break information in bInfos
			if (minRank > 0){
				bInfos[i].rank = minRank;
				bInfos[i].chunk = minChunk;
				bInfos[i].chunkSize = chunkSizes[minRank];

				//set the chunkSize to powa2 so it isnt recorded again
				chunkSizes[minRank] = powa2;
			}
		}
		
		//record the rank for the last chunk worked on
		int maxChunk = 0;

		for (int j = 1; j < size; j++){
			if (chunks[j] > maxChunk){
			lastNode = j;
			maxChunk = chunks[j];
			}
		}
	}


	//WORK NODES RANK > 0
	//receive either more work or kill command
	//loop throught he assigned chunk, checking for breaks, stop loop if break is found
	//return how many numbers have been checked in the chunk
	if (rank > 0){

		bInfos = nullptr; //for clean deletion
		iter = 0; //keeps track of the chunks this node has processed

		//work loop, will terminate when stop signal is received : binnumberHold[0] = 2
		do{
			MPI_Recv(binnumberHold, sizeNum + extra, MPI::INT, 0, 0, MPI_COMM_WORLD, &status);

			auto startTimer = chrono::high_resolution_clock::now(); //start timer
			int streakChunk = 0; //keep track of steps throught he assigned range

			//if work, do CollatzWork
			if (binnumberHold[0] != 2){
				
				 
				//printf("This is rank %i on iteration %i\n", rank, iter);
				

				for (int j = 0; j < sizeNum + extra; j++){
					binnumber[j] = binnumberHold[j];
					//firstnumber[j] = binnumberHold[j];
				}
				//Now we have each process go through their chunk of the range to check!  Not for loop goes from 0 to powa2-1
				for (int i = 0; i < powa2; i++)
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
					/*if (i % runnercheck == 0)
					{
						cout << "Node " << name <<  " -- Rank " << rank << ": We are on iteration " << i << "/" << powa2 << ", and this collatz sequence had " << steps << " steps." << endl;
					}*/
					// if we are on the last number in our chunk to work on skip this, as we do not want to increment for another pass
					// don't add 1 to last number tested so we have it!
					if (i != powa2-1)
					{
						add1Bin(binnumberHold);
						for (int j = 0; j < sizeNum + extra; j++)
						{
							binnumber[j] = binnumberHold[j];
						}
					}
					// else it is the last number worked on, so store that in lastnumber!

				}

				/*
				//store last number
				for (int j = 0; j < sizeNum + extra; j++){
					lastnumber[j] = binnumberHold[j];
				} */

				//stop timer and record time elapsed
				auto stopTimer = chrono::high_resolution_clock::now();
				long duration = chrono::duration_cast<chrono::milliseconds>(stopTimer - startTimer).count();

				if (duration < minTime) minTime = duration;
				if (duration > maxTime) maxTime = duration;

				//cout << "Rank " << rank << " finishes iteration " << iter << " in " << duration << " ms" << endl;
				
				//return chunksize to controller node
				MPI_Send(&streakChunk, 1, MPI::INT, 0, 0, MPI_COMM_WORLD);

				iter++;
			}

		} while (binnumberHold[0] != 2);
			
		
	}

	// ok, so let's let everyone catch up here!  If a process(es) got done early, it will wait here.
    cout << "Node " << name <<  " -- Rank " << rank << ": Waiting at the barrier." ;
	if (rank > 0){
		cout << "Min Time: " << minTime << " ms. Max Time: " << maxTime << " ms." << endl;
	}

	MPI_Barrier(MPI_COMM_WORLD);

	//send the rank of the node that had the last chunk to all other nodes
    //MPI_Bcast(&lastNode, 1, MPI::INT, 0, MPI_COMM_WORLD);

	//send last checked number to 0 if you were the lastNode
	/*if (rank == lastNode){
		MPI_Send(lastnumber, sizeNum + extra, MPI::INT, 0, 0, MPI_COMM_WORLD);
	} */

//	ColData[0] holds number of steps to coalesce, [1] and [2] hold min and max coalescence values

//	cout << "Rank " << rank << ": Shortest coalescence value is  " << ColData[1] << endl;
//	cout << "Rank " << rank << ": Longest coalescence value is  " << ColData[2] << endl;
//	cout << "Rank " << rank << ": Sum of coalescence values is  " << ColSum << endl;
//	cout << "Rank " << rank << ": Average coalescence value is  " << ColSum/i << endl;
	
	MPI_Gather(&iter, 1, MPI::INT, chunkCounts, 1, MPI::INT, 0, MPI_COMM_WORLD);
	MPI_Gather(&minTime, 1, MPI::LONG, minTimes, 1, MPI::LONG, 0, MPI_COMM_WORLD);
	MPI_Gather(&maxTime, 1, MPI::LONG, maxTimes, 1, MPI::LONG, 0, MPI_COMM_WORLD);

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

	//Node 0 Prints Results
	if (rank == 0)
	{
		//print out found breaks
		for (int i = 0; i < breakCount; i++){
			cout << "Node " << name << " -- Rank " << rank << ": A break was found by rank " << bInfos[i].rank << " at " << (long long int)(bInfos[i].chunk) * (long long int) powa2 + (long long int) bInfos[i].chunkSize << " in chunk " << bInfos[i].chunk << endl;
		}

		//find biggest streak
		long long int biggestStreak = (long long int)(bInfos[0].chunk) * (long long int) powa2 + (long long int) bInfos[0].chunkSize;

		for (int i = 1; i < breakCount; i++){
			long long int currentStreak = (long long int)((bInfos[i].chunk * powa2) + (long long int) bInfos[i].chunkSize) - (( (long long int) bInfos[i - 1].chunk * (long long int) powa2) + (long long int) bInfos[i -1].chunkSize);

			if (currentStreak > biggestStreak) biggestStreak = currentStreak;
		}


		//find average min and max times
		long avgMin = 0;
		long avgMax = 0;
		long absMin = LONG_MAX;
		long absMax = 0;
		int minChunk = INT_MAX;
		int maxChunk = 0;
		int avgChunk = 0;

		for (int i = 1; i < size; i++){
			avgMin += minTimes[i];
			avgMax += maxTimes[i];
			if (minTimes[i] < absMin) absMin = minTimes[i];
			if (maxTimes[i] > absMax) absMax = maxTimes[i];

			if(chunkCounts[i] < minChunk) minChunk = chunkCounts[i];
			if(chunkCounts[i] > maxChunk) maxChunk = chunkCounts[i];
			avgChunk += chunkCounts[i];
		}

		avgMin /= (size - 1);
		avgMax /= (size - 1);
		avgChunk /= (size - 1);

/*
		printf("Total Chunks Processed: %i\n", chunkCount);

		printf("Minimum Chunks Processed: %i\n", minChunk);

		printf("Maximum Chunks Processed: %i\n", maxChunk);
		
		printf("AVG Chunks Processed: %i\n", avgChunk);

		printf("AVG Minimum Time (ms): %ld\n", avgMin);

		printf("AVG Maximum Time (ms): %ld\n", avgMax);

		printf("ABS Minimum Times (ms): %ld\n", absMin);

		printf("ABS Maximum Time (ms): %ld\n", absMax);
*/
		printf("Chunks Processed Per Node:\t\t");
		for(int i = 1; i < size; i++){
			printf("%i\t", chunkCounts[i]);
		}
		printf("\n");

		printf("Mininum Times Per Node(ms):\t\t");
		for(int i = 1; i < size; i++){
			printf("%ld\t", minTimes[i]);
		}
		printf("\n");

		printf("Maxmimum Times Per Node(ms):\t");
		for(int i = 1; i < size; i++){
			printf("%ld\t", maxTimes[i]);
		}
		printf("\n");

		/*check that we added 1 the correct amount of times by looking at the VERY LAST number checked
		printf("Rank 0: First number tested was : ");
		printnum(firstnumber, sizeNum + extra);
		printf("\n"); 
		

		//receive last number from last tested chunk
		MPI_Recv(lastnumber, sizeNum + extra, MPI::INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);

		printf("Rank 0: Last number tested was : ");
		printnum(lastnumber, sizeNum + extra);
		printf("\n"); */

		// now the fun stats!



		// print out all the userful info here!  Hopefully we do not have currun == size*powa2

		cout << endl << endl;

		cout << "Node " << name <<  " -- Rank " << rank << ":  Numbers start at (2^" << sizeNum - 1 << ") + 1 with the biggest streak being " << biggestStreak << " long. Range spanned "  << (long long int) (chunkCount)*powa2 << " numbers" << endl;

		cout << "Node " << name <<  " -- Rank " << rank << ": Shortest coalescence value is :  " << ColMinReduce << endl;
		cout << "Node " << name <<  " -- Rank " << rank << ": Longest coalescence value is  :  " << ColMaxReduce << endl;
		cout << "Node " << name <<  " -- Rank " << rank << ": Sum of coalescence values is  :  " << ColSumReduce << endl;
		cout << "Node " << name <<  " -- Rank " << rank << ": Average coalescence value is  :  " << ColSumReduce/ColSumReducei << endl;
		cout << "Node " << name <<  " -- Rank " << rank << ": Total number of values checked is " << ColSumReducei << endl;
		cout << "Node " << name <<  " -- Rank " << rank << ": Total times the sizes were unequal (didn't compare digit-by-digit) is :" << timesNoCompareReduce << endl;
		cout << "Node " << name <<  " -- Rank " << rank << ": Total times the sizes were equal (did compare digit-by-digit) is      :" << timesYesCompareReduce << endl;
		cout << "Node " << name <<  " -- Rank " << rank << ": Total times we skipped a step is: " << skippedStepsReduce << endl;
		cout << "Node " << name <<  " -- Rank " << rank << ": Total times shift was greater than 1 is: " << shiftWasGreaterThan1Reduce << endl;
		cout << "Node " << name <<  " -- Rank " << rank << ": Max number of sequential divide by 2's is: " << maxSkipStepsReduce << endl;

                cout << "Node " << name <<  " -- Rank " << rank << ": Total Chunks Processed is: " <<  chunkCount << endl;
                cout << "Node " << name <<  " -- Rank " << rank << ": Minimum Chunks Processed is: " << minChunk << endl;
                cout << "Node " << name <<  " -- Rank " << rank << ": Maximum Chunks Processed is: " <<  maxChunk << endl;
                cout << "Node " << name <<  " -- Rank " << rank << ": AVG Chunks Processed is is : " << avgChunk << endl;
                cout << "Node " << name <<  " -- Rank " << rank << ": AVG Minimum Time (ms) is : " << avgMin << endl;
                cout << "Node " << name <<  " -- Rank " << rank << ": AVG Maximum Time (ms) is : " << avgMax << endl;
                cout << "Node " << name <<  " -- Rank " << rank << ": ABS Minimum Times (ms) is : " << absMin << endl;
                cout << "Node " << name <<  " -- Rank " << rank << ": ABS Maximum Time (ms) is : " << absMax << endl;


                cout << endl;
                cout << ColMinReduce << " " << ColMaxReduce << " " << ColSumReduce << " 0 " << ColSumReduce/ColSumReducei << " 0 "
                     << ColSumReducei << " " << timesNoCompareReduce << " " << timesYesCompareReduce << " 0 " << skippedStepsReduce
                     << " " << shiftWasGreaterThan1Reduce << " " << maxSkipStepsReduce << " " << chunkCount << " " << minChunk << " "
                     << maxChunk << " " << avgChunk << " " << avgMin << " " << avgMax << " " << absMin << " " << absMax << endl;



	}
	
	// now time to delete the arrays!
	delete[] stepArrGather;
	delete[] binnumber;
	delete[] binnumberHold;
	delete[] firstnumber;
	delete[] lastnumber;
	delete[] ColData;
	delete[] ColSeqSizes;
	delete[] bInfos;

	// now to free that pesky 2D array!
	free(ColSeq[0]);
	free(ColSeq);


    MPI::Finalize();

	return 0;
}

/**
 * Adds a decimal amount to a given binnumber array.
 * 
 * @param bin a binnumber array
 * @param decimal a base 10 integer
 * @param binSize size of binary number
 * @returns new size of bin
*/
int addDecToBin(int bin[], int decimal, int binSize){
	vector<int> binConversion; //store coversion from decimal to binary

	int quotient = decimal; //rename

	//division method of decimal to binary conversion
	while(quotient > 0){
		binConversion.push_back(quotient % 2);
		quotient /= 2;
	}

	if (binConversion.size() > binSize) binSize = binConversion.size();

	//binary addition
	for (int i = 0; i < binConversion.size(); i++){

		if (binConversion[i] == 1){

			if (bin[i] == 1){

				int temp = i;
				while(bin[temp] == 1){
					bin[temp] = 0;
					temp++;
				}
				bin[temp] = 1;
				if ((temp + 1) > binSize) binSize = temp + 1;
			}
			else bin[i] = 1;
		}
	}

	return binSize;
}

/**
 * Updates the lookup table finding modes within a given sample set.
 * 
 * @param ColSeq the 2D array lookup table
 * @param ColSeqSizes the array of the sizes
 * @param ColSteps number of steps in the sequence
 * @param numsize size of the binnumbers
 * @param startPower power of the start of the range
 * @param amountOfSamples number of integers to sample for the table
 * @param spacing the distance between the sampled integers
*/
void updateTable(int** ColSeq, int* ColSeqSizes, int ColSteps, int numsize, int startPower, int amountOfSamples, int spacing){
	vector<int*> samples; //stores the set of integers to sample for the table
	vector<int> frequencies; //keeps track of frequencies
	vector<int> sizes;

	vector<int> searchIndices;
	vector<int> dumpIndices;

	int maxFrequency = 0;
	int modeIndex = 0;

	int INDEX = 0;

	//initialize samples and frequencies
	for(int i = 0; i < amountOfSamples; i++){
		samples.push_back(new int[numsize]());
		samples[i][startPower - 1] = 1;

		frequencies.push_back(1);

		sizes.push_back(startPower);
		
		sizes[i] = addDecToBin(samples[i], spacing * (i + 1), sizes[i]);
	}

	//printf("Initialize Temp\n");
	
	/*
	for(int i = 0; i < samples.size(); i++){
		printnum(samples[i], sizes[i]);
	} */

	//main loop
	while (samples.size() > 1 && INDEX < ColSteps){

		//printf("INDEX %i\n", INDEX);
		//collatz step
		for(int i = 0; i < samples.size(); i++){
			if(samples[i][0] == 1){
				sizes[i] = mul3p1Bin(samples[i], sizes[i]);
			}
			else{
				div2Bin(samples[i], sizes[i]);
				sizes[i]--;
			}
		}
		//FREQUENCIES
		//initialize indices to search
		for(int i = 0; i < samples.size(); i++){
			searchIndices.push_back(i);
		}

		while (searchIndices.size() > 1){
			//search for matches to the first index and up frequency
			for(int i = 1; i < searchIndices.size(); i++){
				if (sizes[searchIndices[0]] == sizes[searchIndices[i]]){

					if (bincompare2(samples[searchIndices[0]], samples[searchIndices[i]], sizes[searchIndices[0]], 0) == -1){
						frequencies[searchIndices[0]]++;
						dumpIndices.push_back(i);
					}
				}
			}

			//erase found matches so they arent searched again
			for(int i = dumpIndices.size() - 1; i >= 0; i--){
				
				searchIndices.erase(searchIndices.begin() + dumpIndices[i]);
			}

			//erase the first index
			searchIndices.erase(searchIndices.begin());

			dumpIndices.clear();
		}

		searchIndices.clear();

		//find MODE
		
		for (int i = 0; i < frequencies.size(); i++){
			if (frequencies[i] > maxFrequency){
				maxFrequency = frequencies[i];
				modeIndex = i;
			}
		}

		//assign to table
		if (maxFrequency > 1){
			//printf("MODE FOUND FOR INDEX %i\n", INDEX);
			for(int i = 0; i < numsize; i++){
				ColSeq[INDEX][i] = samples[modeIndex][i];
			}
			ColSeqSizes[INDEX] = sizes[modeIndex];
		
			//Find and Exclude the MODE
			for (int i = 0; i < samples.size(); i++){
				if (sizes[modeIndex] == sizes[i]){
					if (bincompare2(samples[modeIndex], samples[i], sizes[modeIndex], 0) == -1){
						dumpIndices.push_back(i);
					}
				}
			}

			//delete MODE samples
			for(int i = dumpIndices.size() - 1; i >= 0; i--){
				delete samples[dumpIndices[i]];
				samples.erase(samples.begin() + dumpIndices[i]);
				sizes.erase(sizes.begin() + dumpIndices[i]);
				frequencies.erase(frequencies.begin() + dumpIndices[i]);
			}

			dumpIndices.clear();

			//reset frequencies
			maxFrequency = 1;
			for(int i = 0; i < frequencies.size(); i++){
				frequencies[i] = 1;
			}
		}

		INDEX++;
	}

	//cleanup
	for(int i = 0; i < samples.size(); i++){
		delete samples[i];
	}
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
		for (int j = 0; j < binsize; j++)
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
			cv = bincompare2(binnumber, ColSeq[steps - 1], binsize, lsd);
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


//just like bincompare except:
//returns -1 if numbers are equal
//returns 0 if numbers are not equal
int bincompare2(int num0[], int num1[], int size, int lsd)
{
        int i = 0;
        //int kg = 1; //Keep going
        int wb = 0; //Who is bigger
        int digit = lsd;

        while (digit < size + lsd)
        {
                if (num0[digit] ^ num1[i]) //numbers are not equal
                        return 0;

                i++;
                digit++;
        }

        //all digit pairs are equal so the numbers are equal
        return -1;
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

