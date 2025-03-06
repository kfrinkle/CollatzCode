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

// SR 2025.2.15 -- implemented a functionality to update the lookup table, sampling a range within a threshold

// TO RUN THIS CODE, here is an example....
// sbatch jobfile.mpi 3935 10 30000 4096 30 2 2[65] ... 
// the above command submits the job where the run command in the job file would look like:
// time prun ./executable $1 $2 $3 $4 $5 ... $n
// the example says to test 2^3935 where each process checks chunks of numbers 2^10 long, and add 30000 extra padding to
// the front of the array that holds the binary number.  We always set this close to the height of 2^k, in general.
// Then you would build the table taking 4096 samples in a range up to 2^k + 2^30, builing a new table at 2^30.
// after that are offsets of powers of 2, either by themselves or with a multiplier in the form x[y]
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
#include <stack>
#include <string>
#include <random>

using namespace std;

/**
 * Used to record information about breaks.
 * @param rank int
 * @param chunk long long int
 * @param chunkSize int
*/
struct BreakInfo{
	int rank;
	long long chunk;
	int chunkSize;
};

/**
 * Used to hold and record information of table builds
 * @param startIndex int
 * @param stopIndex int
 * @param totalReplaced int
 * @param generationTime long
 * @param spacing long
*/
struct TableBuildInfo{
	int startIndex;
	int stopIndex;
	int totalReplaced;
	long generationTime;
	long spacing;
};

/**
 *Used to store information about offset arguments
 * @param multiplier int
 * @param power int
*/
struct InitialOffset{
	int multiplier;
	int power;
};

int addDecToBin(int bin[], int decimal, int binSize);//adds a decimal number to a binnumber
void div2Bin(int num[], int size); // divide by 2 function in binary
int mul3p1BinMod(int num[], int size, int lsd); // multiply by 3, then divide by 2 in binary  (returns new size of binary number)
int digitsum(int num[], int mod[], int size); // function to compute binary mod 9 calculation
int bincompare2(int num1[], int num2[], int size, int lsd); // used to compare two binary numbers
void printnum(int num[], int size); // print out number in correct order
void printTableInfo(TableBuildInfo tbInfo); //special print for table build abortion
void add1Bin(int num[]); // just add 1 to a binary number
void shift1Bin(int num[], int pos); // add 2^pos to a binary number
int Collatz(int binnumber[], int binsize); // generate Collatz sequence for binary number of length size
void CollatzSteps(int num[], int sizeNum, int **ColSeq, int ColSeqSizes[]); // generate Collatz sequence, save the values along the way in 2D array
int CollatzCompare(int num[], int sizeNum, int **ColSeq, int steps, int CoalData[], int BinExtra, int ColSeqSizes[]); // generate Collatz sequence, compare the values along the way
InitialOffset parseInitalOffset(char* carr);
TableBuildInfo updateTable(int** ColSeq, int* ColSeqSizes, int ColSteps, int numsize, int startPower, InitialOffset* initialOffsets,int initOffsetSize, int thresholdPower, int threshMultiplier, int amountOfSamples, MPI_Comm comm); //self explanatory

/*DEPRECATED*/
//int bincompare(int num1[], int num2[], int size, int lsd); // used to compare two binary numbers
//int powerfulDiv2Bin(int num[], int size); // divide by 2 many times in binary (returns new size of binary number)
//void div2(int num[], int size[]); // divide by 2 function
//int mul3p1Bin(int num[], int size); // multiply by 3 and add 1 in binary  (returns new size of binary number, only used in generating table)
//int mul3p1d2Bin(int num[], int size, int lsd); // multiply by 3, add 1, then divide by 2 in binary  (returns new size of binary number)

//Here add the global variable that you'll use to store the number of times you don't have to compare because sizes were different.
long long int timesNoCompare = 0;
long long int timesYesCompare = 0;
long long int skippedSteps = 0;
long long int shiftWasGreaterThan1 = 0;
int maxSkipSteps = 0;

/**
 * MAIN
*/
int main(int argc, char *argv[]){	

//ARGUEMENTS
	int sizeNum = atoi(argv[1])+1; // input of exponent k in 2^k+1 !!IMPORTANT +1 ADDED IF USED TO POINT TO THE POWER -1 FROM IT!!
	int powa = atoi(argv[2]); // power of 2 to add to base number for each proccess for the streak (shifted by process)
	int extra = atoi(argv[3]); // how much padding to add to left end of number -- CHANGE IN COLLATZ FUNCTION TO MATCH
	int tableSampleAmount = atoi(argv[4]); //number of integers to build the table with
	int thresholdPower = atoi(argv[5]); //distance between the numbers to build the table with
	InitialOffset initialOffsets[argc-6]; //offsets

//COMMUNICATION
    MPI::Init(argc, argv);
	MPI_Status status;
    int size = MPI::COMM_WORLD.Get_size(); // get total size of the world
	int rank = MPI::COMM_WORLD.Get_rank(); // each process gets its own rank

	char processorName[MPI_MAX_PROCESSOR_NAME];  // create character array for the names of the compute nodes
	int processorNamelength = 0; // used for printing out the name of the compute node
    MPI::Get_processor_name(processorName, processorNamelength);

	//make a group for just the workers
	int color = (rank > 0) ? 1 : 0;
	MPI_Comm workCommunicator;
	MPI_Comm_split(MPI_COMM_WORLD, color, rank, &workCommunicator);

//MAIN VARIABLES
	unsigned int powa2 = 1; // power of 2 value in base 10 (for each node to work on, should be less than INT MAX)

	//We need two copies of the binary number, remember that if you pass an array into a function and change it there, it changes everywhere
	int* binnumber = new int[sizeNum + extra] (); // will hold the binary number
	int *binnumberHold = new int[sizeNum + extra] (); // will hold the binary number, this wont change in Collatz function	

	//data recording for prints
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

	//RANK 0 COMMUNICATOR VARIABLES
		long long chunkCount = 0; //keeps track of current chunk, starts at 0
		long long chunks[size]; //stores chunk assignemnts, index = rank of work node assigned chunk
		int chunkSizes[size]; //chunk sizes returned by work nodes
		int chunkCounts[size]; //gather for local chunk counts
		int breakCount = 0; //keeps track of the amount of breaks found
		BreakInfo* bInfos; //stores info of the breaks in BreakInfo structs

		//printing
		long minTimes[size]; //gather for local min times
		long maxTimes[size]; //gather for local max times
		long duration = 0; // keep track of how long it takes to do a set of chunks
        long oldduration = 0; // ditto

	//WORK NODE VARIABLES
		int iter = 0; //keeps track of chunks processed per node
		int steps = 0; // keep track of the number of steps in Collatz Sequence
		int extra2D = 1000; // just enough  padding on the end of the 2D array to hold extra 3x+1 digits for start of code

		long minTime = LONG_MAX; //keeps track of minimum times locally
		long maxTime = 0;	//keeps track of maximum times locally

		int *ColData = new int[3] (); // holds number of steps to coalesce, [1] and [2] hold min and max coalescence values
		int ColSteps; //compute the number of steps
		
		int **ColSeq; //2D sequential int array to hold Collatz Sequence
		int *ColSeqSizes; //keeps track of the sizes of the numbers stores in ColSeq
		TableBuildInfo tbInfos = {0}; //infos got from table builds
		
	
//INITIALIZAITION

	//initialize initial offsets
	for(int i = 0; i < argc - 6; i++){
		initialOffsets[i] = parseInitalOffset(argv[6+i]);
		//abort if there is an arguement error
		if (initialOffsets[i].power == -1) return 1;
	}

	//init powa2
	for (int i = 1; i <= powa; i++){
		powa2 = powa2*2;
	}

	//init binnumbers
	binnumberHold[sizeNum-1] = 1;//Set the leading value in the array to 1 so we have 2^k
	binnumberHold[0] = 1;//Set the 0th place to 1 so we all have 2^k+1 now.

	//apply initial offsets
	for(int i = 0; i < argc - 6; i++){
		for(int j = 0; j < initialOffsets[i].multiplier; j++){
			shift1Bin(binnumberHold, initialOffsets[i].power);
		}
	}

	//copy to binnumber for destructive procedures
	for (int j = 0; j < sizeNum + extra; j++){
		binnumber[j] = binnumberHold[j];
	}

	//initialize colsteps
	ColSteps = Collatz(binnumber, sizeNum);

	//init ColSteps
	ColData[1] = ColSteps; //used by CollatzCompare()	
	//copy binnumberHold to binnumber for use in destructive functions like Collatz()

		//WORKER VARIABLE INIT

		//now initialize the 2D array to fill with the numbers in the Collatz sequence now that we know the number of steps
		//pay attention to how this is allocated.  This will be sequential memory for a 2D array!!!!
		ColSeq = (int**)malloc(ColSteps * sizeof(int*)); 
		ColSeq[0] = (int*)malloc(ColSteps * (sizeNum + extra2D) * sizeof(int));
		for (int j = 1; j < ColSteps; j++){
			ColSeq[j] = ColSeq[j-1] +  (sizeNum + extra2D);
		}
		//let's make sure these are all set to zero first as malloc does not guarantee this (try calloc?)
		for (int i = 0; i < ColSteps; i++){
			for (int j = 0; j < sizeNum + extra2D; j++){
				ColSeq[i][j] = 0;
			}
		}

		//init ColSeqSizes
		ColSeqSizes = new int[ColSteps]();

		//reset the number since binnumber was modified in the ColSteps function
		for (int j = 0; j < sizeNum + extra; j++){
			binnumber[j] = binnumberHold[j];
		}
		
		//initialize ColSeq on Worknodes
		if (rank > 0){
			CollatzSteps(binnumber, sizeNum, ColSeq, ColSeqSizes);
			tbInfos = updateTable(ColSeq, ColSeqSizes, ColSteps, sizeNum + extra, sizeNum, initialOffsets, argc-6, thresholdPower, 0, tableSampleAmount, workCommunicator);
		}

//END INITIALIZATION

    cout << "Node[" << processorName << "]:  Rank " << rank << " finished initialization " << "\n";
	
	MPI_Barrier(MPI_COMM_WORLD);

	if(rank > 0 && tbInfos.generationTime == 0){
		cout << "[" << processorName << "] Rank " << rank << ": Table Build Aborted." << "\n";
	}
	
	MPI_Barrier(MPI_COMM_WORLD);
	
	if(rank == 1){
		printTableInfo(tbInfos);
	}
	

//START PROGRAM

	//RANK 0 AS CONTROL NODE
	//First Initializes the work nodes
	//then loops till a chunksize is found that != powa2
	//will send rebuild table flag when the current chunk is a multiple of the 2^(thresholdPower - chunksize)
	//then collects remaining work and sends kill command
	//processes and gathers information about the breaks in bInfos
	if (rank == 0){

		bool kg = true; //keep going boolean for work
        auto startTimerPZero = chrono::high_resolution_clock::now(); //start timer
		long long threshold = 1; //number to check against to send rebuild table flag
		int rebuildFlags[size] = {0}; //keeps track of flags sent

		//initialize threshold
		for(int i = 0; i < thresholdPower - powa; i++){
			threshold *= 2;
		}

		//initial assignments
		for (int i = 1; i < size; i++)
		{
			//record chunk assignment
			chunks[i] = chunkCount++;
			//printf("Rank %i receives chunk %i\n", i, chunkCount - 1);
			//send chunk assignment
			MPI_Send(binnumberHold, sizeNum+extra, MPI_INT, i, 0, MPI_COMM_WORLD);

			//shiftBin afterwards for next chunk
			shift1Bin(binnumberHold, powa);
		}

		//work loop
		while(kg){
			int chunkSize; //collection variable received
			MPI_Recv(&chunkSize, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
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

				//if threshold is met, ready flags for rebuild
				if((chunkCount)%threshold == 0){
					printf("Flag Triggered!\n");
					for(int i = 0; i < size; i++){
						rebuildFlags[i]++;
					}
				}

				//if a flag is waiting, send it
				if(rebuildFlags[sender] != 0){
					//printf("Sent flag to rank %i.\n", sender);
					binnumberHold[sizeNum+extra - 1] = rebuildFlags[sender];
					rebuildFlags[sender] = 0;
				}
			}

			//send either more work or kill command to the sender
			MPI_Send(binnumberHold, sizeNum+extra, MPI::INT, sender, 0, MPI_COMM_WORLD);

			//reset flag
			binnumberHold[sizeNum+extra-1] = 0;

			//if more work was sent, shiftBin again
			if (kg){
				shift1Bin(binnumberHold, powa);


                if (chunkCount % 10000 == 0){
					//printf("chunk %i has been sent\n", chunkCount);

                    auto stopTimerPZero = chrono::high_resolution_clock::now();
                    duration = chrono::duration_cast<chrono::milliseconds>(stopTimerPZero - startTimerPZero).count();

                    cout << "Node " << processorName <<  " -- Rank " << rank << ": Chunk " << chunkCount << " has been sent, previous chunk processing time: " << duration - oldduration << "\n";
                    oldduration = duration;

                    auto startTimerPZero = chrono::high_resolution_clock::now(); //start timer over

                }

			}
		}

		//cleanup when break is found, continuing to record any further breaks in lt round
		for(int i = 2; i < size; i++){

			//same as work loop, gathering the remaining chunksizes
			int chunkSize;
			MPI_Recv(&chunkSize, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
			int sender = status.MPI_SOURCE;

			//record received chunk
			chunkSizes[sender] = chunkSize;

			//keeps track if any more breaks are found
			if (chunkSize < powa2){
				breakCount++;
				printf("Rank %i breaks with a chunk size of %i in chunk %i\n", sender, chunkSizes[sender], chunks[sender]);
			}

			//send kill command
			MPI_Send(binnumberHold, sizeNum, MPI_INT, sender, 0, MPI_COMM_WORLD);
		}

		bInfos = new BreakInfo[breakCount]; //initialize bInfos with number of breaks found

		//search for and record information about the breaks in ascending order in bInfos
		for (int i = 0; i < breakCount; i++){

			long long minChunk = LLONG_MAX; // to keep track of the earliest assigned chunk
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
	}

	//WORK NODES RANK > 0
	//receive either more work or kill command
	//rebuild lookup table if it receives the rebuild flag
	//loop through the assigned chunk, checking for breaks, stop loop if break is found
	//return how many numbers have been checked in the chunk
	if (rank > 0){

		bInfos = nullptr; //for clean deletion
		iter = 0; //keeps track of the chunks this node has processed
		int thresholdsReached = 0;

		//work loop, will terminate when stop signal is received : binnumberHold[0] = 2
		do{
			MPI_Recv(binnumberHold, sizeNum + extra, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);

			auto startTimer = chrono::high_resolution_clock::now(); //start timer
			int streakChunk = 0; //keep track of steps through the assigned range

			//rebuild ColSeq lookup table if the flag is received from rank 0
			if (binnumberHold[sizeNum+extra - 1] != 0){
				thresholdsReached += binnumberHold[sizeNum + extra - 1];
				tbInfos = updateTable(ColSeq, ColSeqSizes, ColSteps, sizeNum + extra, sizeNum, initialOffsets, argc - 6, thresholdPower, thresholdsReached, tableSampleAmount, workCommunicator);
				if(rank == 1){
					printTableInfo(tbInfos);
				}
				if(tbInfos.generationTime == 0){
					cout << "[" << processorName << "] Rank " << rank << ": Table Build Aborted." << "\n";
				}
				binnumberHold[sizeNum + extra - 1] = 0; //reset flag
			}

			//if work, do CollatzWork
			if (binnumberHold[0] != 2){

				//printf("This is rank %i on iteration %i\n", rank, iter);

				for (int j = 0; j < sizeNum + extra; j++){
					binnumber[j] = binnumberHold[j];
					//firstnumber[j] = binnumberHold[j];
				}

				//Now we have each process go through their chunk of the range to check!  Not for loop goes from 0 to powa2-1
				for (int i = 0; i < powa2; i++){

					// store the value of steps in the CollatzCompare function
					// remember CollatzCompare compares the numbers at each step in the sequence to that of 2^k+1
					// ColSeq is the array which holds the numbers in the sequence for 2^k+1
					// ColData is the array of useful information (steps, max and min steps)

					steps = CollatzCompare(binnumber, sizeNum, ColSeq, ColSteps, ColData, sizeNum + extra, ColSeqSizes);

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
						cout << "Node " << processorName <<  " -- Rank " << rank << ": Broke streak at " << streakChunk << "\n";
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

				//stop timer and record time elapsed
				auto stopTimer = chrono::high_resolution_clock::now();
				long duration = chrono::duration_cast<chrono::milliseconds>(stopTimer - startTimer).count();

				if (duration < minTime) minTime = duration;
				if (duration > maxTime) maxTime = duration;
				
				//cout << "Rank " << rank << " finishes iteration " << iter << " in " << duration << " ms" << endl;
				//return chunksize to controller node
				MPI_Send(&streakChunk, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);

				iter++;
			}

		} while (binnumberHold[0] != 2);
	}

//PRINT RESULTS

	// ok, so let's let everyone catch up here!  If a process(es) got done early, it will wait here.
    cout << "Node " << processorName <<  " -- Rank " << rank << ": Waiting at the barrier." << "\n" ;
	if (rank > 0){
		cout << "Min Time: " << minTime << " ms. Max Time: " << maxTime << " ms." << "\n";
	}

	MPI_Barrier(MPI_COMM_WORLD);

	MPI_Gather(&iter, 1, MPI_INT, chunkCounts, 1, MPI_INT, 0, MPI_COMM_WORLD); //gather chunks done by each worker
	MPI_Gather(&minTime, 1, MPI_LONG, minTimes, 1, MPI_LONG, 0, MPI_COMM_WORLD); //gather minimum times of each worker
	MPI_Gather(&maxTime, 1, MPI_LONG, maxTimes, 1, MPI_LONG, 0, MPI_COMM_WORLD); //gather max times of each worker

	// simple reduce functions to tally the information about max/min and total number of steps
	MPI_Reduce(&ColSum, &ColSumReduce, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
	MPI_Reduce(&ii, &ColSumReducei, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
	MPI_Reduce(&ColData[1], &ColMinReduce, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
	MPI_Reduce(&ColData[2], &ColMaxReduce, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
	MPI_Reduce(&maxSkipSteps, &maxSkipStepsReduce, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
	MPI_Reduce(&timesNoCompare, &timesNoCompareReduce, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
	MPI_Reduce(&timesYesCompare, &timesYesCompareReduce, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
	MPI_Reduce(&skippedSteps, &skippedStepsReduce, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
	MPI_Reduce(&shiftWasGreaterThan1, &shiftWasGreaterThan1Reduce, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

	//Node 0 Prints Results
	if (rank == 0){

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

		//print out found breaks
		for (int i = 0; i < breakCount; i++){
			cout << "Node " << processorName << " -- Rank " << rank << ": A break was found by rank " << bInfos[i].rank << " at " << (long long int)(bInfos[i].chunk) * (long long int) powa2 + (long long int) bInfos[i].chunkSize << " in chunk " << bInfos[i].chunk << "\n";
		}


		// print out all the userful info here!  Hopefully we do not have currun == size*powa2
		cout << "Node " << processorName <<  " -- Rank " << rank << ":  Numbers start at (2^" << sizeNum - 1 << ")";
		for(int i = 0; i < argc - 6; i++){
			printf(" + %i(2^%i)", initialOffsets[i].multiplier, initialOffsets[i].power);
		}
		cout << " + 1 with the biggest streak being " << biggestStreak << " long. Range spanned "  << (long long int) (chunkCount)*powa2 << " numbers" << "\n";

		// now the fun stats!
		cout << "Node " << processorName <<  " -- Rank " << rank << ": Shortest coalescence value is :  " << ColMinReduce << "\n";
		cout << "Node " << processorName <<  " -- Rank " << rank << ": Longest coalescence value is  :  " << ColMaxReduce << "\n";
		cout << "Node " << processorName <<  " -- Rank " << rank << ": Sum of coalescence values is  :  " << ColSumReduce << "\n";
		cout << "Node " << processorName <<  " -- Rank " << rank << ": Average coalescence value is  :  " << ColSumReduce/ColSumReducei << "\n";
		cout << "Node " << processorName <<  " -- Rank " << rank << ": Total number of values checked is " << ColSumReducei << "\n";
		cout << "Node " << processorName <<  " -- Rank " << rank << ": Total times the sizes were unequal (didn't compare digit-by-digit) is :" << timesNoCompareReduce << "\n";
		cout << "Node " << processorName <<  " -- Rank " << rank << ": Total times the sizes were equal (did compare digit-by-digit) is      :" << timesYesCompareReduce << "\n";
		cout << "Node " << processorName <<  " -- Rank " << rank << ": Total times we skipped a step is: " << skippedStepsReduce << "\n";
		cout << "Node " << processorName <<  " -- Rank " << rank << ": Total times shift was greater than 1 is: " << shiftWasGreaterThan1Reduce << "\n";
		cout << "Node " << processorName <<  " -- Rank " << rank << ": Max number of sequential divide by 2's is: " << maxSkipStepsReduce << "\n";
        cout << "Node " << processorName <<  " -- Rank " << rank << ": Total Chunks Processed is: " <<  chunkCount << "\n";
        cout << "Node " << processorName <<  " -- Rank " << rank << ": Minimum Chunks Processed is: " << minChunk << "\n";
        cout << "Node " << processorName <<  " -- Rank " << rank << ": Maximum Chunks Processed is: " <<  maxChunk << "\n";
        cout << "Node " << processorName <<  " -- Rank " << rank << ": AVG Chunks Processed is is : " << avgChunk << "\n";
        cout << "Node " << processorName <<  " -- Rank " << rank << ": AVG Minimum Time (ms) is : " << avgMin << "\n";
        cout << "Node " << processorName <<  " -- Rank " << rank << ": AVG Maximum Time (ms) is : " << avgMax << "\n";
        cout << "Node " << processorName <<  " -- Rank " << rank << ": ABS Minimum Times (ms) is : " << absMin << "\n";
        cout << "Node " << processorName <<  " -- Rank " << rank << ": ABS Maximum Time (ms) is : " << absMax << endl;
	}

	// now time to delete the arrays!
	delete[] binnumber;
	delete[] binnumberHold;
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
 * @param decimal a base 10 long integer
 * @param binSize size of binary number
 * @returns new size of bin
*/
int addDecToBin(int bin[], long decimal, int binSize){
	vector<int> binConversion; //store coversion from decimal to binary

	long quotient = decimal; //rename

	//division method of decimal to binary conversion
	while(quotient > 0){
		binConversion.push_back((int)(quotient % 2));
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
 * It will first initialise the samples along the range picking randomly in equal clamped sets.
 * It will then parallely check the sample heights to ensure no breaks.
 * Then it will initialize the table 1 integer after the threshold.
 * If a break is found it will abort, returning all 0s in the return TableBuildInfo.
 * Finally it will step through the collatz sequence on each sample, recording the first highest mode in the lookup table.
 * 
 * @param ColSeq the 2D array lookup table
 * @param ColSeqSizes the array of the sizes
 * @param ColSteps number of steps in the sequence
 * @param numsize size of the binnumbers
 * @param startPower power of the start of the range
 * @param initialOffsets array holding the powers of 2 to add to the samples
 * @param initOffsetSize size of initialOffsets array
 * @param thresholdPower power of the size of the sample range
 * @param threshMultiplier which range of samples you are at
 * @param amountOfSamples number of integers to sample for the table
 * @param comm the comminicator for the work group using MPI_Split, rank 0 should never enter this function
 * @returns TableBuildInfo struct, will be all 0s if the table build fails
*/
TableBuildInfo updateTable(int** ColSeq, int* ColSeqSizes, int ColSteps, int numsize, int startPower, InitialOffset* initialOffsets, int initOffsetSize, int thresholdPower, int threshMultiplier, int amountOfSamples, MPI_Comm comm){
	
//VARIABLES
	//group MPI info
	int rank = 0; //local rank
	int size = 0; //local groupsize
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &size);
	int breakFound;
	int breakFlag;
	
	vector<int> threshMConversion; //vector to hold the binary version of threshMultiplier
	long spacing = 1; //spacing between samples

	//main variables to store sample info
	vector<int*> samples; //stores the set of integers to sample for the table
	vector<int> frequencies; //keeps track of frequencies
	vector<int> sizes; //keep tack of number sizes
	int tempBin[numsize] = {0}; //a binnumber array to copy to when doing something destructive, like Collatz()
	int tempTempBin[numsize] = {0}; //same deal as the previous one
	long offsets[amountOfSamples] = {0}; //offsets for numbers

	//start long randomization
	random_device rd; //random device
	mt19937 generator(rd()); //random number generator seeded with random device
	uniform_int_distribution<long> distribution(0, std::numeric_limits<long>::max()); //define range of random number generator

	//for mode searching
	vector<int> searchIndices; //holds current indexes to search through
	stack<int> dumpIndices; //holds indexes marked for deletion

	int maxFrequency = 0; //the highest frequency of a mode found
	int modeIndex = 0; //the index of the highest frequency mode found
	int INDEX = 0; //current index in ColSeq

	//to keep track of build info
	TableBuildInfo tbInfos = {0}; //struct to return info in
	chrono::high_resolution_clock::time_point startTime = chrono::high_resolution_clock::now(); //start of the build timer
	bool firstIndexFound = false; //keeps track of whether an mode has been found in the entire build

//INITIALIZE

	//initialize theshMConversion with binary conversion
	int quotient = threshMultiplier;
	while(quotient > 0){
		threshMConversion.push_back(quotient%2);
		quotient /= 2;
	}

	//initialize spacing : (2^thresholdPower / amountOfSamples) atleast 1
	for (int i = 0; i < thresholdPower; i++){
		spacing*=2;
	}
	spacing /= (long)amountOfSamples;
	if (spacing == 0) spacing = 1;
	tbInfos.spacing = spacing;//record spacing
	
	//initialize offsets
	for(int i = 0; i < amountOfSamples; i++){
		offsets[i] = distribution(generator)%spacing - (spacing/2);
	}

	//INITIALIZE SAMPLES
	//samples are initialized as 2^startpower + spacing(i+1)
	//then they are given a random offset to expand the sample set 
	for(int i = 0; i < amountOfSamples; i++){

		samples.push_back(new int[numsize]());
		samples[i][startPower - 1] = 1;

		//add threshold to number, INITAL POWER MUST BE HIGHER THAN THE THRESHOLD
		for(int j = 0; j < threshMConversion.size(); j++){
			samples[i][thresholdPower + j] = threshMConversion[j];
		}
		
		//add initial offsets
		for(int j = 0; j < initOffsetSize; j++){
			for(int k = 0; k < initialOffsets[j].multiplier; k++){
				shift1Bin(samples[i], initialOffsets[j].power);
			}
		}

		frequencies.push_back(1);

		sizes.push_back(startPower);

		sizes[i] = addDecToBin(samples[i], (spacing * (long)(i + 1)) + offsets[i], sizes[i]);
	}


//CHECK HEIGHTS
	//distribution of ranges for each node to check
	int assignedSize = samples.size() / size;
	int extra = samples.size() % size;
	int assignedSizes[size] = {0};
	int startIndices[size] = {0};

	for(int i = 0; i < size; i++){
		assignedSizes[i] = assignedSize;
	}

	for(int i = 0; i < extra; i++){
		assignedSizes[i]++;
	}

	int runningTotal = 0;
	for(int i = 1; i < size; i++){
		runningTotal += assignedSizes[i-1];
		startIndices[i] = runningTotal;
	}

	//check heights for break, different ranks check their assigned range
	for(int i = 0; i < assignedSizes[rank]; i++){
			
		//array copy because Collatz() is destructive
		for(int j = 0; j < sizes[startIndices[rank] + i]; j++){
			tempBin[j] = samples[startIndices[rank] + i][j];
		}
		
		int ColData[3] = {0, ColSteps, 0};
		int currentSteps = CollatzCompare(tempBin, sizes[startIndices[rank]+i], ColSeq, ColSteps, ColData, numsize, ColSeqSizes);

		if(currentSteps != ColSteps){	
			if(rank == 0) printf("First Trigger: x: %i, colSteps: %i\n", currentSteps, ColSteps);
			 breakFound = 1;
			 break;
		}
	}

// REINITIALIZE COLSEQ with (2^startPower + threshMultiplier(2^thresholdPower) + 1)
	if (threshMultiplier > 0){
		//make sure temp bin is reset
		for(int i = 0; i < numsize; i++){
			tempBin[i] = 0;
		}

		//build the sample
		tempBin[startPower - 1] = 1;
		for(int i = 0; i < threshMConversion.size(); i++){
			tempBin[thresholdPower + i] = threshMConversion[i];
		}
		tempBin[0] = 1;

		//apply initial offsets
		for(int j = 0; j < initOffsetSize; j++){
			for(int k = 0; k < initialOffsets[j].multiplier; k++){
				shift1Bin(tempBin, initialOffsets[j].power);
			}
		}

		//copy the sample for Collatz() because it eats things and doesnt give them back
		for(int i = 0; i < startPower; i++){
			tempTempBin[i] = tempBin[i];
		}
		
		int ColData[3] = {0, ColSteps, 0}; //dummy
		int currentSteps = CollatzCompare(tempTempBin, startPower, ColSeq, ColSteps, ColData, numsize, ColSeqSizes);

		//abort if the initializing number breaks the streak
		if( currentSteps != ColSteps){
			//printf("Second Trigger: x: %i, colSteps: %i\n", currentSteps, ColSteps);
			breakFound = 1;
		}

		//initialize ColSeq with the new sample
		else CollatzSteps(tempBin, startPower, ColSeq, ColSeqSizes);
	}

	//communicate if a break is found
	MPI_Allreduce(&breakFound, &breakFlag, 1, MPI_INT, MPI_BOR, comm);

	//handle a break and ABORT if there is one
	if(breakFlag){
		for(int i = 0; i < samples.size(); i++){
			delete samples[i];
		}
		return tbInfos;
	}

//START TABLE UPDATE
	while (samples.size() > 1 && INDEX < ColSteps){
		
		//collatz step for this INDEX
		for(int i = 0; i < samples.size(); i++){
			if(samples[i][0] == 1){
				sizes[i] = mul3p1BinMod(samples[i], sizes[i], 0);
			}
			else{
				div2Bin(samples[i], sizes[i]);
				sizes[i]--;
			}
		}
		//initialize indices to search
		for(int i = 0; i < samples.size(); i++){
			searchIndices.push_back(i);
		}

		//search through and find frequencies of numbers
		while (searchIndices.size() > 1){
			//search for matches to the first index and up frequency
			for(int i = 1; i < searchIndices.size(); i++){
				if (sizes[searchIndices[0]] == sizes[searchIndices[i]]){

					if (bincompare2(samples[searchIndices[0]], samples[searchIndices[i]], sizes[searchIndices[0]], 0) == -1){
						frequencies[searchIndices[0]]++;
						dumpIndices.push(i);
					}
				}
			}

			//erase found matches so they arent searched again
			while(!dumpIndices.empty()){
				searchIndices.erase(searchIndices.begin() + dumpIndices.top());
				dumpIndices.pop();
			}

			//erase the first index
			searchIndices.erase(searchIndices.begin());
		}

		//erase any indexes left
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
				//if(INDEX == 43) printf("%i\n", iter++);
				ColSeq[INDEX][i] = samples[modeIndex][i];
			}
			ColSeqSizes[INDEX] = sizes[modeIndex];
					
			//update info in tbInfos
			if (!firstIndexFound){
				tbInfos.startIndex = INDEX;
				firstIndexFound = true;
			}
			tbInfos.stopIndex = INDEX;
			tbInfos.totalReplaced++;

			//Find and mark samples equal to the MODE for deletion
			for (int i = 0; i < samples.size(); i++){
				if (sizes[modeIndex] == sizes[i]){
					if (bincompare2(samples[modeIndex], samples[i], sizes[modeIndex], 0) == -1){
						dumpIndices.push(i);
					}
				}
			}
			//delete MODE samples
			while(!dumpIndices.empty()){
				int dumpIndex = dumpIndices.top();

				delete samples[dumpIndex];
				samples.erase(samples.begin() + dumpIndex);
				sizes.erase(sizes.begin() + dumpIndex);
				frequencies.erase(frequencies.begin() + dumpIndex);

				dumpIndices.pop();
			}
			//reset frequencies
			maxFrequency = 1;
			for(int i = 0; i < frequencies.size(); i++){
				frequencies[i] = 1;
			}
		}
		//go to next index in ColSeq
		INDEX++;
	}
	//cleanup
	for(int i = 0; i < samples.size(); i++){
		delete samples[i];
	}

	//record elapsed build time in tbInfos
	tbInfos.generationTime = chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - startTime).count();

	return tbInfos;
}


/** 
 * generate Collatz sequence for binary number binnumber of length binsize, return number of steps
 * 
 * @param binnumber binary to sequence !IS DESTRUCTIVE!
 * @param binsize size of the number
 * @returns Number of steps through the Collatz sequence
 * @warning changes the information in binnumber
*/
int Collatz(int binnumber[], int binsize){
	int steps = 0; // just keep track of the number of steps!
	int lsd = 0;
	// for binary, binsize = 1 means your number is 1, the end of the sequence
	while (binsize > 1)
	{	
		
		// if its off, do 3*x+1, note binsize value is returned
		if(binnumber[lsd] == 1)
		{
			binsize = mul3p1BinMod(binnumber, binsize, lsd);
			steps++;
		}
		// if its even, do x>>1, note binsize value is automatically adjusted

		while(binnumber[lsd] == 0){
			lsd++;
			binsize--;
			steps++;
		}
		//steps ++;
	}
	//printf("Collatz: %i\n", steps);

	return steps;

}
/**
 * generate lookup table with one number, saving all numbers in the sequence as it goes, storing it in ColSeq and ColSeqSizes
 * 
 * @param binnumber binnumber to build the ColSeq table with !IS DESTRUCTIVE!
 * @param binsize size of the initial number
 * @param ColSeq 2D sequential array to store the subsequent binnumbers
 * @param ColSeqSizes array to store sizes of the subsequent binnumbers
 * @warning changes the information in binnumber
*/
void CollatzSteps(int binnumber[], int binsize, int **ColSeq, int ColSeqSizes[])
{
	int j = 0; // loop variable only
	int steps = 0; // keep track of number of steps

	while (binsize > 1)
	{
		// once again, if odd, 3x+1
		if(binnumber[0] == 1)
		{
			binsize = mul3p1BinMod(binnumber, binsize, 0);
			
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

	//printf("CollatzSteps: %i\n", steps);

	//	return steps;
}


/**
 * generate Collatz sequence, stopping when it hits a number at same index in ColSeq array
 * 
 * @param binNumber binnumber to sequence and compare with the lookup table !IS DESTRUCTIVE!
 * @param binsize size of the binnumber
 * @param ColSeq lookup table
 * @param ColSteps size of the lookup table
 * @param ColData size 3 array where [0] should equal ColSteps
 * @param BinExtra total size of the binnumber array
 * @param ColSeqSizes sizes of the numbers stored in the lookup table
 * @returns Number of steps through the Collatz sequence
*/
int CollatzCompare(int binnumber[], int binsize, int **ColSeq, int ColSteps, int ColData[], int BinExtra, int ColSeqSizes[])
{
// ColData[0] holds number of steps to coalesce, [1] and [2] hold min and max coalescence values

	int steps = 0; // the step count variable
	int cv = 0; // compare value between current step and base step
	int shifts = 0;
	int lsd = 0;
	while (binsize > 1)
	{
		// this has been modified from the original approach
		// only shift lsd up one at a time when /2 in hopes of finding match in lookup table earlier
		// only perform 3x+1, not (3x+1)/2
		if(binnumber[lsd] == 1)
		{
//			binsize = mul3p1d2Bin(binnumber, binsize, lsd);
//			steps += 2;
			
			binsize = mul3p1BinMod(binnumber, binsize, lsd);
			steps++;

		}
		else
		{
			lsd++;
			steps++;
			binsize--;

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

	//printf("CollatzCompareSteps: %i\n", steps);
	return steps;

}
/**
 * Used to check if two binnary numbers of equal size are equal.
 * 
 * @warning !BOTH BIN NUMBERS NEED TO BE THE SAME SIZE!
 * @param num0 binnumber to compare
 * @param num1 second binnumber to compare
 * @param size size of the binnumbers
 * @param lsd least significant digit
 * @returns -1 if numbers are equal, 0 if they are not equal
*/
int bincompare2(int num0[], int num1[], int size, int lsd){
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

/**
 * multiply by 3, then add 1 in binary
 * 
 * @param num binnumber to perform the operation on
 * @param size size of binnumber
 * @param lsd least significant digit of binnumber
 * @returns new size of the binnumber
*/
int mul3p1BinMod(int num[], int size, int lsd){
	//SR - Sorry Frinkle, I had to change it :(
    int i = lsd;//least significant digit
	int localSize = lsd + size;
    int carry = 1; //need to be 1 to add 1
    int tempCurr = 0; //dont touch plz, can be 1 instead of carry to add 1

	//if both carry and tempCurr are initialized as 0, this will work as a multiply by 3

    while(i < localSize){
		int tempHold = num[i];
		num[i] = tempCurr + num[i] + carry;
		carry = num[i]>>1; //mod2
		num[i] = num[i]&1; //1 if odd, 0 if even
		tempCurr = tempHold; //keep track of previous digit

		i++;
	}
	
	if(carry){
		num[lsd + size] = 0;
		num[lsd + size + 1] = 1;
		size+=2;
	}
	else{
		num[lsd + size] = 1;
        size++;
	}

	return size;
}

/**
 * binary divide by two simply shifts all the bits down 1
 * 
 * @param num binnumber to perform the operation to
 * @param size size of binnumber
*/
void div2Bin(int num[], int size)
{
	int i = 0;
	for (i = 0; i < size-1; i++)
	{
		num[i] = num[i+1];
	}
	num[size-1] = 0;

}

/**
 * binary add 1, simply adds one at 0th entry and goes until carry is zero
 * 
 * @warning !CAN OVERFLOW!
 * @param num binnumber to perform the operation to
*/
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

/**
 * Adds a power of 2 to a binnumber.
 * 
 * @param num binnumber to perform the operation on
 * @param pow power of 2 to add to the binnumber
*/
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

/**
 * a simple print function for large number arrays.  this assume the entry index corresponds to the base power value
 * i.e. entry 0 is a^0,  entry 1 is a^1, entry 2 is a^2 etc...
 * @param num binnumber to print
 * @param size size of the binnumber
*/
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

/**
 * Prints info from table.
 * 
 * @param tbInfos TableBuildInfo struct from table build.
*/
void printTableInfo(TableBuildInfo tbInfos){
	if (tbInfos.generationTime == 0) printf("Table Build Aborted\n");
	else{
		printf(
			"Table Built. First Index: %i. Last Index %i. Total Replaced %i. Time taken (ms): %ld. Spacing: %ld\n",
			tbInfos.startIndex,
			tbInfos.stopIndex,
			tbInfos.totalReplaced,
			tbInfos.generationTime,
			tbInfos.spacing
		);
	}

}
/**
 * Parses an input argument in the form x(y) with x being a multiplier and y being a power of 2
 * 
 * @param carr character array from the arguments
 * @returns the multiplier and power of the offset
*/
InitialOffset parseInitalOffset(char* carr){
	string a(carr);
	int firstBrace = -1;
	int lastBrace = -1;
	char* multC = nullptr;
	char* powC = nullptr;
	InitialOffset offset;

	for(int i = 0; i < a.length(); i++){
		if(a[i]=='['){
			firstBrace = i;
		}
		if(a[i]==']'){
			lastBrace = i;
		}
	}

	if(firstBrace != -1 && lastBrace != -1){
		multC = new char[firstBrace];
		powC = new char[lastBrace - (firstBrace + 1)];

		for(int i = 0; i < firstBrace; i++){
			multC[i] = a[i];
		}
		for(int i = 0; i < lastBrace - (firstBrace + 1); i++){
			powC[i] = a[firstBrace + 1 + i];
		}
		
		offset = {atoi(multC), atoi(powC)};
	}
	else if(firstBrace != -1 || lastBrace != -1){
		cerr << "missing a brace on an offset, unable to parse " << carr << "\n";
		offset = {0,-1};
	}
	else{
		offset = {1, atoi(carr)};
	}

	delete multC;
	delete powC;

	return offset;
}

/**
 * binary multiply by 3, add 1, divide by 2
 * 
 * @param num binnumber to perform the operation to
 * @param size size of the binnumber
 * @param lsd least significant digit of the binnumber
 * @returns new size of the binnumber
*/
/*
int mul3p1d2Bin(int num[], int size, int lsd){
	skippedSteps++;

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

}
*/

/*
// used to compare two numbers, returns 0 if num0 is larger, 1 if num1 is larger, -1 if they are equal
//With the current changes, please only call this method when the sizes are equal; thanks!
int bincompare(int num0[], int num1[], int size, int lsd){
	int i = 0;
	int kg = 1; //Keep going
	int wb = 0; //Who is bigger
	int digit = lsd;

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
	return wb;
} */

/*
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

} */

/**
 * binary multiply by 3 and add 1 and return the new binary number size
 * 
 * @param num binnumber to mult by 3 and add 1 to
 * @param size size of the binnumber
 * @returns new size of the binnumber after the operation
*/
/*
int mul3p1Bin(int num[], int size){
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

} */


// binary multiply by 3 and add 1 and return the new binary number size
/**
 * binary multiply by 3 and add 1 and return the new binary number size
*/
/*
int mul3p1BinMod(int num[], int size, int lsd)
{

        int i = lsd;
        int carry = 0; // keep track of the carrys
        int tempCurr = 0;  // store b_i temporarily for computations after b_i is overwritten
        int tempNext = 0; // store b_i+1 temporarily for same reason


//        tempCurr = num[0];  // keep this value in temp because it has to get added to the 1st entry while computing X+2X
//        num[0] = num[0] + 1; // add the 1 from the 3X+1
//        carry = num[0]/2;  // compute the carry
//        num[0] = num[0]%2; // now ensure that the entry is 0 or 1.


        tempCurr = num[lsd];  // keep this value in temp because it has to get added to the 1st entry while computing X+2X
        num[lsd] = num[lsd] + 1; // add the 1 from the 3X+1
        carry = num[lsd]>>1;  // compute the carry
        num[lsd] = num[lsd]&1; // now ensure that the entry is 0 or 1.

//      cout << "Original number has " << size << " binary digits." << endl;

//        for (i = 0; i < size; i++)
//        {
//                tempNext = num[i+1]; // store current position so it can be used for next entry in X+2X
//                num[i + 1] = tempCurr + num[i+1] + carry; // add correct entries plus carry to current position
//                carry = num[i+1]/2; // compute the carry
//                num[i+1] = num[i+1]%2; // get the entry to 0 or 1
//                tempCurr = tempNext; // don't lose the new current position!
//        }
//        num[size+1] = carry;


	for (i=lsd; i < lsd + size; i++)
        {
                tempNext = num[i+1]; // store current position so it can be used for next entry in X+2X
                num[i + 1] = tempCurr + num[i+1] + carry; // add correct entries plus carry to current position
                carry = num[i+1]>>1; // compute the carry
                num[i+1] = num[i+1]&1; // get the entry to 0 or 1
                tempCurr = tempNext; // don't lose the new current position!
        }
        num[lsd + size + 1] = carry;

        // now let's figure out how large our number is and return that value! We know that 3X+1 will be at least 1 digit larger than X
        if (carry == 0)
        {
//              cout << "3x+1 number has " << size + 1 << " binary digits." << endl;
                return (size + 1);
        }
        else
        {
//              cout << "3x+1 number has " << size + 2 << " binary digits." << endl;
//                num[size+1] = carry;
                return (size + 2);
        }


} */

/*
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
} */



// this is a "base 10 divide by 2" function which keeps track of the remainder and length of new number after division by 2
// rmsize -- zeroth entry is the remainder, first entry is size of remaining number after >>1
// note that rmsize is an array with two entries and is updated ALONG with thenumber
/*void div2(int num[], int rmsize[])
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

}*/



