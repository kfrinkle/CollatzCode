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

// SR 2025.4.9 -- converted the binnumber to num64, changing the base from binary to b2^64.

// TO RUN THIS CODE, here is an example.... there are three command line arguments
// sbatch jobfile.mpi 3935 10 4096 t31 t2 o2[3] r3[31] skip ptable
// the above command submits the job where the run command in the job file would look like:
// time prun ./executable $1 $2 $3 ....... $n
// the example says to test 2^3935 where each process checks chunks of numbers 2^10 long
// the front of the array that holds the num64.  We always set this close to the height of 2^k, in general.
// Then you would build the table taking 4096 samples
// The remaining commands are offsets, you can have as many as you want, offsets of the same type add togeter
// offsets of the type tx[y] define the sample range of the table build
// offsets of the type ox[y] define the initial offset of the test range
// offsets of the type rx[y] define the total range to be tested and will activate non-stop mode
// an arguement of "skip" will enable skip-evens mode
// an arguement of "ptable" will enable shared-table mode
// in the example, we have a table range of 2^31 + 2^2, an offset of 2^3, and a total test range of 3*2^31 skipping evens and having a shared table build between nodes
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
#include <string>
#include <algorithm>
#include <stack>
#include <random>

using namespace std;

#define ULDIV2 (LONG_MAX + 1UL) // 2^63, used for divide by 2 code
#define ULDIV3 (ULONG_MAX / 3UL) // to see if we overflow with a carry of 1``
#define ULDIV3M2 (ULDIV3 * 2UL) // to see if we overflow with a carry of 2

/**
 * Used to record information about breaks.
 * @param rank int
 * @param chunk int
 * @param chunkSize int
*/
struct BreakInfo{
	int rank;
	long long int chunk;
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
	long long int spacing;
};

/**
 *Used to store information about offset arguments
 * @param multiplier int
 * @param power int
*/
struct Offset{
	int multiplier;
	int power;
};

void printnum(int num[], int size); // print out number in correct order
void printTableInfo(TableBuildInfo tbInfo); //special print for table build abortion
int Collatz(unsigned long int num64[], int sizeNum); // generate Collatz sequence for num64 of length size
void CollatzSteps(unsigned long int num64[], int sizeNum, unsigned long int **ColSeq, int ColSeqSizes[]); // generate Collatz sequence, save the values along the way in 2D array
int CollatzCompare(unsigned long int num64[], int sizeNum, unsigned long int **ColSeq, int steps, int CoalData[], int ColSeqSizes[]); // generate Collatz sequence, compare the values along the way
TableBuildInfo updateTable(unsigned long int** ColSeq, int* ColSeqSizes, int ColSteps, int numsize, int startPower, vector<Offset> initialOffsets, vector<Offset> tableThresholdOffsets, int threshMultiplier, int amountOfSamples, MPI_Comm comm); //self explanatory
Offset parseOffset(string carr);
bool compare64(unsigned long int num0[], unsigned long int num1[], int size); //checks if two num64s are equal
int mul64b3(unsigned long int num64[], int size); // multiply by 3 in base 2^64, returning new size of number
int addUL64(unsigned long int num64[], unsigned long int val, int size); // add single ULL in base 2^64 to array, returning new size of number
int addPow2UL64(unsigned long int num64[], int valExp, int size); // adds 2^valExp to 0th entry of num
int add64b1(unsigned long int num64[], int size); // add 1 in base 2^64, returning new size of number
int div64b2(unsigned long int num64[], int size); // divide by 2 in base 2^64, returning new size of number
void print64(unsigned long int num64[], int size); // print out base 2^64 number entry by entry

/*DEPRECIATED*/
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

//PROGRAM MODES
bool SKIPEVENS = false;
bool NONSTOP = false;
bool PARALLEL_TABLES = false;

/**
 * MAIN
*/
int main(int argc, char *argv[]){	

//ARGUEMENTS
	int expon = atoi(argv[1]); //starting exponent of 2
	int powa = atoi(argv[2]); // power of 2 to add to base number for each proccess for the streak (shifted by process)
	//int extra = atoi(argv[3]); // how much padding to add to left end of number -- CHANGE IN COLLATZ FUNCTION TO MATCH
	int tableSampleAmount = atoi(argv[3]); //number of integers to build the table with
	vector<Offset> initialOffsets; //offset the start of the test range
	vector<Offset> tableThresholdOffsets; //define the range for sampling for the table build
	vector<Offset> testRangeOffsets;

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

	//Base Conversion to 2^64
	int sizeNum = expon/64; //number of entries needed to hold ^expon in base 2^64
	const int extra = 6; //extra padding at the front, first two will be used for flagsin num64
	const int entrybitshift = (expon & 63); //remainder to put in entries position of 2^64
	const int num64Size = sizeNum + extra + 1; //the total size of the num64 array

	//We need two copies of the num64, remember that if you pass an array into a function and change it there, it changes everywhere
	unsigned long int num64[num64Size] = {0UL}; // will hold a copy the num64 for destructive purposes
	unsigned long int num64hold[num64Size] = {0UL}; // will hold the num64 as it goes

	// NUM64 HEADER INFO
		// num64Size -1: is the kill flag, if set to 1, it will tell the work process to stop.
		// num64Size -2: is the rebuild table flag, it holds a number to add to the thresholdsReached variable, if it is greater than 0 it will trigger a rebuild.
		// num64Size -3: is the size of the num64 (not the size of the array), used to set sizeNum on the work nodes.
	
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
		long long int chunkCount = 0; //keeps track of current chunk, starts at 0
		long long int chunks[size]; //stores chunk assignemnts, index = rank of work node assigned chunk
		int chunkSizes[size]; //chunk sizes returned by work nodes
		int lastNode; //keeps track of the last node to process
		long long int chunkCounts[size]; //gather for local chunk counts
		vector<BreakInfo> bInfos; //stores info of the breaks in BreakInfo structs
		TableBuildInfo tbInfos = {0}; //infos got from table builds

		//printing
		long minTimes[size]; //gather for local min times
		long maxTimes[size]; //gather for local max times
		long duration = 0; // keep track of how long it takes to do a set of chunks
        long oldduration = 0; // ditto

	//WORK NODE VARIABLES
		long long int iter = 0; //keeps track of chunks processed per node
		int steps = 0; // keep track of the number of steps in Collatz Sequence
		int extra2D = 2; // just enough  padding on the end of the 2D array to hold extra 3x+1 digits for start of code

		long minTime = LONG_MAX; //keeps track of minimum times locally
		long maxTime = 0;	//keeps track of maximum times locally

		int *ColData = new int[3] (); // holds number of steps to coalesce, [1] and [2] hold min and max coalescence values
		int ColSteps; //compute the number of steps
		
		unsigned long int **ColSeq; //2D sequential int array to hold Collatz Sequence
		int *ColSeqSizes; //keeps track of the sizes of the numbers stores in ColSeq
		
	
//INITIALIZAITION

	//initialize conditional arguements
	for(int i = 4; i < argc; i++){
		string tempS(argv[i]);
		//table build range offsets
		if (argv[i][0] == 't'){
			string tempC = "";
			for(int j = 1; j < tempS.length(); j++){
				tempC.insert(tempC.end(), tempS.at(j));
			}
			tableThresholdOffsets.push_back(parseOffset(tempC));
		}
		//initial offsets
		else if (argv[i][0] == 'o'){
			string tempC = "";
			for(int j = 1; j < tempS.length(); j++){
				tempC.insert(tempC.end(), tempS.at(j));
			}
			initialOffsets.push_back(parseOffset(tempC));
		}
		//range offsets
		else if (argv[i][0] == 'r'){
			string tempC = "";
			for(int j = 1; j < tempS.length(); j++){
				tempC.insert(tempC.end(), tempS.at(j));
			}
			testRangeOffsets.push_back(parseOffset(tempC));

			NONSTOP = true;
		}
		//skip even toggle
		else if(tempS == "skip"){
			SKIPEVENS = true;
		}
		//paralell tables
		else if(tempS == "ptable"){
			PARALLEL_TABLES = true;
		}
	}

	//init powa2
	for (int i = 0; i < powa; i++){
		powa2 = powa2*2;
	}

	//init num64s
	num64hold[sizeNum] = ((1UL << entrybitshift) | num64hold[sizeNum]);//Set the leading value in the array to 1 so we have 2^k
	num64hold[0]++;//Set the 0th place to 1 so we all have 2^k+1 now.
	for(int i = 0; i < initialOffsets.size(); i++){
		for(int j = 0; j < initialOffsets[i].multiplier; j++){
			sizeNum = addPow2UL64(num64hold, initialOffsets[i].power, sizeNum);
		}
	}

	for (int j = 0; j < num64Size; j++){
		num64[j] = num64hold[j];
	}
	
	//init ColSteps
	ColSteps = Collatz(num64, sizeNum);
	ColData[1] = ColSteps; //used by CollatzCompare()	

		//WORKER VARIABLE INIT

		//now initialize the 2D array to fill with the numbers in the Collatz sequence now that we know the number of steps
		//pay attention to how this is allocated.  This will be sequential memory for a 2D array!!!!
		ColSeq = (unsigned long int**)malloc(ColSteps * sizeof(unsigned long int*)); 					//2D sequential int array to hold Collatz Sequence
		ColSeq[0] = (unsigned long int*)malloc(ColSteps * (num64Size) * sizeof(unsigned long int));
		for (int j = 1; j < ColSteps; j++){
			ColSeq[j] = ColSeq[j-1] +  (num64Size);
		}
		//let's make sure these are all set to zero first as malloc does not guarantee this (try calloc?)
		for (int i = 0; i < ColSteps; i++){
			for (int j = 0; j < num64Size; j++){
				ColSeq[i][j] = 0UL;
			}
		}
		ColSeqSizes = new int[ColSteps](); //init ColSeqSizes
		
		//reset the number since num64 was modified in the ColSteps function
		for (int j = 0; j < num64Size; j++){
			num64[j] = num64hold[j];
		}

		//Here I initialize the array storing the size of each number in the collatz sequence.
		//initialize ColSeq on Worknodes
		if (rank > 0){

			//initialize table with 2^expon + 1
            cerr << "[" << processorName << "] Rank " << rank << ": 2D Array Build Initiated." << "\n";
             CollatzSteps(num64, sizeNum, ColSeq, ColSeqSizes);
            cerr << "[" << processorName << "] Rank " << rank << ": 2D Array Build Finalized." << "\n";

			//build table with sampled numbers from 2^expon + 1 to 2^expon + 2^thresholdPower
            cerr << "[" << processorName << "] Rank " << rank << ": Table Build Initiated." << "\n";
			tbInfos = updateTable(ColSeq, ColSeqSizes, ColSteps, num64Size, expon, initialOffsets, tableThresholdOffsets, 0, tableSampleAmount, workCommunicator);;
            cerr << "[" << processorName << "] Rank " << rank << ": Table Build Finalized." << "\n";


		}

//END INITIALIZATION

    cout << "Node[" << processorName << "]:  Rank " << rank << " finished initialization " << endl;
	
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
		long long threshold = 1LL; //number to check against to send rebuild table flag
		int rebuildFlags[size] = {0}; //keeps track of flags sent
		long long maxChunk = 0LL;

		//initialize threshold
		for(int i = 0; i < tableThresholdOffsets.size(); i++){
			for(int j = 0; j < tableThresholdOffsets[i].multiplier; j++){
				long long tempThresh = 1LL;
				for(int k = 0; k < tableThresholdOffsets[i].power - powa; k++){
					tempThresh *= 2LL;
				}
				threshold += tempThresh;
			}
		}

		//initialize test range
		for(int i = 0; i < testRangeOffsets.size(); i++){
			for(int j = 0; j < testRangeOffsets[i].multiplier; j++){
				long long tempRange = 1LL;
				for(int k = 0; k < testRangeOffsets[i].power - powa; k++){
					tempRange *= 2LL;
				}
				maxChunk += tempRange;
			}
		}

		//initial assignments
		for (int i = 1; i < size; i++)
		{
			//record chunk assignment
			chunks[i] = chunkCount++;
			//printf("Rank %i receives chunk %i\n", i, chunkCount - 1);
			//send chunk assignment
			num64hold[num64Size - 3] = (unsigned long int)sizeNum;
			MPI_Send(num64hold, num64Size, MPI_UNSIGNED_LONG, i, 0, MPI_COMM_WORLD);

			//shiftBin afterwards for next chunk
			sizeNum = addPow2UL64(num64hold, powa, sizeNum);
		}

		//work loop
		while(kg){
			int chunkSize; //collection variable received
			
			MPI_Recv(&chunkSize, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
			
			int sender = status.MPI_SOURCE; //rank received from

			chunkSizes[sender] = chunkSize; //store the received value in the appropiate position in chunkSizes

			//condition to find a break
			if (chunkSize < powa2){
				bInfos.push_back({sender, chunks[sender], chunkSize});
				printf("Rank %i breaks with a chunk size of %i in chunk %i\n", sender, chunkSizes[sender], chunks[sender]);
				if(!NONSTOP){
					num64hold[num64Size-1] = 1UL;
					kg = false;
				}
			}

			if(NONSTOP && chunks[sender] >= maxChunk){
				num64hold[num64Size-1] = 1UL;
				kg = false;
			}

			//keep going if a break isnt found, assigning the sender more work
			else{


				chunks[sender] = chunkCount++;
				//printf("Rank %i receives chunk %i\n", sender, chunkCount - 1);

								//if threshold is met, ready flags for rebuild
				if((chunkCount)%threshold == 0LL){
					printf("Flag Triggered!\n");
					for(int i = 0; i < size; i++){
						rebuildFlags[i]++;
					}
				}
			}

			//if a flag is waiting, send it
			if(rebuildFlags[sender] > 0){
				//printf("Sent flag to rank %i.\n", sender);
				num64hold[num64Size - 2] = (unsigned long int) rebuildFlags[sender];
				rebuildFlags[sender] = 0;
			}

			//send either more work or kill command to the sender
			num64hold[num64Size - 3] = (unsigned long int)sizeNum;
			MPI_Send(num64hold, sizeNum+extra + 1, MPI_UNSIGNED_LONG, sender, 0, MPI_COMM_WORLD);

			//reset flag
			num64hold[num64Size - 2] = 0UL;

			//if more work was sent, shiftBin again
			if (kg){
				sizeNum = addPow2UL64(num64hold, powa, sizeNum);


                if (chunkCount % 10000LL == 0LL){
					//printf("chunk %i has been sent\n", chunkCount);

                    auto stopTimerPZero = chrono::high_resolution_clock::now();
                    duration = chrono::duration_cast<chrono::milliseconds>(stopTimerPZero - startTimerPZero).count();

                    cout << "Node " << processorName <<  " -- Rank " << rank << ": Chunk " << chunkCount << " has been sent, previous chunk processing time: " << duration - oldduration << endl;
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
				bInfos.push_back({sender, chunks[sender], chunkSize});
				printf("Rank %i breaks with a chunk size of %i in chunk %i\n", sender, chunkSizes[sender], chunks[sender]);
			}

			if(rebuildFlags[sender] > 0){
					//printf("Sent flag to rank %i.\n", sender);
					num64hold[num64Size - 2] = (unsigned long int) rebuildFlags[sender];
					rebuildFlags[sender] = 0;
			}

			//send kill command
			num64hold[num64Size - 3] = (unsigned long int)sizeNum;
			MPI_Send(num64hold, num64Size, MPI_UNSIGNED_LONG, sender, 0, MPI_COMM_WORLD);
			num64hold[num64Size - 2] = 0UL;
		}

		//sort the breaks by chunk asc
		sort(bInfos.begin(), bInfos.end(),
			[](const BreakInfo& a, const BreakInfo& b){
				return a.chunk < b.chunk;
			} 
		);	
	}

	//WORK NODES RANK > 0
	//receive either more work or kill command
	//rebuild lookup table if it receives the rebuild flag
	//loop through the assigned chunk, checking for breaks, stop loop if break is found
	//return how many numbers have been checked in the chunk
	if (rank > 0){

		iter = 0; //keeps track of the chunks this node has processed
		int thresholdsReached = 0;

		//work loop, will terminate when stop signal is received : num64Hold[num64Size-1] = 1;
		do{
			MPI_Recv(num64hold, num64Size, MPI_UNSIGNED_LONG, 0, 0, MPI_COMM_WORLD, &status);
			sizeNum = (int) num64hold[num64Size - 3];

			auto startTimer = chrono::high_resolution_clock::now(); //start timer
			int streakChunk = 0; //keep track of steps through the assigned range

			//rebuild ColSeq lookup table if the flag is received from rank 0
			if (num64hold[num64Size - 2] > 0UL){

				cerr << "[" << processorName << "] Rank " << rank << ": Table Build Initiated." << "\n";
				thresholdsReached += (int) num64hold[sizeNum+extra - 1];
				tbInfos = updateTable(ColSeq, ColSeqSizes, ColSteps, num64Size, expon, initialOffsets, tableThresholdOffsets, thresholdsReached, tableSampleAmount, workCommunicator);
				cerr << "[" << processorName << "] Rank " << rank << ": Table Build Finalized." << "\n";

				if(rank == 1){
					printTableInfo(tbInfos);
				}
				num64hold[num64Size - 2] = 0UL; //reset flag
			}

			//if work, do CollatzWork
			if (num64hold[num64Size - 1] != 1UL){

				//printf("This is rank %i on iteration %i\n", rank, iter);

				for (int j = 0; j < num64Size; j++){
					num64[j] = num64hold[j];
					//firstnumber[j] = binnumberHold[j];
				}

				//Now we have each process go through their chunk of the range to check!  Not for loop goes from 0 to powa2-1
				for (int i = 0; i < powa2; i++)
				{

					// store the value of steps in the CollatzCompare function
					// remember CollatzCompare compares the numbers at each step in the sequence to that of 2^k+1
					// ColSeq is the array which holds the numbers in the sequence for 2^k+1
					// ColData is the array of useful information (steps, max and min steps)

					steps = CollatzCompare(num64, sizeNum, ColSeq, ColSteps, ColData, ColSeqSizes);

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
						cout << "Node " << processorName <<  " -- Rank " << rank << ": Broke streak at " << streakChunk << endl;
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
						sizeNum = add64b1(num64hold, sizeNum);

						if(SKIPEVENS && (num64hold[0] & 1UL) == 0 && i != powa2 - 2){
							sizeNum = add64b1(num64hold, sizeNum);
							streakChunk++;
							i++;
						}
						
						for (int j = 0; j < sizeNum + extra; j++)
						{
							num64[j] = num64hold[j];
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

		} while (num64hold[num64Size - 1] != 1UL);
	}

//PRINT RESULTS

	// ok, so let's let everyone catch up here!  If a process(es) got done early, it will wait here.
    cout << "Node " << processorName <<  " -- Rank " << rank << ": Waiting at the barrier." ;
	if (rank > 0){
		cout << "Min Time: " << minTime << " ms. Max Time: " << maxTime << " ms." << endl;
	}

	MPI_Barrier(MPI_COMM_WORLD);

	MPI_Gather(&iter, 1, MPI_LONG_LONG, chunkCounts, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD); //gather chunks done by each worker
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
		long long int biggestStreak = (bInfos[0].chunk) * (long long int) powa2 + (long long int) bInfos[0].chunkSize;

		for (int i = 1; i < bInfos.size(); i++){
			long long int currentStreak = ((bInfos[i].chunk * (long long int)powa2) + (long long int) bInfos[i].chunkSize) - (( bInfos[i - 1].chunk * (long long int) powa2) + (long long int) bInfos[i -1].chunkSize);

			if (currentStreak > biggestStreak) biggestStreak = currentStreak;
		}

		//find average min and max times
		long avgMin = 0;
		long avgMax = 0;
		long absMin = LONG_MAX;
		long absMax = 0;

		long long int minChunk = LONG_MAX;
		long long int maxChunk = 0;
		long long int avgChunk = 0;

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
		avgChunk /= (long long int)(size - 1);

/*
		printf("Chunks Processed Per Node:\t\t");
		for(int i = 1; i < size; i++){
			printf("%i\t", chunkCounts[i]);
		}
		printf("\n");
*/
                printf("Chunks Processed Per Node:\t\t");
                cout << endl;
                for(int i = 1; i < size; i++){
                        cout << "Rank " << i << ": " << chunkCounts[i] << endl;
//                      printf("%i\t", chunkCounts[i]);
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



		cout << endl << endl << endl;

		//print out found breaks
		for (int i = 0; i < bInfos.size(); i++){
			cout << "Node " << processorName << " -- Rank " << rank << ": A break was found by rank " << bInfos[i].rank << " at " << (bInfos[i].chunk) * (long long int) powa2 + (long long int) bInfos[i].chunkSize << " in chunk " << bInfos[i].chunk << endl;
		}


		// print out all the userful info here!  Hopefully we do not have currun == size*powa2
		cout << "Node " << processorName <<  " -- Rank " << rank << ":  Numbers start at (2^" << expon;
		for(int i = 0; i < initialOffsets.size(); i++){
			printf(" + ");
			if(initialOffsets[i].multiplier > 1) printf("%i(", initialOffsets[i].multiplier);
			printf("2^%i", initialOffsets[i].power);
			if(initialOffsets[i].multiplier > 1) printf(")");
		}
		cout << ") + 1 with the biggest streak being " << biggestStreak << " long. Range spanned "  << (long long int) (chunkCount)*powa2 << " numbers" << endl;

		// now the fun stats!
		cout << "Node " << processorName <<  " -- Rank " << rank << ": Shortest coalescence value is :  " << ColMinReduce << endl;
		cout << "Node " << processorName <<  " -- Rank " << rank << ": Longest coalescence value is  :  " << ColMaxReduce << endl;
		cout << "Node " << processorName <<  " -- Rank " << rank << ": Sum of coalescence values is  :  " << ColSumReduce << endl;
		cout << "Node " << processorName <<  " -- Rank " << rank << ": Average coalescence value is  :  " << ColSumReduce/ColSumReducei << endl;
		cout << "Node " << processorName <<  " -- Rank " << rank << ": Total number of values checked is " << ColSumReducei << endl;
		cout << "Node " << processorName <<  " -- Rank " << rank << ": Total times the sizes were unequal (didn't compare digit-by-digit) is :" << timesNoCompareReduce << endl;
		cout << "Node " << processorName <<  " -- Rank " << rank << ": Total times the sizes were equal (did compare digit-by-digit) is      :" << timesYesCompareReduce << endl;
		cout << "Node " << processorName <<  " -- Rank " << rank << ": Total times we skipped a step is: " << skippedStepsReduce << endl;
		cout << "Node " << processorName <<  " -- Rank " << rank << ": Total times shift was greater than 1 is: " << shiftWasGreaterThan1Reduce << endl;
		cout << "Node " << processorName <<  " -- Rank " << rank << ": Max number of sequential divide by 2's is: " << maxSkipStepsReduce << endl;
	    cout << "Node " << processorName <<  " -- Rank " << rank << ": Total Chunks Processed is: " <<  chunkCount << endl;
        cout << "Node " << processorName <<  " -- Rank " << rank << ": Minimum Chunks Processed is: " << minChunk << endl;
        cout << "Node " << processorName <<  " -- Rank " << rank << ": Maximum Chunks Processed is: " <<  maxChunk << endl;
        cout << "Node " << processorName <<  " -- Rank " << rank << ": AVG Chunks Processed is is : " << avgChunk << endl;
        cout << "Node " << processorName <<  " -- Rank " << rank << ": AVG Minimum Time (ms) is : " << avgMin << endl;
        cout << "Node " << processorName <<  " -- Rank " << rank << ": AVG Maximum Time (ms) is : " << avgMax << endl;
        cout << "Node " << processorName <<  " -- Rank " << rank << ": ABS Minimum Times (ms) is : " << absMin << endl;
        cout << "Node " << processorName <<  " -- Rank " << rank << ": ABS Maximum Time (ms) is : " << absMax << endl;
	}

	// now time to delete the arrays!
	delete[] ColData;
	delete[] ColSeqSizes;

	// now to free that pesky 2D array!
	free(ColSeq[0]);
	free(ColSeq);


    MPI::Finalize();

	return 0;
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
 * @param numsize size of the num64s
 * @param startPower power of the start of the range
 * @param initialOffsets initial offsets
 * @param numberOfOffsets number of initial offsets
 * @param thresholdPower power of the size of the sample range
 * @param threshMultiplier which range of samples you are at
 * @param amountOfSamples number of integers to sample for the table
 * @param comm the comminicator for the work group using MPI_Split, rank 0 should never enter this function
 * @returns TableBuildInfo struct, will be all 0s if the table build fails
*/
TableBuildInfo updateTable(unsigned long int** ColSeq, int* ColSeqSizes, int ColSteps, int numsize, int startPower, vector<Offset> initialOffsets, vector<Offset> tableThresholdOffsets, int threshMultiplier, int amountOfSamples, MPI_Comm comm){
//STRUCT
	struct Sample{
		unsigned long int* num64;
		int size;
		int frequency;
	};

//VARIABLES
	//group MPI info
	int rank = 0; //local rank
	int size = 0; //local groupsize
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &size);
	
	//for communicating if one of the samples is a break
	int breakFound = 0;
	int breakFlags[size];
	stack<int> breakIndices;
	int dummyData[3] = {0};

	long long int spacing = 1LL; //spacing between samples
	long long int intervalOffsets[amountOfSamples]{0LL}; //offsets

	//start long randomization
	random_device rd; //random device
	mt19937 generator(rd()); //random number generator seeded with random device
	uniform_int_distribution<long long int> distribution(0LL, std::numeric_limits<long long int>::max()); //define range of random number generator

	//main variables to store sample info
	vector<Sample> samples;
	unsigned long int tempBin[numsize]; //a num64 array to copy to when doing something destructive, like Collatz()
	unsigned long int tempTempBin[numsize]; //same deal as the previous one
	stack<int> dumpIndices;

	for(int i = 0; i < numsize; i++){
		tempBin[i] = tempTempBin[i] = 0UL;
	}

	int maxFrequency = 1; //the highest frequency of a mode found
	int modeIndex = -1; //the index of the highest frequency mode found
	int INDEX = 0; //current index in ColSeq

	//to keep track of build info
	TableBuildInfo tbInfos = {0}; //struct to return info in
	chrono::high_resolution_clock::time_point startTime = chrono::high_resolution_clock::now(); //start of the build timer
	bool firstIndexFound = false; //keeps track of whether an mode has been found in the entire build

//INITIALIZE

	//initialize spacing : (2^thresholdPower / amountOfSamples) atleast 1
	for(int i = 0; i < tableThresholdOffsets.size(); i++){
		for(int j = 0; j < tableThresholdOffsets[i].multiplier; j++){
			long long tempThresh = 1LL;
			for(int k = 0; k < tableThresholdOffsets[i].power; k++){
				tempThresh *= 2LL;
			}
			spacing += tempThresh;
		}
	}
	spacing /= (long long int)amountOfSamples;
	if (spacing == 0LL) spacing = 1LL;
	tbInfos.spacing = spacing;//record spacing

	//generate offsets
	for(int i = 0; i < amountOfSamples; i++){
		intervalOffsets[i] = distribution(generator)%spacing - (spacing/2LL);
	}

	if(PARALLEL_TABLES) MPI_Bcast(intervalOffsets, amountOfSamples, MPI_LONG_LONG, 0, comm);

	//INITIALIZE SAMPLES
	//samples are initialized as 2^startpower + spacing(i+1)
	//then they are given a random offset to expand the sample set 
	for(int i = 0; i < amountOfSamples; i++){

		samples.push_back({
			new unsigned long int[numsize]{0UL}, //num64
			startPower/64, //size
			1 //frequency
		});

		samples[i].num64[startPower/64] = ((1UL << (startPower & 63)) | samples[i].num64[startPower/64]);

		//add threshold to number, INITAL POWER MUST BE HIGHER THAN THE THRESHOLD
		for(int j = 0; j < threshMultiplier; j++){
			for(int k = 0; k < tableThresholdOffsets.size(); k++){
				for(int l = 0; l < tableThresholdOffsets[k].multiplier; l++){
					samples[i].size = addPow2UL64(samples[i].num64, tableThresholdOffsets[k].power, samples[i].size);
				}
			}
		}

		//add offsets
		for(int j = 0; j < initialOffsets.size(); j++){
			for(int k = 0; k < initialOffsets[j].multiplier; k++){
				samples[i].size = addPow2UL64(samples[i].num64, initialOffsets[j].power, samples[i].size);
			}
		}
		
		samples[i].size = addUL64(samples[i].num64, (unsigned long int)(spacing * (long long int)(i + 1)) + intervalOffsets[i], samples[i].size);
	}

//CHECK HEIGHTS
	//parallel version
	if(PARALLEL_TABLES){
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
			for(int j = 0; j < numsize; j++){
				tempBin[j] = samples[startIndices[rank] + i].num64[j];
			}
			
			int currentSteps = CollatzCompare(tempBin, samples[startIndices[rank]+i].size, ColSeq, ColSteps, dummyData, ColSeqSizes);

			if(currentSteps != ColSteps){	
				//if(rank == 0) printf("First Trigger: x: %i, colSteps: %i\n", currentSteps, ColSteps);
				breakFound++;
				breakIndices.push(startIndices[rank] + i);
			}
		}

		//gather and mark breaks for dumping
		MPI_Allgather(&breakFound, 1, MPI_INT, breakFlags, 1, MPI_INT, comm);

		for(int i = 0; i < size; i++){
			for(int j = 0; j < breakFlags[i]; j++){
				int tempIndex = 0;
				if(rank == i){
					tempIndex = breakIndices.top();
					breakIndices.pop();
				}
				MPI_Bcast(&tempIndex, 1, MPI_INT, i, comm);
				dumpIndices.push(tempIndex);
			}
		}
	}
	//unique version
	else{
		//check heights for break, different ranks check their assigned range
		for(int i = 0; i < samples.size(); i++){
				
			//array copy because Collatz() is destructive
			for(int j = 0; j < numsize; j++){
				tempBin[j] = samples[i].num64[j];
			}

			int currentSteps = CollatzCompare(tempBin, samples[i].size, ColSeq, ColSteps, dummyData, ColSeqSizes);

			if(currentSteps != ColSteps){	
				//if(rank == 0) printf("First Trigger: x: %i, colSteps: %i\n", currentSteps, ColSteps);
				dumpIndices.push(i);
			}
		}
	}

	//dump breaking samples
	while(!dumpIndices.empty()){
		delete[] samples[dumpIndices.top()].num64;
		samples.erase(samples.begin() + dumpIndices.top());
		dumpIndices.pop();
	}

// REINITIALIZE COLSEQ with (2^startPower + threshMultiplier(2^thresholdPower) + 1)
	if (threshMultiplier > 0){
		//make sure temp bin is reset
		for(int i = 0; i < numsize; i++){
			tempBin[i] = 0UL;
		}

		//build the sample
		int tempSize = startPower/64;
		tempBin[tempSize] = ((1UL << (startPower & 63)) | tempBin[startPower/64]);
		for(int i = 0; i < threshMultiplier; i++){
			for(int j = 0; j < tableThresholdOffsets.size(); j++){
				for(int k = 0; k < tableThresholdOffsets[j].multiplier; k++){
					tempSize = addPow2UL64(tempBin, tableThresholdOffsets[j].power, tempSize);
				}
			}
		}
		//add offsets
		for(int i = 0; i < initialOffsets.size(); i++){
			for(int j = 0; j < initialOffsets[i].multiplier; j++){
				tempSize = addPow2UL64(tempBin, initialOffsets[i].power, tempSize);
			}
		}
		tempSize = add64b1(tempBin, startPower/64);

		//copy the sample for Collatz() because it eats things and doesnt give them back
		for(int i = 0; i < tempSize + 1; i++){
			tempTempBin[i] = tempBin[i];
		}
		
		int currentSteps = CollatzCompare(tempTempBin, tempSize, ColSeq, ColSteps, dummyData, ColSeqSizes);
		//abort if the initializing number breaks the streak
		if( currentSteps == ColSteps){
			//printf("Second Trigger: x: %i, colSteps: %i\n", currentSteps, ColSteps);
			CollatzSteps(tempBin, tempSize, ColSeq, ColSeqSizes);
		}
	}

//START TABLE UPDATE
	while (samples.size() > 1 && INDEX < ColSteps){

		//collatz step for this INDEX
		for(int i = 0; i < samples.size(); i++){
			if(samples[i].num64[0] & 1UL){
				samples[i].size = mul64b3(samples[i].num64, samples[i].size);
				samples[i].size = add64b1(samples[i].num64, samples[i].size);
			}
			else{
				samples[i].size = div64b2(samples[i].num64, samples[i].size);
			}
		}

		//sort the samples
		sort(samples.begin(), samples.end(), 
			[](const Sample& a, const Sample& b){

				if(a.size < b.size) return true;

				if(a.size == b.size){
					for(int i = 0; i < a.size + 1; i++){
						if(a.num64[i] < b.num64[i]) return true;
						else if(a.num64[i] > b.num64[i]) return false;
					}
				}

				return false;
			}
		);

		//find the mode
		for(int i = 1; i < samples.size(); i++){
			if(samples[i-1].size == samples[i].size){
				if(compare64(samples[i-1].num64, samples[i].num64, samples[i].size)){

					samples[i].frequency = samples[i-1].frequency + 1;

					if (samples[i].frequency > maxFrequency){
						maxFrequency = samples[i].frequency;
						modeIndex = i;
					}
				}
			}
		}
		
		//assign to table
		if (maxFrequency > 1){
			//printf("MODE FOUND FOR INDEX %i\n", INDEX);
			int iter = 0;
			for(int i = 0; i < numsize; i++){
				//if(INDEX == 43) printf("%i\n", iter++);
				ColSeq[INDEX][i] = samples[modeIndex].num64[i];
			}
			ColSeqSizes[INDEX] = samples[modeIndex].size;
					
			//update info in tbInfos
			if (!firstIndexFound){
				tbInfos.startIndex = INDEX;
				firstIndexFound = true;
			}
			tbInfos.stopIndex = INDEX;
			tbInfos.totalReplaced++;

			//delete moded samples
			for(int i = 0; i < maxFrequency; i++){
				delete[] samples[modeIndex - i].num64;
				samples.erase(samples.begin() + (modeIndex - i));
			}

			//reset frequencies
			maxFrequency = 1;
			for(int i = 0; i < samples.size(); i++){
				samples[i].frequency = 1;
			}
		}

		//go to next index in ColSeq
		INDEX++;
	}

	for(int i = 0; i < samples.size(); i++){
		delete[] samples[i].num64;
	}

	//record elapsed build time in tbInfos
	tbInfos.generationTime = chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - startTime).count();

	return tbInfos;
}


/** 
 * generate Collatz sequence for binary number num64 of length size, return number of steps
 * 
 * @param num64 number to sequence !IS DESTRUCTIVE!
 * @param size size of the number
 * @returns Number of steps through the Collatz sequence
 * @warning changes the information in num64
*/
int Collatz(unsigned long int num64[], int size){
	int steps = 0; // just keep track of the number of steps!
	while (! (( size == 0) && (num64[0] == 1UL)) ){
        steps++;

        // if number odd, perform 3x+1
         if (num64[0] & 1UL){
			
            size = mul64b3(num64, size);
			
            size = add64b1(num64, size);
             
		}
            // else the number is even, perform /2
         else{
			
            size = div64b2(num64, size);

        }

    }

    return steps;
}

/**
 * generate lookup table with one number, saving all numbers in the sequence as it goes, storing it in ColSeq and ColSeqSizes
 * 
 * @param num64 number to build the ColSeq table with !IS DESTRUCTIVE!
 * @param size size of the initial number
 * @param ColSeq 2D sequential array to store the subsequent num64s
 * @param ColSeqSizes array to store sizes of the subsequent num64s
 * @warning changes the information in num64
*/
void CollatzSteps(unsigned long int num64[], int size, unsigned long int **ColSeq, int ColSeqSizes[]){
	int steps = 0; // keep track of number of steps

    while (! (( size == 0) && (num64[0] == 1UL)) ){
        // if number odd, perform 3x+1
        if (num64[0] & 1UL){
            size = mul64b3(num64, size);
            size = add64b1(num64, size);
        }
        // else the number is even, perform /2
        else{
            size = div64b2(num64, size);
		}

        // copy the number over to array after each step in the process
        // note we only go out to binsize, so the 2D array better be zeroed out first!
        for (int j = 0; j <= size; j++){
            ColSeq[steps][j] = num64[j];
        }
        //Here we add in the binsize of this number in the sequence.
            ColSeqSizes[steps] = size;
            steps ++;
    }

	//printf("CollatzSteps: %i\n", steps);

	//	return steps;
}


/**
 * generate Collatz sequence, stopping when it hits a number at same index in ColSeq array
 * 
 * @param num64 number to sequence and compare with the lookup table !IS DESTRUCTIVE!
 * @param sizeNum size of the num64
 * @param ColSeq lookup table
 * @param ColSteps size of the lookup table
 * @param ColData size 3 array where [0] should equal ColSteps
 * @param ColSeqSizes sizes of the numbers stored in the lookup table
 * @returns Number of steps through the Collatz sequence
*/
int CollatzCompare(unsigned long int num64[], int sizeNum, unsigned long int **ColSeq, int ColSteps, int ColData[], int ColSeqSizes[]){
	// ColData[0] holds number of steps to coalesce, [1] and [2] hold min and max coalescence values

    int steps = 0; // the step count variable
    bool cv = false; // compare value between current step and base step

    while (! (( sizeNum == 0) && (num64[0] == 1UL)) ){

        // if number odd, perform 3x+1
        if (num64[0] & 1UL){
            sizeNum = mul64b3(num64, sizeNum);
            sizeNum = add64b1(num64, sizeNum);
        }
        // else the number is even, perform /2
        else{
            sizeNum = div64b2(num64, sizeNum);
        }
        steps++;

        //First we compare the current size of the number to the size of the original sequence's number
        if (sizeNum == ColSeqSizes[steps - 1]){
            cv = compare64(num64, ColSeq[steps - 1], sizeNum);
            timesYesCompare++;
        }
        else{
            //here you will update your global variable since you didn't have to compare digit-by-digit
            timesNoCompare++;
        }

        // if the current number and that in ColSeq[steps] are the same, gather data and stop!
        if (cv){
            // put number of steps taken into 0th entry
            ColData[0] = steps;
            // then check we have another max/min number of steps
        	if (steps > ColData[2]){
                ColData[2] = steps;
            }
            else if (steps < ColData[1]){
                ColData[1] = steps;
            }
            // so put the number of steps into ColSteps since we know current number will have
            // same sequence length as one we are streak checking
            steps = ColSteps;
            break;
        }

        // if we have gotten all the way to number of steps for base number and  cv != 1, then we need to stop
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
 * Multiply the num64 by 3.
 * @param num64 base 2^64 number to multiply
 * @param size size of the number
 * @returns size of number
*/
int mul64b3(unsigned long int num64[], int size){

    unsigned long int currc; // for the current carry
    unsigned long int nextc; // for the next carry

	// lets perform 3x+1 now, for 2^64 base, its easier to do 3x first, then go back and add 1.
	currc = 0; // start with current carry = 0;
	nextc = 0; // start with next carry = 0;

	for(int i = 0; i <= size; i++){
		// remmeber the following constants:
		// div3 = ulmax/3, so 2*div3 = 2*ulmax/3

		// if num > 2/3*(2^64-1), then num >= 2/3*(2^64-1)+1, so
		//  3*num + c >= 2*2^64-2+3 + c = 2*2^65+1 + c, so carry = 2 always
		// note in worse case scenario, num = 2^64-1, and so 3*num + c = 3*2^64-3 + c, 
		// and since c < 3, next carry = 2 
		if (num64[i] > ULDIV3M2){

			nextc = 2;
			// num[i] = 3*num[i] + currc;
			// this is bit version of 3x + carry 
			num64[i] = (num64[i]<<1) + num64[i] + currc;

		}
		// what happens if num = 2/3*(2^64-1), 3*num + c =  2*2^64-2 + c 
		// so next carry is 2 if c=2, but 1 if c=0 or c=1.
		else if (num64[i] == ULDIV3M2)
		{
			if (currc == 2)
			{

				nextc = 2;
			}
			else
			{

				nextc = 1;
			}
			// num[i] = 3*num[i] + currc;
			// this is bit version of 3x + carry 
			num64[i] = (num64[i]<<1) + num64[i] + currc;

		}
		// now if num > 1/3*(2^64-1), then num >= 1/3*(2^64-1)+1
		// 3*num + c >= 2^64-1+3 + c = 2^64+2 + c,  and next carry is 1
		else if (num64[i] > ULDIV3){

			nextc = 1;
			// num[i] = 3*num[i] + currc;
			// this is bit version of 3x + carry 
			num64[i] = (num64[i]<<1) + num64[i] + currc;

		}
		// so what if num = 1/3*(2^64-1)?  then 3*num + c = 2^64-1 + c, so if c > 0 we have a carry!
        else if (num64[i] == ULDIV3){
            if (currc > 0){
				nextc = 1;
            }
            else{
				nextc = 0;
            }
			// num[i] = 3*num[i] + currc;
			// this is bit version of 3x + carry 
			num64[i] = (num64[i]<<1) + num64[i] + currc;

    	}


		// if num < div3, num <= 1/3*(2^64-1)-1, and so 3*num +c <= 2^64-1-3 + c = 2^64-4 + c < 2^64-1 always
		// thus no carry needed if num < div3 regardless of carry!
		else{
			nextc = 0;
			// num[i] = 3*num[i] + currc;
			// this is bit version of 3x + carry 
			num64[i] = (num64[i]<<1) + num64[i] + currc;

		}
		// now that all that is done, we copy next carry into current carry and zero out current carry
		currc = nextc;
		// theoreticall we do not have to set nextc = 0 since it always gets adjusted above....
		nextc = 0;

	}

	// now let's check to see if we need to add the carry to the next entry, in which case
	// we incremment the size of the number by 1
	if (currc > 0){

		size ++;
		num64[size] = currc;
	}
	return (size);
}

/**
 * Divide num64 by 2.
 * @param num64 number to divide
 * @param size size of number
 * @returns size of number
*/
int div64b2(unsigned long int num64[], int size){

	// we start with the largest entry first, to determine if zeros out
	unsigned long int drop = 0; // if current entry is even, this is 0, if it is odd it is 1, used to carry over to next entry

	// other odd number, will have a remainder to carry over to next lowest entry

	for (int i = size; i>=0; i--){
		// if number is odd, we have to add some 2^63 to the next entry's result due to the carry (drop)
		if (num64[i] & 1UL)
		{
			num64[i] = (num64[i]>>1) + drop*ULDIV2;
			drop = 1;
		}
		// else the number is even, /2 works out evenly for this entry, no need to add a carry (drop)
		else
		{
			num64[i] = (num64[i]>>1) + drop*ULDIV2;
			drop = 0;
		}
	}
	if (num64[size] == 0)
	{
		return (--size);
	}
	else
	{
		return (size);
	}

}

/**
 * checks if two num64s are the same or not
 * @warning !ONLY USE IF THE NUMBERS ARE THE SAME SIZE!
 * @param num0 the first number
 * @param num1 the second number
 * @param size the size of the numbers
 * @returns boolean of equality
*/
bool compare64(unsigned long int num0[], unsigned long int num1[], int size){
    int i = 0;

    while (i <= size)
    {
         if (num0[i] ^ num1[i]) //numbers are not equal
             return false;

        i++;
    }

    //all digit pairs are equal so the numbers are equal
     return true;
}

/**
 * Add a power of two to num64
 * @param num64 number to add to
 * @param valExp the power of two to add
 * @param size size of number
 * @returns size of number
*/
int addPow2UL64(unsigned long int num64[], int valExp, int size){

    int start = valExp/64; //where to add the power of 2
    int entrybitshift = (valExp & 63); //remainder to put in entries position of 2^64

	unsigned long int val = 0;
	val =  ((1UL << entrybitshift) | val);
	int carry = 0;
	// we start at entry 0 and keep going until we have no more carry
	if (num64[start] > ULONG_MAX - val){
		carry = 1;
	}
	num64[start] = num64[start] + val;
	// if there is a carry, we are adding 1 to the next entry, which means that we need
	// to ensure that it is not ULMAX

	int i = start + 1;
	if (carry == 1)
	{
		while (num64[i] == ULONG_MAX)
		{
			num64[i] = 0;
			i++;
		}
		num64[i] = num64[i] + 1;
	}
	// if we went over the current size limit, update!
	if (i > size)
	{
		return (i);
	}
	else
	{
		return (size);
	}
}

/**
 * Add a number to the num64
 * @param num64 number to add to
 * @param val number to add
 * @param size size of number
 * @returns size of number
*/
int addUL64(unsigned long int num64[], unsigned long int val, int size)
{
	int i = 1;
	int carry = 0;
	// we start at entry 0 and keep going until we have no more carry
	if (num64[0] > ULONG_MAX - val)
	{
		carry = 1;
	}
	num64[0] = num64[0] + val;
	// if there is a carry, we are adding 1 to the next entry, which means that we need
	// to ensure that it is not ULMAX
	if (carry == 1)
	{
		while (num64[i] == ULONG_MAX)
		{
			num64[i] = 0;
			i++;
		}
		num64[i] = num64[i] + 1;
	}
	// if we went over the current size limit, update!
	if (i > size)
	{
		return (i);
	}
	else
	{
		return (size);
	}
}

/**
 * increment num64 by 1
 * @param num64 number to increment
 * @param size size of number
 * @returns size of number
*/
int add64b1(unsigned long int num64[], int size){
	int i = 0;
	// the only way to overflow when adding 1 is to have num[i] = ULMAX
	// then we have to zero out that entry and add 1 to the next etc....
	while (num64[i] == ULONG_MAX)
	{
		num64[i] = 0;
		i++;

	}
	num64[i]++;
	if (i > size)
	{
		return (i);
	}
	else
	{
		return (size);
	}
}

/**
 * prints num64
 * @param num64 the number to print
 * @param size size of the number
*/
void print64(unsigned long int num[], int size)
{

	for(int i = 0; i <= size; i++)
	{
		cout <<  "b64[" << i << "]: " << num[i] << endl;
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
Offset parseOffset(string carr){
	int firstBrace = -1;
	int lastBrace = -1;
	char* multC = nullptr;
	char* powC = nullptr;
	Offset offset;

	for(int i = 0; i < carr.length(); i++){
		if(carr.at(i)=='['){
			firstBrace = i;
		}
		if(carr.at(i)==']'){
			lastBrace = i;
		}
	}

	if(firstBrace != -1 && lastBrace != -1){
		multC = new char[firstBrace];
		powC = new char[lastBrace - (firstBrace + 1)];

		for(int i = 0; i < firstBrace; i++){
			multC[i] = carr.at(i);
		}
		for(int i = 0; i < lastBrace - (firstBrace + 1); i++){
			powC[i] = carr.at(i);
		}
		
		offset = {atoi(multC), atoi(powC)};
	}
	else if(firstBrace != -1 || lastBrace != -1){
		cerr << "missing a brace on an offset, unable to parse " << carr << "\n";
		offset = {0,-1};
	}
	else{
		offset = {1, atoi(carr.c_str())};
	}

	delete[] multC;
	delete[] powC;

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



