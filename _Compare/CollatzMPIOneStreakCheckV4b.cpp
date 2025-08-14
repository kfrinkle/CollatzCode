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

// 2023.09.26 -- Updated to only check streak starting at 2^k+1, code similar to V3.cpp in CollatzNew now
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

using namespace std;

void div2(int num[], int size[]); // divide by 2 function
void div2Bin(int num[], int size); // divide by 2 function in binary
int mul3p1Bin(int num[], int size); // multiply by 3 and add 1 in binary  (returns new size of binary number)
int digitsum(int num[], int mod[], int size); // function to compute binary mod 9 calculation
int bincompare(int num1[], int num2[], int size); // used to compare two numbers
void printnum(int num[], int size); // print out number in correct order
void add1Bin(int num[]); // just add 1 to a binary number

void shift1Bin(int num[], int pos); // just add 1 to a binary number at position pos
int Collatz(int num[], int sizeNum); // generate Collatz sequence for random number of length size


int main(int argc, char *argv[])
{
	int i = 0; // loop variable only
	int j = 0; // loop variable only

	// do not change the value of size anywhere, we need that...
	int sizeNum = 0; //length of number, will be 2^sizeNum + 1 for starters
//	int runner = 0; // amount of sequential numbers to test
	int powa = 0; // power of 2 to add to number to break up streak
	int powa2 = 1; // power of 2 value in base 10
	int extra = 50; // how much padding to add to left end of number -- CHANGE IN COLLATZ FUNCTION TO MATCH
        int namelength = 0; // used for printing out the name of the compute node
        int rank, size; // for COMM_WORLD info
	int steps; // keep track of the number of steps in Collatz Sequence
	int longestrun = 0; // for keeping track of longest sequence of constant steps
	long long int currrun = 0; // current run of sequence variable
	int firststep = 0; // first entry for the longest run
	int laststep = 0; // last entry for the longest run
	int firststeptemp = 0; // temp variable for current run
	int laststeptemp = 0; // temp variable for current run
	int currval = 0; // number of steps 
	int streaker = 0; // check to see if you actually change values during the test
	int runnercheck = 0; // check the iterations, output every runnercheck amount
	int ColSteps = 0; // length of collatz sequence for starting value 2^k+1
	int streakChunk = 0; // how many steps in each chunk of a streak there is

        sizeNum = atoi(argv[1]); // input of largest possible number for guessing game, done command line
	powa = atoi(argv[2]); // power of 2 to add to base number for each proccess for the streak

	for (i = 1; i <= powa; i++)
	{
		powa2 = powa2*2;
	}

	runnercheck = powa2/8;

	cout << "powa = " << powa << ", and powa2 = " << powa2 << ", and runnercheck = " << runnercheck << endl;


//	clock_t start; // used for timing functions
//	clock_t end; // used for timing functions
//	double cpu_time_used; // used for computing length of function run

//      MPI:Status variable status is a class, and can be used for debugging purposes if need be
        MPI::Status status;

        MPI::Init(argc, argv);

        char name[MPI_MAX_PROCESSOR_NAME];  // create character array for the names of the compute nodes

        size = MPI::COMM_WORLD.Get_size(); // get total size of the world
        rank = MPI::COMM_WORLD.Get_rank(); // each process gets its own rank

        // we can use this for debugging, each process can now tell us what compute node they are on
        MPI::Get_processor_name(name, namelength);

//	cout << "Rank " << rank << ": has a size value of " << sizeNum << endl;

	sizeNum = sizeNum + 1;

	int *binnumber = new int[sizeNum + extra] (); // will hold the binary number
	int *binnumberHold = new int[sizeNum + extra] (); // will hold the binary number, this wont change in Collatz function

	int *firstnumber = new int[sizeNum + extra] (); // will hold the first binary number run
	int *lastnumber = new int[sizeNum + extra] (); // will hold the last binary number run through the sequence

        int *stepArrGather = new int[size] (); // will hold the number of steps for all processes via Gatherv
        long long int ColSumReducei = 0; // actual numbers in sequence that were checked (might be less than estimate at runtime

        cout << "Rank " << rank << ": Large array will have " << (long long int)powa2 * (long long int)size << " entries." << endl;
//      binnumber[sizeNum-1] = 1;
//      Set the leading value in the array to 1 so we have 2^k
        binnumberHold[sizeNum-1] = 1;

//      binnumber[0] = 1;
//      Set the 0th place to 1 so we all have 2^k+1 now.
        binnumberHold[0] = 1;

//      this is for incremental testing. I you need to break up the problem into smaller pieces starting at a power of 2, do it here
//      This would give you starting values of 2^k+2^r+1 where r is any of the 21....30 below.  Uncomment more for even more options!
//      binnumberHold[21] = 1;
//      binnumberHold[22] = 1;
//      binnumberHold[23] = 1;
//      binnumberHold[24] = 1;
//      binnumberHold[25] = 1;
//      binnumberHold[26] = 1;
//      binnumberHold[27] = 1;
//      binnumberHold[28] = 1;
//      binnumberHold[29] = 1;
//      binnumberHold[30] = 1;


//      First ALL processes compute the Collatz sequence for 2^k+1 and store the number of steps in that sequence
//      set up the number that can be modified by copying from binnumberHold first
        for (j = 0; j < sizeNum + extra; j++)
        {
                binnumber[j] = binnumberHold[j];
        }
        // compute the number of steps
        ColSteps = Collatz(binnumber, sizeNum);

        // reset the number since binnumber was modified in the ColSteps function
        for (j = 0; j < sizeNum + extra; j++)
        {
                binnumber[j] = binnumberHold[j];
        }


	// ok, so let's shift ranks' binary number the appropriate amount by successively adding 2^powa 
	for (i = 0; i < rank; i++)
	{
		shift1Bin(binnumberHold, powa);
	//	printnum(binnumber, sizeNum + extra);
	}

	for (j = 0; j < sizeNum + extra; j++)
	{
		binnumber[j] = binnumberHold[j];
		firstnumber[j] = binnumberHold[j];
	}

	MPI::COMM_WORLD.Barrier();

	for (i = 0; i < powa2; i++)
	{

		steps = Collatz(binnumber, sizeNum);


                if (steps == ColSteps)
                {
                        streakChunk++;
                }
                // else we print out the break info! Keep checking the .out file for this statement!
                else
                {
                        cout << "Rank " << rank << ": Broke streak at " << streakChunk << " and steps is " << steps << endl;
                        break;
                }


		if (i % runnercheck == 0)
		{
			cout << "Rank " << rank << ": We are on iteration " << i << "/" << powa2 << ", and this collatz sequence had " << steps << " steps." << endl;
		}
		// don't add 1 to last number tested so we have it!
		if (i != powa2-1)
		{
			add1Bin(binnumberHold);
			for (j = 0; j < sizeNum + extra; j++)
			{
				binnumber[j] = binnumberHold[j];
			}


		}
		// if it is the last number store that!
		else
		{
			for (j = 0; j < sizeNum + extra; j++)
			{
				lastnumber[j] = binnumberHold[j];
			}
		}

	}

        // ok, so let's let everyone catch up here!  If a process(es) got done early, it will wait here.
        MPI::COMM_WORLD.Barrier();


//      now we have to comptue the length of the streak!  So process 0 gathers everyone's streak values.
        MPI::COMM_WORLD.Gather(&streakChunk, 1, MPI::INT, stepArrGather, 1, MPI::INT, 0);

        MPI::COMM_WORLD.Reduce(&i, &ColSumReducei, 1, MPI::INT, MPI::SUM, 0);


        if (rank == 0)
        {
                // let's print out everyone's streak value just to make sure things look good
                cout << "Rank 0: Streak Chunk array is: " ;
                for (i = 0; i < size; i++)
                {
                        cout << " " << stepArrGather[i];
                }
                cout << endl;
                // now keep adding, starting at rank 0, everyone's value for streakChunk
                // if someone's streakChunk does not equal powa2, they stopped before they finished their chunk
                // hence a break!  So stop adding at this point to total streak value
                for (i = 0; i < size; i++)
                {
                        currrun = currrun + stepArrGather[i];
                        if (stepArrGather[i] < powa2)
                        {
                                break;
                        }
                }

                // check that we added 1 the correct amount of times by looking at the VERY LAST number checked
                cout << "Rank 0: First number tested was : " << endl;
                printnum(firstnumber, sizeNum + extra);
                MPI::COMM_WORLD.Recv(lastnumber, sizeNum + extra, MPI::INT, size-1, 0, status);
                cout << "Rank 0: Last number tested was : " << endl;
                printnum(lastnumber, sizeNum + extra);


                // print out all the userful info here!  Hopefully we do not have surrun == size*powa2
                cout << "Rank " << rank << ": Numbers start at (2^" << sizeNum - 1 << ") + 1 and has a run which is " << currrun << " long. " << endl;
		cout << "Rank " << rank << ": Range of numbers was " <<  (long long) size*powa2 << " numbers" << endl;
                cout << "Rank " << rank << ": Total number of values checked is " << ColSumReducei << endl;
                if (currrun == (long long int)size * (long long int)powa2)
                {
                        cout << "Rank " << rank << ": We need to test further!!!!!!" << endl;
                }

	}


	if (rank == size-1)
	{
		MPI::COMM_WORLD.Send(lastnumber, sizeNum + extra, MPI::INT, 0, 0);
//		cout << "Rank " << size-1 << ": Last number tested was : " << endl;
//		printnum(lastnumber, sizeNum + extra);
	}

        delete[] stepArrGather;
        delete[] binnumber;
        delete[] binnumberHold;
        delete[] firstnumber;
        delete[] lastnumber;

        MPI::Finalize();

	return 0;
}


// generate Collatz sequence for random number of length size
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
                // if its even, do x/2, note binsize value is automatically adjusted
                else
                {
                        div2Bin(binnumber, binsize);
                        binsize--;
                }

                steps ++;
        }

        return steps;

}

// used to compare two numbers, returns 0 if num0 is larger, 1 if num1 is larger, -1 if they are equal
int bincompare(int num0[], int num1[], int size)
{
        int i = size - 1;
        int kg = 1; // keep going!
        int wb = 0; // who is bigger
        while (kg && i > -1)
        {
                if (num0[i] > num1[i])
                {
                        wb = 0;
                        kg = 0;
                }
                else if (num0[i] < num1[i])
                {
                        wb = 1;
                        kg = 0;
                }

                i--;

        }
        // if kg = 1 still here, then the two numbers are equal!
        if (kg == 1)
        {
                wb = -1;
        }
//      cout << endl << "compared  " << size-i << " entries out of " << size << " entries, returning value " << wb << endl;
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



// binary multiply by 3 and add 1 and return the new bunary number size
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
	carry = num[0]/2;  // compute the carry
	num[0] = num[0]%2; // now ensure that the entry is 0 or 1.

//	cout << "Original number has " << size << " binary digits." << endl;

	for (i = 0; i < size; i++)
	{
		tempNext = num[i+1]; // store current position so it can be used for next entry in X+2X
		num[i+1] = tempCurr + num[i+1] + carry; // add correct entries plus carry to current position
		carry = num[i+1]/2; // compute the carry
		num[i+1] = num[i+1]%2; // get the entry to 0 or 1
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

void add1Bin(int num[])
{
	int i = 0;
	int carry = 1;
	int temp = 0;
	while (carry > 0)
	{
		temp = num[i];
		num[i] = (temp + carry)%2;
		carry = (temp + carry)/2;
		i ++;
	}


}


void shift1Bin(int num[], int pow) // just add 1 to a binary number at position pos
{
	int i = pow;
	int carry = 0;
	int temp = 0;
	// add 1  to the pow + 1 position
	temp = num[i]+1;
	num[i] = (temp + carry)%2;
	carry = (temp + carry)/2;
	i ++;
	while (carry > 0)
	{
		temp = num[i];
		num[i] = (temp + carry)%2;
		carry = (temp + carry)/2;
		i ++;
	}

}

// this is a "base 10 divide by 2" function which keeps track of the remainder and length of new number after division by 2
// rmsize -- zeroth entry is the remainder, first entry is size of remaining number after /2
// note that rmsize is an array with two entries and is updated ALONG with thenumber
void div2(int num[], int rmsize[])
{
	int size = rmsize[1];
	int dvr = 0;
	int i = 0;
	for (i = size-1; i > 0; i--)
	{
		dvr = num[i]%2;
		num[i] = num[i]/2;
		num[i-1] = num[i-1]+10*dvr;
	}
	dvr = num[0]%2;
	num[0] = num[0]/2;
	while (num[size-1] == 0&&size>0)
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
