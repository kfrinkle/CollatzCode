//MACROS///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//STD
#include <iostream>
#include <cstdlib>
#include <vector>
#include <string>
#include <random>
#include <chrono>
#include <algorithm>
#include <thread>

//CUDA
#include <cooperative_groups.h>

//COMMON
#include "Common/Device.cuh"
#include "Common/Data.cuh"
#include "Common/Sort.cuh"

//DEFINITIONS
#define ULDIV2 (LONG_MAX + 1UL) // 2^63, used for divide by 2 code
#define ULDIV3 (ULONG_MAX / 3UL) // to see if we overflow with a carry of 1
#define ULDIV3M2 (ULDIV3 * 2UL) // to see if we overflow with a carry of 2

//MODES
bool NONSTOP = false; //set if checking a specific range, disables ending the program at the first break
bool SKIPEVENS = false; //set if skipping checks for even numbers

//STRUCTURES///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * Comparitor for sorting the samples in the table build with.
 * @tparam type of data inside the sample
*/
template <typename T>
struct SampleCompare{
	/**
	 * operator for the comparison
	 * @param a left sample
	 * @param b right sample
	 * @param len length of the sample arrays
	 * @returns whether a < b
	*/
	__device__ bool operator()(const T* a, const T* b, unsigned int len) const {
		if(len == 0) return false;

		unsigned int aSize = static_cast<unsigned int>(a[len-1]);
		unsigned int bSize = static_cast<unsigned int>(b[len-1]);

		if(aSize >= len || bSize >= len) return false;

		if(aSize != bSize){
			return aSize < bSize;
		}
		else{
			for(unsigned int i = 0U; i <= aSize; i++){
				if(a[i] < b[i]) return true;
				if(a[i] > b[i]) return false;
			}
		}
		return false;
	};
};

/**
 * Used to hold information about found breaks
 * @param chunk unsigned long int
 * @param offset unsigned long int
 */
struct BreakInfo{
	unsigned long int chunk;
	unsigned long int offset;
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
	unsigned int startIndex;	//first index a mode was found
	unsigned int stopIndex;		//last index a mode was found
	unsigned int totalReplaced; //total Modes found
	long generationTime;		//time the table build took
	long long int spacing;		//spacing betweens the centers of the sample intervals
	unsigned int memoryUsed;	//memory used by the table build

	/**
	 * Prints the TableBuildInfo's information to the terminal
	*/
	__host__ void print(){
		if (generationTime == 0) printf("Table Build Aborted\n");
		else{
			printf(
				"Table Built. First Index: %u. Last Index %u. Modes Found %u. Time taken (ms): %ld. Spacing: %ld. Memory Used: %u.\n",
				startIndex,
				stopIndex,
				totalReplaced,
				generationTime,
				spacing,
				memoryUsed
			);
		}
	}
};

/**
 * Used to hold and record information of a frame
 * @param time 	long long
 * @param range unsigned int
 * @param chunksize unsigned int
 * @param avgStep unsigned int
 * @param minStep unsigned int
 * @param maxStep unsigned int
 * @param memoryUsed unsigned int
 * @param breaks std::vector<unsigned int>
*/
struct CollatzFrameInfo{
	long long time; 					//time the frame took to complete
	unsigned int range; 				//range of the frame
	unsigned int chunksize; 			//numbers done by each proccess of the--force-grab-cursor frame
	unsigned int avgStep; 				//average steps of comparison taken
	unsigned int minStep; 				//minimum comparison steps taken
	unsigned int maxStep; 				//maximum comparison steps taken
	unsigned int memoryUsed; 			//memory used by the frame
	unsigned long int startChunk;
	std::vector<unsigned long int> breaks; 	//vector containing any breaks found in the frame

	/**
	 * Prints the statistics of the frame
	*/
	__host__ void print(){
		if(time == 0) printf("Frame Failed!");
		else{
			printf(
				"Frame Finished. Range: %u, ChunkSize: %u, avgStep: %u, minStep: %u, maxStep %u, usedMemory: %u, time: %lld\n",
				range,
				chunksize,
				avgStep,
				minStep,
				maxStep,
				memoryUsed,
				time
			);
			for(unsigned int i = 0; i < breaks.size(); i++){
				printf("Break found at %lu!\n", breaks[i]);
			}
		}
	}
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

//DECLARATIONS//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * Device kernel for the table build.
 * @param ColSeq binding for the 2D array to build the table in
 * @param Colsteps height of 2^k+1
 * @param num64 binding for the starting num64 of the table range
 * @param numLength size of the num64 array
 * @param sampleVec binding for the sample vector
 * @param tempVec binding for the temp vector used for sorting
 * @param offsets binding for the offsets for each sample
 * @param vecSize binding for the number of entries in the sample vector
 * @param modeCheck binding for the boolean array
 * @param tbInfos binding for the table build information struct
*/
__global__ void d_tableBuild(
	SE_CUDA::Binding<unsigned long int> ColSeq, 
	unsigned long int Colsteps,
	SE_CUDA::Binding<unsigned long int> num64,
	unsigned int numLength,
	SE_CUDA::Binding<unsigned long int*> sampleVec,
	SE_CUDA::Binding<unsigned long int*> tempVec,
	SE_CUDA::Binding<unsigned long int> offsets,
	SE_CUDA::Binding<unsigned int> vecSize,
	SE_CUDA::Binding<bool> modeCheck,
	SE_CUDA::Binding<TableBuildInfo> tbInfos
);

/**
 * Host side of the table build for a given device
 * @param device device to build the table on
 * @param ColSeq local SE_CUDA::Cuda_2D<unsigned long int> to build the table with
 * @param ColSteps height of 2^k+1
 * @param num64 2^k+1 num64
 * @param sampleAmount amount of samples to build with
 * @param tableThresholdOffsets offset used to determine the range of the table build and where it ends
 * @param tableExtraOffsets further range for the table build that doesnt effect when it is rebuilt
 * @returns information of the table build after completion
*/
__host__ TableBuildInfo tableBuild(
	SE_CUDA::Device& device, 
	SE_CUDA::Cuda_2D<unsigned long int>& ColSeq, 
	unsigned long int ColSteps, 
	SE_CUDA::Cuda_1D<unsigned long int>& num64,
	unsigned int sampleAmount,
	std::vector<Offset>& tableThresholdOffsets,
	std::vector<Offset>& tableExtraOffsets
);

/**
 * Device kernel for the main collatz work
 * @param alloc1 Binding for array of num64 pointers per thread
 * @param alloc2 Binding for array of temp num64 pointers per thread
 * @param offsets Binding for array to record the amount of chunk completed per thread
 * @param num64  Binding for the starting num64 of the work
 * @param numLength length of the num64 arrays
 * @param tables Binding for array of Bindings of ColSeq tables per threadblock
 * @param ColSteps height of 2^k+1
 * @param powa power of 2 for the chunksize
 * @param powa2 expanded powa for the chunksize
 * @param avgSteps Binding for an array to record average steps taken to match per thread
 * @param minSteps Binding for an array to record minimum steps taken to match per thread
 * @param maxSteps Binding for an array to record maximum steps taken to match per thread
 * @param skipEvens set to true if skipping evens
 * @param sharedBytes the number of bytes allocated in shared memory for each block
*/
__global__ void d_collatzJob(
	SE_CUDA::Binding<unsigned long int*> alloc1,
	SE_CUDA::Binding<unsigned long int*> alloc2,
	SE_CUDA::Binding<unsigned int> offsets,
	SE_CUDA::Binding<unsigned long int> num64,
	unsigned int numLength,
	SE_CUDA::Binding<SE_CUDA::Binding<unsigned long int>> tables,
	unsigned int ColSteps,
	unsigned int powa2,
	SE_CUDA::Binding<unsigned int> avgSteps,
	SE_CUDA::Binding<unsigned int> minSteps,
	SE_CUDA::Binding<unsigned int> maxSteps,
	bool skipEvens,
	size_t sharedBytes
);

__host__ Offset parseOffset(std::string carr); //parse offset arguements
__host__ __device__ long long applyOffset(long long num, unsigned int multiplier, unsigned int power); //apply arguement offset to a long long
__host__ __device__ void applyOffset(unsigned long int* num64, unsigned int numLength, unsigned int multiplier, unsigned int power); //apply an arguement offset to a num64
__host__ __device__ unsigned int Collatz(unsigned long int num64[], unsigned long int& size); //check height
__host__ void CollatzSteps(SE_CUDA::Cuda_1D<unsigned long int>& num64, SE_CUDA::Cuda_2D<unsigned long int>& ColSeq); //Generate ColSeq on the host side with one number
__device__ bool CollatzCompare(unsigned long int* num64,
	unsigned int numLength,
	SE_CUDA::Binding<unsigned long int> ColSeq,
	unsigned int ColSteps,
	unsigned int* _steps = nullptr,
	const unsigned long int* shColSeq = nullptr,
	unsigned int shRows = 0U
);
__host__ __device__ __forceinline__ void mul64b3(unsigned long int num64[], unsigned long int& size); //num64 * 3
__host__ __device__ __forceinline__ void div64b2(unsigned long int* num64, unsigned long int& size); // num64 / 2
__host__ __device__ bool compare64(unsigned long int* num0, unsigned long int* num1, unsigned int numLength); //compare 2 num64s of the same size
__host__ __device__ void addPow2UL64(unsigned long int* num64, int valExp, unsigned long int& size); //increment num64 by a power of 2
__host__ void addPow2UL64(SE_CUDA::Cuda_1D<unsigned long int>& num64, unsigned long int valExp, unsigned long int& size); //increment num64 by a power of 2
__host__ __device__ void addUL64(unsigned long int* num64, unsigned long int val, unsigned long int& size); //add an unsigned integer to the num64
__host__ void addUL64(SE_CUDA::Cuda_1D<unsigned long int>& num64, unsigned long int val, unsigned long int& size); //add an unsigned integer to the num64
__host__ __device__ __forceinline__ void add64b1(unsigned long int* num64, unsigned long int& size); //increment the num64 by 1
__host__ __device__ void print64(unsigned long int num[], unsigned long int& size); //print num64 device side
__host__  void print64(SE_CUDA::Cuda_1D<unsigned long int>& num, unsigned long int& size); //print num64 host side
__host__ void recordToTable(unsigned long int* num64, SE_CUDA::Cuda_2D<unsigned long int>& ColSeq, unsigned int step, unsigned int numlength); //record to lookup table
__device__ inline void recordToTable(unsigned long int* num64, SE_CUDA::Binding<unsigned long int> ColSeq, unsigned int step, unsigned int numlength); //record to lookup table

//CLASSES///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * This class is used to async jobs for devices, each used device should have a CollatzJob loaded up into the heap to keep track of the device's progress and
 * start / stop the d_collatzJob() kernel based on the state of the device.
 */
class CollatzJob{
public:

	/**
	 * CONSTRUCTOR
	 * @param device reference to the device this job is tied to
	 * @param ColSeq reference to the lookup table
	 * @param ColSteps the height of 2k+1
	 * @param num64 reference to the main num64_hold variable
	 */
	CollatzJob(
		SE_CUDA::Device& device, 
		SE_CUDA::Cuda_2D<unsigned long int>& ColSeq,
		unsigned int ColSteps,
		unsigned int powa2,
		SE_CUDA::Cuda_1D<unsigned long int>& num64 
	){
		runState = false;

		_device = &device;
		_ColSeq = &ColSeq;
		_ColSteps = ColSteps;
		_powa2 = powa2;
		_num64 = &num64;

		_alloc1 = new SE_CUDA::CudaUniqueAlloc<unsigned long int>(_device, _num64->size(), _device->getGrid().blockCount * _device->getGrid().threadCount);
		_alloc2 = new SE_CUDA::CudaUniqueAlloc<unsigned long int>(_device, _num64->size(), _device->getGrid().blockCount * _device->getGrid().threadCount);
		_offsets = new SE_CUDA::Cuda_1D<unsigned int>(_device->getGrid().blockCount * _device->getGrid().threadCount);
		_tables = new SE_CUDA::Cuda_1D<SE_CUDA::Binding<unsigned long int>>(_device->getGrid().blockCount);
		_avgSteps = new SE_CUDA::Cuda_1D<unsigned int>(_device->getGrid().blockCount * _device->getGrid().threadCount);
		_minSteps = new SE_CUDA::Cuda_1D<unsigned int>(_device->getGrid().blockCount * _device->getGrid().threadCount);
		_maxSteps = new SE_CUDA::Cuda_1D<unsigned int>(_device->getGrid().blockCount * _device->getGrid().threadCount);

		_d_alloc1 = _alloc1->getBinding();
		_d_alloc2 = _alloc2->getBinding();
		_d_offsets = _offsets->bind(_device);
		_d_num64 = _num64->bind(_device);
		_d_tables = _tables->bind(_device);
		_d_avgSteps = _avgSteps->bind(_device);
		_d_minSteps = _minSteps->bind(_device);
		_d_maxSteps = _maxSteps->bind(_device);

		for(unsigned int i = 0; i < _tables->size(); i++){
			_tables->operator[](i) = _ColSeq->bind(_device);
		}

		_tables->push(_d_tables);
	}

	/**
	 * DESTRUCTOR
	 */
	~CollatzJob(){
		for(unsigned int i = 0; i < _tables->size(); i++){
			_ColSeq->unbind(_tables->operator[](i));
		}

		_num64->unbind(_d_num64);

		delete _alloc1;
		delete _alloc2;
		delete _offsets;
		delete _tables;
		delete _avgSteps;
		delete _minSteps;
		delete _maxSteps;
	}

	/**
	 * Poll the device to see if it is still running a job
	 * 
	 * @returns whether the device is free or not
	 */
	bool poll(){
		return _device->poll();
	}

	/**
	 * Start a d_collatzjob() on the device.
	 * 
	 * @param startChunk the begining chunk of this job, used for recording
	 * 
	 * @throws runtime-error if a job is already running or the previous job hasnt been handled by stop()
	 */
	void start(unsigned long int startChunk){
		_device -> select();
		if (runState) throw std::runtime_error("ERROR CollatzJob: Double Job Allocation\n");
		if (!poll()) throw std::runtime_error("ERROR CollatzJob: Tried to start a job when one is already running!\n");
		
		_cInfo.startChunk = startChunk;
		_cInfo.chunksize = _powa2;
		_cInfo.range = _powa2 * _device->getGrid().blockCount * _device->getGrid().threadCount;
		_startTime = std::chrono::high_resolution_clock::now();

		
		for(unsigned int i = 0; i < _tables->size(); i++){
			_ColSeq->push(_tables->operator[](i));
		}
		
		_num64->push(_d_num64);

		_cInfo.memoryUsed = _device->getUsedMemory();
		
		d_collatzJob<<<_device->getGrid().blockCount, _device->getGrid().threadCount, _device->getProperties().sharedMemPerBlock, _device->getStream()>>>(
			_d_alloc1,
			_d_alloc2,
			_d_offsets,
			_d_num64,
			_num64->size(),
			_d_tables,
			_ColSteps,
			_powa2,
			_d_avgSteps,
			_d_minSteps,
			_d_maxSteps,
			SKIPEVENS,
			_device->getProperties().sharedMemPerBlock
		);

		runState = true;
	}

	/**
	 * Used to retreive information from a completed d_collatzJob()
	 * 
	 * @throws runtime-error if a job is currently running or if no job was started previously
	 */
	CollatzFrameInfo stop(){
		_device->select();
		if (!runState) throw std::runtime_error("ERROR CollatzJob: No job to retreive from!\n");
		if (!poll()) throw std::runtime_error("ERROR CollatzJob: Tried to end a job prematurely!\n");

		//Retreive offsets and required info from device
		_offsets->pull(_d_offsets);
		_avgSteps->pull(_d_avgSteps);
		_minSteps->pull(_d_minSteps);
		_maxSteps->pull(_d_maxSteps);

		//find and record breaks and step information
		unsigned int minStep = _ColSteps;
		unsigned int maxStep = 0;
		unsigned int stepSum = 0;
		for(unsigned int i = 0; i < (_device->getGrid().blockCount * _device->getGrid().threadCount); i++){

			if (_offsets->operator[](i) < _powa2) _cInfo.breaks.push_back((unsigned long int)(_offsets->operator[](i)) + ((unsigned long int)(_powa2) * (unsigned long int)(i))); //record breaks

			stepSum += _avgSteps -> operator[](i);
			if(_minSteps -> operator[](i) < minStep) minStep = _minSteps->operator[](i);
			if(_maxSteps->operator[](i) > maxStep) maxStep = _maxSteps->operator[](i);

		}

		//record step information
		_cInfo.minStep = minStep;
		_cInfo.maxStep = maxStep;
		_cInfo.avgStep = stepSum / (_device->getGrid().blockCount * _device->getGrid().threadCount);
		_cInfo.time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - _startTime).count();

		runState = false;

		return _cInfo;
	}

private:
	//Main
	bool runState; //keeps track if a job has been started from the host side
	SE_CUDA::Device* _device; //reference to the SE_CUDA::Device this job is tied to
	unsigned int _ColSteps; //height of 2k+1
	unsigned int _powa2; //the range each thread has to check
	CollatzFrameInfo _cInfo; //frame info
	std::chrono::high_resolution_clock::time_point _startTime; //time point from when a job is started

	//CPU
	SE_CUDA::Cuda_1D<unsigned long int>* _num64; //refernce to the num64_hold variable
	SE_CUDA::Cuda_2D<unsigned long int>* _ColSeq; //reference to the lookup table
	SE_CUDA::CudaUniqueAlloc<unsigned long int>* _alloc1; //num64 per each thread
	SE_CUDA::CudaUniqueAlloc<unsigned long int>* _alloc2; //temp num64 per each thread
	SE_CUDA::Cuda_1D<unsigned int>* _offsets; //records the offsets of the completed work for each thread, is a break if < powa2
	SE_CUDA::Cuda_1D<SE_CUDA::Binding<unsigned long int>>* _tables; //unique allocation of ColSeq pre threadblock
	SE_CUDA::Cuda_1D<unsigned int>* _avgSteps; //records average steps per thread
	SE_CUDA::Cuda_1D<unsigned int>* _minSteps; //records min steps per thread
	SE_CUDA::Cuda_1D<unsigned int>* _maxSteps; //records max steps per thread

	//GPU
	SE_CUDA::Binding<unsigned long int*> _d_alloc1; //device binding of alloc1
	SE_CUDA::Binding<unsigned long int*> _d_alloc2; //device binding of alloc2
	SE_CUDA::Binding<unsigned int> _d_offsets; //device binding of offsets
	SE_CUDA::Binding<unsigned long int> _d_num64; //device binding of num64
	SE_CUDA::Binding<SE_CUDA::Binding<unsigned long int>> _d_tables; //device binding of tables (ColSeq per Threadblock)
	SE_CUDA::Binding<unsigned int> _d_avgSteps; //device binding of avgSteps
	SE_CUDA::Binding<unsigned int> _d_minSteps; //device binding of minSteps
	SE_CUDA::Binding<unsigned int> _d_maxSteps; //device binding of maxSteps
};

//FUNCTIONS/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * MAIN
*/
int main(int argc, char** argv){
	
//ARGUEMENTS
	int expon = atoi(argv[1]); 									//starting exponent of 2
	int powa = atoi(argv[2]); 									// power of 2 to add to base number for each proccess for the streak (shifted by process)
	unsigned int sampleAmount = (unsigned int)atoi(argv[3]); 	//amount of samples
	std::vector<Offset> initialOffsets; 						//offset the start of the test range
	std::vector<Offset> tableThresholdOffsets; 					//define the range for sampling for the table build
	std::vector<Offset> tableExtraOffsets; 						//add to the right of the sample range
	std::vector<Offset> testRangeOffsets; 						//offsets for the nonstop version's range

	//initialize conditional arguements
	for(int i = 4; i < argc; i++){
		std::string tempS(argv[i]);
		//table build range offsets
		if (argv[i][0] == 't'){
			std::string tempC = "";
			for(int j = 1; j < tempS.length(); j++){
				tempC.insert(tempC.end(), tempS.at(j));
			}
			tableThresholdOffsets.push_back(parseOffset(tempC));
		}
		//table range extra offsets
		if(argv[i][0] == 'e'){
			std::string tempC = "";
			for(int j = 1; j < tempS.length(); j++){
				tempC.insert(tempC.end(), tempS.at(j));
			}
			tableExtraOffsets.push_back(parseOffset(tempC));
		}
		//initial offsets
		else if (argv[i][0] == 'o'){
			std::string tempC = "";
			for(int j = 1; j < tempS.length(); j++){
				tempC.insert(tempC.end(), tempS.at(j));
			}
			initialOffsets.push_back(parseOffset(tempC));
		}
		//range offsets
		else if (argv[i][0] == 'r'){
			std::string tempC = "";
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
	}

//GPU INFO
    std::vector<std::unique_ptr<SE_CUDA::Device>> devices = SE_CUDA::initDevices(); //container for device info and selection method
	
//MAIN VARIABLES
    unsigned int powa2 = 1U << powa; 			// power of 2 value in base 10 (for each node to work on, should be less than INT MAX)
	unsigned long int tableThreshold = 1UL; 	//the chunk where a table should be rebuilt
	unsigned long int testRange = 1UL; 			//range used if a defined range of the test is given for the NONSTOP mode
	unsigned long int chunkCount = 0UL; 		//number of chunks started total
	unsigned long int runningTableChunk = 0UL; 	//number of chunks started since the last table build
	unsigned int minStep = UINT_MAX; 			//minimum step ever done by CollatzCompare()
	unsigned int maxStep = 0U; 					//maximum step ever done by CollatzCompare()
	unsigned long int avgStepSum = 0UL; 		//sum of all the average steps done by CollatzCompare(), used for getting the average steps at the finish
	long long minTime = LLONG_MAX; 				//minimum time for a CollatzJob frame completion
	long long maxTime = 0LL; 					//maximum time for a CollatzJob frame completion
	long long avgTimeSum = 0LL; 				//sum of all times for CollatzJob frame completeion, used for getting the average time
	unsigned int runCount = 0; 					//number of frames completed
	std::vector<BreakInfo> bInfos; 				//vector to store information about breaks
	CollatzFrameInfo cInfo; 					//record used to store information about frames

    //Base Conversion to 2^64
	unsigned long int sizeNum = expon/64; 		//number of entries needed to hold ^expon in base 2^64
	const unsigned int extra = 3; 				//extra padding at the front, first two will be used for flags in num64
    unsigned int num64entries = sizeNum + extra; //total num64 length
	
	SE_CUDA::Cuda_1D<unsigned long int> h_num64hold = SE_CUDA::Cuda_1D<unsigned long int>(num64entries); //num64
	unsigned long int h_num64[num64entries] = {0UL}; //temp num64
	for(unsigned int i = 0; i < num64entries; i++){
		h_num64hold[i] = 0UL;
	}

	//init num64s
	h_num64hold[sizeNum] = ((1UL << (expon & 63)) | h_num64hold[sizeNum]);//Set the leading value in the array to 1 so we have 2^k
	h_num64hold[0]++;//Set the 0th place to 1 so we all have 2^k+1 now.
	h_num64hold[h_num64hold.size()-1] = (unsigned long int)sizeNum;
    for (int i = 0; i < num64entries; i++){
        h_num64[i] = h_num64hold[i];
    }
	
	//init colsteps
    unsigned int ColSteps = Collatz(h_num64, h_num64[h_num64hold.size()-1]);

	//init ColSeq
	SE_CUDA::Cuda_2D<unsigned long int> ColSeq = SE_CUDA::Cuda_2D<unsigned long int>(ColSteps,5U);
	for(unsigned int i = 0; i < ColSeq.getRows(); i++){
		for(unsigned int j = 0; j < ColSeq.getCols(); j++){
			ColSeq[i][j] = 0UL;
		}
	}
	//initialize table to 2k+1
	CollatzSteps(h_num64hold, ColSeq);

	//apply initial offsets to the num64
	for(unsigned int i = 0; i < initialOffsets.size(); i++){
		for(unsigned int j = 0; j < initialOffsets[i].multiplier; i++){
			addPow2UL64(h_num64hold, initialOffsets[i].power, h_num64hold[h_num64hold.size() -1]);
		}
	}
	
	//init table range
	for(unsigned int i = 0; i < tableThresholdOffsets.size(); i++){
		for(unsigned int j = 0; j < tableThresholdOffsets[i].multiplier; j++){
			for(unsigned int k = 0; k < tableThresholdOffsets[i].power - powa; k++){
				tableThreshold *= 2;
			}
		}
	}

	//init test range
	if(NONSTOP){
		for(unsigned int i = 0; i < testRangeOffsets.size(); i++){
			for(unsigned int j = 0; j < testRangeOffsets[i].multiplier; j++){
				for(unsigned int k = 0; k < testRangeOffsets[i].power - powa; k++){
					testRange *= 2;
				}
			}
		}
	}
	
	printf("Table Threshold %lu\n", tableThreshold);
	printf("powa2 %u\n", powa2);
	printf("begin size: %lu\n", h_num64hold[h_num64hold.size()-1]);
	printf("Steps: %u\n", ColSteps);

//INITIALIZATION

	//create array of CollatzJobs for each device
	CollatzJob* collatzJobs[devices.size()] = {nullptr};
	for(int i = 0; i < devices.size(); i++){
		collatzJobs[i] = new CollatzJob(*devices[i], ColSeq, ColSteps, powa2, h_num64hold);
	}
	
	//initial table build
	tableBuild(*devices[0], ColSeq, ColSteps, h_num64hold, sampleAmount, tableThresholdOffsets, tableExtraOffsets).print();

	//initial job assignment
	for(unsigned int i = 0; i < devices.size(); i++){
		//get number of chunks to assign to device
		unsigned long int assignedChunks = (unsigned long int)devices[i]->getGrid().blockCount * (unsigned long int)devices[i]->getGrid().threadCount;

		//start job on device
		collatzJobs[i]->start(chunkCount);

		//get the num64 ready for the next job
		addUL64(h_num64hold, assignedChunks * (unsigned long int)powa2, h_num64hold[h_num64hold.size()-1]);
		
		//record place in chunks
		chunkCount += assignedChunks;
		runningTableChunk += assignedChunks;
	}	

//MAIN LOOP
	
	//Will keep running jobs and recording data until conditions are met
	bool kg = true;
	while (kg){
		unsigned int currentDevice = 0; //current index of device

		//wait for a device to be done with a job
		bool deviceFound = false; 
		while(!deviceFound){
			for(unsigned int i = 0; i < devices.size(); i++){
				if (collatzJobs[i] != nullptr && collatzJobs[i] -> poll()){
					cInfo = collatzJobs[i]->stop();
					currentDevice = i;
					deviceFound = true;
					break;
				}
			}
			if(!deviceFound) std::this_thread::sleep_for(std::chrono::milliseconds(1));
		}

		cInfo.print();

		//record info from frame retrived from device
		minStep = (minStep > cInfo.minStep) ? cInfo.minStep : minStep;
		maxStep = (maxStep < cInfo.maxStep) ? cInfo.maxStep : maxStep;
		avgStepSum += cInfo.avgStep;
		minTime = (minTime > cInfo.time) ? cInfo.time : minTime;
		maxTime = (maxTime < cInfo.time) ? cInfo.time : maxTime;
		avgTimeSum += cInfo.time;
		runCount++;

		//if breaks are found, record them
		if (!cInfo.breaks.empty()){
			for (unsigned int i = 0; i < cInfo.breaks.size(); i++){
				bInfos.push_back({cInfo.startChunk, cInfo.breaks[i]});
			}

			//if not nonstop, stop the loop
			if(!NONSTOP){
				delete collatzJobs[currentDevice];
				collatzJobs[currentDevice] = nullptr;
				kg = false;
			}
		}

		//if nonstop and the range has been met, stop the loop
		else if(NONSTOP && chunkCount >= testRange){
			delete collatzJobs[currentDevice];
			collatzJobs[currentDevice] = nullptr;
			kg = false;
		}

		//if not, keep going
		else{
			unsigned long int assignedChunks = (unsigned long int)devices[currentDevice]->getGrid().blockCount * (unsigned long int)devices[currentDevice]->getGrid().threadCount;

			//check for a table rebuild
			if(runningTableChunk + assignedChunks >= tableThreshold){
				tableBuild(*devices[currentDevice], ColSeq, ColSteps, h_num64hold, sampleAmount, tableThresholdOffsets, tableExtraOffsets).print();
				runningTableChunk = 0UL;
			}
			
			//start the next job
			collatzJobs[currentDevice]->start(chunkCount);
			addUL64(h_num64hold, assignedChunks * (unsigned long int)powa2, h_num64hold[h_num64hold.size()-1]);
			chunkCount += assignedChunks;
			runningTableChunk += assignedChunks;
		}
	}

//CLEANUP

	//retreive frames from all remaining devices and record the data
	for(unsigned int i = 0; i < devices.size() - 1; i++){
		unsigned int currentDevice = 0;

		//wait for devices to be free
		bool deviceFound = false;
		while(!deviceFound){
			for(unsigned int i = 0; i < devices.size(); i++){
				if (collatzJobs[i] != nullptr && collatzJobs[i] -> poll()){
					cInfo = collatzJobs[i]->stop();
					currentDevice = i;
					deviceFound = true;
					break;
				}
			}
			if(!deviceFound) std::this_thread::sleep_for(std::chrono::milliseconds(1));
		}

		cInfo.print();

		//record info from the frame
		minStep = (minStep > cInfo.minStep) ? cInfo.minStep : minStep;
		maxStep = (maxStep < cInfo.maxStep) ? cInfo.maxStep : maxStep;
		avgStepSum += cInfo.avgStep;
		minTime = (minTime > cInfo.time) ? cInfo.time : minTime;
		maxTime = (maxTime < cInfo.time) ? cInfo.time : maxTime;
		avgTimeSum += cInfo.time;
		runCount++;

		//delete the job
		delete collatzJobs[currentDevice];
		collatzJobs[currentDevice] = nullptr;

		//if there were breaks, record them
		if (!cInfo.breaks.empty()){
			for (unsigned int i = 0; i < cInfo.breaks.size(); i++){
				bInfos.push_back({cInfo.startChunk, cInfo.breaks[i]});
			}
		}
	}

	//if breaks were found, sort them from earliest to latest
	if (!bInfos.empty()) std::sort(bInfos.begin(), bInfos.end(), [](const BreakInfo& a, const BreakInfo&b){
		if (a.chunk < b.chunk) return true;
		else if (a.chunk > b.chunk) return false;
		else return a.offset < b.offset;
	});

//RESULTS

	//if a break is found
	if(!bInfos.empty()){
		printf("Break found at %lu. Range is %lu.\n", 
			bInfos[0].chunk * (unsigned long int)powa2 + (unsigned long int)bInfos[0].offset, //offset
			chunkCount * (unsigned long int)powa2 //range
		);
	}

	//no break is found
	else{
		printf("No breaks found. Checked range was %lu\n", chunkCount * (unsigned long int)powa2);
	}

	//stats
	printf("Average Step: %u\n Min Step: %u\n Max Step: %u\n Avg Time: %ld\n Min Time: %ld\n MaxTime: %ld\n",
		avgStepSum / runCount,
		minStep,
		maxStep,
		avgTimeSum / runCount,
		minTime,
		maxTime
	);
	
    return 0;
}

/**
 * Device kernel for the main collatz work
 * @param alloc1 Binding for array of num64 pointers per thread
 * @param alloc2 Binding for array of temp num64 pointers per thread
 * @param offsets Binding for array to record the amount of chunk completed per thread
 * @param num64  Binding for the starting num64 of the work
 * @param numLength length of the num64 arrays
 * @param tables Binding for array of Bindings of ColSeq tables per threadblock
 * @param ColSteps height of 2^k+1
 * @param powa power of 2 for the chunksize
 * @param powa2 expanded powa for the chunksize
 * @param avgSteps Binding for an array to record average steps taken to match per thread
 * @param minSteps Binding for an array to record minimum steps taken to match per thread
 * @param maxSteps Binding for an array to record maximum steps taken to match per thread
 * @param skipEvens set to true if skipping evens
 * @param sharedBytes the number of bytes allocated in shared memory for each block
*/
__global__ void d_collatzJob(
	SE_CUDA::Binding<unsigned long int*> alloc1,
	SE_CUDA::Binding<unsigned long int*> alloc2,
	SE_CUDA::Binding<unsigned int> offsets,
	SE_CUDA::Binding<unsigned long int> num64,
	unsigned int numLength,
	SE_CUDA::Binding<SE_CUDA::Binding<unsigned long int>> tables,
	unsigned int ColSteps,
	unsigned int powa2,
	SE_CUDA::Binding<unsigned int> avgSteps,
	SE_CUDA::Binding<unsigned int> minSteps,
	SE_CUDA::Binding<unsigned int> maxSteps,
	bool skipEvens,
	size_t sharedBytes
){

//INIT
	unsigned int rank = blockDim.x*blockIdx.x+threadIdx.x; //thread identity

	SE_CUDA::Binding<unsigned long int> ColSeq = tables.d_ptr[blockIdx.x]; //ColSeq, lookupTable

	extern __shared__ unsigned long int shrColSeq[]; //shared memory for lookup table
	const unsigned int sharedCols = 5U; //columns in shared memory lookup table
	unsigned int sharedRows = (unsigned int)(sharedBytes / (sizeof(unsigned long int) * sharedCols)); //rows stored in shared lookup table
	const unsigned int totalShared = sharedRows * sharedCols; //total amount of unsigned long ints stored in shared lookup table

	//copy from global lookup table to shared memory lookup table
	for(unsigned idx = threadIdx.x; idx < totalShared; idx += blockDim.x){
		unsigned int row = idx / sharedCols;
		unsigned int col = idx % sharedCols;
		shrColSeq[idx] = SE_CUDA::read2D<unsigned long int>(ColSeq.d_ptr, row, col, ColSeq.pitch);	
	}

	__syncthreads();

	unsigned long int* localNum = alloc1.d_ptr[rank]; //local num64
	unsigned long int* tempNum = alloc2.d_ptr[rank]; //local temp num64

	//print64(localNum, localNum[numLength-1]);
	//Copy starting num64 to localNum
	for(unsigned int i = 0; i < numLength; i++){
		localNum[i] = num64.d_ptr[i];
	}

	//apply start of this thread's chunk
	addUL64(localNum, (unsigned long int)(powa2 * rank), localNum[numLength-1]);
	
//WORK
	//Main Work Loop, 
	//the thread will check its assigned chunk for any number that does not match the table, returning an iter < powa2 if a break is found
	
	bool kg = true; 				//to keep track of whether a match is found in the table or not
	unsigned int currentStep = 0; 	//current steps taken to match
	unsigned int stepSum = 0; 		//sum of all the steps taken to be averaged
	unsigned int minStep = ColSteps; //minimum step
	unsigned int maxStep = 0; 		//maximum step
	unsigned int iter = 0; 			//current position in assigned range

	while(kg && iter < powa2){
		//copy to temp
		for(unsigned int i = 0; i < numLength; i++){
			tempNum[i] = localNum[i];
		}

		//check for a match with the table
		kg = CollatzCompare(tempNum, numLength, ColSeq, ColSteps, &currentStep, shrColSeq, sharedRows);

		//record step information
		stepSum += currentStep;
		if(currentStep < minStep) minStep = currentStep;
		if(currentStep > maxStep) maxStep = currentStep;

		//break the loop if a match isnt found !IMPORTANT for breaks position powa2-1!
		if(!kg) break;

		//start next iteration
		add64b1(localNum, localNum[numLength-1]);
		iter++;
		if(skipEvens && (num64.d_ptr[0] & 1UL) == 0 && iter != powa2 - 2){
			add64b1(localNum, localNum[numLength-1]);
			iter++;
		}
	}

	//record local stats for retreival
	offsets.d_ptr[rank] = iter;
	avgSteps.d_ptr[rank] = stepSum / iter;
	minSteps.d_ptr[rank] = minStep;
	maxSteps.d_ptr[rank] = maxStep;
	//if (rank == 0) printf("Collatz job finished\n");
}

/**
 * Host side of the table build for a given device
 * @param device device to build the table on
 * @param ColSeq local SE_CUDA::Cuda_2D<unsigned long int> to build the table with
 * @param ColSteps height of 2^k+1
 * @param num64 2^k+1 num64
 * @param sampleAmount amount of samples to build with
 * @param tableThresholdOffsets offset used to determine the range of the table build and where it ends
 * @param tableExtraOffsets further range for the table build that doesnt effect when it is rebuilt
 * @returns information of the table build after completion
*/
__host__ TableBuildInfo tableBuild(
	SE_CUDA::Device& device, 
	SE_CUDA::Cuda_2D<unsigned long int>& ColSeq, 
	unsigned long int ColSteps, 
	SE_CUDA::Cuda_1D<unsigned long int>& num64,
	unsigned int sampleAmount,
	std::vector<Offset>& tableThresholdOffsets,
	std::vector<Offset>& tableExtraOffsets
	)
{
//START
	printf("Start Build CPU side\n");
	std::chrono::high_resolution_clock::time_point startTime = std::chrono::high_resolution_clock::now(); //start timer
	int threadsTotal = sampleAmount; //total amount of threads required for the table build
    int threadsPerBlock = std::min(threadsTotal, device.getProperties().maxThreadsPerBlock); //threads per block of the table build job
    int blocks = (threadsTotal + threadsPerBlock - 1) / threadsPerBlock; //threadblocks required for the tablebuild

	//initialize spacing
	long long int spacing = 1LL; //spacing between samples

	//base range
	for (unsigned int i = 0; i < tableThresholdOffsets.size(); i++){
		for(unsigned int j = 0; j < tableThresholdOffsets[i].multiplier; j++){
			for(unsigned int k = 0; k < tableThresholdOffsets[i].power; k++){
				spacing *= 2LL;
			}
		}
	}

	//extra range
	if(!tableExtraOffsets.empty()){
		long long tempSpacing = 1LL;
		for (unsigned int i = 0; i < tableExtraOffsets.size(); i++){
			for(unsigned int j = 0; j < tableExtraOffsets[i].multiplier; j++){
				for(unsigned int k = 0; k < tableExtraOffsets[i].power; k++){
					tempSpacing *= 2LL;
				}
			}
		}
		spacing += tempSpacing;
	}

	//guard spacing
	spacing /= sampleAmount;
	if (spacing < 1) spacing = 1;

	//start long randomization
	std::random_device rd; //random device
	std::mt19937 generator(rd()); //random number generator seeded with random device
	std::uniform_int_distribution<long long int> distribution(0LL, std::numeric_limits<long long int>::max()); //define range of random number generator

//ALLOCATION
	//CPU
	SE_CUDA::CudaUniqueAlloc<unsigned long int> sampleVec = SE_CUDA::CudaUniqueAlloc<unsigned long int>(&device, num64.size(), sampleAmount); //array of num64s to build samples with
	SE_CUDA::Cuda_1D<unsigned long int*> tempVec = SE_CUDA::Cuda_1D<unsigned long int*>(sampleAmount); 	//temp array of num64s used for sorting
	SE_CUDA::Cuda_1D<unsigned int> vecSize = SE_CUDA::Cuda_1D<unsigned int>(1); 						//size of the sampleVec
	SE_CUDA::Cuda_1D<bool> modeCheck = SE_CUDA::Cuda_1D<bool>(sampleAmount);							//array to do boolean checks with
	SE_CUDA::Cuda_1D<unsigned long int> offsets = SE_CUDA::Cuda_1D<unsigned long int>(sampleAmount); 	//offsets of each sample
	SE_CUDA::Cuda_1D<TableBuildInfo> tbInfos = SE_CUDA::Cuda_1D<TableBuildInfo>(1); 					//table build info

	SE_CUDA::Binding<unsigned long int> d_num64 = num64.bind(&device); 		//device binding for the start 2^k+1 of the tablebuild
	SE_CUDA::Binding<unsigned long int> d_colSeq = ColSeq.bind(&device); 	//device binding for the table to build
	SE_CUDA::Binding<unsigned long int*> d_vec = sampleVec.getBinding(); 	//device binding for the sample vector
	SE_CUDA::Binding<unsigned long int*> d_tempVec = tempVec.bind(&device); //device binding for the temp sample vector
	SE_CUDA::Binding<unsigned int> d_vecSize = vecSize.bind(&device); 		//device binding for the size of the sample vector
	SE_CUDA::Binding<unsigned long int> d_offsets = offsets.bind(&device); 	//device binding for the offsets of each sample
	SE_CUDA::Binding<bool> d_modeCheck = modeCheck.bind(&device); 			//device binding for the boolean array
	SE_CUDA::Binding<TableBuildInfo> d_tbInfos = tbInfos.bind(&device); 	//device binding for the table infos

	//init table infos
	tbInfos[0] = {0U,0U,0U,0L,0LL,0U};
	tbInfos[0].spacing = spacing;
	tbInfos[0].memoryUsed = device.getUsedMemory();

	//populate the table with 2^k+1
	CollatzSteps(num64, ColSeq);

	//set vecSize
	vecSize[0] = sampleAmount;

	//initialize offsets
	for(int i = 0; i < sampleAmount; i++){
		offsets[i] = (unsigned long int)((spacing * (long long int)(i)) + (distribution(generator)%spacing - (spacing/2LL)));
	}

	//push required values to the device
	num64[0]--; //subtract 1 from num64, changing it to 2^k from 2^k+1
	num64.push(d_num64);
	ColSeq.push(d_colSeq);
	vecSize.push(d_vecSize);
	offsets.push(d_offsets);
	tbInfos.push(d_tbInfos);
	num64[0]++; //restor num64's orginal state

//WORK
	//required info for cooperative kernel launch
	cudaStream_t stream = 0; 				//stream of the launch
	unsigned int numLength = num64.size(); 	//length of the num64
	dim3 gridDim(blocks); 					//conversion for execution
	dim3 blockDim(threadsPerBlock); 		//conversion for execution

	//kernel arguements
	void* kernelArgs[] ={
		&d_colSeq,
		&ColSteps,
		&d_num64,
		&numLength,
		&d_vec,
		&d_tempVec,
		&d_offsets,
		&d_vecSize,
		&d_modeCheck,
		&d_tbInfos
	};

	//launch cooperative kernel
	cudaLaunchCooperativeKernel(
		(void*)d_tableBuild,
		gridDim,
		blockDim,
		kernelArgs,
		0,
		stream
	);

	//await kernel completion
	cudaDeviceSynchronize();

	//retreive table and infos
	ColSeq.pull(d_colSeq);
	tbInfos.pull(d_tbInfos);

	//deallocate non-local bindings
	num64.unbind(d_num64);
	ColSeq.unbind(d_colSeq);

	std::chrono::high_resolution_clock::time_point stoptime = std::chrono::high_resolution_clock::now();//stop time

	printf("End Build CPU side\n");

	//record time
	tbInfos[0].generationTime = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - startTime).count();

	return tbInfos[0];
}

/**
 * Device kernel for the table build. DOES NOT SUPPORT SAMPLE AMOUNTS GREATER THAN THE AMOUNT OF AVAILABLE THREADS
 * @param ColSeq binding for the 2D array to build the table in
 * @param Colsteps height of 2^k+1
 * @param num64 binding for the starting num64 of the table range
 * @param numLength size of the num64 array
 * @param sampleVec binding for the sample vector
 * @param tempVec binding for the temp vector used for sorting
 * @param offsets binding for the offsets for each sample
 * @param vecSize binding for the number of entries in the sample vector
 * @param modeCheck binding for the boolean array
 * @param tbInfos binding for the table build information struct
*/
__global__ void d_tableBuild(
	SE_CUDA::Binding<unsigned long int> ColSeq, 
	unsigned long int Colsteps,
	SE_CUDA::Binding<unsigned long int> num64,
	unsigned int numLength,
	SE_CUDA::Binding<unsigned long int*> sampleVec,
	SE_CUDA::Binding<unsigned long int*> tempVec,
	SE_CUDA::Binding<unsigned long int> offsets,
	SE_CUDA::Binding<unsigned int> vecSize,
	SE_CUDA::Binding<bool> modeCheck,
	SE_CUDA::Binding<TableBuildInfo> tbInfos
	)
{	
//INIT
	cooperative_groups::grid_group grid = cooperative_groups::this_grid(); 	//grid for syncing
	unsigned int rank = blockDim.x*blockIdx.x+threadIdx.x; 					//thread id
	unsigned int worldSize = gridDim.x*blockDim.x; 							//size of the world
	bool startIndexFound = false; 											//used to check if the first mode replacement has happened

	//Build Samples
	if (rank < vecSize.d_ptr[0]){
		for(unsigned int i = 0; i < numLength; i++){
			sampleVec.d_ptr[rank][i] = num64.d_ptr[i];
		}

		addUL64(sampleVec.d_ptr[rank], offsets.d_ptr[rank], sampleVec.d_ptr[rank][numLength -1]);

		//check the height of the built sample
		modeCheck.d_ptr[rank] = CollatzCompare(sampleVec.d_ptr[rank], numLength, ColSeq, Colsteps);
	}

	grid.sync();
	
	//Check and remove breaking samples from the set
	if(rank == 0){
		unsigned int breaks = 0;
		
		for(unsigned int i = 0; i < vecSize.d_ptr[0]; i++){
			if (!modeCheck.d_ptr[i]){
				offsets.d_ptr[i] = ULONG_MAX;

				unsigned int j = vecSize.d_ptr[0] - 1;
				while( j > i && !modeCheck.d_ptr[j]) j--;

				if (j <= i){
					breaks = vecSize.d_ptr[0] - i;
					break;
				}

				unsigned long int temp = offsets.d_ptr[j];
				offsets.d_ptr[j] = offsets.d_ptr[i];
				offsets.d_ptr[i] = temp;

				bool tmpFlag = modeCheck.d_ptr[j];
                modeCheck.d_ptr[j] = modeCheck.d_ptr[i];
                modeCheck.d_ptr[i] = tmpFlag;

				breaks++;
			}
		}
		vecSize.d_ptr[0] -= breaks;
	}

	grid.sync();

	//rebuild Samples after the height check is complete
	if (rank < vecSize.d_ptr[0]){
		for(unsigned int i = 0; i < numLength; i++){
			sampleVec.d_ptr[rank][i] = num64.d_ptr[i];
		}

		addUL64(sampleVec.d_ptr[rank], offsets.d_ptr[rank], sampleVec.d_ptr[rank][numLength -1]);
	}

	//store the first sample as the running sample
	if(rank == 0){
		for(int i = 0; i < numLength; i++){
			num64.d_ptr[i] = sampleVec.d_ptr[rank][i];
		}
	}
	grid.sync();

//WORK

	//Main Loop
	//Will iterate the samples with the collatz function each round and check for modes, if a mode is found, it is recorded as the running sample and duplicates are deleted
	//if no mode is found, the running sample is placed into the table
	//if a mode is found, it is placed into the table

	unsigned int INDEX = 0U;//current step in ColSeq
	while(vecSize.d_ptr[0] > 1U && INDEX < Colsteps){

		//apply collatz to the running sample
		if(rank == 0){
			if(num64.d_ptr[0] & 1UL){
				mul64b3(num64.d_ptr, num64.d_ptr[numLength - 1]);
				add64b1(num64.d_ptr, num64.d_ptr[numLength - 1]);
			}
			else{
				div64b2(num64.d_ptr, num64.d_ptr[numLength - 1]);
			}
		}

		//apply collatz to all the samples in sampleVec
		if(rank < vecSize.d_ptr[0]){
			if(sampleVec.d_ptr[rank][0] & 1UL){
				mul64b3(sampleVec.d_ptr[rank], sampleVec.d_ptr[rank][numLength - 1]);
				add64b1(sampleVec.d_ptr[rank], sampleVec.d_ptr[rank][numLength - 1]);
			}
			else{
				div64b2(sampleVec.d_ptr[rank], sampleVec.d_ptr[rank][numLength - 1]);
			}
		}

		grid.sync();

		//Sort the samples
		SE_CUDA::deviceSort<unsigned long int*, SampleCompare<unsigned long int>>(sampleVec.d_ptr, tempVec.d_ptr, vecSize.d_ptr[0], rank, worldSize, grid, numLength);
		
		grid.sync();

		//mark modes
		if(rank < vecSize.d_ptr[0] - 1){
			modeCheck.d_ptr[rank] = compare64(sampleVec.d_ptr[rank], sampleVec.d_ptr[rank + 1], numLength);
		}
		
		grid.sync();
		
		//rank 0 iterates and finds the least greatest mode
		if(rank == 0){
			unsigned int currentFrequency = 1; 	//current frequency of veiwed number
			unsigned int maxFrequency = 1; 		//maximum frequency found
			unsigned int modeIndex = 0; 		//index of the number with the max frequency

			//find max frequency and the index of the mode with the max frequency
			for (unsigned int i = 0; i < vecSize.d_ptr[0] - 1 ; i++){
				if (modeCheck.d_ptr[i]) currentFrequency++;
				else currentFrequency = 1;

				if (currentFrequency > maxFrequency){
					maxFrequency = currentFrequency;
					modeIndex = i + 1;
				}
			}
			
			//if a mode was found, record it into the table and delete duplicates
			if(modeIndex != 0){

				//record start index if not done so already
				if (!startIndexFound){
					tbInfos.d_ptr[0].startIndex = INDEX;
					startIndexFound = true;
				}

				//add the mode found to the cumulative record
				tbInfos.d_ptr[0].totalReplaced++;

				//delete duplicates of the mode
				for(unsigned int i = 0; i < maxFrequency - 1; i++){
					unsigned long int* temp = sampleVec.d_ptr[vecSize.d_ptr[0] - i - 1];
					sampleVec.d_ptr[vecSize.d_ptr[0] - i - 1] = sampleVec.d_ptr[modeIndex - (i + 1)];
					sampleVec.d_ptr[modeIndex - (i + 1)] = temp;
				}
				vecSize.d_ptr[0] -= maxFrequency - 1;

				//record the mode to the table
				recordToTable(sampleVec.d_ptr[modeIndex], ColSeq, INDEX, numLength);
				for(unsigned int i = 0; i < numLength; i++){
					num64.d_ptr[i] = sampleVec.d_ptr[modeIndex][i];
				}
			}
			
			//if no mode is found, simply record the running sample to the table
			else{
				recordToTable(num64.d_ptr, ColSeq, INDEX, numLength);
			}
		}

		grid.sync();

		//iterate to the next index of the table
		INDEX++;
	}

	//record max index reached
	if(rank == 0) tbInfos.d_ptr[0].stopIndex = INDEX - 1;
}

/**
 * generate lookup table with one number, saving all numbers in the sequence as it goes, storing it in ColSeq and ColSeqSizes
 * 
 * @param num64 number to build the ColSeq table with !IS DESTRUCTIVE!
 * @param size size of the initial number
 * @param ColSeq 2D sequential array to store the subsequent num64s
 * @warning changes the information in num64
*/
__host__ void CollatzSteps(SE_CUDA::Cuda_1D<unsigned long int>&num64, SE_CUDA::Cuda_2D<unsigned long int>& ColSeq){

	unsigned long int temp64[num64.size()] = {0UL};

	for (int i = 0; i < num64.size(); i++){
		temp64[i] = num64[i];
	}

	int steps = 0; // keep track of number of steps

    while (! (( temp64[num64.size()-1] == 0UL) && (temp64[0] == 1UL)) ){
        // if number odd, perform 3x+1
        if (temp64[0] & 1UL){
            mul64b3(temp64, temp64[num64.size()-1]);
            add64b1(temp64, temp64[num64.size()-1]);
        }
        // else the number is even, perform /2
        else{
            div64b2(temp64, temp64[num64.size()-1]);
		}

        // copy the number over to array after each step in the process
        // note we only go out to binsize, so the 2D array better be zeroed out first!
        recordToTable(temp64, ColSeq, steps, num64.size());

        steps ++;
    }

	//printf("CollatzSteps: %i\n", steps);

	//	return steps;
}


/**
 * Host side record a num64 ID to the lookup table
 * 
 * @param num64 number to record
 * @param ColSeq lookup table to record to
 * @param step row to record to
 * @param numlength length of the num64 container
 */
__host__ void recordToTable(unsigned long int* num64, SE_CUDA::Cuda_2D<unsigned long int>& ColSeq, unsigned int step, unsigned int numlength){

	unsigned long int size = num64[numlength-1];

	ColSeq[step][0] = num64[0];
	if (size > 0UL) ColSeq[step][1] = num64[1];
	if (size > 1UL) ColSeq[step][2] = num64[size - 1];
	if (size > 2UL) ColSeq[step][3] = num64[size];
	ColSeq[step][4] = size;
}

/**
 * Device side record a num64 ID to the lookup table
 * 
 * @param num64 number to record
 * @param ColSeq binding of lookup table to record to
 * @param step row to record to
 * @param numlength length of the num64 container
 */
__device__ inline void recordToTable(unsigned long int* num64, SE_CUDA::Binding<unsigned long int> ColSeq, unsigned int step, unsigned int numlength){

	unsigned long int size = num64[numlength-1];

	SE_CUDA::access2D<unsigned long int>(ColSeq.d_ptr, step, 0, ColSeq.pitch) = num64[0];
	if(size > 0UL)SE_CUDA::access2D<unsigned long int>(ColSeq.d_ptr, step, 1, ColSeq.pitch) = num64[1];
	if(size > 1UL) SE_CUDA::access2D<unsigned long int>(ColSeq.d_ptr, step, 2, ColSeq.pitch) = num64[size-1];
	if(size > 2UL) SE_CUDA::access2D<unsigned long int>(ColSeq.d_ptr, step, 3, ColSeq.pitch) = num64[size];
	SE_CUDA::access2D<unsigned long int>(ColSeq.d_ptr, step, 4, ColSeq.pitch) = size;
}

/**
 * generate Collatz sequence, stopping when the given num64 matches with an entry in the lookup table
 * 
 * @param num64 number to sequence and compare with the lookup table !IS DESTRUCTIVE!
 * @param numLength size of the num64 array
 * @param ColSeq lookup table
 * @param ColSteps size of the lookup table
 * @param steps optional pointer to keep track of steps internally
 * @param shColSeq optional pointer to shared memory storing the early lookup table entries
 * @param shRows optional number of rows of the lookup table stored in the shared memory
 * @returns Whether the given num64 matches with the lookup table
*/
__device__ bool CollatzCompare(unsigned long int* num64,
	unsigned int numLength,
	SE_CUDA::Binding<unsigned long int> ColSeq,
	unsigned int ColSteps,
	unsigned int* _steps,
	const unsigned long int* shColSeq,
	unsigned int shRows
){

    unsigned int steps = 0U; // the step count variable
    bool cv = false; // compare value between current step and base step

    while (! (( num64[numLength-1] == 0UL) && (num64[0] == 1UL)) ){

        // if number odd, perform 3x+1
        if (num64[0] & 1UL){
            mul64b3(num64, num64[numLength-1]);
            add64b1(num64, num64[numLength-1]);
        }
        // else the number is even, perform /2
        else{
            div64b2(num64, num64[numLength-1]);
        }
		
        // Compare using shared memory if the requested row is cached
        unsigned long int tableSize;
        unsigned long int b0, b1, b2, b3;

        if (steps < shRows && shColSeq != nullptr) {
            // read entire row from shared: layout row*numCols + col
            const unsigned int base = steps * 5U;
            b0 = shColSeq[base + 0];
            b1 = shColSeq[base + 1];
            b2 = shColSeq[base + 2];
            b3 = shColSeq[base + 3];
            tableSize = shColSeq[base + 4];
        } else {
            // fallback to global
            b0 = SE_CUDA::read2D<unsigned long int>(ColSeq.d_ptr, steps, 0, ColSeq.pitch);
            b1 = SE_CUDA::read2D<unsigned long int>(ColSeq.d_ptr, steps, 1, ColSeq.pitch);
            b2 = SE_CUDA::read2D<unsigned long int>(ColSeq.d_ptr, steps, 2, ColSeq.pitch);
            b3 = SE_CUDA::read2D<unsigned long int>(ColSeq.d_ptr, steps, 3, ColSeq.pitch);
            tableSize = SE_CUDA::read2D<unsigned long int>(ColSeq.d_ptr, steps, 4, ColSeq.pitch);
        }

        // quick size check then element checks (same as compareToTable)
        unsigned long int size = num64[numLength-1];
        if (tableSize != size) {
            cv = false;
        } else {
            cv = (b0 == num64[0]);
            if (cv && size > 0UL) cv = (b1 == num64[1]);
            if (cv && size > 1UL) cv = (b2 == num64[size-1]);
            if (cv && size > 2UL) cv = (b3 == num64[size]);
        }
		
        // if the current number and that in ColSeq[steps] are the same, gather data and stop!
        if (cv){
			if(_steps != nullptr) *_steps = steps + 1;
            steps = ColSteps;
            break;
        }

        // if we have gotten all the way to number of steps for base number and  cv != 1, then we need to stop
        // as this is a break!
        else if (steps >= ColSteps)
        {
			if(_steps != nullptr) *_steps = steps + 1;
            steps = 0U;
            break;
        }

		steps++;
    }

     //printf("CollatzCompareSteps: %i\n", steps);
    return (steps == ColSteps);

}

/**
 * Parses an input argument in the form x(y) with x being a multiplier and y being a power of 2
 * 
 * @param carr character array from the arguments
 * @returns the multiplier and power of the offset
*/
__host__ Offset parseOffset(std::string carr){
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
		std::cerr << "missing a brace on an offset, unable to parse " << carr << "\n";
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
 * Apply offset to a long long
 * @param num long long to apply the offset to
 * @param multiplier multiplier of the offset
 * @param power power of the offset
 * @returns size of the number in the num64
*/
__host__ __device__ long long applyOffset(long long num, unsigned int multiplier, unsigned int power){
    for(unsigned int i = 0; i < multiplier; i++){
		long long tempThresh = 1LL;
		for(unsigned int j = 0; j < power; j++){
			tempThresh *= 2LL;
		}
		num += tempThresh;
	}
    return num;
}

/**
 * Apply offset to a num64
 * @param num64 num64 array to apply offset to
 * @param numLength length of the num64 array
 * @param multiplier multiplier of the offset
 * @param power power of the offset
 * @returns size of the number in the num64
*/
__host__ __device__ void applyOffset(unsigned long int* num64, unsigned int numLength, unsigned int multiplier, unsigned int power){

	for(unsigned int i = 0; i < multiplier; i++){
		addPow2UL64(num64, power, num64[numLength-1]);
	}

}

/** 
 * generate Collatz sequence for binary number num64 of length size, return number of steps
 * 
 * @param num64 number to sequence !IS DESTRUCTIVE!
 * @param numLength length of the array
 * @returns Number of steps through the Collatz sequence
 * @warning changes the information in num64
*/
__host__ __device__ unsigned int Collatz(unsigned long int num64[], unsigned long int& size){
	unsigned int steps = 0; // just keep track of the number of steps!
	while (! (( size == 0UL) && (num64[0] == 1UL)) ){
        steps++;

        // if number odd, perform 3x+1
         if (num64[0] & 1UL){
			
            mul64b3(num64, size);
			
            add64b1(num64, size);
             
		}
            // else the number is even, perform /2
         else{
			
            div64b2(num64, size);

        }

    }

    return steps;
}

/**
 * Multiply the num64 by 3.
 * @param num64 base 2^64 number to multiply
 * @param size reference to the size of the number
*/

__host__ __device__ __forceinline__ void mul64b3(unsigned long int num64[], unsigned long int& size){
	unsigned long int sum = 0UL;
	unsigned long int currentDigit;
	unsigned long int size_hold = size;
	unsigned long int carry_in = 0UL;
	unsigned long int carry_out = 0UL;

	for(unsigned int i = 0; i <= size_hold; i++){

		currentDigit = num64[i];

		carry_out = (currentDigit >> 63);

		sum = ((currentDigit << 1) + currentDigit);

		carry_out += (sum < (currentDigit << 1));

		currentDigit = sum + carry_in;

		carry_out += (currentDigit < sum);

		num64[i] = currentDigit;

		carry_in = carry_out;
	}

	if (carry_in){
		size_hold++;
		size = size_hold;
		num64[size_hold] = carry_in;
	}
}

/**
 * Multiply the num64 by 3.
 * @param num64 base 2^64 number to multiply
 * @param size reference to the size of the number
*/
/*
__host__ __device__ __forceinline__ void mul64b3(unsigned long int num64[], unsigned long int& size){

    unsigned long int currc; // for the current carry
    unsigned long int nextc; // for the next carry

	// lets perform 3x+1 now, for 2^64 base, its easier to do 3x first, then go back and add 1.
	currc = 0; // start with current carry = 0;
	nextc = 0; // start with next carry = 0;

	for(unsigned long int i = 0; i <= size; i++){
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
}

*/

/**
 * Divide num64 by 2.
 * @param num64 number to divide
 * @param size reference to the size of number
 * @returns size of number
*/

__host__ __device__ __forceinline__ void div64b2(unsigned long int* num64, unsigned long int& size){

	unsigned long int size_hold = size;
	unsigned long int carry_in = 0UL;
	unsigned long int carry_out = 0UL;
	unsigned long int currentDigit;

	for(unsigned int i = 0U; i <= size_hold; i++){
		currentDigit = num64[size_hold - i];
		carry_out = currentDigit & 1UL;
		currentDigit = (currentDigit >> 1) | (carry_in << 63);
		carry_in = carry_out;
		num64[size_hold - i] = currentDigit;
	}

	if(size_hold != 0UL && num64[size_hold] == 0UL){
		size--;
	}
}

/**
 * Divide num64 by 2.
 * @param num64 number to divide
 * @param size reference to the size of number
 * @returns size of number
*/
/*
__host__ __device__ __forceinline__ void div64b2(unsigned long int* num64, unsigned long int& size){

	// we start with the largest entry first, to determine if zeros out
	unsigned long int drop = 0; // if current entry is even, this is 0, if it is odd it is 1, used to carry over to next entry

	// other odd number, will have a remainder to carry over to next lowest entry

	for (unsigned long int i = 0; i<=size; i++){
		// if number is odd, we have to add some 2^63 to the next entry's result due to the carry (drop)
		if (num64[size-i] & 1UL)
		{
			num64[size-i] = (num64[size-i]>>1) + drop*ULDIV2;
			drop = 1;
		}
		// else the number is even, /2 works out evenly for this entry, no need to add a carry (drop)
		else
		{
			num64[size-i] = (num64[size-i]>>1) + drop*ULDIV2;
			drop = 0;
		}
	}
	if (num64[size] == 0)
	{
		size--;
	}

}
*/

/**
 * checks if two num64s are the same or not
 * @warning !ONLY USE IF THE NUMBERS ARE THE SAME SIZE!
 * @param num0 the first number
 * @param num1 the second number
 * @param numLength length of the number array
 * @returns boolean of equality
*/
__host__ __device__ bool compare64(unsigned long int* num0, unsigned long int* num1, unsigned int numLength){
    
	if(num0[numLength-1 ] ^ num1[numLength-1]){
		return false;
	}

	unsigned long int i = 0;
    while (i <= num0[numLength-1])
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
__host__ void addPow2UL64(SE_CUDA::Cuda_1D<unsigned long int>& num64, unsigned long int valExp, unsigned long int& size){

    unsigned long int start = valExp/64; //where to add the power of 2
    unsigned long int entrybitshift = (valExp & 63); //remainder to put in entries position of 2^64

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

	unsigned long int i = start + 1;
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
		size = i;
	}
}

/**
 * Add a power of two to num64
 * @param num64 number to add to
 * @param valExp the power of two to add
 * @param size size of number
 * @returns size of number
*/
__host__ __device__ void addPow2UL64(unsigned long int* num64, int valExp, unsigned long int& size){

    unsigned long int start = valExp/64; //where to add the power of 2
    unsigned long int entrybitshift = (valExp & 63); //remainder to put in entries position of 2^64

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

	unsigned long int i = start + 1;
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
		size = i;
	}
}

/**
 * Add a number to the num64
 * @param num64 number to add to
 * @param val number to add
 * @param size size of number
 * @returns size of number
*/
__host__ void addUL64(SE_CUDA::Cuda_1D<unsigned long int>& num64, unsigned long int val, unsigned long int& size)
{
	unsigned long int i = 1UL;
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
			num64[i] = 0UL;
			i++;
		}
		num64[i] = num64[i] + 1;
	}
	// if we went over the current size limit, update!
	if (i > size)
	{
		size = i;
	}
}

/**
 * Add a number to the num64
 * @param num64 number to add to
 * @param val number to add
 * @param size size of number
 * @returns size of number
*/
__host__ __device__ void addUL64(unsigned long int* num64, unsigned long int val, unsigned long int& size)
{
	unsigned long int i = 1;
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
		size = i;
	}
}

/**
 * increment num64 by 1, usable on both host and device
 * @param num64 number to increment
 * @param size size of number
 * @returns size of number
*/
__host__ __device__ __forceinline__ void add64b1(unsigned long int* num64, unsigned long int& size){
	unsigned long int i = 0;
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
		size = i;
	}
}

/**
 * prints num64
 * @param num64 the number to print
 * @param size size of the number
*/
__host__ void print64(SE_CUDA::Cuda_1D<unsigned long int>& num, unsigned long int& size)
{

	for(unsigned long int i = 0; i <= size; i++)
	{
		printf("b64[%lu]: %lu\n", i, num[i]);
        }
	printf("\n");

}

/**
 * prints num64
 * @param num64 the number to print
 * @param size size of the number
*/
__host__ __device__ void print64(unsigned long int num[], unsigned long int& size)
{

	for(unsigned long int i = 0; i <= size; i++)
	{
		printf("b64[%lu]: %lu\n", i, num[i]);
        }
	printf("\n");
}