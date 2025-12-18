#pragma once
#include <memory>
#include <vector>
#include <stdexcept>
#include <cassert>

namespace SE_CUDA
{
    /**
     * Used to store grid info for the device
     * @param blockCount suggested block amount
     * @param threadCount suggest thread per block
     */
    struct Grid
    {
        unsigned int blockCount;  // block
        unsigned int threadCount; // thread
    };

    /**
     * Used to store device profile
     *
     * Profiles connected GPUs and allows for easy information collection and selection of devices
     */
    class Device
    {
    public:
        /**
         * CONSTRUCTOR
         * @param id id of the device
         * @throws runtime error
         */
        Device(unsigned int id)
        {
            _id = id;

            select();
            cudaGetDeviceProperties(&_properties, id);
            cudaStreamCreateWithFlags(&_stream, cudaStreamNonBlocking);

            unsigned int totalThreads = _properties.multiProcessorCount * _properties.maxThreadsPerMultiProcessor;
            _maxBlock = totalThreads / _properties.maxThreadsPerBlock;
            _maxBlockThreads = totalThreads / _maxBlock;
        }

        /**
         * DESTRUCTOR
         */
        ~Device() {
            cudaStreamDestroy(_stream);
        }

        // Disable Copy Operator
        Device(const Device &) = delete;
        Device &operator=(const Device &) = delete;


        /**
         * get suggested memory grid for this device
         * @returns suggested Device::Grid
         */
        Grid getGrid()
        {

            unsigned int threads = _maxBlockThreads;

            return {_maxBlock, threads};
        }

        /**
         * Get id of the device
         * @returns id of the device
         */
        unsigned int getId() { return _id; }

        /**
         * get device properties
         * @returns cudaDeviceProp struct
         */
        cudaDeviceProp getProperties()
        {
            return _properties;
        }

        /**
         * get used global memory on this device
         * @returns number of bytes currently used in global memory
         */
        size_t getUsedMemory()
        {
            select();

            size_t free_mem = 0;
            size_t total_mem = 0;

            cudaMemGetInfo(&free_mem, &total_mem);

            return total_mem - free_mem;
        }

        /**
         * get kernel stream for asynchronous kernel runs
         * @returns cuda stream for this device
         */
        cudaStream_t getStream(){
            return _stream;
        }

        /**
         * poll device to see if it is ready for more work
         * @returns whether the device is free or not
         */
        bool poll(){
            select();
            return (cudaStreamQuery(_stream) == cudaSuccess);
        }

        /**
         * select the device to be executed on
         * @throws runtime error
         */
        void select()
        {
            cudaError_t error = cudaSetDevice(_id);
            if (error != cudaSuccess)
                throw std::runtime_error("Device: ID does not exist.");
        }

    private:
        unsigned int _id;              // id of the device
        unsigned int _maxBlock;        // max blocks of the device
        unsigned int _maxBlockThreads; // max threads per block
        cudaDeviceProp _properties;    // device properties
        cudaStream_t _stream;
    };

    /**
     * initialize Device objects for each available GPU
     * @returns std::vector containing the initialized Device objects
     * @throws runtime error
     */
    std::vector<std::unique_ptr<Device>> initDevices()
    {
        // get device count
        int deviceCount = 0;
        cudaError_t device_error_id = cudaGetDeviceCount(&deviceCount);

        // error check
        if (device_error_id != cudaSuccess)
            throw std::runtime_error("Fatal Error: Initialization of devices failed!");
        if (deviceCount == 0)
            throw std::runtime_error("No devices found to initialize!");

        // create Device Vector
        std::vector<std::unique_ptr<Device>> devices;

        for (int i = 0; i < deviceCount; i++)
        {
            devices.push_back(std::make_unique<Device>(i));
        }

        return devices;
    }
}