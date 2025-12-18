#pragma once
#include "Device.cuh"
#include <vector>
#include <cassert>
#include <stdexcept>
#include <type_traits>
#include <bits/stdc++.h>
#include <string>

namespace SE_CUDA
{
    /**
     * used to keep track of bindings
     * @tparam T type of the binding
     *
     * @param d_ptr device handle
     * @param device device that is bound to
     */
    template <typename T>
    struct Binding
    {
        T *d_ptr;
        Device *device;
        size_t pitch;
    };

    /**
     * 1D array, used to bind to one or more device handles.
     *
     * @tparam T type to build Cuda_1D with.
     *
     * Enforces one-to-many relationship between CPU and multiple GPUs, with one copy on the CPU able to push to and pull from one
     * or multiple copies located in Device memory.
     */
    template <typename T>
    class Cuda_1D
    {

    public:
        /**
         * Constructor
         * @param size number of elements in the array
         * @throws bad_alloc
         */
        Cuda_1D(unsigned int size)
        {
            _local_ptr = (T *)malloc(sizeof(T) * size);
            if (_local_ptr == nullptr)
                throw std::bad_alloc();
            _size = size;
        }

        /**
         * Destructor
         */
        ~Cuda_1D()
        {
            unbindAll();
            free(_local_ptr);
        }

        // Disable Copy Operator
        Cuda_1D(const Cuda_1D &) = delete;
        Cuda_1D &operator=(const Cuda_1D &) = delete;

        /**
         * access operator to mimic array access
         * @returns element in that index of the array
         */
        T &operator[](unsigned int index)
        {
            assert(index < _size);
            return _local_ptr[index];
        }

        /**
         * access operator to mimic array access
         * @returns element in that index of the array
         */
        const T &operator[](unsigned int index) const
        {
            assert(index < _size);
            return _local_ptr[index];
        }

        /**
         * Number of elements in the array
         * @returns number of elements in the array
         */
        unsigned int size()
        {
            return _size;
        }

        /**
         * checks to see if a given binding belongs to this Cuda_1D
         * @param binding Data::Binding<T> to check
         * @returns true if binding belongs to this Cuda_1D
         */
        bool hasBinding(Binding<T> binding)
        {
            for (int i = 0; i < _bindings.size(); i++)
            {
                if (binding.d_ptr == _bindings[i].d_ptr && binding.device == _bindings[i].device)
                {
                    return true;
                }
            }
            return false;
        }

        /**
         * bind to a device, allocating memory on that device
         * @param device device to bind to
         * @returns Data::Binding<T> that contains information about the binding
         * @warning do not delete or modify device handle, use unbind() instead
         * @throws runtime error
         */
        Binding<T> bind(Device *device)
        {
            T *d_ptr;
            device->select();
            cudaError_t error = cudaMalloc((void **)&d_ptr, sizeof(T) * _size);
            if (error != cudaSuccess)
                throw std::runtime_error(std::string("Cuda_1D: Error while binding:\n") + cudaGetErrorString(error));
            Binding<T> binding = {d_ptr, device, 0};
            _bindings.push_back(binding);

            return binding;
        }

        /**
         * unbind given device handle from given device
         * @param binding the binding to unbind
         * @throws runtime error
         */
        void unbind(Binding<T> binding)
        {
            for (int i = 0; i < _bindings.size(); i++)
            {
                if (_bindings[i].d_ptr == binding.d_ptr && _bindings[i].device == binding.device)
                {
                    binding.device->select();
                    cudaError_t error = cudaFree(binding.d_ptr);
                    if (error != cudaSuccess)
                        throw std::runtime_error(std::string("Cuda_1D: Error while unbinding, check for a double free:\n") + cudaGetErrorString(error));
                    _bindings.erase(_bindings.begin() + i);
                }
            }
        }

        /**
         * unbinds all bound device handles for this array
         * @throws runtime error
         */
        void unbindAll()
        {
            for (int i = 0; i < _bindings.size(); i++)
            {
                _bindings[i].device->select();
                cudaError_t error = cudaFree(_bindings[i].d_ptr);
                if (error != cudaSuccess)
                    throw std::runtime_error(std::string("Cuda_1D: Error while unbinding, check for a double free:\n") + cudaGetErrorString(error));
            }
            _bindings.clear();
        }

        /**
         * push data to a given device handle on a given device
         * @param binding the binding to push to
         * @throws runtime error
         */
        void push(Binding<T> binding)
        {
            if (hasBinding(binding))
            {
                binding.device->select();
                cudaError_t error = cudaMemcpy(
                    binding.d_ptr,
                    _local_ptr,
                    sizeof(T) * _size,
                    cudaMemcpyHostToDevice);
                if (error != cudaSuccess)
                    throw std::runtime_error(std::string("Cuda_1D: Error while pushing to device, check bindings\n") + cudaGetErrorString(error));
            }
            else
                throw std::runtime_error("Cuda_1D: Binding given for push does not exist!\n");
        }

        /**
         * push data to all bindings
         * @throws runtime error
         */
        void pushAll()
        {
            for (int i = 0; i < _bindings.size(); i++)
            {
                _bindings[i].device->select();
                cudaError_t error = cudaMemcpy(
                    _bindings[i].d_ptr,
                    _local_ptr,
                    sizeof(T) * _size,
                    cudaMemcpyHostToDevice);
                if (error != cudaSuccess)
                    throw std::runtime_error(std::string("Cuda_1D: Error while pushing to all devices, check bindings\n") + cudaGetErrorString(error));
            }
        }

        /**
         * pull data from a given device handle and given device
         * @param binding the binding to pull from
         * @throws runtime error
         */
        void pull(Binding<T> binding)
        {
            if (hasBinding(binding))
            {
                binding.device->select();
                cudaError_t error = cudaMemcpy(
                    _local_ptr,
                    binding.d_ptr,
                    sizeof(T) * _size,
                    cudaMemcpyDeviceToHost);
                if (error != cudaSuccess)
                    throw std::runtime_error(std::string("Cuda_1D: Error while pulling from device, check bindings\n") + cudaGetErrorString(error));
            }
            else
                throw std::runtime_error("Cuda_1D: Binding to pull from does not exist!\n");
        }

    private:
        T *_local_ptr;                     // local array
        unsigned int _size;                // number of elements in the array
        std::vector<Binding<T>> _bindings; // bindings linked to this array
    };

    /**
     * 2D array, used to bind to one or more device handles. Only allows primitive types.
     *
     * @tparam T type to build Cuda_2D with. Must be a primitive type, or a typedef of one.
     *
     * Enforces one-to-many relationship between CPU and multiple GPUs, with one copy on the CPU able to push to and pull from one
     * or multiple copies located in Device memory.
     */
    template <typename T>
    class Cuda_2D
    {
        static_assert(std::is_arithmetic_v<T>, "Cuda_1D<T> requires an arithmetic (primitive) type");

    public:
        /**
         * Proxy class to enable [x][y] access of the 2d array on the host side
         */
        class RowProxy
        {
        public:
            RowProxy(T *row_start) : _row_start(row_start) {}

            T &operator[](size_t col)
            {
                return _row_start[col];
            }

        private:
            T *_row_start;
        };

        /**
         * Constructor
         * @param row height of the array
         * @param col width of the array
         */
        Cuda_2D(unsigned int row, unsigned int col)
        {
            _local_ptr = (T *)malloc(sizeof(T) * row * col);
            _col = col;
            _row = row;
        }

        /**
         * Destructor
         */
        ~Cuda_2D()
        {
            unbindAll();
            free(_local_ptr);
        }

        // Disable Copy Operator
        Cuda_2D(const Cuda_2D &) = delete;
        Cuda_2D &operator=(const Cuda_2D &) = delete;

        /**
         * access operator to mimic array access
         * @returns element in that index of the array
         */
        RowProxy operator[](unsigned int row)
        {
            assert(row < _row);
            return RowProxy(_local_ptr + row * _col);
        }

        /**
         * access operator to mimic array access
         * @returns element in that index of the array
         */
        const RowProxy operator[](unsigned int row) const
        {
            assert(row < _row);
            return RowProxy(_local_ptr + row * _col);
        }

        /**
         * @returns height of the array
         */
        unsigned int getRows()
        {
            return _row;
        }

        /**
         * @returns width of the array
         */
        unsigned int getCols()
        {
            return _col;
        }

        /**
         * checks to see if a given binding belongs to this Cuda_2D
         * @param binding Data::Binding<T> to check
         * @returns true if binding belongs to this Cuda_2D
         */
        bool hasBinding(Binding<T> binding)
        {
            for (int i = 0; i < _bindings.size(); i++)
            {
                if (binding.d_ptr == _bindings[i].d_ptr && binding.device == _bindings[i].device)
                {
                    return true;
                }
            }
            return false;
        }

        /**
         * bind to a device, allocating memory on that device
         * @param device device to bind to
         * @returns Data::Binding<T> that contains information about the binding
         * @warning do not delete or modify device handle, use unbind() instead
         * @throws runtime error
         */
        Binding<T> bind(Device *device)
        {
            T *d_ptr;
            size_t pitch;
            device->select();
            cudaError_t error = cudaMallocPitch((void **)&d_ptr, &pitch, sizeof(T) * _col, _row);
            if (error != cudaSuccess)
                throw std::runtime_error(std::string("Cuda_2D: Error while binding:\n") + cudaGetErrorString(error));
            Binding<T> binding = {d_ptr, device, pitch};
            _bindings.push_back(binding);

            return binding;
        }

        /**
         * unbind given device handle from given device
         * @param binding the binding to unbind
         * @throws runtime error
         */
        void unbind(Binding<T> binding)
        {
            for (int i = 0; i < _bindings.size(); i++)
            {
                if (_bindings[i].d_ptr == binding.d_ptr && _bindings[i].device == binding.device)
                {
                    binding.device->select();
                    cudaError_t error = cudaFree(binding.d_ptr);
                    if (error != cudaSuccess)
                        throw std::runtime_error(std::string("Cuda_2D: Error while unbinding, check for a double free:\n") + cudaGetErrorString(error));
                    _bindings.erase(_bindings.begin() + i);
                }
            }
        }

        /**
         * unbinds all bound device handles for this array
         * @throws runtime error
         */
        void unbindAll()
        {
            for (int i = 0; i < _bindings.size(); i++)
            {
                _bindings[i].device->select();
                cudaError_t error = cudaFree(_bindings[i].d_ptr);
                if (error != cudaSuccess)
                    throw std::runtime_error(std::string("Cuda_2D: Error while unbinding, check for a double free:\n") + cudaGetErrorString(error));
            }
            _bindings.clear();
        }

        /**
         * push data to a given device handle on a given device
         * @param binding the binding to push to
         * @throws runtime error
         */
        void push(Binding<T> binding)
        {
            if (hasBinding(binding))
            {
                binding.device->select();
                cudaError_t error = cudaMemcpy2D(
                    binding.d_ptr,
                    binding.pitch,
                    _local_ptr,
                    sizeof(T) * _col,
                    sizeof(T) * _col,
                    _row,
                    cudaMemcpyHostToDevice);
                if (error != cudaSuccess)
                    throw std::runtime_error(std::string("Cuda_2D: Error while pushing to device, check bindings\n") + cudaGetErrorString(error));
            }
            else
                throw std::runtime_error("Cuda_2D: Binding given for push does not exist!\n");
        }

        /**
         * push data to all bindings
         * @throws runtime error
         */
        void pushAll()
        {
            for (int i = 0; i < _bindings.size(); i++)
            {
                _bindings[i].device->select();
                cudaError_t error = cudaMemcpy2D(
                    _bindings[i].d_ptr,
                    _bindings[i].pitch,
                    _local_ptr,
                    sizeof(T) * _col,
                    sizeof(T) * _col,
                    _row,
                    cudaMemcpyHostToDevice);
                if (error != cudaSuccess)
                    throw std::runtime_error(std::string("Cuda_2D: Error while pushing to all devices, check bindings\n") + cudaGetErrorString(error));
            }
        }

        /**
         * pull data from a given device handle and given device
         * @param binding the binding to pull from
         * @throws runtime error
         */
        void pull(Binding<T> binding)
        {
            if (hasBinding(binding))
            {
                binding.device->select();
                cudaError_t error = cudaMemcpy2D(
                    _local_ptr,
                    sizeof(T) * _col,
                    binding.d_ptr,
                    binding.pitch,
                    sizeof(T) * _col,
                    _row,
                    cudaMemcpyDeviceToHost);
                if (error != cudaSuccess)
                    throw std::runtime_error(std::string("Cuda_2D: Error while pulling from device, check bindings\n") + cudaGetErrorString(error));
            }
            else
                throw std::runtime_error("Cuda_2D: Binding to pull from does not exist!\n");
        }

    private:
        T *_local_ptr;
        unsigned int _col;
        unsigned int _row;
        std::vector<Binding<T>> _bindings;
    };

    /**
     * Allocates a unique array for each thread of a given thread number
     *
     * @tparam T type to be stored in the containers
     *
     * Creates an array of pointers to more arrays inside of the device, creating an array per increment of given threadcount
     */
    template <typename T>
    class CudaUniqueAlloc
    {
    public:
        /**
         * Constructor
         * @param device device to allocate on
         * @param size size of the array to allocate
         * @param threadCount number of threads to allocate for
         * @throws runtime error
         */
        CudaUniqueAlloc(Device *device, unsigned int size, unsigned int threadCount)
        {
            _device = device;
            device->select();
            _thread_count = threadCount;
            _size = size;

            // allocate pointer arrays
            _local_ptr = (T **)malloc(sizeof(T *) * threadCount);
            cudaError_t error = cudaMalloc((void **)&_d_ptr, sizeof(T *) * _thread_count);
            if (error != cudaSuccess)
                throw std::runtime_error(std::string("CudaUniqueAlloc: Error while binding:\n") + cudaGetErrorString(error));

            // allocate and record actual arrays
            for (unsigned int i = 0; i < threadCount; i++)
            {
                T *d_ptr;
                error = cudaMalloc((void **)&d_ptr, sizeof(T) * _size);
                if (error != cudaSuccess)
                    throw std::runtime_error(std::string("CudaUniqueAlloc: Error while binding:\n") + cudaGetErrorString(error));
                _local_ptr[i] = d_ptr;
            }

            // send pointers to device array
            error = cudaMemcpy(
                _d_ptr,
                _local_ptr,
                sizeof(T *) * _thread_count,
                cudaMemcpyHostToDevice);
            if (error != cudaSuccess)
                throw std::runtime_error(std::string("CudaUniqueAlloc: Error while pushing to device, check bindings\n") + cudaGetErrorString(error));
        }

        /**
         * Destructor
         */
        ~CudaUniqueAlloc()
        {
            _device->select();

            // free arrays
            for (unsigned int i = 0; i < _thread_count; i++)
            {
                cudaFree((void *)_local_ptr[i]);
            }

            // free device pointer array
            cudaFree((void *)_d_ptr);

            // free host pointer array
            free(_local_ptr);
        }

        // Disable Copy Operator
        CudaUniqueAlloc(const CudaUniqueAlloc &) = delete;
        CudaUniqueAlloc &operator=(const CudaUniqueAlloc &) = delete;

        /**
         * retreive the binding for this alloc
         * @returns binding for this alloc
         */
        Binding<T *> getBinding()
        {
            return {_d_ptr, _device, 0};
        }

        /**
         * checks if the given binding is for this alloc
         * @returns true if the binding belongs to this alloc
         */
        bool hasBinding(Binding<T *> binding)
        {
            return (_d_ptr == binding.d_ptr && _device == binding.device);
        }

        /**
         * Number of arrays that belong to this alloc
         * @returns number of arrays in the alloc
         */
        unsigned int numberOfArrays()
        {
            return _thread_count;
        }

        /**
         * size of each array in the alloc
         * @returns size of the arrays in the alloc
         */
        unsigned int arraySize()
        {
            return _size;
        }

    private:
        unsigned int _size;         // size of the arrays
        unsigned int _thread_count; // number of threads allocate for, or number of arrays created
        Device *_device;            // bound device
        T **_d_ptr;                 // device handle
        T **_local_ptr;             // host handle
    };

    /**
     * Device side access for Cuda2D arrays
     *
     * @tparam T type of the array
     * @param arr pointer to the 2D array
     * @param row row to access
     * @param col column to access
     * @param pitch pitch given by binding
     *
     * @return read/write reference for index
     */
    template <typename T>
    static inline __device__ T &access2D(T *arr, unsigned int row, unsigned int col, size_t pitch)
    {
        T *temprow = (T *)((char *)arr + (row * pitch));
        return temprow[col];
    }

     /**
     * Device side read only access for Cuda2D arrays
     *
     * @tparam T type of the array
     * @param arr pointer to the 2D array
     * @param row row to access
     * @param col column to access
     * @param pitch pitch given by binding
     *
     * @return read only reference for index
     */
    template <typename T>
    static inline __device__ T read2D(T *arr, unsigned int row, unsigned int col, size_t pitch)
    {
        T *temprow = (T *)((char *)arr + (row * pitch));
        return __ldg(&temprow[col]);
    }
}