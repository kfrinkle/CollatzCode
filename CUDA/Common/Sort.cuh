#pragma once
#include <cooperative_groups.h>

namespace SE_CUDA
{

    namespace Comp
    {
        /**
         * Comparitor for lesser to greater sorting
         * @tparam T type of the primitives to compare
         */
        template <typename T>
        struct LessertoGreater
        {
            __device__ bool operator()(T a, T b, unsigned int len = 0) const
            {
                return a < b;
            }
        };

        /**
         * Comparitor for greater to lesser sorting
         * @tparam T type of the primitives to compare
         */
        template <typename T>
        struct GreatertoLesser
        {
            __device__ bool operator()(T a, T b, unsigned len = 0) const
            {
                return a > b;
            }
        };
    }

    /**
     * Internal merge function used by deviceSort
     *
     * @tparam T type to be sorted
     * @tparam Comp comparitor to use
     *
     * @param vec array to sort
     * @param start index of the start of the sort chunk
     * @param mid index of the middle of the sort chunk
     * @param end index of the end of the sort chunk
     */
    template <typename T, typename Comp> __device__ void _deviceMerge(T *vec, T* tempVec, unsigned int start, unsigned int mid, unsigned int end, unsigned int numSize = 0)
    {
        Comp comp;

        int n1 = mid - start + 1;
        int n2 = end - mid;

        int i = 0;
        int j = 0;
        int k = start;

        while (i < n1 && j < n2)
        {
            if (comp(vec[start +i], vec[mid + 1 + j], numSize))
            {
                tempVec[k] = vec[start +i];
                i++;
            }
            else
            {
                tempVec[k] = vec[mid + 1 + j];
                j++;
            }
            k++;
        }

        while (i < n1)
        {
            tempVec[k] = vec[start + i];
            i++;
            k++;
        }

        while (j < n2)
        {
            tempVec[k] = vec[mid + 1 + j];
            j++;
            k++;
        }
    }

    /**
     * Merge sort an array in a device kernel
     *
     * @tparam T type of the array to sort
     * @tparam Comp optional - Comparitor, defaults to Comp::LessertoGreater<T>
     *
     * @param vec array to sort
     * @param len length of the array
     * @param rank local rank
     * @param worldSize total number of threads available
     * @param grid cooperative grid for syncing
     * @param -optional- numsize, size of the array object used in compares
     * 
     * By default sorts numbers from least to greatest.
     * 
     * Custom comparitors must take the form of:
     * 
     *  struct Custom{
     *      __device__ bool operator()(T a, T b, unsigned int numsize){
     * 
     *          ...Code to detemine ordering...
     * 
     *          return (ordering_comparison);
     *      } 
     *  };
     */
    template <typename T, typename Comp = Comp::LessertoGreater<T>> __device__ void deviceSort(T *vec, T* tempVec, unsigned int len, unsigned int rank, unsigned int worldSize, cooperative_groups::grid_group grid, unsigned int numSize = 0)
    {   T* holdVec = vec;
        T* temp;
        int maxSize = 2;
        Comp comp;
        while (maxSize < len)
        {

            unsigned int passes = len / (worldSize*maxSize) + 1;

            for (int i = 0; i < passes; i++)
            {
                int maxRank = (i == passes - 1) ? len%(worldSize*maxSize) + 1 : worldSize;
                if (rank < maxRank)
                {
                    int start = maxSize * rank + (i * maxSize * worldSize);
                    int end = min(start + maxSize - 1, len - 1);
                    int middle = min(start + (maxSize / 2) - 1, len - 2);
                    if (start < len && middle < len)
                        _deviceMerge<T, Comp>(vec, tempVec, start, middle, end, numSize);
                }
            }
            maxSize *= 2;
            grid.sync();

            temp = vec;
            vec = tempVec;
            tempVec = temp;
        }

        // final pass
        if (rank == 0)
        {
            _deviceMerge<T, Comp>(vec, tempVec, 0, min(maxSize / 2 - 1, len - 2), len - 1, numSize);

            temp = vec;
            vec = tempVec;
            tempVec = temp;

            if (vec != holdVec){
                temp = vec;
                vec = tempVec;
                tempVec = temp;

                for(int i = 0; i < len; i++){
                    vec[i] = tempVec[i];
                }
            }
        }

        grid.sync();
    }
}