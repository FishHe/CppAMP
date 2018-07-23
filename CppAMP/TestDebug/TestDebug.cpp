#include <stdio.h>  
#include <tchar.h>  
#include <amp.h>  

const int BLOCK_DIM = 32;

using namespace concurrency;

void sum_kernel_tiled(tiled_index<BLOCK_DIM> t_idx, array<int, 1> &A, int stride_size) restrict(amp)
{
	tile_static int localA[BLOCK_DIM];

	index<1> globalIdx = t_idx.global * stride_size;
	index<1> localIdx = t_idx.local;

	localA[localIdx[0]] = A[globalIdx];

	t_idx.barrier.wait();

	// Aggregate all elements in one tile into the first element.  
	for (int i = BLOCK_DIM / 2; i > 0; i /= 2)
	{
		if (localIdx[0] < i)
		{

			localA[localIdx[0]] += localA[localIdx[0] + i];
		}

		t_idx.barrier.wait();
	}

	if (localIdx[0] == 0)
	{
		A[globalIdx] = localA[0];
	}
}

int size_after_padding(int n)
{
	// The extent might have to be slightly bigger than num_stride to   
	// be evenly divisible by BLOCK_DIM. You can do this by padding with zeros.  
	// The calculation to do this is BLOCK_DIM * ceil(n / BLOCK_DIM)  
	return ((n - 1) / BLOCK_DIM + 1) * BLOCK_DIM;
}

int reduction_sum_gpu_kernel(array<int, 1> input)
{
	int len = input.extent[0];

	//Tree-based reduction control that uses the CPU.  
	for (int stride_size = 1; stride_size < len; stride_size *= BLOCK_DIM)
	{
		// Number of useful values in the array, given the current  
		// stride size.  
		int num_strides = len / stride_size;

		extent<1> e(size_after_padding(num_strides));

		// The sum kernel that uses the GPU.  
		parallel_for_each(extent<1>(e).tile<BLOCK_DIM>(), [&input, stride_size](tiled_index<BLOCK_DIM> idx) restrict(amp)
		{
			sum_kernel_tiled(idx, input, stride_size);
		});
	}

	array_view<int, 1> output = input.section(extent<1>(1));
	return output[0];
}

int cpu_sum(const std::vector<int> &arr) {
	int sum = 0;
	for (size_t i = 0; i < arr.size(); i++) {
		sum += arr[i];
	}
	return sum;
}

std::vector<int> rand_vector(unsigned int size) {
	srand(2011);

	std::vector<int> vec(size);
	for (size_t i = 0; i < size; i++) {
		vec[i] = rand();
	}
	return vec;
}

array<int, 1> vector_to_array(const std::vector<int> &vec) {
	array<int, 1> arr(vec.size());
	copy(vec.begin(), vec.end(), arr);
	return arr;
}

int _tmain(int argc, _TCHAR* argv[])
{
	std::vector<int> vec = rand_vector(10000);
	array<int, 1> arr = vector_to_array(vec);

	int expected = cpu_sum(vec);
	int actual = reduction_sum_gpu_kernel(arr);

	bool passed = (expected == actual);
	if (!passed) {
		printf("Actual (GPU): %d, Expected (CPU): %d", actual, expected);
	}
	printf("sum: %s\n", passed ? "Passed!" : "Failed!");

	getchar();

	return 0;
}
