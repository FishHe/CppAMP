#include <stdio.h>  
#include <tchar.h> 

#include <amp.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>

using namespace Concurrency;

//the size of matrix
#define N 2048

void default_properties() {
	accelerator default_acc;
	std::wcout << "Device Path: " << default_acc.device_path << "\n";
	std::wcout << "Dedicated Memory: " << default_acc.dedicated_memory << "\n";
	std::vector<accelerator> accs = accelerator::get_all();
	for (size_t i = 0; i < accs.size(); i++)
	{
		std::wcout << (accs[i].supports_cpu_shared_memory ?
			"CPU shared memory: true" : "CPU shared memory: false") << "\n";
		std::wcout << (accs[i].supports_double_precision ?
			"double precision: true" : "double precision: false") << "\n";
		std::wcout << (accs[i].supports_limited_double_precision ?
			"limited double precision: true" : "limited double precision: false") << "\n";
		std::wcout << (accs[i].is_debug ?
			"GPU can debug : ture" : "GPU can debug : false") << "\n";
	}
}

void pick_with_most_memory()
{
	std::vector<accelerator> accs = accelerator::get_all();
	accelerator acc_chosen = accs[0];
	//acc_chosen.set_default_cpu_access_type(access_type_none);
	for (size_t i = 0; i < accs.size(); i++)
	{
		if (accs[i].dedicated_memory > acc_chosen.dedicated_memory) {
			acc_chosen = accs[i];
		}
	}

	//set the default
	accelerator::set_default(acc_chosen.device_path);

	std::wcout << "The accelerator with the most memory: "<< acc_chosen.device_path << "\n";
	std::wcout << "Dedicated Memory: " << acc_chosen.dedicated_memory << "\n";
}

void pick_to_debug()
{
	////chose debug
	//accelerator defaultAcc(accelerator::default_accelerator);
	//accelerator_view defaultView = defaultAcc.default_view;

#ifndef DEBUG
	std::vector<accelerator> allAccelerators = accelerator::get_all();
	accelerator::set_default(accelerator::direct3d_warp);
	for (size_t i = 0; i < allAccelerators.size(); i++)
	{
		if (allAccelerators[i].device_path== accelerator::direct3d_warp)
		{
			allAccelerators[i].set_default_cpu_access_type(access_type_read_write);
		}
	}
	std::wcout << "Now use the direct3d_warp.\n";
	
	//allAccelerators.erase(std::remove_if(allAccelerators.begin(), allAccelerators.end(),
	//	[](const accelerator& acc) { return (acc.is_emulated) ||
	//	(acc.device_path == accelerator::direct3d_ref); }),
	//	allAccelerators.end());

	//if (allAccelerators.size() > 0)
	//	defaultView = allAccelerators[0].default_view;
#endif

}


void multiplyGPU(float * A, float * B, float * P)
{
	// for tick
	DWORD start, end;

	// copy data to default memory (shared memory)
	start = GetTickCount();
	array_view<float, 2> a(N, N, A);
	array_view<float, 2> b(N, N, B);
	array_view<float, 2> product(N, N, P);	
	end = GetTickCount();
	std::wcout << "GPU Data Copy Time Cost :" << end - start << std::endl;

	// set the function
	start = GetTickCount();
	parallel_for_each(
		product.extent,
		[=](index<2> idx) restrict(amp) {
		int row = idx[0];
		int col = idx[1];
		for (int inner = 0; inner < N; inner++) {
			product[idx] += a(row, inner) * b(inner, col);
			int testa = a(row, inner);
			int testb = b(inner, col);
		}
	}
	);	
	end = GetTickCount();
	std::wcout << "GPU Setting Function Time Cost :" << end - start << std::endl;

	// calculate
	start = GetTickCount();
	product.synchronize(access_type_read_write);
	end = GetTickCount();
	std::wcout << "GPU Calculation Time Cost :" << end - start << std::endl;
}

void multiplyCPU(float * A, float *B, float * P)
{
	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++)
			for (int k = 0; k < N; k++)
			{
				P[i*N + j] += A[i*N + k] * B[k*N + j];
			}
}

void main()
{
	//// properties test
	////default_properties();
	//// pick a ideal accelerator
	//pick_with_most_memory();
	// debug
	/*pick_to_debug();*/

	//accelerator::set_default(accelerator::direct3d_warp);

	// the arrray of matrix A
	float *A = new float[N*N];
	for (int i = 0; i < N*N; i++) A[i] = 1.0f;
	// the arrray of matrix A
	float *B=new float[N*N];
	for (int i = 0; i < N*N; i++) B[i] = 1.0f;
	// the array of product matrix
	float *P= new float[N*N];
	for (int i = 0; i < N*N; i++) P[i] = 0.0f;

	DWORD start, end;

	// GPU
	start = GetTickCount();
	multiplyGPU(A, B, P);
	end = GetTickCount();
	std::wcout << "GPU Total Time Cost :" << end - start << std::endl;

	//// CPU
	//start = GetTickCount();
	//multiplyCPU(A, B, P);
	//end = GetTickCount();
	//std::wcout << "CPU Time Cost :" << end - start << std::endl;
}