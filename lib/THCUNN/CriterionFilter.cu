#include "THCUNN.h"
#include "common.h"

#include <stdio.h>
#include <assert.h>

#include <thrust/functional.h>
__global__ void cunn_CriterionFilter_updateOutput_kernel(
          float *target,
          float *ignored_label,
          int bound,
          int batch_size,
          int map_nelem,
          int blocks_per_sample)
{
  int i;
  int sample = blockIdx.x / blocks_per_sample;
  int step = blockDim.x * blocks_per_sample;
  int toffset = sample * map_nelem;
  int ignored_label_num = (int)(ignored_label[0]);
  for (i = (blockIdx.x % blocks_per_sample) * blockDim.x + threadIdx.x; i < map_nelem; i += step) {
    if (target[toffset + i] == ignored_label_num) {
      target[toffset + i] = (float) bound + 1;
    }
  }
}

__global__ void cunn_CriterionFilter_updateGradInput_kernel(
          float *gradInput,
          float *target,
          float *ignored_label,
          int batch_size,
          int n_classes,
          int map_nelem,
          int blocks_per_sample)
{
  int i, t;
  int sample = blockIdx.x / blocks_per_sample;
  int step = blockDim.x * blocks_per_sample;
  int toffset = sample * map_nelem;
  int ioffset = sample * map_nelem * n_classes;
  int ignored_label_num = (int)(ignored_label[0]);
  for (i = (blockIdx.x % blocks_per_sample) * blockDim.x + threadIdx.x; i < map_nelem; i += step) {
    t = (int)target[toffset + i];
    if (t == ignored_label_num) {
      int j;
      for (j = 0; j < n_classes; j++) gradInput[ioffset + j * map_nelem + i] = 0;
    }
  }
}

void THNN_CudaCriterionFilter_updateOutput(THCState *state, THCudaTensor *target, THCudaTensor *input, THCudaTensor *ignored_label) {
  int n_dims = THCudaTensor_nDimension(state, target);
  int bound = THCudaTensor_size(state, input, 1);

  target = THCudaTensor_newContiguous(state, target);
  ignored_label = THCudaTensor_newContiguous(state, ignored_label);

  float *target_data = THCudaTensor_data(state, target);
  float *ignored_label_data = THCudaTensor_data(state, ignored_label);

  long batch_size = THCudaTensor_size(state, target, 0);
  long map_nelem = THCudaTensor_nElement(state, target) / batch_size;
  int blocks_per_sample = GET_BLOCKS(map_nelem) / 512;

  blocks_per_sample = (blocks_per_sample == 0) ? 1 : blocks_per_sample;
  int total_blocks = blocks_per_sample * batch_size;

  int size_1, size_2, size_3;
  if (n_dims == 1) {
    size_1 = THCudaTensor_size(state, target, 0);
    size_2 = 1;
    size_3 = 1;
  } else if (n_dims == 2) {
    size_1 = 1;
    size_2 = THCudaTensor_size(state, target, 0);
    size_3 = THCudaTensor_size(state, target, 1);
  } else if (n_dims == 3) {
    size_1 = THCudaTensor_size(state, target, 0);
    size_2 = THCudaTensor_size(state, target, 1);
    size_3 = THCudaTensor_size(state, target, 2);
  } else {THError("Target Tensor should be 1D~3D tensor!");}
  cunn_CriterionFilter_updateOutput_kernel<<<total_blocks, CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state)>>>(
    target_data,
    ignored_label_data,
    bound,
    size_1,
    size_2 * size_3,
    blocks_per_sample
  );

}

void THNN_CudaCriterionFilter_updateGradInput(THCState *state, THCudaTensor *target, THCudaTensor *gradInput, THCudaTensor *ignored_label) {

  int n_dims = THCudaTensor_nDimension(state, target);
  
  ignored_label = THCudaTensor_newContiguous(state, ignored_label);
  target = THCudaTensor_newContiguous(state, target);

  float *target_data = THCudaTensor_data(state, target);
  float *gradInput_data = THCudaTensor_data(state, gradInput);
  float *ignored_label_data = THCudaTensor_data(state, ignored_label);
  long batch_size = THCudaTensor_size(state, target, 0);
  long map_nelem = THCudaTensor_nElement(state, target) / batch_size;
  int blocks_per_sample = GET_BLOCKS(map_nelem) / 128; //128 is the number of tasks one thread should processed.
  blocks_per_sample = (blocks_per_sample == 0) ? 1 : blocks_per_sample;
  int total_blocks = blocks_per_sample * batch_size;
  
  int size_1, size_2, size_3, size_4;
  //TODO:when the dimension of target tensor is 1, the block number is the length of it, it's too large.
  if (n_dims == 1) {
    size_1 = THCudaTensor_size(state, target, 0);
    size_2 = THCudaTensor_size(state, gradInput, 1);
    size_3 = 1;
    size_4 = 1;
  } else if (n_dims == 2) {
    size_1 = 1;
    size_2 = THCudaTensor_size(state, gradInput, 1);
    size_3 = THCudaTensor_size(state, target, 0);
    size_4 = THCudaTensor_size(state, target, 1);
  } else if (n_dims == 3) {
    size_1 = THCudaTensor_size(state, target, 0);
    size_2 = THCudaTensor_size(state, gradInput, 1);
    size_3 = THCudaTensor_size(state, target, 1);
    size_4 = THCudaTensor_size(state, target, 2);
  } else {THError("Target Tensor should be 1D~3D tensor!");}
  cunn_CriterionFilter_updateGradInput_kernel<<<total_blocks, CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state)>>>(
    gradInput_data,
    target_data,
    ignored_label_data,
    size_1,
    size_2,
    size_3 * size_4,
    blocks_per_sample
  );
  THCudaTensor_free(state, target);
  THCudaTensor_free(state, ignored_label);
}
