#include <iostream>
#include <chrono>
#include <vector>
#include "consts.h"
#include "conv_op1.h"

using namespace std;
using namespace std::chrono;

void conv(
	  float out[OUT_K][IMG_HEIGHT][IMG_WIDTH],
	  float kernels[OUT_K][IMG_CHANNELS][K_SIZE][K_SIZE],
	  float image[IMG_CHANNELS][IMG_HEIGHT][IMG_WIDTH],
	  int H,
	  int W,
	  int C,
	  int k_size,
	  int nK
	  ){
  for(int i = 0; i < nK ; ++i){
    for (int j = 0; j < C; ++j){
      for (int h = 0; h < H ; ++h){
	for (int w = 0; w < W; ++w){
	  for (int p = 0; p < k_size ; ++p){
	    for (int q = 0; q < k_size ; q++){
	      out[i][h][w] += kernels[i][j][p][q] * image[j][h+p][w+q];
	    }
	  }
	}
      }
    }
  }
}


void generate_random_image(
			   float image[IMG_CHANNELS][IMG_HEIGHT][IMG_WIDTH],
			   int H,
			   int W,
			   int C
			   ){
  for(int c = 0; c < C ; ++c){
    for(int h = 0; h < H; ++h){
      for(int w = 0; w < W; ++w){
	image[c][h][w] = (float)rand()/RAND_MAX;
      }
    }
  }
}

void generate_fake_kernels(
			   float kernels[OUT_K][IMG_CHANNELS][K_SIZE][K_SIZE],
			   int C,
			   int k_size,
			   int nK){
  for(int i = 0; i < nK; ++i){
    for(int j = 0; j < C; ++j){
      for(int p = 0; p < k_size; ++p){
	for(int q = 0; q < k_size; ++q){
	  kernels[i][j][p][q] = (float)rand()/RAND_MAX;
	}
      }
    }
  }
}

void print_conv(
		float image[IMG_CHANNELS][IMG_HEIGHT][IMG_WIDTH],
			   int H,
			   int W,
		int C){
  for(int h = 0; h < H; ++h){
    for(int w = 0; w < W; ++w){
      cout << image[C][h][w] << " ";
    }
    cout << endl;
  }
}

int main(){
  // MNIST Dataset
  const int H = IMG_HEIGHT;
  const int W = IMG_WIDTH;
  const int C = IMG_CHANNELS;
  const int k_size = K_SIZE;
  const int nK = OUT_K;
  float out [nK][H][W];
  float image [C][H][W];
  float kernels [nK][C][k_size][k_size];

  srand(time(0));
  generate_random_image(image, H, W, C);
  generate_fake_kernels(kernels, C, k_size, nK);
  
  // Time Convolution
  const int run_times = 20;
  double durations [run_times];
  for(int i = 0; i < run_times; i++){
    auto start = high_resolution_clock::now();
    conv(out, kernels, image, H, W, C, k_size, nK);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    durations[i] = duration.count();
  }
  // auto start = high_resolution_clock::now();
  // conv(out, kernels, image, H, W, C, k_size, nK);
  // auto stop = high_resolution_clock::now();
  // auto duration = duration_cast<microseconds>(stop - start);
  // cout << "Time taken by function: "
  //      << duration.count() << " microseconds" << endl;
  double mean = 0;
  for(int i = 0; i < run_times; i++){
    mean += durations[i];
  }
  mean = mean / run_times;
  cout << "Time taken by baseline: "
       << mean << " microseconds" << endl;

  for(int i = 0; i < run_times; i++){
    auto start = high_resolution_clock::now();
    conv_op1(out, kernels, image, H, W, C, k_size, nK);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    durations[i] = duration.count();
  }
  mean = 0;
  for(int i = 0; i < run_times; i++){
    mean += durations[i];
  }
  mean = mean / run_times;
  cout << "Time taken op1: "
       << mean << " microseconds" << endl;
  return 0;
}
