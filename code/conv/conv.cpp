#include <iostream>
#include <chrono>
#include <vector>
#include "consts.h"
#include "conv_op1.h"
#include "utils.h"

using namespace std;
using namespace std::chrono;

void conv(
	  float out[OUT_K][IMG_HEIGHT][IMG_WIDTH],
	  float kernels[OUT_K][IMG_CHANNELS][K_SIZE][K_SIZE],
	  float image[IMG_CHANNELS][PADDED_IMG_HEIGHT][PADDED_IMG_WIDTH],
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
  float out2 [nK][H][W];
  initOut(out);
  initOut(out2);
  float image [C][H + k_size][W + k_size];
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
  
  double mean = 0;
  for(int i = 0; i < run_times; i++){
    mean += durations[i];
  }
  mean = mean / run_times;
  cout << "Time taken by baseline: "
       << mean << " microseconds" << endl;
  for(int i = 0; i < run_times; i++){
    auto start = high_resolution_clock::now();
    conv_op1(out2, kernels, image, H, W, C, k_size, nK);
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
  if (!isEqual(out, out2)){
    cout << "Outputs are not equal." << endl;
  }
  return 0;
}
