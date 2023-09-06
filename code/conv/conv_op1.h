#include <iostream>
#include "consts.h"


using namespace std;


int conv_op2(
	  float out[OUT_K][IMG_HEIGHT][IMG_WIDTH],
	  float kernels[OUT_K][IMG_CHANNELS][K_SIZE][K_SIZE],
	  float image[IMG_CHANNELS][PADDED_IMG_HEIGHT][PADDED_IMG_WIDTH],
	  int H,
	  int W,
	  int C,
	  int k_size,
	  int nK
	     ){
  int B = 10;
   for(int i = 0; i < nK ; ++i){
    for (int j = 0; j < C; ++j){
      for(int b1 = 0; b1 < H; b1 += B){
	for(int h = b1; h < b1+B; ++h){
	  for(int b = 0; b < W; b+=B){
	    for (int w = b; w < b+B; ++w){
	      for (int p = 0; p < k_size ; ++p){
		for(int q = 0; q < k_size ; ++q){
		  out[i][h][w] += kernels[i][j][p][q] * image[j][h+p][w+q];
		}
	      }
	    }
	  }
	}
      }
    }
  }
  return 0;
}


int conv_op1(
	  float out[OUT_K][IMG_HEIGHT][IMG_WIDTH],
	  float kernels[OUT_K][IMG_CHANNELS][K_SIZE][K_SIZE],
	  float image[IMG_CHANNELS][PADDED_IMG_HEIGHT][PADDED_IMG_WIDTH],
	  int H,
	  int W,
	  int C,
	  int k_size,
	  int nK
	     ){
  int B = 5;
  for (int j = 0; j < C; ++j){
    for(int b1 = 0; b1 < H; b1 += B){
      for(int h = b1; h < b1+B; ++h){
	for(int b = 0; b < W; b+=B){
	  for (int w = b; w < b+B; ++w){
	    for(int i = 0; i < nK ; ++i){
	      for (int p = 0; p < k_size ; ++p){
		for(int q = 0; q < k_size ; ++q){
		  out[i][h][w] += kernels[i][j][p][q] * image[j][h+p][w+q];
		}
	      }
	    }
	  }
	}
      }
    }
  }
  return 0;
}
