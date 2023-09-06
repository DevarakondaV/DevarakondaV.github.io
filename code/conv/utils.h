#include <cstdlib>
#include <iostream>
#include "consts.h"

using namespace std;

bool isEqual(
	     float out1[OUT_K][IMG_HEIGHT][IMG_WIDTH],
	     float out2[OUT_K][IMG_HEIGHT][IMG_WIDTH]
	     ){
  bool equal = true;
  for(int k = 0 ; k < OUT_K; ++k){
    for(int i = 0; i < IMG_HEIGHT; ++i){
      for(int j = 0; j < IMG_WIDTH; ++j){
	float diff = abs(out1[k][i][j] - out2[k][i][j]);
	cout << "k: " << k << " i: " << i << " j: " << j << " "  << out1[k][i][j] << " " << out2[k][i][j] << " " ;
	cout << "Diff: " << diff << endl;
	if (diff > 0.0001){
	  equal = false;
	  /* return false; */
	}
      }
    }
  }
  return equal;
  /* return true; */
}

void generate_random_image(
			   float image[IMG_CHANNELS][PADDED_IMG_HEIGHT][PADDED_IMG_WIDTH],
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


void initOut(float out[OUT_K][IMG_HEIGHT][IMG_WIDTH]){
  for(int k = 0 ; k < OUT_K ; ++k){
    for(int h = 0; h < IMG_HEIGHT; ++h){
      for(int w = 0; w < IMG_WIDTH; ++w){
	out[k][h][w] = 0;
      }
    }
  }
}
