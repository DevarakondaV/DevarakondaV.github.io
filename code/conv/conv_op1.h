#include <iostream>
#include "consts.h"


using namespace std;


int conv_op1(
	     float out[OUT_K][IMG_HEIGHT][IMG_WIDTH],
	  float kernels[OUT_K][IMG_CHANNELS][K_SIZE][K_SIZE],
	  float image[IMG_CHANNELS][IMG_HEIGHT][IMG_WIDTH],
	  int H,
	  int W,
	  int C,
	  int k_size,
	  int nK
	     ){
  int B = 10;
   for(int i = 0; i < nK ; ++i){
    for (int j = 0; j < C; ++j){

      for(int b = 0; b < H ; b+=B){

	
	for (int h = b; h < b+B ; ++h){
	  for (int w = b; w < b+B; ++w){
	    /* cout << "h: " << h << " w: " << w << endl; */
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
  return 0;
}
