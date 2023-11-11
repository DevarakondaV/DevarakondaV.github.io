#include <iostream>
#include <vector>
#include <chrono>
#include <thread>

#define N 1000000

using namespace std;
using namespace std::chrono;

void generate_random_vector(float vec[N]){
  for(int i = 0; i < N; i++){
    vec[i] = (float)rand()/RAND_MAX;
  }
}

void dot_product(float vec1[N], float vec2[N], float & out){
  for(int i = 0; i < N; i++){
    out += vec1[i] * vec2[i];
  }
}

void dot_thread(float vec1[N], float vec2[N], float& out, int x){
  out += vec1[x] * vec2[x];
}

// Parallelize
void dot_product_op(float vec1[N], float vec2[N], float & out){
  int B = 5;
  vector<thread> threads;
  for(int b = 0; b < N; b+=B){
    int bb = 0;
    for(int i = b; i < b+B; ++i){
      threads.push_back(thread(dot_thread, ref(vec1), ref(vec2), ref(out), i));
      ++bb;
      // out += vec1[i] * vec2[i];
    }
  }
  for(int i = 0; i < threads.size(); ++i){
      threads[i].join();
  }
}

int main(){
  const int n = N;
  float vec1[N];
  float vec2[N];
  float out1 = 0;
  float out2 = 0;
  generate_random_vector(vec1);
  generate_random_vector(vec2);

  const int run_times = 20;
  double durations [run_times];
  for(int i = 0; i < run_times; i++){
    auto start = high_resolution_clock::now();
    dot_product(vec1, vec2, out1);
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
    dot_product(vec1, vec2, out1); // 88
    dot_product_op(vec1, vec2, out2); // 21
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
  cout << "E: " << out1 << " " << out2 << endl;
  return 0;
}
