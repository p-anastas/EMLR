#include "cpu_utils.hpp"

#include <float.h>
#include <stdlib.h>
#include <time.h>
#include <algorithm>
#include <omp.h>
#include <math.h>

double Gval_per_s(long long value, double time){
  return value / (time * 1e9);
}

void massert(bool condi, const char* msg) {
  if (!condi) {
    fprintf(stderr, "Error: %s\n", msg);
    exit(1);
  }
}

void warning(const char* string) { fprintf(stderr, "WARNING ( %s )\n", string); }

void error(const char* string) {
  fprintf(stderr, "ERROR ( %s ) halting execution\n", string);
  exit(1);
}


double csecond(void) {
  struct timespec tms;

  if (clock_gettime(CLOCK_REALTIME, &tms)) {
    return (0.0);
  }
  /// seconds, multiplied with 1 million
  int64_t micros = tms.tv_sec * 1000000;
  /// Add full microseconds
  micros += tms.tv_nsec / 1000;
  /// round up if necessary
  if (tms.tv_nsec % 1000 >= 500) {
    ++micros;
  }
  return ((double)micros / 1000000.0);
}



