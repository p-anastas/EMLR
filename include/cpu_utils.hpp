///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief C++ utilities for timing and error-checking
///

#ifndef CPU_UTILS_H
#define CPU_UTILS_H

#include <stdio.h>
#include <cstring>

double Gval_per_s(long long value, double time);

void massert(bool condi, const char* msg);
void error(const char* string);
void warning(const char* string);

double csecond();

#endif
