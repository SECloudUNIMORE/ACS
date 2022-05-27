#ifndef C_CODE_TIMINGS_H
#define C_CODE_TIMINGS_H

#include <time.h>
#ifdef __arm__
#indlude "clock_arch_arm.h"
#else
#include "clock_arch_x86.h"
#endif

unsigned long time_stats(double *avg, double *std_dev, const long *times, unsigned long size, double scale_factor_1, double scale_factor_2);

#endif //C_CODE_TIMINGS_H
