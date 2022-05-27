#include "timings.h"



#include <math.h>

#include <limits.h>
#include <float.h>
//#include <stdlib.h>

//#define SCALE_FACTOR 1

unsigned long time_stats(double *avg, double *std_dev, const long *times, unsigned long size, double scale_factor_1, double scale_factor_2){
    unsigned long i;
    unsigned long long sum, v;

    for(i=0,sum=0;i<size;i++){
        v = (unsigned long long)((double)times[i]/scale_factor_1);
        if( ULLONG_MAX - v < sum || ((unsigned long long)DBL_MAX - v < sum)){
            return i;
        }
        sum += v;

    }

    sum /= (unsigned long long)scale_factor_2;

    *avg = (double)sum/(double)size;

    for(i=0,sum=0;i<size;i++){
        sum += (unsigned long long)(pow((double)times[i]/(scale_factor_1*scale_factor_2) - *avg, 2));
    }
    *std_dev = (double)sum/(double)size;
    *std_dev = sqrt(*std_dev);
    return size;
}