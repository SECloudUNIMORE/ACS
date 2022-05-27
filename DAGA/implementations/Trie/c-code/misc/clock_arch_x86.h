#ifndef C_CODE_CLOCK_ARCH_X86_H
#define C_CODE_CLOCK_ARCH_X86_H

#define timediff(end,start) (((end).tv_sec - (start).tv_sec)*(timedelta)1e9 + ((end).tv_nsec - (start).tv_nsec))
//#define crosstime_t struct timespec
typedef long timedelta;
typedef struct timespec crosstime_t;
#define clock_arch(t) clock_gettime(CLOCK_MONOTONIC_RAW, (t))

#endif
