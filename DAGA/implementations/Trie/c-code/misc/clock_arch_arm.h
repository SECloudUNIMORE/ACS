#ifndef C_CODE_CLOCK_ARCH_ARM_H
#define C_CODE_CLOCK_ARCH_ARM_H

#define clock_arch localtime
typedef time_t crosstime_t
#define timediff(end,start) ((end) - (start))

#endif