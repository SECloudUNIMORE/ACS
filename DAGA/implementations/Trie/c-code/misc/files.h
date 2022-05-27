#ifndef C_CODE_FILES_H
#define C_CODE_FILES_H

#include <stdio.h>

// 8 kilobyte
#define MEM_PREALLOC (1UL<<14UL)

size_t load_data(const char *trace_name, void **trace_p);

#endif //C_CODE_FILES_H
