#include "../src/byod.h"

#include <stdio.h>

#include "../misc/timings.h"

#include "../data.h"
#include "../trace.h"

#ifndef limit
#define limit 0
#endif


int main(int argc, char *argv[]) {
    mysize_t i,j;
    int64_t r;
    symbol_type value;
    crosstime_t time_begin, time_end;
    timedelta time;

    SequenceIt trace_it[1];
    populate_trace_it(trace_it, trace);

    printf("Testing %u sequences\n", limit);

    clock_arch(&time_begin);
    for(r=schema.window_size,i=0,j=0;
        i < trace_size && (limit==0 || j<limit) && r == schema.window_size;
        j++)
    {
        r= verify_sequence_sparse_tree(*trace_it, schema, model, skips);
        i+=0<next_sequence_value(&value, trace_it);
    }
    clock_arch(&time_end);

    time = timediff(time_end, time_begin);

    if(r == schema.window_size ){
        printf("All %llu sequences match!\n", (unsigned long long)j);
//        printf("Start time: %llu\nEnd time:  %llu\n", (unsigned long long)time_begin.tv_nsec, (unsigned long long)time_end.tv_nsec);
        printf("Total time: %f s\n", (double)time/1e9);
        printf("Avg time:   %f us\n", (double)time/(1000.0*j));
    } else {
        printf("Sequence %llu mismatch at symbol %lld\n", (unsigned long long)j, (unsigned long long)r);
    }

    return (int)r;
}