#include "../src/byod.h"
#include "../misc/files.h"

#include <stdlib.h>
#include <time.h>
#include "../misc/timings.h"

void help(const char *command_name){
    fprintf(stderr, "%s: <trace> <testsize> <format> <counters> <counters-sizes> <values> <values-sizes> <skips>\n", command_name);
}


#define _printf printf
#define printf(...)


int main(int argc, char *argv[]) {
    if(argc!=7){
        fprintf(stderr, "Invalid number of arguments\n");
        help(argv[0]);
        return(1);
    }

    const char *trace_filename = argv[1];
    const char *testsize_s = argv[2];
    const char *format_filename = argv[3];
    const char *counters_filename = argv[4];
    const char *values_filename = argv[5];
    const char *skips_filename = argv[6];

    crosstime_t *times_begin=NULL, *times_end;
    long *times;
    double devstd, avg;

    unsigned long testsize;
    char *endptr;


    testsize = strtoul(testsize_s, &endptr, 10);
    if(endptr==testsize_s || *endptr!='\0') {
        fprintf(stderr, "Invalid limit value\n");
        return 2;
    }

    times_begin = (crosstime_t*) malloc(sizeof(crosstime_t) * testsize);
    if(times_begin==NULL){
        fprintf(stderr, "Memory error\n");
        return 3;
    }
    times_end = (crosstime_t*) malloc(sizeof(crosstime_t) * testsize);
    if(times_end==NULL){
        fprintf(stderr, "Memory error\n");
        return 3;
    }

    times = (long*) malloc(sizeof(long) * testsize);
    if(times==NULL){
        fprintf(stderr, "Memory error\n");
        return 3;
    }
//
//    block_type *trace, *indexes, *values, value;
//    uint8_t *schema_data, *skips_data;
//    mysize_t *indexes_sizes, *values_sizes;
//    mysize_t i, j, file_size, indexes_size, model_size, skips_size;
//    int64_t r;
//    SequenceIt trace_it;
//    Skips skips;
//    Schema schema;
//    SRLEModel srle_model;
//
//    const char *trace_filename = argv[1];
//    const char *format_filename = argv[2];
//    const char *counters_filename = argv[3];
//    const char *values_filename = argv[4];
//    const char *skips_filename = argv[5];

    block_type *trace, *indexes_data, *values_data, value;
    uint8_t *schema_data, *skips_data;
    size_t i, j, file_size, indexes_size, model_size, skips_size;
    int64_t r;
    SequenceIt trace_it;
    Skips skips;
    Schema schema;
    SRLEModel srle_model;

    // TODO: use a proper conversion function for big/little-endian representations
    //  (now using the native representation of the architecture)
    printf("Opening file %s\n", format_filename);
    file_size = load_data(format_filename, (void**)&schema_data);
    if(schema_data==NULL || file_size == 0){
        return 1;
    }
    printf("Schema: ");
    for(i=0;i<file_size;i++){
        printf("%llu ", (unsigned long long) schema_data[i]);
    }
    printf("\n");

    printf("Size of values tree sizes is %llu\n", (unsigned long long) file_size);
    printf("Opening trace file %s\n", trace_filename);
    file_size = load_data(trace_filename, (void**)&trace);
    if(trace==NULL || file_size == 0){
        return 1;
    }
    printf("Size of trace is %llu\n", (unsigned long long) file_size);
    printf("Opening counters tree file %s\n", counters_filename);
    indexes_size = load_data(counters_filename, (void**)&indexes_data);
    if(indexes_data == NULL || indexes_size == 0){
        return 1;
    }
    printf("Size of indexes_tree is %llu\n", (unsigned long long) indexes_size);
    printf("Opening values tree file %s\n", values_filename);
    model_size = load_data(values_filename, (void**)&values_data);
    if(values_data == NULL || model_size == 0){
        return 1;
    }
    printf("Size of values tree is %llu\n", (unsigned long long) model_size);
    printf("Opening skips file %s\n", skips_filename);
    skips_size = load_data(skips_filename, (void**)&skips_data);
    if(skips_data == NULL || skips_size == 0){
        return 1;
    }
    printf("Size of skips is %llu\n", (unsigned long long) skips_size);

    load_skips_data(&skips, skips_data);
    load_schema_data(&schema, schema_data);
    populate_trace_it(&trace_it, trace);

    srle_model.counters = load_counters(indexes_data, schema.window_size - 1U);
    srle_model.values = load_values(values_data, schema.window_size - 1U);

    i=0;
    j=0;
    do {
        clock_arch(times_begin + j);
        r= verify_sequence_sparse_tree(trace_it, schema, srle_model, skips);
        clock_arch(times_end + j);
        i+=0<next_sequence_value(&value, &trace_it);
        j++;
    } while(i < file_size - schema.window_size && j < testsize && r == schema.window_size);

    if(r != schema.window_size ){
        fprintf(stderr,"Sequence %llu mismatch at symbol %lld\n", (unsigned long long)j, (unsigned long long)r);
        return 4;
    }

    for(i=0;i<j;i++){
        times[i] = timediff(times_end[i], times_begin[i]);
    }

    i = time_stats(&avg, &devstd, times, j, 1000, 1);
    if( i != j){
        fprintf(stderr, "Timings overflow at value %llu\n", (unsigned long long) i);
        return 20;
    }
    fprintf(stdout, "%llu\n", (unsigned long long) j);
    fprintf(stdout, "%f\n", avg);
    fprintf(stdout, "%f\n", devstd);
//
//    for(i=0;i<j;i++){
//        fprintf(stdout, "%lu\n", times[i]);
//    }

    return 0;
}