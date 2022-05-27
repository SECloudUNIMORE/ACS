//#include "byod.h"
#include "../misc/files.h"
#include "../src/types.h"

#include <stdlib.h>
#include <string.h>

extern SRLEModel model;
extern Skips skips;
extern Schema schema;
extern mysize_t trace_size;
extern block_type trace[];

void help(const char *command_name){
    fprintf(stderr, "%s: <trace> <format> <counters> <counters-sizes> <values> <values-sizes> <skips>\n", command_name);
}

int main(int argc, char *argv[]) {
    if(argc!=8){
        fprintf(stderr, "Invalid number of arguments\n");
        help(argv[0]);
        return(1);
    }

    const char *trace_filename = argv[1];
    const char *format_filename = argv[2];
    const char *indexes_filename = argv[3];
    const char *indexes_sizes_filename = argv[4];
    const char *model_filename = argv[5];
    const char *model_sizes_filename = argv[6];
    const char *skips_filename = argv[7];

    block_type *indexes, *data;
    uint8_t *schema_data, *skips_data;
    mysize_t *indexes_sizes, *values_sizes;
    mysize_t i, file_size, indexes_size, model_size, skips_size;
    Skips skips_local;
    Schema schema_local;
    block_type *trace_local;
    SRLEModel model_local;

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
    printf("Opening counters tree sizes file %s\n", indexes_sizes_filename);
    file_size = load_data(indexes_sizes_filename, (void**)&indexes_sizes);
    if(indexes_sizes==NULL || file_size == 0){
        return 1;
    }
    printf("Size of counters tree sizes is %llu\n", (unsigned long long) file_size);
    printf("Opening values tree sizes file %s\n", model_sizes_filename);
    file_size = load_data(model_sizes_filename, (void**)&values_sizes);
    if(values_sizes==NULL || file_size == 0){
        return 1;
    }
    printf("Size of values tree sizes is %llu\n", (unsigned long long) file_size);
    printf("Opening trace file %s\n", trace_filename);
    file_size = load_data(trace_filename, (void**)&trace_local);
    if(trace_local==NULL || file_size == 0){
        return 1;
    }
    printf("Size of trace is %llu\n", (unsigned long long) file_size);
    printf("Opening counters tree file %s\n", indexes_filename);
    indexes_size = load_data(indexes_filename, (void**)&indexes);
    if(indexes == NULL || indexes_size == 0){
        return 1;
    }
    printf("Size of indexes_tree is %llu\n", (unsigned long long) indexes_size);
    printf("Opening values tree file %s\n", model_filename);
    model_size = load_data(model_filename, (void**)&data);
    if(data == NULL || model_size == 0){
        return 1;
    }
    printf("Size of values tree is %llu\n", (unsigned long long) model_size);
    printf("Opening skips file %s\n", skips_filename);
    skips_size = load_data(skips_filename, (void**)&skips_data);
    if(skips_data == NULL || skips_size == 0){
        return 1;
    }
    printf("Size of skips is %llu\n", (unsigned long long) skips_size);


    model_local.counters = indexes;
    model_local.indexes_sizes = indexes_sizes;
    model_local.values = data;
    model_local.data_sizes = values_sizes;
    load_skips_data(&skips_local, skips_data);
    load_schema_data(&schema_local, schema_data);

    mysize_t tot_size;

    for(i=0,tot_size=0;i<schema.window_size-2;i++){
        if(model_local.indexes_sizes[i] != model.indexes_sizes[i]){
            fprintf(stderr, "Model counters sizes mismatch at position %lu: %u %u\n", i, model_local.indexes_sizes[i], model.indexes_sizes[i]);
        }
        tot_size+=model_local.indexes_sizes[i];
    }

    if(0!=memcmp(model_local.counters, model.counters, tot_size * sizeof(block_type))){
        fprintf(stderr, "Models counters mismatch\n");
    }
    printf("Total counters size is %u\n", tot_size);

    for(i=0,tot_size=0;i<schema.window_size-2;i++){
        if(model_local.data_sizes[i] != model.data_sizes[i]){
            fprintf(stderr, "Models values sizes mismatch at position %lu\n", i);
        }
        tot_size+=model_local.data_sizes[i];
    }

    if(0!=memcmp(model_local.values, model.values, tot_size * sizeof(block_type))){
        fprintf(stderr, "Models values mismatch\n");
    }
    printf("Total counters size is %u\n", tot_size);

//
//    free(values_sizes);
//    free(indexes_sizes);
//    free(counters);
//    free(trace_local);
//    free(values);
//    free(skips_data);
//    free(schema_data);
    return 0;
}