#include "files.h"

#include <stdlib.h>

size_t load_data(const char *trace_name, void **trace_p){
    FILE *trace_fp;
    size_t trace_size, read_size, i;
    *trace_p = NULL;
    void *tmp;

    trace_fp = fopen(trace_name, "r");
    if(trace_fp == NULL){
        fprintf(stderr, "Cannot open file %s\n", trace_name);
        return 0;
    }

    *trace_p = malloc(MEM_PREALLOC);
//    fseek(trace_fp, 0L, SEEK_END);
//    trace_size = (mysize_t)ftell(trace_fp);
//    rewind(trace_fp);
//    *trace_p = malloc(trace_size);
    if(*trace_p == NULL){
        fprintf(stderr, "Memory error: cannot allocate %lu bytes of memory\n", MEM_PREALLOC);
        if(fclose(trace_fp) != 0) {
            fprintf(stderr, "Error while closing file\n");
        }
        return 0;
    }
    trace_size=0;
    for(i=2; (read_size=fread(*trace_p + (MEM_PREALLOC*(i-2)), 1, MEM_PREALLOC, trace_fp)) == MEM_PREALLOC ; i++){
        trace_size+=read_size;
        tmp = realloc(*trace_p, MEM_PREALLOC*i);
        if(tmp == NULL){
            fprintf(stderr, "Memory error: cannot allocate %lu bytes of memory\n", MEM_PREALLOC*i);
            free(*trace_p);
            if(fclose(trace_fp) != 0) {
                fprintf(stderr, "Error while closing file\n");
            }
            return 0;
        } else {
            *trace_p = tmp;
        }
    }

    trace_size += read_size;
    tmp = realloc(*trace_p, trace_size);
    if(tmp == NULL){
        fprintf(stderr, "Memory error: cannot allocate %lu bytes of memory\n", MEM_PREALLOC*i);
        free(*trace_p);
        if(fclose(trace_fp) != 0) {
            fprintf(stderr, "Error while closing file\n");
        }
        return 0;
    } else {
        *trace_p = tmp;
    }

//
//    if((read_size=fread(*trace_p, 1, MEM_PREALLOC, trace_fp)) != MEM_PREALLOC){
//        fprintf(stderr, "Read error: read only %lu instead of %lu\n", read_size, trace_size);
//        free(*trace_p);
//        if(fclose(trace_fp) != 0) {
//            fprintf(stderr, "Error while closing file\n");
//        }
//        return 0;
//    }

    if(fclose(trace_fp) != 0){
        fprintf(stderr, "Error while closing file\n");
        free(*trace_p);
        return 0;
    }
    return trace_size;
}
