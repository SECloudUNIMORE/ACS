
#include "types.h"
#include <stdlib.h>

void load_skips_data(Skips *t, const uint8_t *data){
    mysize_t i;
    uint32_t *sizes;
    t->number = data[0];
    t->counters = (const uint16_t**) malloc(sizeof(const uint16_t*) * t->number);
    t->values = (const uint16_t**) malloc(sizeof(const uint16_t*) * t->number);
    t->start = data[1];
    t->size = (uint16_t*)(data + 2);
    sizes = (uint32_t*)(t->size + t->number);
    t->counters[0] = (uint16_t*)(sizes + t->number);
    for(i=1;i<t->number;i++){
        t->counters[i] = t->counters[i-1] + sizes[i-1];
    }
    t->values[0] = t->counters[i-1] + sizes[i-1];
    for(i=1;i<t->number;i++){
        t->values[i] = t->values[i-1] + sizes[i-1];
    }
}


void load_schema_data(Schema *t, const uint8_t *data){
    t->n_symbols = data[0];
    t->window_size = data[1];
}


const values_t** load_values(const uint8_t *data, uint8_t n){
    int i;
    const values_t **r;
    const mysize_t *sizes;
    const values_t *p;

    sizes = (const mysize_t *)data;
    data += (sizeof(mysize_t) * n);
    p = (values_t*) data;

    r = (const values_t**) malloc(sizeof(values_t*) * n);
    if(r==NULL){
        return NULL;
    }
    for(i=0;i<n;i++){
        r[i] = p;
        p += sizes[i];
    }
    return r;
}


const counters_t** load_counters(const uint8_t *data, uint8_t n){
    int i;
    const counters_t **r;
    const mysize_t *sizes;
    const counters_t *p;

    sizes = (const mysize_t *)data;
    data += (sizeof(mysize_t) * n);
    p = (counters_t*) data;

    r = (const counters_t**) malloc(sizeof(counters_t*) * n);
    if(r==NULL){
        return NULL;
    }
    for(i=0;i<n;i++){
        r[i] = p;
        p += sizes[i];
    }
    return r;
}