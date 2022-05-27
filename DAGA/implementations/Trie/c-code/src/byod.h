
#ifndef C_CODE_BYOD_H
#define C_CODE_BYOD_H

#include "config.h"
#include "types.h"

#include <stdint.h>


#ifndef LOOKUP
#define LOOKUP 0
#endif


#define BIT_OFFSET(N) ((N) * _symbol_size % block_bitsize)

#define search linear_search


void populate_trace_it(SequenceIt *it, const block_type *trace);

SequenceIt *get_trace_it(const block_type *trace);



// get the value of the sequence at the current state of the iterator
// return 0 if the read has been performed within one block of the sequence and there is still some space left
// return 1 if the read has been performed within one block of the sequence and there is no more space left
// return 3 if the read has been performed by also accessing the next block of the sequence
static inline block_type extract_value(block_type *pr, const SequenceIt *sequence){
#define r (*pr)
#define pt (sequence->trace_p)
#define voffset (sequence->offset)

#if use_tables && block_bitsize == 8 && symbol_bitsize == 6
    switch(sequence->offset){
        case block_const(0):
            r = *pt >> block_const(2);
            return 0;
        case block_const(2):
            r = *pt & block_const(0b111111);
            return 1;
        case block_const(4):
            r = (block_type) ((((*pt) & block_const(0b1111U)) << block_const(2)) | (*(pt + 1) >> block_const(6)));
            return 3;
        case block_const(6):
            r = (block_type) ((((*pt) & block_const(0b11U)) << block_const(4U)) | (*(pt + 1) >> block_const(4)));
            return 3;
    }
#else
    int16_t rightpad = block_bitsize - symbol_bitsize - sequence->offset;
    if(voffset==0){
        r = *pt >> (uint16_t) rightpad;
        return 0;
    }
    r = (block_type ) (*pt & ((one << (block_bitsize - voffset)) - one));
    if(rightpad==0) {
        return 1;
    }
    if(rightpad>0){
        r >>= (uint16_t) rightpad;
        return 0;
    }
//    if(rsize==0)
//        return voffset == (block_bitsize - symbol_bitsize);
//    if(rsize<0){
//        r >>= voffset + symbol_bitsize - block_bitsize;
//        return 0;
//    }
    r <<= (uint16_t)(-rightpad);
//    r &= ~((one << (rsize - one)) - one);
    r |= (*(pt+1) >> (block_bitsize - (uint16_t)(-rightpad)));
    return 3U;
#endif
#undef r
#undef pt
#undef voffset
}
//intranode_search

#if block_bitsize == 8 && symbol_bitsize == 6 && rle_bitsize == 2

static inline void rle_sequence_next(block_type *counter, block_type *value, RleIt *sequence){
    *value = **sequence & 0b00111111U;
    *counter = **sequence >> 6U;
    (*sequence)++;
}

#elif use_tables && block_bitsize == 8 && symbol_bitsize == 6

static inline void rle_sequence_next(block_type *counter, block_type *value, RleIt *sequence){
#define pt (sequence->trace_p)

    if(!(sequence->offset)) {
//        case 0:
        *value = *pt;
        *counter = *value >> 2U;
        pt++;
        *value = (block_type) (((*value) & 0b11U) << 4U) | (*pt >> 4U);
        sequence->offset = 4U;
        return;
    }
//        case 4U:
    *counter = (block_type) ((((*pt) & 0b1111U) << 2U) | ((*(++pt)) >> 6U));
    *value = *pt & 0b111111U;
    pt++;
    sequence->offset = 0U;
//    return;
//    }
#undef pt
}


#else // block_size == block_const(8) && symbol_bitsize == block_const(6)

static inline block_type sequence_next_2_values(block_type *v, SequenceIt *sequence){

#error Not Implemented

}
#endif



#if use_tables && block_bitsize == 8 && symbol_bitsize == 6


//block_type next_sequence_value(block_type *v, SequenceIt *sequence, void *scratch);
static inline block_type next_sequence_value(block_type *v, SequenceIt *sequence){
#define r (*v)
#define pt (sequence->trace_p)

    switch(sequence->offset){
        case 0:
            r = (*pt) >> block_const(2);
            sequence->offset = block_const(6);
            return 0;
        case 2U:
            r = (*pt++) & block_const(0b111111);
            sequence->offset = block_const(0U);
            return 1;
        case 4U:
            r = (block_type) ((((*pt) & 0b1111U) << 2U) | ((*(++pt)) >> 6U));
            sequence->offset = block_const(2);
            return 3;
        case 6U:
            r = (block_type) ((((*pt) & 0b11U) << 4U) | ((*(++pt)) >> 4U));
            sequence->offset = block_const(4);
            return 3;
    }

#undef pt
#undef r
}

#else // block_size == block_const(8) && symbol_size == block_const(6)

static inline block_type next_sequence_value(block_type *v, SequenceIt *sequence){

    block_type r;
        r=extract_value(v, sequence);
        if(r) {
            (sequence->trace_p)++;
        }
        sequence->offset = (sequence->offset + symbol_bitsize) & block_maxvalue;
        return r;
}
#endif


mysize_t sequence_seek(SequenceIt *r, mysize_t n);

// search *value* within *trace* of size *size*. Return the pointer to the value
block_type intranode_search(SequenceIt trace_it, block_type size, block_type value);

mysize_t seq_count(SequenceIt *sequence_it, mysize_t size);

mysize_t sparse_plain_lookup(mysize_t *r_c, block_type *r_next, SequenceIt *sequence_it, const mysize_t *seek);

//uint64_t bitmap_lookup(SequenceIt *trace, uint8_t trace_size,  SequenceIt *counters, const uint64_t *indexes_sizes,
//                       const uint64_t bitmap_size);

void sparse_compressed_lookup(mysize_t *r_c, block_type *r_next, RleIt *seq, mysize_t seek);

// given *trace*, verify if the next elements have been seen by the model
int64_t verify_sequence_sparse_tree(SequenceIt trace_it, Schema schema, SRLEModel model, Skips skips);
//                        SequenceIt model_indexes_it, mysize_t indexes_sizes,
//                        SequenceIt model_data_it, mysize_t data_sizes,
//

int64_t verify_sequence_bitmap_plain(SequenceIt trace_it, Schema schema, const bitmap_blocktype* bitmaps);

// search *value* within *trace* of size *size*. Return the pointer to the value
block_type inmemory_intranode_search(SequenceIt trace_it, block_type size, block_type value);

#ifdef __USE_AVR_FLASH

#include "byod_extra.hh"

#define intranode_search flash_intranode_search

#else

#define intranode_search inmemory_intranode_search

#endif


#endif //C_CODE_BYOD_H
