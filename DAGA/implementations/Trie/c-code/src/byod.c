#include "byod.h"
#include <stdlib.h>


bitmap_blocktype check_bitmap(const bitmap_blocktype *bitmaps, mysize_t node_seek, mysize_t target, uint8_t n_symbols){
    mysize_t bits_offset = n_symbols * node_seek + target;
    mysize_t offset = (target + bits_offset) / bitmap_blocksize;
    mysize_t index = bits_offset - offset;
    return 1U & (bitmaps[offset] >> (bitmap_blocksize - 1U - index));
}


uint8_t count_bits(block_type v){
    uint8_t c;
    for(c=block_bitsize;c>0;c--){
        c+=(v & 0b1U);
        v>>=1U;
    }
    return c;
}

mysize_t count_bitmap_ones(const block_type *a, mysize_t state_seek, symbol_type v){
    mysize_t c, i, n= (state_seek + v) / block_bitsize;
    const block_type *pt=a+n;
    for(c=0;a!=pt;a++){
        v=*a;
        for(i=block_bitsize;i>0;i--){
            c+=(v & 0b1U);
            v>>=1U;
        }
    }
    v=*a;
    for(i=(state_seek-(n*8));i>0;i--){
        c+=(v & 0b1U);
        v>>=1U;
    }
    return c;
}

void populate_trace_it(SequenceIt *it, const block_type *trace){
    it->trace_p=trace;
    it->offset=0;
}

#ifdef ALIGNED_RLE
void populate_rle_it(RleIt *it, const block_type *trace){
    *it = trace;
}
#else
#define populate_rle_it populate_trace_it
#endif

SequenceIt *get_trace_it(const block_type *trace){
    SequenceIt *pt = (SequenceIt*)malloc(sizeof(SequenceIt));
    populate_trace_it(pt, trace);
    return pt;
}

mysize_t sequence_seek(SequenceIt *r, mysize_t n){
    mysize_t bytes;
    if(n==0){
        return 0;
    }
    bytes = (r->offset + n * symbol_bitsize);
    (r->offset) = bytes & (block_bitsize - 1);
    bytes >>= block_bitshift;
    (r->trace_p) += bytes;
    return bytes;
}

#ifdef ALIGNED_RLE
mysize_t rle_seek(RleIt *r, mysize_t n){
    (*r) += n;
    return n;
}
#elif symbol_bitsize == rle_bitsize
mysize_t rle_seek(RleIt *r, mysize_t n){
    return sequence_seek(r, n*2);
}
#else
#error
#endif

int vb01_search(uint8_t first_child, const uint8_t *deltas, uint8_t size, uint8_t target){
    uint8_t buff;
    if(first_child == target){
        return 1;
    }
    if(first_child > target){
        return 0;
    }
    size--;
    while(size>0) {
        switch((*deltas) & (uint8_t )(0b11000000)){
            case (uint8_t ) 0b00000000:
                if(target == *deltas) {
                    return 1;
                }
                size--;
                break;
            case (uint8_t ) 0b01000000:
                buff = (*deltas) & (uint8_t )(0b00111111);
                if(((target - first_child) % 2) == 0 && (target - first_child) < 2 * buff){
                    return 1;
                }
                size = (uint8_t)(size - buff);
                break;
            // 0b1xyyyyyy
            default:
                buff = (*deltas) & (uint8_t )(0b01111111);
                if((target - first_child) < buff){
                    return 1;
                }
                size = (uint8_t)(size - buff);
                break;
        }
        deltas++;
    }
    return 0;
}

// buffer points to three bytes that includes 4 values of six bits
int vb01_pack_search(const uint8_t *buffer, uint32_t node_index, const uint8_t *nodes_sizes, uint8_t target){
    uint8_t first_child;
    int r;
    switch(node_index){
        case 0:
            // first value
            first_child = *buffer >> 2U;
            r = vb01_search(first_child, buffer+3, nodes_sizes[0], target);
            break;
        case 1:
            // second value
            first_child = (uint8_t)(((uint8_t )(*buffer & (uint8_t ) 0b00000011)) << 4U);
            ++buffer;
            first_child = (uint8_t)(first_child | (uint8_t)(*buffer >> 4U));
            r= vb01_search(first_child, buffer+2, nodes_sizes[1], target);
            break;
        case 2:
            // third value
            ++buffer;
            first_child = (uint8_t)(((uint8_t )(*buffer & (uint8_t ) 0b00001111)) << 2U);
            ++buffer;
            first_child = (uint8_t)(first_child | (uint8_t)(*buffer >> 6U));
            r = vb01_search(first_child, buffer+1, nodes_sizes[2], target);
        case 3:
            // fourth value
            buffer+=2;
            first_child = (uint8_t)(*buffer >> (uint8_t) 2U);
            r = vb01_search(first_child, buffer+1, nodes_sizes[3], target);
            break;
        default:
            r = -1;
    }
    return r;
}



// search *value* within *trace* of size *size*. Return the pointer to the value
block_type inmemory_intranode_search(SequenceIt trace_it, const block_type size, const block_type value){
    block_type tmp1;
    block_type i;
    for(i=1;i<=size;i++){
        next_sequence_value(&tmp1, &trace_it);
        if(tmp1==value) {
            return i;
        }
    }
    return 0;
}

#ifdef __USE_AVR_FLASH


#include "byod_extra.hh"


#define intranode_search flash_intranode_search

#else


#define intranode_search inmemory_intranode_search

#endif

mysize_t seq_count(SequenceIt *sequence_it, mysize_t size){
    block_type buff;
    mysize_t counter;
    for (counter = 0; size > 0; size--) {
        // here n_symbols is just used as a buffer
        next_sequence_value(&buff, sequence_it);
        counter += buff;
    }
    return counter;
}

mysize_t sparse_plain_lookup(mysize_t *r_c, block_type *r_next, SequenceIt *sequence_it, const mysize_t *seek){
    mysize_t i;
    for (*r_c = 0,i=0; i<*seek; i++) {
        // here n_symbols is just used as a buffer
        next_sequence_value(r_next, sequence_it);
        *r_c += (*r_next);
    }
    next_sequence_value(r_next, sequence_it);
    return *seek+1;
}

void sparse_compressed_lookup(mysize_t *r_c, block_type *r_next, RleIt *seq, mysize_t seek){
    block_type c,buff2;
    mysize_t tot, buff, buff3;
//#define buff (*r_c)
    buff=0;
    rle_sequence_next(&c, &buff2, seq);
    tot = (buff3 = (mysize_t)c + 1);
    while(tot < seek) {
        buff += (buff3 * buff2);
        rle_sequence_next(&c, &buff2, seq);
        tot += (buff3 = (mysize_t)c + 1);
    }
    buff += ((buff3 + seek - tot) * buff2);
    if(tot==seek){
        rle_sequence_next(&c, &buff2, seq);
    }
    *r_c = buff;
    *r_next = buff2;
//#undef buff
}

int64_t verify_sequence_bitmap_plain(SequenceIt trace_it, Schema schema, const bitmap_blocktype* bitmaps){
    block_type trace_value;
    mysize_t pos, depth_size, offset, state_index, b, c;
    // the first value of the trace acts as the seek value for the first depth
    next_sequence_value(&trace_value, &trace_it);
    // verifying that the first value is a valid symbol
    if (trace_value >= schema.n_symbols)
        return 0;
    for( pos=1, depth_size=schema.n_symbols*schema.n_symbols, offset=0, state_index=0 ;
         pos < schema.window_size ;
         pos++ , depth_size *= schema.n_symbols, offset *= schema.n_symbols){

        offset += trace_value;
//        offset *= schema.n_symbols;
        next_sequence_value(&trace_value, &trace_it);
        b = (offset*schema.n_symbols + state_index + trace_value);
//        c = b >> 6U;
        if ( !(1U & ((bitmaps[b >> bitmap_bitshift] >> (bitmap_blocksize_bits - 1U - (b & (bitmap_blocksize_bits - 1)))))) )
            break;
        b = state_index + depth_size;
        c = b >> bitmap_bitshift;
        bitmaps += (c);
        state_index = b & (bitmap_blocksize_bits - 1U);
    }
    return pos;
}


int64_t verify_sequence_sparse_tree(SequenceIt trace_it, Schema schema, SRLEModel model, Skips skips) {
    block_type trace_value, branch_size;
    mysize_t seek_data, skip_index_seek, jump_number, seek_indexes;
    uint32_t skip_index_it, skip_data_seek;
    mysize_t state_seek, pos;
    const uint16_t *endp, *skip_indexes_p, *jump_data_p;
    SequenceIt data_it;
    RleIt indexes_it;
    // the first value of the trace acts as the seek value of the first depth
    next_sequence_value(&trace_value, &trace_it);
    // verifying that the first value is a valid symbol
    if (trace_value >= (block_type) schema.n_symbols)
        return 0;
    state_seek = (mysize_t)trace_value;

    for(seek_data=0, pos=0;
        pos < (schema.window_size - 1) ;
        pos++
        ) {

        // get the next trace value
        next_sequence_value(&trace_value, &trace_it);
        // seek through the counters to get the values seek value
        seek_indexes = seek_data + state_seek;
        populate_trace_it(&data_it, model.values[pos]);
        populate_rle_it(&indexes_it, model.counters[pos]);
        if(skips.number==0 || pos < skips.start)
        {
            sparse_compressed_lookup(&seek_data, &branch_size, &indexes_it, seek_indexes);
        } else {
            // number of skip values available at the current search depth
//            skip_numbers = skips.sizes[pos - skips.start];
            // amount of counters skipped for each skip value
            skip_index_it = skips.size[pos - skips.start];
            // compute the number of skip values that we can use
            jump_number = seek_indexes / skip_index_it;
            // compute the total amount of counters that we can skip
            skip_index_seek = skip_index_it * jump_number;
            // compute the offset of the counters for the corresponding total skip amount

            for (skip_index_it = 0 ,
                 skip_indexes_p = skips.counters[pos - skips.start],
                 endp = skip_indexes_p + jump_number;
                 skip_indexes_p != endp; skip_indexes_p++){
                skip_index_it += *skip_indexes_p;
            }
            // advance the pointer to index skips for the next iteration
//            skip_indexes_p += (skip_numbers - jump_number);
            // skip over the counters (the skip value is doubled due the RLE compression)
            rle_seek(&indexes_it, skip_index_it );
            sparse_compressed_lookup(&seek_data, &branch_size, &indexes_it, seek_indexes - skip_index_seek);

            for (skip_data_seek = 0 ,
                 jump_data_p = skips.values[pos - skips.start],
                 endp = jump_data_p + jump_number;
                 jump_data_p != endp; jump_data_p++){
                skip_data_seek += *jump_data_p;
            }
            // advance the pointer of values skip for the next iteration
//            jump_data_p += (skip_numbers - jump_number);
            seek_data += skip_data_seek;
        }
        // seek within the values values to the branch of the trace value
        sequence_seek(&data_it, seek_data);

        // look if trace value exists
        state_seek = intranode_search(data_it, branch_size, trace_value);
        // NOTE: the search function returns an index by using a [1, N] notation
        if(0 == state_seek){
//            pos-=1;
            break;
        }
        state_seek -= 1;
    }
    // if sequence exists the caller verifies that the return value is equal to sequence_size
    return (int64_t)pos+1;
}
