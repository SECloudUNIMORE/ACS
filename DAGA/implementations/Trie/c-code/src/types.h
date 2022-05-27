#ifndef C_CODE_TYPES_H
#define C_CODE_TYPES_H

#include <stdint.h>
#include "config.h"

#define MYUINT8_C(x) ((uint8_t)UINT8_C(x))

typedef uint32_t mysize_t;
#define size_type_size 4

typedef uint8_t symbol_type;
#define symbol_type_size 1

#if block_type_size==1
typedef uint8_t block_type;
#define block_const(x) MYUINT8_C(x)
#define block_maxvalue 255U
#define block_bitshift 3U
//#warning "Compiling with block_type_size = 1"
#elif block_type_size==2
typedef uint16_t block_type;
#define block_maxvalue 0b1111111111111111U
#define block_const(x) (UINT16_C(x))
#define block_bitshift 4U
//#warning "Compiling with block_type_size = 2"
#elif block_type_size==4
typedef uint32_t block_type;
#define block_const(x) (UINT32_C(x))
#define block_maxvalue 0b11111111111111111111111111111111U
#define block_bitshift 5U
//#warning "Compiling with block_type_size = 4"
#elif block_type_size==8
//////
typedef uint64_t block_type;
#define block_const(x) (UINT64_C(x))
#define block_maxvalue 0b1111111111111111111111111111111111111111111111111111111111111111U
#define block_bitshift 16U

//#warning "Compiling with block_type_size = 8"
///////
#else
#error "Invalid block type size"
#endif

#define one block_const(1)
#define eight block_const(8U)
#define six block_const(6U)
#define four block_const(4U)
#define ten block_const(10)

#define block_bitsize (8U * block_type_size)

typedef struct SequenceIt_struct {
    const block_type *trace_p;
    uint8_t offset;
#if LOOKUP
    int (*next)(symbol_type *, struct SequenceIt_struct *);
#endif
} SequenceIt;


#if block_bitsize == (symbol_bitsize + rle_bitsize)
#define ALIGNED_RLE
#endif

#ifdef ALIGNED_RLE
typedef block_type const *RleIt;
#else
typedef SequenceIt RleIt;
#endif

#define bitmap_blocktype uint64_t
#define bitmap_blocksize 8U
#define bitmap_blocksize_bits 64U
#define bitmap_bitshift 6U
#define bitmap_bitmask 63U

typedef struct {
    uint8_t number;
    uint8_t start;
    const uint16_t *size;
    const uint16_t **counters;
    const uint16_t **values;
} Skips;

typedef struct {
    uint8_t n_symbols;
    uint8_t window_size;
} Schema;

typedef block_type values_t;
typedef block_type counters_t;

typedef struct {
    const counters_t **counters;
    const values_t **values;
} SRLEModel;

void load_skips_data(Skips *t, const uint8_t *data);

void load_schema_data(Schema *t, const uint8_t *data);

const values_t** load_values(const uint8_t *data, uint8_t n);

const counters_t** load_counters(const uint8_t *data, uint8_t n);

#endif //C_CODE_TYPES_H
