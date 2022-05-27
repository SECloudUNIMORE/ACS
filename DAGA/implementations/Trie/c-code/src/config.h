
#ifndef SPT_CONFIG_H
#define SPT_CONFIG_H

//#ifndef
//#error "Must define ''"
//#endif
//#define  10

#define LOOKUP 0

#define symbol_bitsize 6

#ifndef rle_bitsize
#error "Missing rle_bitsize define"
#endif

#if rle_bitsize != 6 && rle_bitsize != 2
#error "Current version only supports rle_bitsize values 2 and 6"
#endif
//#define rle_bitsize 6


#ifndef block_type_size
#define block_type_size 1
#endif

#ifndef compressed
#define compressed 1
#endif

#ifndef use_tables
#define use_tables 1
#endif

#ifndef limit
#define limit 0
#endif

#endif //SPT_CONFIG_H
