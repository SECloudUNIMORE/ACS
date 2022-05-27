#include "byod_extra.hh"

#include <avr/pgmspace.h>
#include <HardwareSerial.h>

extern "C"
{
#include "config.h"
#include "types.h"
#include "byod.h"
}

#define BUFF_SIZE 50

block_type flash_intranode_search(SequenceIt trace_it, const block_type branch_size, const block_type value) {
    static block_type buff[BUFF_SIZE];
    static SequenceIt buff_it = { .trace_p=buff, .offset=0 };
    block_type tmp1;
    block_type i;

    //uint16_t l = size * symbol_bitsize / block_bitsize + 1;
    buff_it.offset = trace_it.offset;
    memcpy_PF(buff, trace_it.trace_p, branch_size<BUFF_SIZE?branch_size:BUFF_SIZE);
    Serial.print("Size: ");
    Serial.println(branch_size);
    for(i=1;i<=branch_size;i++){
        next_sequence_value(&tmp1, &trace_it);
        Serial.print("Value: ");
        Serial.println(tmp1);
        if(tmp1==value) {
            return i;
        }
    }
    return 0;
}
