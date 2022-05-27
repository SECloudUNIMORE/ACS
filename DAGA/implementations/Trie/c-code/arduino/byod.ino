extern "C" {
#include <byod.h>

unsigned long t_begin, t_end;
const unsigned long test_size=50;
}

#include <data2.h>
#include <trace50.h>

void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);
  while (!Serial);
  Serial.println(F("*** Lookup Time [microseconds] ***"));

  unsigned i,r;
  symbol_type value;
  SequenceIt trace_it[1];
  populate_trace_it(trace_it, trace);
  
  for(i=0,r=schema.window_size;i<test_size && r == schema.window_size;i++) {
        t_begin = micros();
        r = verify_sequence_sparse_tree(*trace_it, 0, 0, schema, model, skips);
        t_end = micros();
        Serial.println(t_end-t_begin);
        next_sequence_value(&value, trace_it);
        Serial.flush();
  }
  if(r!=schema.window_size){
        Serial.println(F("Errore Lunghezza sequenza!!!"));
        Serial.flush();
  }
  
}

void loop() {
  // put your main code here, to run repeatedly:

}
