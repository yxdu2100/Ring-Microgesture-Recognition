#ifndef CLF_MLC_H
#define CLF_MLC_H

#include <stdint.h>

int clf_mlc_poll_result(uint8_t *class_id, int16_t *score, uint8_t *raw_code);

#endif
