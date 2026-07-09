#ifndef CLASSIFIER_H
#define CLASSIFIER_H

#include <stdint.h>

#define CLF_WINDOW_SAMPLES 128U
#define CLF_CHANNELS 6U
#define CLF_CLASS_COUNT 5U

#ifdef __cplusplus
extern "C" {
#endif

int clf_init(void);
/* window: 128 samples x 6 channels int16, plus metadata */
int clf_process_window(const int16_t (*win)[CLF_CHANNELS], uint16_t n);
/* returns class id 0..4 or -EAGAIN if no decision */
const char *clf_name(void);
int16_t clf_last_score(void);

#ifdef __cplusplus
}
#endif

#endif
