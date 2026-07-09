#include <errno.h>

#include <zephyr/sys/util.h>

#include "classifier.h"

int clf_init(void)
{
	return 0;
}

int clf_process_window(const int16_t (*win)[CLF_CHANNELS], uint16_t n)
{
	ARG_UNUSED(win);
	ARG_UNUSED(n);

	return -EAGAIN;
}

const char *clf_name(void)
{
	return "none";
}

int16_t clf_last_score(void)
{
	return 0;
}
