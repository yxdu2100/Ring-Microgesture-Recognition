#ifndef IMU_REG_H
#define IMU_REG_H

#include <stddef.h>
#include <stdint.h>

int imu_reg_write(uint8_t reg, uint8_t val);
int imu_reg_read(uint8_t reg, uint8_t *val);
int imu_reg_read_buf(uint8_t start_reg, uint8_t *buf, size_t len);
int imu_reg_update(uint8_t reg, uint8_t mask, uint8_t value);

#endif
