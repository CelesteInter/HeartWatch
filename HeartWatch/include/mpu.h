#ifndef HEARTWATCH_MPU_H
#define HEARTWATCH_MPU_H

#include <stdbool.h>
#include <stdint.h>
#include "driver/i2c_master.h"

// MPU device and register configuration.
#define MPU_I2C_ADDRESS 0x68
#define MPU_DATA_LENGTH 14
#define MPU_REG_WHO_AM_I 0x75
#define MPU_REG_ACCEL_XOUT_H 0x3B
#define MPU_WHO_AM_I_VALUE 0x70

/**
 * @brief Initializes the MPU and its communication interface.
 *
 * Configures the sensor and prepares it for accelerometer and gyroscope
 * measurements. Returns false when the sensor cannot be detected or
 * configured.
 *
 * @param bus_handle I2C bus created by the application.
 * @return true when initialization succeeds; otherwise false.
 */
bool mpu_init(i2c_master_bus_handle_t bus_handle);

/**
 * @brief Reads the latest accelerometer and gyroscope measurements.
 *
 * The output values are the raw signed 16-bit register values. Any output
 * pointer may be NULL when that measurement is not needed.
 *
 * @param accel_x Pointer receiving the X-axis acceleration.
 * @param accel_y Pointer receiving the Y-axis acceleration.
 * @param accel_z Pointer receiving the Z-axis acceleration.
 * @param gyro_x Pointer receiving the X-axis angular velocity.
 * @param gyro_y Pointer receiving the Y-axis angular velocity.
 * @param gyro_z Pointer receiving the Z-axis angular velocity.
 * @return true when a complete sample is read; otherwise false.
 */
bool mpu_read(int16_t *accel_x, int16_t *accel_y, int16_t *accel_z,
              int16_t *gyro_x, int16_t *gyro_y, int16_t *gyro_z);

/**
 * @brief Reports whether the MPU is available and responding.
 *
 * @return true when the sensor can be communicated with; otherwise false.
 */
bool mpu_is_ready(void);

#endif
