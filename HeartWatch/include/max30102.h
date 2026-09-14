#ifndef MAX30102_H
#define MAX30102_H

#include "driver/i2c_master.h"
#include "esp_err.h"
#include <stdbool.h>
#include <stdint.h>

// MAX30102 address macros
#define MAX30102_ADDRESS 0x57
#define MAX30102_REG_FIFO_WR_PTR 0x04
#define MAX30102_REG_OVF_COUNTER 0x05
#define MAX30102_REG_FIFO_RD_PTR 0x06
#define MAX30102_REG_FIFO_DATA 0x07
#define MAX30102_REG_MODE_CONFIG 0x09
#define MAX30102_REG_SPO2_CONFIG 0x0A
#define MAX30102_REG_LED1_PA 0x0C
#define MAX30102_REG_LED2_PA 0x0D
#define MAX30102_REG_PART_ID 0xFF
#define MAX30102_PART_ID 0x15

/**
 * @brief Initialize the max30102 to read heartrate, uses handler
 */
esp_err_t max30102_init(i2c_master_dev_handle_t handle);

/**
 * @brief Reads the current IR value, returns either true or false
 */
bool max30102_read_ir(i2c_master_dev_handle_t handle, uint32_t *ir_value);

#endif