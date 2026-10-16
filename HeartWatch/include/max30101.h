#ifndef MAX30101_H
#define MAX30101_H

#include "driver/i2c_master.h"
#include "esp_err.h"
#include <stdbool.h>
#include <stdint.h>

// MAX30101 address and register macros
#define MAX30101_ADDRESS 0x57 // 7-bit address; 0xAE write byte, 0xAF read byte
#define MAX30101_REG_FIFO_WR_PTR 0x04 // Write Pointer 
#define MAX30101_REG_OVF_COUNTER 0x05 // Overflow counter - detects if samples lost
#define MAX30101_REG_FIFO_RD_PTR 0x06 // FIFO RD Pointer
#define MAX30101_REG_FIFO_DATA 0x07 // FIFO Data Register
#define MAX30101_REG_MODE_CONFIG 0x09 // Configuration Register (SpO2, HR, Temp)
#define MAX30101_REG_SPO2_CONFIG 0x0A // SpO2 configuration Register (Range control, sample rate)
#define MAX30101_REG_LED1_PA 0x0C // LED1 Pulse Amplitude
#define MAX30101_REG_LED2_PA 0x0D // LED2 Pulse Amplitude
#define MAX30101_REG_PART_ID 0xFF // Part ID (Whoami)
#define MAX30101_PART_ID 0x15 // Return from whoami

/**
 * @brief Initialize the MAX30101 to read heart rate using the I2C device handle.
 */
esp_err_t max30101_init(i2c_master_dev_handle_t handle);

/**
 * @brief Read the current IR value, returning true when a FIFO sample is available.
 */
bool max30101_read_ir(i2c_master_dev_handle_t handle, uint32_t *ir_value);

#endif