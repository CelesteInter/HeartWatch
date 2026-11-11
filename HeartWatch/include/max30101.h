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
#define MAX30101_REG_LED3_PA 0x0E // LED3 Pulse Amplitude
#define MAX30101_REG_MULTI_LED_CTRL1 0x11 // Multi-LED slots 1 and 2
#define MAX30101_REG_MULTI_LED_CTRL2 0x12 // Multi-LED slot 3
#define MAX30101_REG_PART_ID 0xFF // Part ID (Whoami)
#define MAX30101_PART_ID 0x15 // Return from whoami

typedef struct {
    uint32_t red;
    uint32_t ir;
    uint32_t green;
} max30101_sample_t;

/**
 * @brief Initialize the MAX30101 to read heart rate using the I2C device handle.
 */
esp_err_t max30101_init(i2c_master_dev_handle_t handle);

/**
 * @brief Read the current red, IR, and green values from the FIFO.
 */
bool max30101_read_sample(i2c_master_dev_handle_t handle,
                          max30101_sample_t *sample);

#endif