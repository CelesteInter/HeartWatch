#ifndef SH1107_H
#define SH1107_H

#include "driver/gpio.h"
#include "driver/spi_master.h"
#include "esp_err.h"

// HiLetgo sh1107 dimensions - 128 x 128
#define SH1107_WIDTH 128
#define SH1107_HEIGHT 128

// Configuration struct for sh1107
typedef struct {
    spi_host_device_t host;
    gpio_num_t sclk_gpio;
    gpio_num_t mosi_gpio;
    gpio_num_t cs_gpio;
    gpio_num_t dc_gpio;
    gpio_num_t reset_gpio;
} sh1107_config_t;

/**
 * @brief initialize the sh1107 using configuration struct
 */
esp_err_t sh1107_init(const sh1107_config_t *config);

/**
 * @brief Clear the OLED display
 */
esp_err_t sh1107_clear(void);

/**
 * @brief Set the cursor location to a specified column and row
 */
esp_err_t sh1107_set_cursor(uint8_t column, uint8_t row);

/**
 * @brief Write a singular character to the OLED
 */
esp_err_t sh1107_write_char(char character);

/**
 * @brief print a string to the OLED
 */
esp_err_t sh1107_print(const char *text);

#endif