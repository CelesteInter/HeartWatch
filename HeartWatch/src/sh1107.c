#include <string.h>
#include "font8x8_basic.h"
#include "esp_check.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "sh1107.h"

// Macros for page and buffer size
#define SH1107_PAGES (SH1107_HEIGHT / 8)
#define SH1107_BUFFER_SIZE (SH1107_WIDTH * SH1107_PAGES)
#define SH1107_COLUMN_OFFSET 96

static spi_device_handle_t display_device;
static const char *TAG = "sh1107";
static gpio_num_t display_dc_gpio;
static gpio_num_t display_reset_gpio;
static uint8_t display_buffer[SH1107_BUFFER_SIZE];
static uint8_t cursor_column;
static uint8_t cursor_row;

// Send command/data to sh1107 using buffer and length. data determines interpreted as command or data
static esp_err_t sh1107_send(bool data, const uint8_t *buffer, size_t length)
{
    //Set command or data for DC wire 
    gpio_set_level(display_dc_gpio, data ? 1 : 0);

    //Create transaction
    spi_transaction_t transaction = {
        .length = length * 8,
        .tx_buffer = buffer,
    };

    // Transmit transaction to sh1107, return error to function call
    return spi_device_transmit(display_device, &transaction);
}

// Send a command to the sh1107
static esp_err_t sh1107_command(uint8_t command)
{
    return sh1107_send(false, &command, 1);
}

/**
 *  Flush the buffer onto the OLED display
 * 
 * The sh1107 stores pixels in pages, and with a 128x128 display we have 16 pages:
 *      Pages = 128 vertical / 8 vertical pixels per address
 * This function effectively copies the entire framebuffer over into the OLED over the SPI bus.
 */
static esp_err_t sh1107_flush(void)
{
    // Loop over all pages for the sh1107 (16 pages total)
    for (uint8_t page = 0; page < SH1107_PAGES; page++) {
        // Page select starts at 0xB0, up to page 15 (0xBF)
        uint8_t page_command = 0xB0 | page;

        // Starting column (start at leftmost)
        uint8_t column = SH1107_COLUMN_OFFSET;

        uint8_t column_commands[] = {
            0x00 | (column & 0x0F),
            0x10 | ((column >> 4) & 0x0F)
        };

        // Send page select command
        ESP_ERROR_CHECK(sh1107_command(page_command));
        // Send column select command
        ESP_ERROR_CHECK(sh1107_send(false, column_commands, sizeof(column_commands)));
        // Send data to sh1107 - starting at page
        ESP_ERROR_CHECK(sh1107_send(true, &display_buffer[page * SH1107_WIDTH], SH1107_WIDTH));
    }

    // Return success
    return ESP_OK;
}

// Initialize the SPI bus for the sh117 using the configuration struct
esp_err_t sh1107_init(const sh1107_config_t *config)
{
    // error if config is undefined
    if (config == NULL) {
        return ESP_ERR_INVALID_ARG;
    }

    // Setup SPI bus
    spi_bus_config_t bus_config = {
        .sclk_io_num = config->sclk_gpio,
        .mosi_io_num = config->mosi_gpio,
        .miso_io_num = -1,
        .quadwp_io_num = -1,
        .quadhd_io_num = -1,
        .max_transfer_sz = SH1107_WIDTH,
    };
    // Initialize bus
    ESP_RETURN_ON_ERROR(spi_bus_initialize(config->host, &bus_config, SPI_DMA_CH_AUTO), TAG, "SPI bus init failed");

    // Setup device interface for SPI bus
    spi_device_interface_config_t device_config = {
        .clock_speed_hz = 10 * 1000 * 1000,
        .mode = 0,
        .spics_io_num = config->cs_gpio,
        .queue_size = 1,
    };
    // Add spi bus device
    ESP_RETURN_ON_ERROR(spi_bus_add_device(config->host, &device_config, &display_device), TAG, "SPI device init failed");

    display_dc_gpio = config->dc_gpio;
    display_reset_gpio = config->reset_gpio;

    // Setup GPIO for DC (data/command) and reset 
    gpio_config_t control_gpio_config = {
        .pin_bit_mask = (1ULL << display_dc_gpio) | (1ULL << display_reset_gpio),
        .mode = GPIO_MODE_OUTPUT,
        .pull_up_en = GPIO_PULLUP_DISABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type = GPIO_INTR_DISABLE,
    };
    // Configure GPIO
    ESP_RETURN_ON_ERROR(gpio_config(&control_gpio_config), TAG, "OLED GPIO init failed");

    // Reset sh1107
    gpio_set_level(display_reset_gpio, 0);
    vTaskDelay(pdMS_TO_TICKS(10));
    gpio_set_level(display_reset_gpio, 1);
    vTaskDelay(pdMS_TO_TICKS(10));

    // Commands to initialize sh1107
    const uint8_t initialization_commands[] = {
        0xAE,       // Display off
        0xDC, 0x00, // Display start line
        0x81, 0x2F, // Contrast
        0xA0,       // Segment remap
        0xC0,       // Common output scan direction
        0xA8, 0x7F, // Multiplex ratio: 128 rows
        0xD3, 0x60, // Display offset
        0xD5, 0x51, // Display clock
        0xD9, 0x22, // Pre-charge period
        0xDB, 0x35, // VCOM deselect level
        0xAF,       // Display on
    };

    // Run through setup command list
    for (size_t index = 0; index < sizeof(initialization_commands); index++) {
        ESP_RETURN_ON_ERROR(sh1107_command(initialization_commands[index]), TAG, "OLED init command failed");
    }

    // Clear the display
    return sh1107_clear();
}

// Clear the sh1107 display
esp_err_t sh1107_clear(void)
{
    // Clear memory buffer
    memset(display_buffer, 0, sizeof(display_buffer));
    // Reset cursor values
    cursor_column = 0;
    cursor_row = 0;
    // Flush to display
    return sh1107_flush();
}

// Move the cursor for the display
esp_err_t sh1107_set_cursor(uint8_t column, uint8_t row)
{
    // filter invalid commands (out of bounds)
    if (column >= SH1107_WIDTH || row >= SH1107_HEIGHT / 8) {
        return ESP_ERR_INVALID_ARG;
    }

    // Update cursor location
    cursor_column = column;
    cursor_row = row;

    // Return ok code
    return ESP_OK;
}

// Write a single character to buffer
esp_err_t sh1107_write_char(char character)
{
    // Linebreak (adjust cursor to next row)
    if (character == '\n') {
        cursor_column = 0;
        cursor_row++;
    } else {
        // non 8x8 font character handling
        if ((unsigned char)character >= 128) {
            character = '?';
        }

        // newline if overextended
        if (cursor_column > SH1107_WIDTH - 8) {
            cursor_column = 0;
            cursor_row++;
        }

        // Overwrite to 1st row if at end
        if (cursor_row >= SH1107_HEIGHT / 8) {
            cursor_row = 0;
        }

        // Write to display buffer using font header macros
        memcpy(&display_buffer[cursor_row * SH1107_WIDTH + cursor_column],
               font8x8_basic_tr[(unsigned char)character], 8);
        // Increase cursor location
        cursor_column += 8;
    }

    // reset to first row if cursor exceeds size
    if (cursor_row >= SH1107_HEIGHT / 8) {
        cursor_row = 0;
    }

    // Send buffer to sh1107
    return sh1107_flush();
}

// Print a string to sh1107 - repeatedly calls write_char until EOF
esp_err_t sh1107_print(const char *text)
{
    // empty string handling
    if (text == NULL) {
        return ESP_ERR_INVALID_ARG;
    }

    // read until EOF
    while (*text != '\0') {
        ESP_RETURN_ON_ERROR(sh1107_write_char(*text), TAG, "OLED write failed");
        text++;
    }

    // Return ok
    return ESP_OK;
}