// Include libraries here
#include <stdio.h>
#include <inttypes.h>
#include "driver/gpio.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/i2c_master.h"
#include "driver/spi_master.h"
#include "driver/gpio.h"
#include "esp_log.h"
#include "esp_heap_caps.h"
#include "esp_psram.h"
#include "esp_timer.h"

// Custom headers
#include "max30101.h"
#include "mpu.h"
#include "sh1107.h"
#include "hud.h"

// Macros for GPIO
#define USER_LED GPIO_NUM_21
#define I2C_SCL GPIO_NUM_6
#define I2C_SDA GPIO_NUM_5
#define SPI_SCLK GPIO_NUM_7
#define SPI_MOSI GPIO_NUM_9
#define BUTTON_1_PIN GPIO_NUM_1
#define BUTTON_2_PIN GPIO_NUM_8
#define SPI_DC GPIO_NUM_43
#define SPI_CS GPIO_NUM_44
#define SPI_RST GPIO_NUM_4

// Configs

// Tags
static const char *MPU_TAG = "mpu";
static const char *MAX30101_TAG = "max30101";
static const char *MAIN_TAG = "esp";

// Logging
struct log {
    int16_t ax; // 16
    int16_t ay; // 32
    int16_t az; // 48
    int16_t gx; // 64
    int16_t gy; // 80
    int16_t gz; // 96
    uint8_t heartrate; // 104
    // pad 8 bits = 112 bits or 14 bytes
};

#define LOG_CAPACITY 500000

// Buffer that holds all the logs the user does. This is heap allocation
struct log *log_buffer;
uint32_t log_pointer = 0; // points to the starting log, then use to traverse through logs
uint32_t log_start_time = 0;

// Enumerable to determine device state
typedef enum {
    STATE_DISPLAY,
    STATE_LOG,
    STATE_LOG_AND_DISPLAY,
    STATE_TRANSMIT_DATA,
    STATE_SLEEP
} state_t;
state_t device_state = STATE_SLEEP;

static const char *state_name(state_t state)
{
    switch (state) {
        case STATE_DISPLAY:
            return "DISPLAY";
        case STATE_LOG:
            return "LOG";
        case STATE_LOG_AND_DISPLAY:
            return "LOG_AND_DISPLAY";
        case STATE_TRANSMIT_DATA:
            return "TRANSMIT_DATA";
        case STATE_SLEEP:
            return "SLEEP";
        default:
            return "UNKNOWN";
    }
}

// Variable declarations
    //Button Setup
 // Button Setup (Polling Style)
int button1State = 1;
int lastButton1State = 1;
uint32_t lastDebounceTime1 = 0;
int button2State = 1;
int lastButton2State = 1;
uint32_t lastDebounceTime2 = 0;
    // MAX30101
uint32_t green_value;
uint32_t green_baseline = 0;
uint32_t green_peak = 0;
uint32_t previous_green_signal = 0;
uint32_t prior_green_signal = 0;
int64_t previous_green_signal_ms = 0;
int64_t last_green_beat_ms = 0;
uint16_t stable_bpm = 0;
uint16_t bpm_history[3] = {0};
uint8_t bpm_history_count = 0;
    // Timing
uint32_t day_time = 0;
uint32_t display_time = 0;

// Helper Functions
static void reset_heart_rate_state(void)
{
    green_baseline = 0;
    green_peak = 0;
    previous_green_signal = 0;
    prior_green_signal = 0;
    previous_green_signal_ms = 0;
    last_green_beat_ms = 0;
    stable_bpm = 0;
    bpm_history_count = 0;
}

/**
 * Gets the current runtime in milliseconds
 */
uint32_t millis() {
    return (uint32_t)(esp_timer_get_time() / 1000ULL);
}

// Main Application
void app_main(void)
{
    // Configurations (initialization)

    // PSRAM & Logging Buffer
    ESP_LOGI(MAIN_TAG, "PSRAM detected: %zu bytes", esp_psram_get_size());
    log_buffer = heap_caps_calloc(LOG_CAPACITY, sizeof(*log_buffer), MALLOC_CAP_SPIRAM);
    ESP_ERROR_CHECK(log_buffer != NULL ? ESP_OK : ESP_ERR_NO_MEM); // Error if buffer is null (memory failed)

    // Button Configurations
        // BUTTON 1 config
	gpio_config_t button_1_config = {
		.pin_bit_mask = (1ULL << BUTTON_1_PIN),
		.mode = GPIO_MODE_INPUT,
		.pull_up_en = GPIO_PULLUP_ENABLE,
		.pull_down_en = GPIO_PULLDOWN_DISABLE,
	};
	ESP_ERROR_CHECK(gpio_config(&button_1_config));
        // Button 2 config
    gpio_config_t button_2_config = {
        .pin_bit_mask = (1ULL << BUTTON_2_PIN),
		.mode = GPIO_MODE_INPUT,
		.pull_up_en = GPIO_PULLUP_ENABLE,
		.pull_down_en = GPIO_PULLDOWN_DISABLE,
    };
    ESP_ERROR_CHECK(gpio_config(&button_2_config));

    // SPI & sh1107 (OLED Display)
    sh1107_config_t display_config = {
        .host = SPI2_HOST,
        .sclk_gpio = SPI_SCLK,
        .mosi_gpio = SPI_MOSI,
        .cs_gpio = SPI_CS,
        .dc_gpio = SPI_DC,
        .reset_gpio = SPI_RST,
    };
    ESP_ERROR_CHECK(sh1107_init(&display_config));
    ESP_ERROR_CHECK(sh1107_clear());
    displayHUD(0);

    // I2C Master Configuration
    i2c_master_bus_config_t i2c_mst_config = {
        .clk_source = I2C_CLK_SRC_DEFAULT,
        .i2c_port = I2C_NUM_1,
        .scl_io_num = I2C_SCL,
        .sda_io_num = I2C_SDA,
        .glitch_ignore_cnt = 7,
        .flags.enable_internal_pullup = true,
    };
        // bus handler (for calls after setup)
    i2c_master_bus_handle_t bus_handle;
        // Error checking & bus setup (verify I2C bus)
    ESP_ERROR_CHECK(i2c_new_master_bus(&i2c_mst_config, &bus_handle));

    // MPU I2C Configuration
    ESP_ERROR_CHECK(mpu_init(bus_handle) ? ESP_OK : ESP_FAIL);
        //MPU read buffer
    int16_t accel_x, accel_y, accel_z;
    int16_t gyro_x, gyro_y, gyro_z;

    // MAX30101
    ESP_ERROR_CHECK(max30101_init(bus_handle));

    // Main state-driven polling loop. GPIO interrupts are intentionally not used.
    while (1) {
        
        // update "current" time, report the state as sleep, read GPIO buttons
        uint32_t now = millis();
        static state_t reported_state = STATE_SLEEP;
        int reading1 = gpio_get_level(BUTTON_1_PIN);
        int reading2 = gpio_get_level(BUTTON_2_PIN);

        // Debounce buttons (still fluctuating)
        if (reading1 != lastButton1State) {
            lastDebounceTime1 = now;
        }
        if (reading2 != lastButton2State) {
            lastDebounceTime2 = now;
        }

        // Update booleans if buttons are pressed
        bool button1_pressed =
            reading1 == 0 && button1State != 0 &&
            now - lastDebounceTime1 >= 50;
        bool button2_pressed =
            reading2 == 0 && button2State != 0 &&
            now - lastDebounceTime2 >= 50;

        // Update state if button1 is pressed
        if (button1_pressed) {
            button1State = 0;
            if (device_state == STATE_LOG) {
                device_state = STATE_LOG_AND_DISPLAY;
            } else if (device_state == STATE_LOG_AND_DISPLAY) {
                device_state = STATE_LOG;
            } else if (device_state == STATE_DISPLAY) {
                device_state = STATE_TRANSMIT_DATA;
            } else {
                device_state = STATE_DISPLAY;
                display_time = now;
            }
        } else if (reading1 != 0) {
            button1State = 1;
        }

        if (button2_pressed) {
            button2State = 0;
            if (device_state == STATE_LOG ||
                device_state == STATE_LOG_AND_DISPLAY) {
                device_state = STATE_SLEEP;
            } else if (device_state == STATE_SLEEP ||
                       device_state == STATE_DISPLAY) {
                log_pointer = 0;
                log_start_time = now;
                device_state = STATE_LOG;
                displayHUD(1);
            }
        } else if (reading2 != 0) {
            button2State = 1;
        }

        if (device_state == STATE_DISPLAY &&
            now - display_time >= 5000) {
            displayHUD(0);
            device_state = STATE_SLEEP;
            reset_heart_rate_state();
        }

        if (device_state != reported_state) {
            ESP_LOGI(MAIN_TAG, "State changed: %s -> %s",
                     state_name(reported_state), state_name(device_state));
            reported_state = device_state;
        }

        if (device_state == STATE_DISPLAY ||
            device_state == STATE_LOG ||
            device_state == STATE_LOG_AND_DISPLAY) {
            for (int sample_count = 0; sample_count < 8; sample_count++) {
                if (!max30101_read_green(&green_value)) {
                    break;
                }

                uint32_t green = green_value;
                int64_t sample_time = esp_timer_get_time() / 1000;
                if (green < 1000) {
                    reset_heart_rate_state();
                    continue;
                }
                if (green_baseline == 0) {
                    green_baseline = green;
                }

                green_baseline = (green_baseline * 31 + green) / 32;
                uint32_t raw_signal = green > green_baseline
                    ? green - green_baseline
                    : green_baseline - green;
                uint32_t signal = (previous_green_signal * 3 + raw_signal) / 4;

                if (signal > green_peak) {
                    green_peak = signal;
                } else {
                    green_peak = (green_peak * 15) / 16;
                }

                uint32_t threshold = green_peak / 4;
                if (threshold < 8) {
                    threshold = 8;
                }

                bool is_peak = prior_green_signal < previous_green_signal &&
                               previous_green_signal >= signal &&
                               previous_green_signal > threshold;
                if (is_peak &&
                    (last_green_beat_ms == 0 ||
                     previous_green_signal_ms - last_green_beat_ms > 450)) {
                    int64_t beat_time = previous_green_signal_ms;
                    if (last_green_beat_ms != 0) {
                        int64_t interval = beat_time - last_green_beat_ms;
                        if (interval >= 400 && interval <= 2000) {
                            uint16_t bpm = (uint16_t)(60000 / interval);
                            if (bpm >= 30 && bpm <= 150) {
                                if (bpm_history_count < 3) {
                                    bpm_history[bpm_history_count++] = bpm;
                                } else {
                                    bpm_history[0] = bpm_history[1];
                                    bpm_history[1] = bpm_history[2];
                                    bpm_history[2] = bpm;
                                }
                                if (bpm_history_count == 1) {
                                    stable_bpm = bpm;
                                }
                                if (bpm_history_count == 3) {
                                    uint16_t minimum = bpm_history[0];
                                    uint16_t maximum = bpm_history[0];
                                    uint16_t total = bpm_history[0];
                                    for (int i = 1; i < 3; i++) {
                                        if (bpm_history[i] < minimum) {
                                            minimum = bpm_history[i];
                                        }
                                        if (bpm_history[i] > maximum) {
                                            maximum = bpm_history[i];
                                        }
                                        total += bpm_history[i];
                                    }
                                    if (maximum - minimum <= 30) {
                                        stable_bpm = total / 3;
                                    }
                                }
                            } else {
                                bpm_history_count = 0;
                            }
                        } else if (interval > 2000) {
                            bpm_history_count = 0;
                        }
                    }
                    last_green_beat_ms = beat_time;
                }

                prior_green_signal = previous_green_signal;
                previous_green_signal = signal;
                previous_green_signal_ms = sample_time;
                if (last_green_beat_ms != 0 &&
                    sample_time - last_green_beat_ms > 2500) {
                    reset_heart_rate_state();
                }
            }
        }

        if (device_state == STATE_LOG ||
            device_state == STATE_LOG_AND_DISPLAY) {
            if (!mpu_read(&accel_x, &accel_y, &accel_z,
                          &gyro_x, &gyro_y, &gyro_z)) {
                ESP_LOGW(MPU_TAG, "MPU read failed");
            }

            if (log_pointer < LOG_CAPACITY) {
                log_buffer[log_pointer++] = (struct log) {
                    .ax = accel_x, .ay = accel_y, .az = accel_z,
                    .gx = gyro_x, .gy = gyro_y, .gz = gyro_z,
                    .heartrate = stable_bpm
                };
            }
        }

        if (device_state == STATE_LOG_AND_DISPLAY) {
            char message[32];
            snprintf(message, sizeof(message),
                     " HR: %" PRIu16 " BPM\nTime: %" PRIu32 "s",
                     stable_bpm, (now - log_start_time) / 1000);
            ESP_ERROR_CHECK(sh1107_set_cursor(0, 0));
            ESP_ERROR_CHECK(sh1107_print(message));
        } else if (device_state == STATE_DISPLAY) {
            char message[32];
            snprintf(message, sizeof(message), " HR: %" PRIu16 " BPM",
                     stable_bpm);
            ESP_ERROR_CHECK(sh1107_set_cursor(5, 7));
            ESP_ERROR_CHECK(sh1107_print(message));
        }

        if (device_state == STATE_TRANSMIT_DATA) {
            for (uint32_t i = 0; i < log_pointer; i++) {
                struct log *entry = &log_buffer[i];
                ESP_LOGI(MAIN_TAG,
                         "log%" PRIu32 ": ax: %d, ay: %d, az: %d, "
                         "gx: %d, gy: %d, gz: %d, heartrate: %u",
                         i, entry->ax, entry->ay, entry->az,
                         entry->gx, entry->gy, entry->gz,
                         entry->heartrate);
            }
            displayHUD(0);
            reset_heart_rate_state();
            log_pointer = 0;
            device_state = STATE_SLEEP;
        }

        lastButton1State = reading1;
        lastButton2State = reading2;
        vTaskDelay(pdMS_TO_TICKS(10));
    }
}