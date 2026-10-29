// Include libraries here
#include <stdio.h>
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

/// Data length
#define MPU_DATA_LENGTH 14

// Addresses
#define WHO_AM_I 0x75
#define MPU_REG_ACCEL_XOUT_H 0x3B

// Tags
static const char *TAG = "mpu";
static const char *HEART_RATE_TAG = "heartrate";
static const char *MAIN = "esp";

 // Button Setup (Polling Style)
int button1State = 1;
int lastButton1State = 1;
uint32_t lastDebounceTime1 = 0;
int button2State = 1;
int lastButton2State = 1;
uint32_t lastDebounceTime2 = 0;

int display = 0;

// Variables to hold MAX30101 values
uint32_t ir_value;
uint32_t ir_baseline = 0;
uint32_t signal_peak = 0;
int64_t last_beat_ms = 0;
uint16_t stable_bpm = 0;

uint32_t millis() {
    return (uint32_t)(esp_timer_get_time() / 1000ULL);
}

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

struct log *log_buffer;
// Include a pointer to point to the current log
uint32_t log_pointer = 0;

// Main application loop
void app_main(void)
{
    // Configurations

    ESP_LOGI(TAG, "PSRAM detected: %zu bytes", esp_psram_get_size());
    log_buffer = heap_caps_calloc(LOG_CAPACITY, sizeof(*log_buffer), MALLOC_CAP_SPIRAM);
    ESP_ERROR_CHECK(log_buffer != NULL ? ESP_OK : ESP_ERR_NO_MEM);

    // Button Config
    gpio_reset_pin(BUTTON_1_PIN);
    gpio_set_direction(BUTTON_1_PIN, GPIO_MODE_INPUT);
    gpio_set_pull_mode(BUTTON_1_PIN, GPIO_PULLUP_ONLY);

    gpio_reset_pin(BUTTON_2_PIN);
    gpio_set_direction(BUTTON_2_PIN, GPIO_MODE_INPUT);
    gpio_set_pull_mode(BUTTON_2_PIN, GPIO_PULLUP_ONLY);

    const uint32_t debounceDelay = 10; // 50ms delay
    
    // Heartbeat LED config
	gpio_config_t led_config = {
		.pin_bit_mask = 1ULL << USER_LED,
		.mode = GPIO_MODE_OUTPUT,
		.pull_up_en = GPIO_PULLUP_DISABLE,
		.pull_down_en = GPIO_PULLDOWN_DISABLE,
		.intr_type = GPIO_INTR_DISABLE,
	};
	gpio_config(&led_config);

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
    displayHUD(display);

    vTaskDelay(200);

    //ESP_ERROR_CHECK(sh1107_clear());

    vTaskDelay(100);

    //displayHUD(2);
    // I2C master configuration
    i2c_master_bus_config_t i2c_mst_config = {
        .clk_source = I2C_CLK_SRC_DEFAULT,
        .i2c_port = I2C_NUM_1,
        .scl_io_num = I2C_SCL,
        .sda_io_num = I2C_SDA,
        .glitch_ignore_cnt = 7,
        .flags.enable_internal_pullup = true,
    };

    // bus handler (to be used for calls after setup)
    i2c_master_bus_handle_t bus_handle;

    // Error checking & bus setup (verifies I2C bus setup correctly)
    ESP_ERROR_CHECK(i2c_new_master_bus(&i2c_mst_config, &bus_handle));

    // MPU I2C configuration
    i2c_device_config_t i2c_mpu_config = {
        .dev_addr_length = I2C_ADDR_BIT_LEN_7,
        .device_address = 0x68,
        .scl_speed_hz = 100000,
    };
    // bus handler
    i2c_master_dev_handle_t mpu_handle;
    // Error checking & bus setup (verify config & setup)
    ESP_ERROR_CHECK(i2c_master_bus_add_device(bus_handle, &i2c_mpu_config, &mpu_handle));

    // Delay for terminal to catchup for debug
    // vTaskDelay(500);

    // MPU read buffer
    uint8_t mpu_data[MPU_DATA_LENGTH];
    uint8_t register_address;
    esp_err_t result;
    int16_t accel_x, accel_y, accel_z;
    int16_t gyro_x, gyro_y, gyro_z;

    // Read WHO_AM_I to verify that the MPU is responding at address 0x68.
    register_address = WHO_AM_I;
    result = i2c_master_transmit_receive(mpu_handle, &register_address, 1,
                                         mpu_data, 1, -1);
    ESP_ERROR_CHECK(result);
    ESP_LOGI(TAG, "WHO_AM_I: 0x%02X", mpu_data[0]);

    // MAX30101 I2C configuration
    i2c_device_config_t i2c_heartrate_config = {
        .dev_addr_length = I2C_ADDR_BIT_LEN_7,
        .device_address = MAX30101_ADDRESS,
        .scl_speed_hz = 100000,
    };
    // bus handler
    i2c_master_dev_handle_t heartrate_handle;
    // Error checking & bus setup
    ESP_ERROR_CHECK(i2c_master_bus_add_device(bus_handle, &i2c_heartrate_config, &heartrate_handle));

    ESP_ERROR_CHECK(max30101_init(heartrate_handle));
    ESP_LOGI(HEART_RATE_TAG, "MAX30101 initialized");


    // Main polling loop
	while (1) {
            
        // Button Debounce Routines and Display Selections
        int reading1 = gpio_get_level(BUTTON_1_PIN);
        if (reading1 != lastButton1State) {
            lastDebounceTime1 = millis();
        }

        // Read heart rate BPM
        for (int sample_count = 0; sample_count < 8; sample_count++) {
            if (!max30101_read_ir(heartrate_handle, &ir_value)) {
                break;
            }

            if (ir_baseline == 0) {
                ir_baseline = ir_value;
            }

            // Track the slowly changing DC level and detect a pulse above it.
            // baseline value for ambient light
            ir_baseline = (ir_baseline * 31 + ir_value) / 32;
            // Remove slowly changing DC component from IR reading
            uint32_t signal = ir_value > ir_baseline ? ir_value - ir_baseline : 0;
            // update previous peak from signal
            uint32_t previous_peak = signal_peak;
            // update current signal peak (new peak is 7/8ths old peak, 1/8th new signal)
            signal_peak = (signal_peak * 7 + signal) / 8;
            // Get the current time on the timer (ms)
            int64_t now_ms = esp_timer_get_time() / 1000;

            /**
             * Heartbeat recognition
             * ir_value > 5000 : assumes finger is present, otherwise ambient pulse
             * signal > 500 : pulse must be at least 500 counts above baseline
             * signal > previous_peak * 3/2 : signal must be 50% larger than recent smoothed level
             * now_ms - last_beat_ms > 300 : theoretical maximum limit for BPM of 200 
             *          -> 60000 ms in a minute / 300 ms = 200 BPM
             */
            if (ir_value > 5000 && signal > 500 && signal > previous_peak * 3 / 2 &&
                now_ms - last_beat_ms > 300) {
                // Ensure first detected pulse doesn't provide a data entry
                if (last_beat_ms != 0) {
                    // calculate bpm (60000 ms in a minute / change in time)
                    uint32_t bpm = 60000 / (now_ms - last_beat_ms);
                    // If the BPM is reasonable, print it to log
                    if (bpm >= 40 && bpm <= 220) {
                        stable_bpm = bpm;
                    } 
                }
                // update last time for beat
                last_beat_ms = now_ms;
            }
        } // End of while loop

        // MPU data read
        register_address = MPU_REG_ACCEL_XOUT_H;
        result = i2c_master_transmit_receive(mpu_handle, &register_address, 1,
                                mpu_data, MPU_DATA_LENGTH, -1);
        ESP_ERROR_CHECK(result);
        accel_x = (int16_t)(((uint16_t)mpu_data[0] << 8) | mpu_data[1]);
        accel_y = (int16_t)(((uint16_t)mpu_data[2] << 8) | mpu_data[3]);
        accel_z = (int16_t)(((uint16_t)mpu_data[4] << 8) | mpu_data[5]);
        gyro_x = (int16_t)(((uint16_t)mpu_data[8] << 8) | mpu_data[9]);
        gyro_y = (int16_t)(((uint16_t)mpu_data[10] << 8) | mpu_data[11]);
        gyro_z = (int16_t)(((uint16_t)mpu_data[12] << 8) | mpu_data[13]);

        if ((millis() - lastDebounceTime1) > debounceDelay) {
            if (reading1 != button1State) {
                button1State = reading1;

                if (button1State == 0) {
                    ESP_ERROR_CHECK(sh1107_set_cursor(0, 0));
                    display++;
                    if (display > 3) {
                        display = 0;
                    }   
               
                    switch (display) {
                        case 0: // Bootup Screen 
                            displayHUD(0);
                        break;
                        case 1: // Clock Display
                            displayHUD(1);
                        break;
                        case 2: // Heart Rate
                            displayHUD(2);
                        break;
                        case 3: // Motion Tracking
                            displayHUD(3);
                        break; 
                    }
                }
            }
        }
        lastButton1State = reading1;
        int reading2 = gpio_get_level(BUTTON_2_PIN);
        if (reading2 != lastButton2State) {
            lastDebounceTime2 = millis();
        }

        if ((millis() - lastDebounceTime2) > debounceDelay) {
            if (reading2 != button2State) {
                button2State = reading2;

                if (button2State == 0) {
                    ESP_ERROR_CHECK(sh1107_set_cursor(0, 0));
                    if (display == 4) {
                        for (uint32_t i = 0; i < log_pointer; i++) {
                            uint16_t temp = (log_buffer+i)->ax;
                            ESP_LOGI(MAIN, "log%d: ax: %d", i, temp);
                        }
                        display = 0;
                    } else {
                        display = 4;   
                    }
                    
                    // displayHUD(display); eventually revert to shutoff
                }
            }
        }
        lastButton2State = reading2;


        // Display 1: Timer
        if (display == 1) {
            unsigned long time = millis() % 10000;
            char message[14];
            snprintf(message, sizeof(message),"     %lu   ", time);
            ESP_ERROR_CHECK(sh1107_set_cursor(5, 7));
            ESP_ERROR_CHECK(sh1107_print(message));
        }


        // Display 2: Heart Rate
        if (display == 2) {
        
            bool sensor_success = false;
            int read_attempts = 0;

            // Try to read the buffer
            // while (max30101_read_ir(heartrate_handle, &ir_value)) {
            //     sensor_success = true;
            //     read_attempts++;
            //      // Read heart rate BPM
            //     for (int sample_count = 0; sample_count < 8; sample_count++) {
            //         if (!max30101_read_ir(heartrate_handle, &ir_value)) {
            //             break;
            //         }

            //         if (ir_baseline == 0) {
            //             ir_baseline = ir_value;
            //         }

            //         // Track the slowly changing DC level and detect a pulse above it.
            //         // baseline value for ambient light
            //         ir_baseline = (ir_baseline * 31 + ir_value) / 32;
            //         // Remove slowly changing DC component from IR reading
            //         uint32_t signal = ir_value > ir_baseline ? ir_value - ir_baseline : 0;
            //         // update previous peak from signal
            //         uint32_t previous_peak = signal_peak;
            //         // update current signal peak (new peak is 7/8ths old peak, 1/8th new signal)
            //         signal_peak = (signal_peak * 7 + signal) / 8;
            //         // Get the current time on the timer (ms)
            //         int64_t now_ms = esp_timer_get_time() / 1000;

            //         /**
            //          * Heartbeat recognition
            //          * ir_value > 5000 : assumes finger is present, otherwise ambient pulse
            //          * signal > 500 : pulse must be at least 500 counts above baseline
            //          * signal > previous_peak * 3/2 : signal must be 50% larger than recent smoothed level
            //          * now_ms - last_beat_ms > 300 : theoretical maximum limit for BPM of 200 
            //          *          -> 60000 ms in a minute / 300 ms = 200 BPM
            //          */
            //         if (ir_value > 5000 && signal > 500 && signal > previous_peak * 3 / 2 &&
            //             now_ms - last_beat_ms > 300) {
            //             // Ensure first detected pulse doesn't provide a data entry
            //             if (last_beat_ms != 0) {
            //                 // calculate bpm (60000 ms in a minute / change in time)
            //                 uint32_t bpm = 60000 / (now_ms - last_beat_ms);
            //                 // If the BPM is reasonable, print it to log
            //                 if (bpm >= 40 && bpm <= 220) {
            //                     stable_bpm = bpm;
            //                 } 
            //             }
            //             // update last time for beat
            //             last_beat_ms = now_ms;
            //         }
            //     }
            // } // End of while loop

            // // Reset on failure (used for when buffer fills up after changing displays)
            // static uint32_t fail_count = 0;
            // if (!sensor_success) {
            //     fail_count++;
            //     if (fail_count > 20) { // If it fails for 20 consecutive loops (~200ms)
            //         ESP_LOGW(HEART_RATE_TAG, "Sensor locked up. Re-initializing...");
            //         max30101_init(heartrate_handle); // Reset the sensor hardware
            //         fail_count = 0;                  // Reset counter
            //     }
            // } else {
            //     fail_count = 0; // Reset counter on successful read
            // }

            // Display Data to screen
            //ESP_LOGE(HEART_RATE_TAG, "(stable) Heart rate: %3lu BPM", (unsigned long)stable_bpm);
            char message[32];
            snprintf(message, sizeof(message),"  HR: %3lu BPM", (unsigned long)stable_bpm);
            ESP_ERROR_CHECK(sh1107_set_cursor(5, 7));
            ESP_ERROR_CHECK(sh1107_print(message));
        }   



        // Display 3: MPU
        if (display == 3) {
            static uint32_t last_mpu_read = 0;

            if (millis() - last_mpu_read > 250) {
                last_mpu_read = millis();
            
                
                char message[128];
                snprintf(message, sizeof(message),
                        "   AX:%6d\n   AY:%6d\n   AZ:%6d\n   GX:%6d\n   GY:%6d\n   GZ:%6d", 
                         accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z);
                ESP_ERROR_CHECK(sh1107_set_cursor(0, 7));
                ESP_ERROR_CHECK(sh1107_print(message));
                
                //MPU reading output to error log
                ESP_LOGI(TAG, "WHO_AM_I: 0x%02X", mpu_data[0]);
                register_address = MPU_REG_ACCEL_XOUT_H;
                result = i2c_master_transmit_receive(mpu_handle, &register_address, 1,
                                                        mpu_data, sizeof(mpu_data), -1);
                ESP_ERROR_CHECK(result);
                ESP_LOG_BUFFER_HEXDUMP(TAG, mpu_data, sizeof(mpu_data), ESP_LOG_INFO);
            }
        }

        // Display 4 - temporary logging
        if (display == 4) {
            ESP_ERROR_CHECK(sh1107_set_cursor(0,7));
            ESP_ERROR_CHECK(sh1107_print("Logging..."));
            struct log new_log = {
                .ax = accel_x, // 16
                .ay = accel_y, // 32
                .az = accel_z, // 48
                .gx = gyro_x, // 64
                .gy = gyro_y, // 80
                .gz = gyro_z, // 96
                .heartrate = stable_bpm // 104
            };
            if (log_pointer < LOG_CAPACITY) {
                log_buffer[log_pointer++] = new_log;
            }
        }

        
        // Print to serial
        // ESP_LOGI(TAG, "AX: %d, AY: %d, AZ: %d, GX: %d, GY: %d, GZ: %d\n", accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z);
        // ESP_LOGI(MAIN, "\n");
        vTaskDelay(pdMS_TO_TICKS(10));
	}
}