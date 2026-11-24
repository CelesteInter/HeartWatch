// Include libraries here
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <sys/time.h>
#include <stdlib.h>
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


// LVGL Headers
#include "esp_lcd_panel_io.h"
#include "esp_lcd_panel_vendor.h"
#include "esp_lcd_panel_ops.h"
#include "lvgl.h"

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

 // MPU
uint8_t mpu_data[MPU_DATA_LENGTH];
uint8_t register_address;
esp_err_t result;
int16_t ax, ay, az;
int16_t gx, gy, gz;
i2c_master_dev_handle_t mpu_handle;

// LGVL
esp_lcd_panel_io_handle_t io_handle = NULL;
esp_lcd_panel_handle_t panel_handle = NULL;
uint8_t ui_div = 0;
uint8_t clock_div = 0;  


// Get the millisecond count from the internal esp timer
uint32_t millis() {
    return (uint32_t)(esp_timer_get_time() / 1000ULL);
}


void init_I2C(void) {
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
    
    // Error checking & bus setup (verify config & setup)
    ESP_ERROR_CHECK(i2c_master_bus_add_device(bus_handle, &i2c_mpu_config, &mpu_handle));

    // Delay for terminal to catchup for debug
    vTaskDelay(500);

    // Read WHO_AM_I to verify that the MPU is responding at address 0x68.
    register_address = WHO_AM_I;
    result = i2c_master_transmit_receive(mpu_handle, &register_address, 1,
                                         mpu_data, 1, -1);
    ESP_ERROR_CHECK(result);
    //ESP_LOGI(TAG, "WHO_AM_I: 0x%02X", mpu_data[0]);

    // // MAX30101 I2C configuration
    // i2c_device_config_t i2c_heartrate_config = {
    //     .dev_addr_length = I2C_ADDR_BIT_LEN_7,
    //     .device_address = MAX30101_ADDRESS,
    //     .scl_speed_hz = 100000,
    // };
    // // bus handler
    // i2c_master_dev_handle_t heartrate_handle;
    // // Error checking & bus setup
    // ESP_ERROR_CHECK(i2c_master_bus_add_device(bus_handle, &i2c_heartrate_config, &heartrate_handle));

    // ESP_ERROR_CHECK(max30101_init(heartrate_handle));
    // ESP_LOGI(HEART_RATE_TAG, "MAX30101 initialized");
}


// Initialize SPI and the ST7789 TFT LCD Screen
void init_st7789(void) {
    // Initialize SPI Bus
    spi_bus_config_t buscfg = {
        .sclk_io_num = SPI_SCLK,
        .mosi_io_num = SPI_MOSI,
        .miso_io_num = -1, // not needed for the display
        .quadwp_io_num = -1,
        .quadhd_io_num = -1,
        .max_transfer_sz = 240 * 240 * 2 + 8 // matches max resolution in bytes
    };
    // SPI2_HOST is standard for ESP32-S3 displays
    ESP_ERROR_CHECK(spi_bus_initialize(SPI2_HOST, &buscfg, SPI_DMA_CH_AUTO));

    // Attach the LCD to the SPI Bus
    esp_lcd_panel_io_spi_config_t io_config = {
        .dc_gpio_num = SPI_DC,
        .cs_gpio_num = SPI_CS,
        .pclk_hz = 40 * 1000 * 1000, // 40 MHz clock speed
        .lcd_cmd_bits = 8,
        .lcd_param_bits = 8,
        .spi_mode = 0, // ST7789 standard SPI mode
        .trans_queue_depth = 10,
    };
    ESP_ERROR_CHECK(esp_lcd_new_panel_io_spi((esp_lcd_spi_bus_handle_t)SPI2_HOST, &io_config, &io_handle));

   // Initialize ST7789 
    esp_lcd_panel_dev_config_t panel_config = {
        .reset_gpio_num = SPI_RST,
        .rgb_ele_order = LCD_RGB_ELEMENT_ORDER_RGB, // modern ESP-IDF v5.2+ syntax
        .bits_per_pixel = 16, // RGB565 format
    };
    ESP_ERROR_CHECK(esp_lcd_new_panel_st7789(io_handle, &panel_config, &panel_handle));
    
    // Reset and wake up the display
    ESP_ERROR_CHECK(esp_lcd_panel_reset(panel_handle));
    ESP_ERROR_CHECK(esp_lcd_panel_init(panel_handle));

    // Invert colors for ST7789 screen to look normal
    ESP_ERROR_CHECK(esp_lcd_panel_invert_color(panel_handle, true)); 
    
    // Turn the screen on
    ESP_ERROR_CHECK(esp_lcd_panel_disp_on_off(panel_handle, true));
}


// Tick timer callback (tells LVGL how much time has passed)
void lvgl_tick_timer_cb(void *arg) {
    lv_tick_inc(1);
}

// Flush callback (The bridge between LVGL and your ST7789)
static void disp_flush(lv_disp_drv_t *disp_drv, const lv_area_t *area, lv_color_t *color_p) {
    esp_lcd_panel_handle_t panel = (esp_lcd_panel_handle_t) disp_drv->user_data;
    
    // Push the LVGL pixel buffer to the screen using ESP-IDF
    esp_lcd_panel_draw_bitmap(panel, area->x1, area->y1, area->x2 + 1, area->y2 + 1, color_p);
}

bool notify_lvgl_flush_ready(esp_lcd_panel_io_handle_t panel_io, esp_lcd_panel_io_event_data_t *edata, void *user_ctx) {
    lv_disp_drv_t *disp_driver = (lv_disp_drv_t *)user_ctx;
    lv_disp_flush_ready(disp_driver);
    return false;
}

// Main LVGL Initialization
void init_lvgl(void) {
    lv_init();

    // Allocate a draw buffer (1/10th of a 240x240 screen is 24 rows)
    static lv_disp_draw_buf_t draw_buf;
    static lv_color_t buf1[240 * 24]; 
    lv_disp_draw_buf_init(&draw_buf, buf1, NULL, 240 * 24);

    // Register the display driver with LVGL
    static lv_disp_drv_t disp_drv;
    lv_disp_drv_init(&disp_drv);
    disp_drv.hor_res = 240;
    disp_drv.ver_res = 240;
    disp_drv.flush_cb = disp_flush;
    disp_drv.draw_buf = &draw_buf;
    disp_drv.user_data = panel_handle;
    lv_disp_drv_register(&disp_drv);

    // Tell the ESP-IDF SPI driver to fire our callback when DMA finishes
    const esp_lcd_panel_io_callbacks_t cbs = {
        .on_color_trans_done = notify_lvgl_flush_ready,
    };
    esp_lcd_panel_io_register_event_callbacks(io_handle, &cbs, &disp_drv);

    // Start the 1ms tick timer
    const esp_timer_create_args_t tick_timer_args = {
        .callback = &lvgl_tick_timer_cb,
        .name = "lvgl_tick"
    };
    esp_timer_handle_t tick_timer;
    ESP_ERROR_CHECK(esp_timer_create(&tick_timer_args, &tick_timer));
    ESP_ERROR_CHECK(esp_timer_start_periodic(tick_timer, 1000));
}

// Button Setup (Interrupt-Driven)
static volatile bool button1Pressed = false;
static volatile bool button2Pressed = false;
static volatile int64_t lastIsrTime1 = 0;
static volatile int64_t lastIsrTime2 = 0;
static watch_screen_t currentDisplay = SCREEN_BOOTUP;
#define DEBOUNCE_MS 200

// ISR for Button 1
static void IRAM_ATTR button1_isr_handler(void *arg) {
    int64_t now = millis();
    if ((now - lastIsrTime1) > DEBOUNCE_MS) {
        lastIsrTime1 = now;
        button1Pressed = true;
    }
}
static void IRAM_ATTR button2_isr_handler(void *arg) {
    int64_t now = millis();
    if ((now - lastIsrTime2) > DEBOUNCE_MS) {
        lastIsrTime2 = now;
        button2Pressed = true;
    }
}

// Initialize Buttons
void init_buttons(void) {
    gpio_config_t io_conf = {
        .pin_bit_mask = (1ULL << BUTTON_1_PIN) | (1ULL << BUTTON_2_PIN),
        .mode = GPIO_MODE_INPUT,
        .pull_up_en = GPIO_PULLUP_ENABLE,    // Active-low buttons (pressed == 0)
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type = GPIO_INTR_NEGEDGE       // Trigger immediately on press
    };
    gpio_config(&io_conf);

    // Install GPIO ISR service
    esp_err_t err = gpio_install_isr_service(0);
    if (err != ESP_OK && err != ESP_ERR_INVALID_STATE) {
        ESP_ERROR_CHECK(err);
    }

    gpio_isr_handler_add(BUTTON_1_PIN, button1_isr_handler, NULL);
    gpio_isr_handler_add(BUTTON_2_PIN, button2_isr_handler, NULL);
}

// Loads the screen based on the desired option
void load_screen(watch_screen_t target) {
    currentDisplay = (watch_screen_t)(target % SCREEN_COUNT);

    switch(currentDisplay) {
        case SCREEN_BOOTUP:
            displayBootup();
            break;
        case SCREEN_SELECTION:
            displaySelection();
            break;
        case SCREEN_HOME:
            displayHome();
            break;
        case SCREEN_STOPWATCH:
            displayStopwatch();
            break;
        case SCREEN_ACTIVE:
            displayActive();
            break;
        case SCREEN_TRANSMISSION:
            displayTransmission();
            break;
        default:
            currentDisplay = SCREEN_SELECTION;
            displaySelection();
            break;
    }
}

// Switches the screen to the next screen, looping around when necessary
static watch_screen_t next_runtime_screen(watch_screen_t current) {
    watch_screen_t next = (watch_screen_t)((current + 1) % SCREEN_COUNT);
    if (next == SCREEN_BOOTUP) {
        next = SCREEN_SELECTION; // Skip bootup logo during normal watch navigation
    }
    return next;
}

static watch_screen_t selector_screen_control(watch_screen_t current) {
    watch_screen_t next = (watch_screen_t)((current + 1) % SCREEN_COUNT);
    if (next == SCREEN_BOOTUP) {
        next = SCREEN_SELECTION; // Skip bootup logo during normal watch navigation
    }
    return next;
}

// Gets the time of day
void set_rtc_epoch(time_t epoch_sec) {
    struct timeval tv = {
        .tv_sec = epoch_sec,
        .tv_usec = 0
    };
    settimeofday(&tv, NULL);
}

void init_rtc_default(void) {
    setenv("TZ", "MST7MDT,M3.2.0,M11.1.0", 1);
    tzset();
    set_rtc_epoch(1795460400);
}

// Estimates battery percent in a dead-reckoning style
static uint8_t get_estimated_battery_pct(void) {
    const int64_t full_runtime_sec = 14400; // 4 hours, 1% drop every 2.4 minutes
    int64_t elapsed_sec = esp_timer_get_time() / 1000000;
    int64_t percent = 100 - ((elapsed_sec * 100) / full_runtime_sec);

    if (percent < 5) {
        percent = 5; // Floor of 5% while board still has power
    }
    return (uint8_t)percent;
}

// Detects IMU Movement
static bool detect_imu_movement(float ax_g, float ay_g, float az_g) {
    static int64_t last_motion_us = 0;
    float mag = sqrtf(ax_g * ax_g + ay_g * ay_g + az_g * az_g);

    if (fabsf(mag - 1.0f) > 0.18f) {
        last_motion_us = esp_timer_get_time();
    }
    return (esp_timer_get_time() - last_motion_us) < 1200000;
}


// Main application loop
void app_main(void) {
    // Set up the ST7789 screen
    init_st7789();
    init_lvgl();
    init_I2C();
    init_buttons();
    setupScreens();
    vTaskDelay(pdMS_TO_TICKS(1500));
    displayBootup();

    // const uint32_t debounceDelay = 10; // 50ms delay
    
    // Heartbeat LED config
	gpio_config_t led_config = {
		.pin_bit_mask = 1ULL << USER_LED,
		.mode = GPIO_MODE_OUTPUT,
		.pull_up_en = GPIO_PULLUP_DISABLE,
		.pull_down_en = GPIO_PULLDOWN_DISABLE,
		.intr_type = GPIO_INTR_DISABLE,
	};
	gpio_config(&led_config);

    // Main polling loop
	while (1) {
        if (button1Pressed) {
            button1Pressed = false;
            load_screen(next_runtime_screen(currentDisplay));
        } else if (button2Pressed) {
            displayHome();
        } else {

            // Time Read
            if (clock_div >= 50) {
                clock_div = 0;
                time_t now;
                struct tm timeinfo;
                time(&now);
                localtime_r(&now, &timeinfo);
                //updateHomeScreen(&timeinfo);
                updateHomeScreen(&timeinfo, get_estimated_battery_pct());
            }
            clock_div++;

            // MPU data read)
            register_address = MPU_REG_ACCEL_XOUT_H;
            result = i2c_master_transmit_receive(mpu_handle, &register_address, 1,
                                                mpu_data, MPU_DATA_LENGTH, 50);

            if (result == ESP_OK) {
                ax = (int16_t)(((uint16_t)mpu_data[0] << 8) | mpu_data[1]);
                ay = (int16_t)(((uint16_t)mpu_data[2] << 8) | mpu_data[3]);
                az = (int16_t)(((uint16_t)mpu_data[4] << 8) | mpu_data[5]);
                gx = (int16_t)(((uint16_t)mpu_data[8] << 8) | mpu_data[9]);
                gy = (int16_t)(((uint16_t)mpu_data[10] << 8) | mpu_data[11]);
                gz = (int16_t)(((uint16_t)mpu_data[12] << 8) | mpu_data[13]);
                
                if (ui_div >= 15) { // 150ms
                    ui_div = 0;
                    float ax_g = ax / 16384.0f;
                    float ay_g = ay / 16384.0f;
                    float az_g = az / 16384.0f;
                    bool moving = detect_imu_movement(ax_g, ay_g, az_g);
                    uint16_t simulated_bpm = 60;
                    //moving = false;
                    //uint16_t simulated_bpm = moving ? 118 : 72; // Replace with MAX30101 BPM  later
                    updateActiveScreen(simulated_bpm, moving);
                    //updateActiveScreen(ax, ay, az, gx, gy, gz);
                }
                ui_div++;
            } else {
                ESP_LOGW("IMU", "MPU read failed: %s", esp_err_to_name(result));
            }
        }
        lv_timer_handler();
        vTaskDelay(pdMS_TO_TICKS(10));
	}
}




        // // Button Debounce Routines and Display Selections
        // int reading1 = gpio_get_level(BUTTON_1_PIN);
        // if (reading1 != lastButton1State) {
        //     lastDebounceTime1 = millis();
        // }
        //  if ((millis() - lastDebounceTime1) > debounceDelay) {
        //     if (reading1 != button1State) {
        //         button1State = reading1;

        //         if (button1State == 0) {
        //             if (currentDisplay > 2) {
        //                 currentDisplay = 0;
        //             }
        //             currentDisplay = nextDisplay;
        //             switch (currentDisplay) {
        //                 case SCREEN_BOOTUP: // Bootup Screen 
        //                     displayBootup();
        //                 break;
        //                 case SCREEN_HOME: // Clock Display
        //                     displayHome();
        //                 break;
        //                 case SCREEN_ACTIVE: // Heart Rate & Movement Data
        //                     displayActive();
        //             }
        //             nextDisplay++;

        //         }
        //     }
        // }
        // lastButton1State = reading1;

        // int reading2 = gpio_get_level(BUTTON_2_PIN);
        // if (reading2 != lastButton2State) {
        //     lastDebounceTime2 = millis();
        // }
        // if ((millis() - lastDebounceTime2) > debounceDelay) {
        //     if (reading2 != button2State) {
        //         button2State = reading2;

        //         if (button2State == 0) {
        //             //displayActive();
        //             //if (current_display == 2) {
                        
        //             //     for (uint32_t i = 0; i < log_pointer; i++) {
        //             //         uint16_t temp = (log_buffer+i)->ax;
        //             //         ESP_LOGI(MAIN, "log%d: ax: %d", i, temp);
        //             //     }
        //             // } else {
        //             //     ESP_ERROR_CHECK(esp_lcd_panel_disp_sleep(panel_handle, true));
        //             // }
        //         } else  {
        //             //displayBootup();
        //         }
        //     }
        // }
        // lastButton2State = reading2;

        //  // MPU data read
        // register_address = MPU_REG_ACCEL_XOUT_H;
        // result = i2c_master_transmit_receive(mpu_handle, &register_address, 1,
        //                         mpu_data, MPU_DATA_LENGTH, -1);
        // ESP_ERROR_CHECK(result);
        // ax = (int16_t)(((uint16_t)mpu_data[0] << 8) | mpu_data[1]);
        // ay = (int16_t)(((uint16_t)mpu_data[2] << 8) | mpu_data[3]);
        // az = (int16_t)(((uint16_t)mpu_data[4] << 8) | mpu_data[5]);
        // gx = (int16_t)(((uint16_t)mpu_data[8] << 8) | mpu_data[9]);
        // gy = (int16_t)(((uint16_t)mpu_data[10] << 8) | mpu_data[11]);
        // gz = (int16_t)(((uint16_t)mpu_data[12] << 8) | mpu_data[13]);




        




        // // Read heart rate BPM
        // for (int sample_count = 0; sample_count < 8; sample_count++) {
        //     if (!max30101_read_ir(heartrate_handle, &ir_value)) {
        //         break;
        //     }

        //     if (ir_baseline == 0) {
        //         ir_baseline = ir_value;
        //     }

        //     // Track the slowly changing DC level and detect a pulse above it.
        //     // baseline value for ambient light
        //     ir_baseline = (ir_baseline * 31 + ir_value) / 32;
        //     // Remove slowly changing DC component from IR reading
        //     uint32_t signal = ir_value > ir_baseline ? ir_value - ir_baseline : 0;
        //     // update previous peak from signal
        //     uint32_t previous_peak = signal_peak;
        //     // update current signal peak (new peak is 7/8ths old peak, 1/8th new signal)
        //     signal_peak = (signal_peak * 7 + signal) / 8;
        //     // Get the current time on the timer (ms)
        //     int64_t now_ms = esp_timer_get_time() / 1000;

        //     /**
        //      * Heartbeat recognition
        //      * ir_value > 5000 : assumes finger is present, otherwise ambient pulse
        //      * signal > 500 : pulse must be at least 500 counts above baseline
        //      * signal > previous_peak * 3/2 : signal must be 50% larger than recent smoothed level
        //      * now_ms - last_beat_ms > 300 : theoretical maximum limit for BPM of 200 
        //      *          -> 60000 ms in a minute / 300 ms = 200 BPM
        //      */
        //     if (ir_value > 5000 && signal > 500 && signal > previous_peak * 3 / 2 &&
        //         now_ms - last_beat_ms > 300) {
        //         // Ensure first detected pulse doesn't provide a data entry
        //         if (last_beat_ms != 0) {
        //             // calculate bpm (60000 ms in a minute / change in time)
        //             uint32_t bpm = 60000 / (now_ms - last_beat_ms);
        //             // If the BPM is reasonable, print it to log
        //             if (bpm >= 40 && bpm <= 220) {
        //                 stable_bpm = bpm;
        //             } 
        //         }
        //         // update last time for beat
        //         last_beat_ms = now_ms;
        //     }
        // } // End of while loop



       
        
      

   


        // // // Display 1: Timer
        // // if (display == 1) {
        // //     unsigned long time = millis() % 10000;
        // //     char message[14];
        // //     snprintf(message, sizeof(message),"     %lu   ", time);
        // //     ESP_ERROR_CHECK(sh1107_set_cursor(5, 7));
        // //     ESP_ERROR_CHECK(sh1107_print(message));
        // // }


        // // // Display 2: Heart Rate
        // // if (display == 2) {
        
        // //     bool sensor_success = false;
        // //     int read_attempts = 0;

        // //     // Try to read the buffer
        // //     // while (max30101_read_ir(heartrate_handle, &ir_value)) {
        // //     //     sensor_success = true;
        // //     //     read_attempts++;
        // //     //      // Read heart rate BPM
        // //     //     for (int sample_count = 0; sample_count < 8; sample_count++) {
        // //     //         if (!max30101_read_ir(heartrate_handle, &ir_value)) {
        // //     //             break;
        // //     //         }

        // //     //         if (ir_baseline == 0) {
        // //     //             ir_baseline = ir_value;
        // //     //         }

        // //     //         // Track the slowly changing DC level and detect a pulse above it.
        // //     //         // baseline value for ambient light
        // //     //         ir_baseline = (ir_baseline * 31 + ir_value) / 32;
        // //     //         // Remove slowly changing DC component from IR reading
        // //     //         uint32_t signal = ir_value > ir_baseline ? ir_value - ir_baseline : 0;
        // //     //         // update previous peak from signal
        // //     //         uint32_t previous_peak = signal_peak;
        // //     //         // update current signal peak (new peak is 7/8ths old peak, 1/8th new signal)
        // //     //         signal_peak = (signal_peak * 7 + signal) / 8;
        // //     //         // Get the current time on the timer (ms)
        // //     //         int64_t now_ms = esp_timer_get_time() / 1000;

        // //     //         /**
        // //     //          * Heartbeat recognition
        // //     //          * ir_value > 5000 : assumes finger is present, otherwise ambient pulse
        // //     //          * signal > 500 : pulse must be at least 500 counts above baseline
        // //     //          * signal > previous_peak * 3/2 : signal must be 50% larger than recent smoothed level
        // //     //          * now_ms - last_beat_ms > 300 : theoretical maximum limit for BPM of 200 
        // //     //          *          -> 60000 ms in a minute / 300 ms = 200 BPM
        // //     //          */
        // //     //         if (ir_value > 5000 && signal > 500 && signal > previous_peak * 3 / 2 &&
        // //     //             now_ms - last_beat_ms > 300) {
        // //     //             // Ensure first detected pulse doesn't provide a data entry
        // //     //             if (last_beat_ms != 0) {
        // //     //                 // calculate bpm (60000 ms in a minute / change in time)
        // //     //                 uint32_t bpm = 60000 / (now_ms - last_beat_ms);
        // //     //                 // If the BPM is reasonable, print it to log
        // //     //                 if (bpm >= 40 && bpm <= 220) {
        // //     //                     stable_bpm = bpm;
        // //     //                 } 
        // //     //             }
        // //     //             // update last time for beat
        // //     //             last_beat_ms = now_ms;
        // //     //         }
        // //     //     }
        // //     // } // End of while loop

        // //     // // Reset on failure (used for when buffer fills up after changing displays)
        // //     // static uint32_t fail_count = 0;
        // //     // if (!sensor_success) {
        // //     //     fail_count++;
        // //     //     if (fail_count > 20) { // If it fails for 20 consecutive loops (~200ms)
        // //     //         ESP_LOGW(HEART_RATE_TAG, "Sensor locked up. Re-initializing...");
        // //     //         max30101_init(heartrate_handle); // Reset the sensor hardware
        // //     //         fail_count = 0;                  // Reset counter
        // //     //     }
        // //     // } else {
        // //     //     fail_count = 0; // Reset counter on successful read
        // //     // }

        // //     // Display Data to screen
        // //     //ESP_LOGE(HEART_RATE_TAG, "(stable) Heart rate: %3lu BPM", (unsigned long)stable_bpm);
        // //     char message[32];
        // //     snprintf(message, sizeof(message),"  HR: %3lu BPM", (unsigned long)stable_bpm);
        // //     ESP_ERROR_CHECK(sh1107_set_cursor(5, 7));
        // //     ESP_ERROR_CHECK(sh1107_print(message));
        // // }   



        // // // Display 3: MPU
        // // if (display == 3) {
        // //     static uint32_t last_mpu_read = 0;

        // //     if (millis() - last_mpu_read > 250) {
        // //         last_mpu_read = millis();
            
                
        // //         char message[128];
        // //         snprintf(message, sizeof(message),
        // //                 "   AX:%6d\n   AY:%6d\n   AZ:%6d\n   GX:%6d\n   GY:%6d\n   GZ:%6d", 
        // //                  accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z);
        // //         ESP_ERROR_CHECK(sh1107_set_cursor(0, 7));
        // //         ESP_ERROR_CHECK(sh1107_print(message));
                
        // //         //MPU reading output to error log
        // //         ESP_LOGI(TAG, "WHO_AM_I: 0x%02X", mpu_data[0]);
        // //         register_address = MPU_REG_ACCEL_XOUT_H;
        // //         result = i2c_master_transmit_receive(mpu_handle, &register_address, 1,
        // //                                                 mpu_data, sizeof(mpu_data), -1);
        // //         ESP_ERROR_CHECK(result);
        // //         ESP_LOG_BUFFER_HEXDUMP(TAG, mpu_data, sizeof(mpu_data), ESP_LOG_INFO);
        // //     }
        // // }

        // // // Display 4 - temporary logging
        // // if (display == 4) {
        // //     ESP_ERROR_CHECK(sh1107_set_cursor(0,7));
        // //     ESP_ERROR_CHECK(sh1107_print("Logging..."));
        // //     struct log new_log = {
        // //         .ax = accel_x, // 16
        // //         .ay = accel_y, // 32
        // //         .az = accel_z, // 48
        // //         .gx = gyro_x, // 64
        // //         .gy = gyro_y, // 80
        // //         .gz = gyro_z, // 96
        // //         .heartrate = stable_bpm // 104
        // //     };
        // //     if (log_pointer < LOG_CAPACITY) {
        // //         log_buffer[log_pointer++] = new_log;
        // //     }
        // // }

        
        // // // Print to serial
        // // // ESP_LOGI(TAG, "AX: %d, AY: %d, AZ: %d, GX: %d, GY: %d, GZ: %d\n", accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z);
        // // // ESP_LOGI(MAIN, "\n");
        // static int counter = 0;
        // if (counter++ % 200 == 0) {
        //     printf("Heartbeat: System alive. Free Heap: %d bytes\n", (int)esp_get_free_heap_size());
        // }