// Standard & FreeRTOS libraries
#include <stdio.h>
#include <stdbool.h>
#include <inttypes.h>
#include <time.h>
#include <math.h>
#include <sys/time.h>
#include <stdlib.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/gpio.h"
#include "driver/i2c_master.h"
#include "driver/spi_master.h"
#include "esp_log.h"
#include "esp_heap_caps.h"
#include "esp_psram.h"
#include "esp_timer.h"

// LVGL & LCD Headers
#include "esp_lcd_panel_io.h"
#include "esp_lcd_panel_vendor.h"
#include "esp_lcd_panel_ops.h"
#include "lvgl.h"

// Custom Headers
#include "max30101.h"
#include "mpu.h"
#include "hud.h"

// GPIO Pin Definitions
#define USER_LED     GPIO_NUM_21
#define I2C_SCL      GPIO_NUM_6
#define I2C_SDA      GPIO_NUM_5
#define SPI_SCLK     GPIO_NUM_7
#define SPI_MOSI     GPIO_NUM_9
#define BUTTON_1_PIN GPIO_NUM_1   // Multi-function mechanical button (1x / 2x / 3x press)
#define BUTTON_2_PIN GPIO_NUM_8   // Capacitive touch sensor (Wake display)
#define SPI_DC       GPIO_NUM_43
#define SPI_CS       GPIO_NUM_44
#define SPI_RST      GPIO_NUM_4

// Timing Constants
#define BTN_DEBOUNCE_US      50000    // 50 ms hardware debounce (short enough for triple-clicks)
#define MULTI_CLICK_GAP_US   100000   // 500 ms window to group single/double/triple presses
#define WAKE_TIMEOUT_MS      7000     // Keep display awake for 7 seconds after touch wake

// Log Tags
static const char *MPU_TAG      = "mpu";
static const char *MAX30101_TAG = "max30101";
static const char *MAIN_TAG     = "esp";

// PSRAM Logging Buffer
typedef struct {
    int16_t ax;
    int16_t ay;
    int16_t az;
    int16_t gx;
    int16_t gy;
    int16_t gz;
    uint16_t heartrate;
} sensor_log_t;
#define LOG_CAPACITY 500000
static sensor_log_t *log_buffer = NULL;
static uint32_t log_pointer   = 0;
static bool is_logging        = false;

// Display, Sleep & LVGL State
static esp_lcd_panel_io_handle_t io_handle    = NULL;
static esp_lcd_panel_handle_t    panel_handle = NULL;
static watch_screen_t currentDisplay          = SCREEN_BOOTUP;
static bool display_awake                     = true;
static uint32_t last_wake_ms                  = 0;
static uint8_t ui_div                         = 0;
static uint8_t clock_div                      = 0;

// Stopwatch State (Controlled by Button 1 double/triple press)
static bool stopwatch_running       = false;
static uint32_t stopwatch_elapsed_ms = 0;
static uint32_t stopwatch_last_tick_ms = 0;

// Button Interrupt & Multi-Click State
static portMUX_TYPE btn_spinlock          = portMUX_INITIALIZER_UNLOCKED;
static volatile uint8_t btn1_click_count  = 0;
static volatile int64_t btn1_last_press_us = 0;
static volatile bool touch_wake_pending   = false;
static volatile int64_t btn2_last_touch_us = 0;

// MAX30101 Heart-Rate Peak Detection State
static uint32_t green_value              = 0;
static uint32_t green_baseline           = 0;
static uint32_t green_peak               = 0;
static uint32_t previous_green_signal    = 0;
static uint32_t prior_green_signal       = 0;
static int64_t  previous_green_signal_ms = 0;
static int64_t  last_green_beat_ms       = 0;
static uint16_t stable_bpm               = 0;
static uint16_t bpm_history[3]           = {0};
static uint8_t  bpm_history_count        = 0;

// ============================================================================
// Utility, RTC & Sleep Helpers
// ============================================================================

uint32_t millis(void) {
    return (uint32_t)(esp_timer_get_time() / 1000ULL);
}

static void wake_display(void) {
    last_wake_ms = millis();
    if (!display_awake) {
        display_awake = true;
        esp_lcd_panel_disp_on_off(panel_handle, true);
        ESP_LOGI(MAIN_TAG, "Display woke");
    }
}

static void sleep_display(void) {
    if (display_awake) {
        display_awake = false;
        esp_lcd_panel_disp_on_off(panel_handle, false);
        ESP_LOGI(MAIN_TAG, "Display sleeping");
    }
}

static void reset_heart_rate_state(void) {
    green_baseline           = 0;
    green_peak               = 0;
    previous_green_signal    = 0;
    prior_green_signal       = 0;
    previous_green_signal_ms = 0;
    last_green_beat_ms       = 0;
    stable_bpm               = 0;
    bpm_history_count        = 0;
}

void set_rtc_epoch(time_t epoch_sec) {
    struct timeval tv = {
        .tv_sec  = epoch_sec,
        .tv_usec = 0
    };
    settimeofday(&tv, NULL);
}

void init_rtc_default(void) {
    setenv("TZ", "MST7MDT,M3.2.0,M11.1.0", 1);
    tzset();
    set_rtc_epoch(1795460400);
}

static uint8_t get_estimated_battery_pct(void) {
    const int64_t full_runtime_sec = 14400; // 4 hours (~1% drop per 2.4 min)
    int64_t elapsed_sec = esp_timer_get_time() / 1000000;
    int64_t percent = 100 - ((elapsed_sec * 100) / full_runtime_sec);
    return (uint8_t)(percent < 5 ? 5 : percent);
}

static bool detect_imu_movement(float ax_g, float ay_g, float az_g) {
    static float baseline_mag     = 0.0f;
    static uint8_t motion_hits    = 0;
    static int64_t last_motion_us = 0;

    float mag = sqrtf(ax_g * ax_g + ay_g * ay_g + az_g * az_g);
    if (baseline_mag == 0.0f) {
        baseline_mag = mag;
    }

    baseline_mag = 0.85f * baseline_mag + 0.15f * mag;
    float dynamic_g = fabsf(mag - baseline_mag);

    if (dynamic_g > 0.08f) {
        if (motion_hits < 3) {
            motion_hits++;
        }
        if (motion_hits >= 3) {
            last_motion_us = esp_timer_get_time();
        }
    } else {
        motion_hits = 0;
    }

    return (last_motion_us > 0) && ((esp_timer_get_time() - last_motion_us) < 800000);
}

// ============================================================================
// Hardware Initialization (I2C Sensors, ST7789 SPI, LVGL, Button ISRs)
// ============================================================================

void init_I2C(void) {
    i2c_master_bus_config_t i2c_mst_config = {
        .clk_source = I2C_CLK_SRC_DEFAULT,
        .i2c_port   = I2C_NUM_1,
        .scl_io_num = I2C_SCL,
        .sda_io_num = I2C_SDA,
        .glitch_ignore_cnt = 7,
        .flags.enable_internal_pullup = true,
    };

    i2c_master_bus_handle_t bus_handle;
    ESP_ERROR_CHECK(i2c_new_master_bus(&i2c_mst_config, &bus_handle));

    ESP_ERROR_CHECK(mpu_init(bus_handle) ? ESP_OK : ESP_FAIL);
    ESP_ERROR_CHECK(max30101_init(bus_handle));
    ESP_LOGI(MAIN_TAG, "I2C sensors initialized");
}

void init_st7789(void) {
    spi_bus_config_t buscfg = {
        .sclk_io_num     = SPI_SCLK,
        .mosi_io_num     = SPI_MOSI,
        .miso_io_num     = -1,
        .quadwp_io_num   = -1,
        .quadhd_io_num   = -1,
        .max_transfer_sz = 240 * 240 * 2 + 8
    };
    ESP_ERROR_CHECK(spi_bus_initialize(SPI2_HOST, &buscfg, SPI_DMA_CH_AUTO));

    esp_lcd_panel_io_spi_config_t io_config = {
        .dc_gpio_num       = SPI_DC,
        .cs_gpio_num       = SPI_CS,
        .pclk_hz           = 40 * 1000 * 1000,
        .lcd_cmd_bits      = 8,
        .lcd_param_bits    = 8,
        .spi_mode          = 0,
        .trans_queue_depth = 10,
    };
    ESP_ERROR_CHECK(esp_lcd_new_panel_io_spi((esp_lcd_spi_bus_handle_t)SPI2_HOST, &io_config, &io_handle));

    esp_lcd_panel_dev_config_t panel_config = {
        .reset_gpio_num = SPI_RST,
        .rgb_ele_order  = LCD_RGB_ELEMENT_ORDER_RGB,
        .bits_per_pixel = 16,
    };
    ESP_ERROR_CHECK(esp_lcd_new_panel_st7789(io_handle, &panel_config, &panel_handle));
    ESP_ERROR_CHECK(esp_lcd_panel_reset(panel_handle));
    ESP_ERROR_CHECK(esp_lcd_panel_init(panel_handle));
    ESP_ERROR_CHECK(esp_lcd_panel_invert_color(panel_handle, true));
    ESP_ERROR_CHECK(esp_lcd_panel_disp_on_off(panel_handle, true));
}

static void lvgl_tick_timer_cb(void *arg) {
    lv_tick_inc(1);
}

static void disp_flush(lv_disp_drv_t *disp_drv, const lv_area_t *area, lv_color_t *color_p) {
    esp_lcd_panel_handle_t panel = (esp_lcd_panel_handle_t)disp_drv->user_data;
    esp_lcd_panel_draw_bitmap(panel, area->x1, area->y1, area->x2 + 1, area->y2 + 1, color_p);
}

static bool notify_lvgl_flush_ready(esp_lcd_panel_io_handle_t panel_io, esp_lcd_panel_io_event_data_t *edata, void *user_ctx) {
    lv_disp_drv_t *disp_driver = (lv_disp_drv_t *)user_ctx;
    lv_disp_flush_ready(disp_driver);
    return false;
}

void init_lvgl(void) {
    lv_init();

    static lv_disp_draw_buf_t draw_buf;
    static lv_color_t buf1[240 * 24];
    lv_disp_draw_buf_init(&draw_buf, buf1, NULL, 240 * 24);

    static lv_disp_drv_t disp_drv;
    lv_disp_drv_init(&disp_drv);
    disp_drv.hor_res   = 240;
    disp_drv.ver_res   = 240;
    disp_drv.flush_cb  = disp_flush;
    disp_drv.draw_buf  = &draw_buf;
    disp_drv.user_data = panel_handle;
    lv_disp_drv_register(&disp_drv);

    const esp_lcd_panel_io_callbacks_t cbs = {
        .on_color_trans_done = notify_lvgl_flush_ready,
    };
    esp_lcd_panel_io_register_event_callbacks(io_handle, &cbs, &disp_drv);

    const esp_timer_create_args_t tick_timer_args = {
        .callback = &lvgl_tick_timer_cb,
        .name     = "lvgl_tick"
    };
    esp_timer_handle_t tick_timer;
    ESP_ERROR_CHECK(esp_timer_create(&tick_timer_args, &tick_timer));
    ESP_ERROR_CHECK(esp_timer_start_periodic(tick_timer, 1000));
}

// ISR for Button 1: Counts debounced presses for single/double/triple actions
static void IRAM_ATTR button1_isr_handler(void *arg) {
    int64_t now = esp_timer_get_time();
    taskENTER_CRITICAL_ISR(&btn_spinlock);
    if ((now - btn1_last_press_us) > BTN_DEBOUNCE_US) {
        btn1_last_press_us = now;
        if (btn1_click_count < 3) {
            btn1_click_count++;
        }
    }
    taskEXIT_CRITICAL_ISR(&btn_spinlock);
}

// ISR for Button 2: Capacitive touch sensor wakes the display
static void IRAM_ATTR button2_isr_handler(void *arg) {
    int64_t now = esp_timer_get_time();
    if ((now - btn2_last_touch_us) > BTN_DEBOUNCE_US) {
        btn2_last_touch_us = now;
        touch_wake_pending = true;
    }
}

void init_buttons(void) {
    // Button 1: Active-low mechanical button with internal pull-up
    gpio_config_t btn1_conf = {
        .pin_bit_mask = (1ULL << BUTTON_1_PIN),
        .mode         = GPIO_MODE_INPUT,
        .pull_up_en   = GPIO_PULLUP_ENABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type    = GPIO_INTR_NEGEDGE
    };
    gpio_config(&btn1_conf);

    // Button 2: Touch sensor (configured active-high; switch to PULLUP + NEGEDGE if active-low)
    gpio_config_t btn2_conf = {
        .pin_bit_mask = (1ULL << BUTTON_2_PIN),
        .mode         = GPIO_MODE_INPUT,
        .pull_up_en   = GPIO_PULLDOWN_DISABLE,
        .pull_down_en = GPIO_PULLDOWN_ENABLE,
        .intr_type    = GPIO_INTR_POSEDGE
    };
    gpio_config(&btn2_conf);

    esp_err_t err = gpio_install_isr_service(0);
    if (err != ESP_OK && err != ESP_ERR_INVALID_STATE) {
        ESP_ERROR_CHECK(err);
    }

    gpio_isr_handler_add(BUTTON_1_PIN, button1_isr_handler, NULL);
    gpio_isr_handler_add(BUTTON_2_PIN, button2_isr_handler, NULL);
}

// ============================================================================
// Screen Navigation & Multi-Press Action Handlers
// ============================================================================

void load_screen(watch_screen_t target) {
    currentDisplay = (watch_screen_t)(target % SCREEN_COUNT);

    switch (currentDisplay) {
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
            currentDisplay = SCREEN_HOME;
            displayHome();
            break;
    }
}

static watch_screen_t next_runtime_screen(watch_screen_t current) {
    watch_screen_t next = (watch_screen_t)((current + 1) % SCREEN_COUNT);
    if (next == SCREEN_BOOTUP) {
        next = SCREEN_SELECTION;
    }
    return next;
}

static void transmit_psram_logs(void) {
    ESP_LOGI(MAIN_TAG, "Transmitting %" PRIu32 " logged entries...", log_pointer);
    for (uint32_t i = 0; i < log_pointer; i++) {
        sensor_log_t *entry = &log_buffer[i];
        ESP_LOGI(MAIN_TAG,
                 "log%" PRIu32 ": ax: %d, ay: %d, az: %d, "
                 "gx: %d, gy: %d, gz: %d, heartrate: %u",
                 i, entry->ax, entry->ay, entry->az,
                 entry->gx, entry->gy, entry->gz,
                 (unsigned int)(entry->heartrate / 2));
    }
    reset_heart_rate_state();
    log_pointer = 0;
    ESP_LOGI(MAIN_TAG, "Transmission complete");
}

static void toggle_psram_logging(void) {
    if (!is_logging) {
        log_pointer = 0;
        is_logging  = true;
        ESP_LOGI(MAIN_TAG, "PSRAM logging started");
    } else {
        if (log_pointer < LOG_CAPACITY) {
            log_buffer[log_pointer++] = (sensor_log_t){
                .ax = 0, .ay = 0, .az = 0,
                .gx = 0, .gy = 0, .gz = 0,
                .heartrate = 0
            };
        }
        is_logging = false;
        ESP_LOGI(MAIN_TAG, "PSRAM logging stopped (%" PRIu32 " entries)", log_pointer);
    }
}

// 1x Press: Strictly screen navigation
static void handle_button1_single_press(void) {
    load_screen(next_runtime_screen(currentDisplay));
}

// 2x Press: Contextual "Select" action
static void handle_button1_double_press(void) {
    switch (currentDisplay) {
        case SCREEN_STOPWATCH:
            stopwatch_running = !stopwatch_running;
            if (stopwatch_running) {
                stopwatch_last_tick_ms = millis();
            }
            ESP_LOGI(MAIN_TAG, "Stopwatch %s", stopwatch_running ? "STARTED" : "STOPPED");
            break;

        case SCREEN_SELECTION:
            // Screen selection logic will be wired here once selection UI is ready
            ESP_LOGI(MAIN_TAG, "Selection screen select triggered");
            break;

        case SCREEN_ACTIVE:
            toggle_psram_logging();
            break;

        case SCREEN_TRANSMISSION:
            if (is_logging) {
                toggle_psram_logging(); // Close active log cleanly before transmitting
            }
            transmit_psram_logs();
            break;

        default:
            break;
    }
}

// 3x Press: Stopwatch restart (and optional logging reset)
static void handle_button1_triple_press(void) {
    stopwatch_running    = false;
    stopwatch_elapsed_ms = 0;
    ESP_LOGI(MAIN_TAG, "Stopwatch RESET");

    // Optional: If triple-press should also clear logged data
    // is_logging = false;
    // log_pointer = 0;
}

// ============================================================================
// ======================== MAX30101 Heart-Rate Sampling ======================
// ============================================================================

static void sample_max30101_bpm(void) {
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
                            uint16_t total   = bpm_history[0];
                            for (int i = 1; i < 3; i++) {
                                if (bpm_history[i] < minimum) minimum = bpm_history[i];
                                if (bpm_history[i] > maximum) maximum = bpm_history[i];
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

        prior_green_signal       = previous_green_signal;
        previous_green_signal    = signal;
        previous_green_signal_ms = sample_time;
        if (last_green_beat_ms != 0 &&
            sample_time - last_green_beat_ms > 2500) {
            reset_heart_rate_state();
        }
    }
}

// ============================================================================
// Main Application
// ============================================================================

void app_main(void) {
    // Allocate PSRAM logging buffer
    ESP_LOGI(MAIN_TAG, "PSRAM detected: %zu bytes", esp_psram_get_size());
    log_buffer = heap_caps_calloc(LOG_CAPACITY, sizeof(*log_buffer), MALLOC_CAP_SPIRAM);
    ESP_ERROR_CHECK(log_buffer != NULL ? ESP_OK : ESP_ERR_NO_MEM);

    // Initialize Display, LVGL, Sensors, RTC, and Button/Touch ISRs
    init_st7789();
    init_lvgl();
    init_I2C();
    init_rtc_default();
    init_buttons();
    setupScreens();

    // Show boot splash briefly, then switch to Home screen and start wake timer
    load_screen(SCREEN_BOOTUP);
    lv_timer_handler();
    vTaskDelay(pdMS_TO_TICKS(1500));
    load_screen(SCREEN_HOME);
    wake_display();

    // Heartbeat LED config
    gpio_config_t led_config = {
        .pin_bit_mask = 1ULL << USER_LED,
        .mode         = GPIO_MODE_OUTPUT,
        .pull_up_en   = GPIO_PULLUP_DISABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type    = GPIO_INTR_DISABLE,
    };
    gpio_config(&led_config);

    int16_t accel_x = 0, accel_y = 0, accel_z = 0;
    int16_t gyro_x  = 0, gyro_y  = 0, gyro_z  = 0;
    bool max30101_active = true;

    while (1) {
        uint32_t now_ms = millis();
        int64_t  now_us = esp_timer_get_time();

        // Handle Button 2 (Touch Sensor): Wake screen and refresh 7s timer
        if (touch_wake_pending) {
            touch_wake_pending = false;
            wake_display();
        }

        // Evaluate Button 1 multi-click state once 350 ms gap elapses (or 3 clicks reached)
        uint8_t finalized_clicks = 0;
        taskENTER_CRITICAL(&btn_spinlock);
        if (btn1_click_count > 0 &&
            (btn1_click_count >= 3 || (now_us - btn1_last_press_us) >= MULTI_CLICK_GAP_US)) {
            finalized_clicks = btn1_click_count;
            btn1_click_count = 0;
        }
        taskEXIT_CRITICAL(&btn_spinlock);

        if (finalized_clicks > 0) {
            wake_display(); // Any Button 1 interaction also keeps the screen awake
            if (finalized_clicks == 1) {
                handle_button1_single_press();
            } else if (finalized_clicks == 2) {
                handle_button1_double_press();
            } else {
                handle_button1_triple_press();
            }
        } else {
            // Auto-sleep display after 7 seconds of inactivity
            if (display_awake && (now_ms - last_wake_ms >= WAKE_TIMEOUT_MS)) {
                sleep_display();
            }

            // Update running stopwatch elapsed time
            if (stopwatch_running) {
                stopwatch_elapsed_ms += (now_ms - stopwatch_last_tick_ms);
                stopwatch_last_tick_ms = now_ms;
                // TODO: Call updateStopwatchScreen(stopwatch_elapsed_ms) once implemented in hud.c
            }

            // Manage MAX30101 power (active when viewing Active screen awake, or when logging)
            bool heart_rate_needed = (display_awake && currentDisplay == SCREEN_ACTIVE) || is_logging;
            if (heart_rate_needed && !max30101_active) {
                if (max30101_wakeup() == ESP_OK) {
                    max30101_active = true;
                    reset_heart_rate_state();
                } else {
                    ESP_LOGW(MAX30101_TAG, "Failed to wake MAX30101");
                }
            } else if (!heart_rate_needed && max30101_active) {
                if (max30101_sleep() == ESP_OK) {
                    max30101_active = false;
                } else {
                    ESP_LOGW(MAX30101_TAG, "Failed to sleep MAX30101");
                }
            }

            if (heart_rate_needed && max30101_active) {
                sample_max30101_bpm();
            }

            // Read MPU-6050 (keeps running in background if PSRAM logging is active)
            bool mpu_ok = mpu_read(&accel_x, &accel_y, &accel_z, &gyro_x, &gyro_y, &gyro_z);
            if (!mpu_ok) {
                ESP_LOGW(MPU_TAG, "MPU read failed");
            } else if (is_logging && log_pointer < LOG_CAPACITY) {
                log_buffer[log_pointer++] = (sensor_log_t){
                    .ax = accel_x, .ay = accel_y, .az = accel_z,
                    .gx = gyro_x,  .gy = gyro_y,  .gz = gyro_z,
                    .heartrate = stable_bpm
                };
            }

            // Update active LVGL screen widgets only while display is awake
            if (display_awake) {
                if (++clock_div >= 50) { // ~500 ms
                    clock_div = 0;
                    time_t now;
                    struct tm timeinfo;
                    time(&now);
                    localtime_r(&now, &timeinfo);
                    updateHomeScreen(&timeinfo, get_estimated_battery_pct());
                }

                if (++ui_div >= 15) { // ~150 ms
                    ui_div = 0;
                    float ax_g = accel_x / 16384.0f;
                    float ay_g = accel_y / 16384.0f;
                    float az_g = accel_z / 16384.0f;
                    bool moving = mpu_ok ? detect_imu_movement(ax_g, ay_g, az_g) : false;
                    updateActiveScreen(stable_bpm / 2, moving);
                }
            }
        }

        lv_timer_handler();
        vTaskDelay(pdMS_TO_TICKS(10));
    }
}