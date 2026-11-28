/**
 * Code for the HUD of the Screen Display
 */

// Custom headers
#include "sh1107.h"

// LVGL Headers
#include "esp_lcd_panel_io.h"
#include "esp_lcd_panel_vendor.h"
#include "esp_lcd_panel_ops.h"
#include "lvgl.h"
#include "hud.h"

// ============================================================================
// 32x32 1-Bit Pixel Art Bitmaps (128 bytes each, MSB = left pixel)
// ============================================================================


static const uint8_t heart_large_data[128] = {
    0b00000000, 0b00000000, 0b00000000, 0b00000000,
    0b00000000, 0b00000000, 0b00000000, 0b00000000,
    0b00000000, 0b00000000, 0b00000000, 0b00000000,
    0b00000111, 0b11000000, 0b00000011, 0b11100000,
    0b00001111, 0b11100000, 0b00000111, 0b11110000,
    0b00011111, 0b11110000, 0b00001111, 0b11111000,
    0b00111111, 0b11111000, 0b00011111, 0b11111100,
    0b01111111, 0b11111100, 0b00111111, 0b11111110,
    0b01111111, 0b11111110, 0b01111111, 0b11111110,
    0b11111111, 0b11111111, 0b11111111, 0b11111111,
    0b11111111, 0b11111111, 0b11111111, 0b11111111,
    0b11111111, 0b11111111, 0b11111111, 0b11111111,
    0b11111111, 0b11111111, 0b11111111, 0b11111111,
    0b11111111, 0b11111111, 0b11111111, 0b11111111,
    0b01111111, 0b11111111, 0b11111111, 0b11111110,
    0b01111111, 0b11111111, 0b11111111, 0b11111110,
    0b00111111, 0b11111111, 0b11111111, 0b11111100,
    0b00111111, 0b11111111, 0b11111111, 0b11111100,
    0b00011111, 0b11111111, 0b11111111, 0b11111000,
    0b00001111, 0b11111111, 0b11111111, 0b11110000,
    0b00000111, 0b11111111, 0b11111111, 0b11100000,
    0b00000011, 0b11111111, 0b11111111, 0b11000000,
    0b00000001, 0b11111111, 0b11111111, 0b10000000,
    0b00000000, 0b11111111, 0b11111111, 0b00000000,
    0b00000000, 0b01111111, 0b11111110, 0b00000000,
    0b00000000, 0b00111111, 0b11111100, 0b00000000,
    0b00000000, 0b00011111, 0b11111000, 0b00000000,
    0b00000000, 0b00001111, 0b11110000, 0b00000000,
    0b00000000, 0b00000111, 0b11100000, 0b00000000,
    0b00000000, 0b00000011, 0b11000000, 0b00000000,
    0b00000000, 0b00000001, 0b10000000, 0b00000000,
    0b00000000, 0b00000000, 0b00000000, 0b00000000,
};

static const uint8_t heart_small_data[128] = {
    0b00000000, 0b00000000, 0b00000000, 0b00000000,
    0b00000000, 0b00000000, 0b00000000, 0b00000000,
    0b00000000, 0b00000000, 0b00000000, 0b00000000,
    0b00000000, 0b00000000, 0b00000000, 0b00000000,
    0b00000000, 0b00000000, 0b00000000, 0b00000000,
    0b00000000, 0b00000000, 0b00000000, 0b00000000,
    0b00000000, 0b00000000, 0b00000000, 0b00000000,
    0b00000001, 0b11100000, 0b00000111, 0b10000000,
    0b00000011, 0b11110000, 0b00001111, 0b11000000,
    0b00000111, 0b11111000, 0b00011111, 0b11100000,
    0b00001111, 0b11111100, 0b00111111, 0b11110000,
    0b00001111, 0b11111110, 0b01111111, 0b11110000,
    0b00001111, 0b11111111, 0b11111111, 0b11110000,
    0b00001111, 0b11111111, 0b11111111, 0b11110000,
    0b00001111, 0b11111111, 0b11111111, 0b11110000,
    0b00000111, 0b11111111, 0b11111111, 0b11100000,
    0b00000111, 0b11111111, 0b11111111, 0b11100000,
    0b00000011, 0b11111111, 0b11111111, 0b11000000,
    0b00000001, 0b11111111, 0b11111111, 0b10000000,
    0b00000000, 0b11111111, 0b11111111, 0b00000000,
    0b00000000, 0b01111111, 0b11111110, 0b00000000,
    0b00000000, 0b00111111, 0b11111100, 0b00000000,
    0b00000000, 0b00011111, 0b11111000, 0b00000000,
    0b00000000, 0b00001111, 0b11110000, 0b00000000,
    0b00000000, 0b00000111, 0b11100000, 0b00000000,
    0b00000000, 0b00000011, 0b11000000, 0b00000000,
    0b00000000, 0b00000001, 0b10000000, 0b00000000,
    0b00000000, 0b00000000, 0b00000000, 0b00000000,
    0b00000000, 0b00000000, 0b00000000, 0b00000000,
    0b00000000, 0b00000000, 0b00000000, 0b00000000,
    0b00000000, 0b00000000, 0b00000000, 0b00000000,
    0b00000000, 0b00000000, 0b00000000, 0b00000000,
};

// Frame 0: Full stride
static const uint8_t runner_f0_data[128] = {
    0b00000000, 0b00000000, 0b00000000, 0b00000000,
    0b00000000, 0b00000000, 0b00000000, 0b00000000,
    0b00000000, 0b00011110, 0b00000000, 0b00000000,
    0b00000000, 0b00111111, 0b00000000, 0b00000000,
    0b00000000, 0b00111111, 0b00000000, 0b00000000,
    0b00000000, 0b00011110, 0b00000000, 0b00000000,
    0b00000000, 0b00001100, 0b00000000, 0b00000000,
    0b00000000, 0b00011110, 0b00000000, 0b00000000,
    0b00000000, 0b01111111, 0b11000000, 0b00000000,
    0b00000001, 0b11001100, 0b01100000, 0b00000000,
    0b00000011, 0b00001100, 0b00110000, 0b00000000,
    0b00000110, 0b00001100, 0b00110000, 0b00000000,
    0b00000000, 0b00001100, 0b00000000, 0b00000000,
    0b00000000, 0b00001100, 0b00000000, 0b00000000,
    0b00000000, 0b00001100, 0b00000000, 0b00000000,
    0b00000000, 0b00001100, 0b00000000, 0b00000000,
    0b00000000, 0b00011110, 0b00000000, 0b00000000,
    0b00000000, 0b00110011, 0b00000000, 0b00000000,
    0b00000000, 0b01100001, 0b10000000, 0b00000000,
    0b00000000, 0b11000000, 0b11000000, 0b00000000,
    0b00000001, 0b10000000, 0b01100000, 0b00000000,
    0b00000011, 0b00000000, 0b01100000, 0b00000000,
    0b00000110, 0b00000000, 0b00110000, 0b00000000,
    0b00001100, 0b00000000, 0b00110000, 0b00000000,
    0b00011000, 0b00000000, 0b00011000, 0b00000000,
    0b00011100, 0b00000000, 0b00011110, 0b00000000,
    0b00000000, 0b00000000, 0b00000000, 0b00000000,
    0b00000000, 0b00000000, 0b00000000, 0b00000000,
    0b00000000, 0b00000000, 0b00000000, 0b00000000,
    0b00000000, 0b00000000, 0b00000000, 0b00000000,
    0b00000000, 0b00000000, 0b00000000, 0b00000000,
    0b00000000, 0b00000000, 0b00000000, 0b00000000,
};

// Frame 1: Passing / idle stance
static const uint8_t runner_f1_data[128] = {
    0b00000000, 0b00000000, 0b00000000, 0b00000000,
    0b00000000, 0b00000000, 0b00000000, 0b00000000,
    0b00000000, 0b00011110, 0b00000000, 0b00000000,
    0b00000000, 0b00111111, 0b00000000, 0b00000000,
    0b00000000, 0b00111111, 0b00000000, 0b00000000,
    0b00000000, 0b00011110, 0b00000000, 0b00000000,
    0b00000000, 0b00001100, 0b00000000, 0b00000000,
    0b00000000, 0b00111111, 0b00000000, 0b00000000,
    0b00000000, 0b01101101, 0b10000000, 0b00000000,
    0b00000000, 0b01101101, 0b10000000, 0b00000000,
    0b00000000, 0b01101101, 0b10000000, 0b00000000,
    0b00000000, 0b01101101, 0b10000000, 0b00000000,
    0b00000000, 0b00001100, 0b00000000, 0b00000000,
    0b00000000, 0b00001100, 0b00000000, 0b00000000,
    0b00000000, 0b00001100, 0b00000000, 0b00000000,
    0b00000000, 0b00001100, 0b00000000, 0b00000000,
    0b00000000, 0b00011110, 0b00000000, 0b00000000,
    0b00000000, 0b00011011, 0b00000000, 0b00000000,
    0b00000000, 0b00011011, 0b00000000, 0b00000000,
    0b00000000, 0b00011011, 0b00000000, 0b00000000,
    0b00000000, 0b00011011, 0b00000000, 0b00000000,
    0b00000000, 0b00011011, 0b00000000, 0b00000000,
    0b00000000, 0b00011011, 0b00000000, 0b00000000,
    0b00000000, 0b00011011, 0b00000000, 0b00000000,
    0b00000000, 0b00011011, 0b00000000, 0b00000000,
    0b00000000, 0b00011111, 0b10000000, 0b00000000,
    0b00000000, 0b00000000, 0b00000000, 0b00000000,
    0b00000000, 0b00000000, 0b00000000, 0b00000000,
    0b00000000, 0b00000000, 0b00000000, 0b00000000,
    0b00000000, 0b00000000, 0b00000000, 0b00000000,
    0b00000000, 0b00000000, 0b00000000, 0b00000000,
    0b00000000, 0b00000000, 0b00000000, 0b00000000,
};

// Frame 2: Knee drive
static const uint8_t runner_f2_data[128] = {
    0b00000000, 0b00000000, 0b00000000, 0b00000000,
    0b00000000, 0b00000000, 0b00000000, 0b00000000,
    0b00000000, 0b00011110, 0b00000000, 0b00000000,
    0b00000000, 0b00111111, 0b00000000, 0b00000000,
    0b00000000, 0b00111111, 0b00000000, 0b00000000,
    0b00000000, 0b00011110, 0b00000000, 0b00000000,
    0b00000000, 0b00001100, 0b00000000, 0b00000000,
    0b00000000, 0b00011110, 0b00000000, 0b00000000,
    0b00000000, 0b00111111, 0b10000000, 0b00000000,
    0b00000000, 0b01101100, 0b11000000, 0b00000000,
    0b00000000, 0b11001100, 0b11000000, 0b00000000,
    0b00000000, 0b11001100, 0b00000000, 0b00000000,
    0b00000000, 0b00001100, 0b00000000, 0b00000000,
    0b00000000, 0b00001100, 0b00000000, 0b00000000,
    0b00000000, 0b00001100, 0b00000000, 0b00000000,
    0b00000000, 0b00001100, 0b00000000, 0b00000000,
    0b00000000, 0b00011110, 0b00000000, 0b00000000,
    0b00000000, 0b00110111, 0b00000000, 0b00000000,
    0b00000000, 0b01100011, 0b00000000, 0b00000000,
    0b00000000, 0b11000011, 0b00000000, 0b00000000,
    0b00000001, 0b10000110, 0b00000000, 0b00000000,
    0b00000001, 0b10001100, 0b00000000, 0b00000000,
    0b00000000, 0b00001100, 0b00000000, 0b00000000,
    0b00000000, 0b00011000, 0b00000000, 0b00000000,
    0b00000000, 0b00011100, 0b00000000, 0b00000000,
    0b00000000, 0b00000000, 0b00000000, 0b00000000,
    0b00000000, 0b00000000, 0b00000000, 0b00000000,
    0b00000000, 0b00000000, 0b00000000, 0b00000000,
    0b00000000, 0b00000000, 0b00000000, 0b00000000,
    0b00000000, 0b00000000, 0b00000000, 0b00000000,
    0b00000000, 0b00000000, 0b00000000, 0b00000000,
    0b00000000, 0b00000000, 0b00000000, 0b00000000,
};

#define MAKE_PIXEL_SPRITE(arr) { \
    .header.cf = LV_IMG_CF_ALPHA_1BIT, \
    .header.always_zero = 0, \
    .header.reserved = 0, \
    .header.w = 32, \
    .header.h = 32, \
    .data_size = 128, \
    .data = (arr), \
}

static const lv_img_dsc_t img_heart_large = MAKE_PIXEL_SPRITE(heart_large_data);
static const lv_img_dsc_t img_heart_small = MAKE_PIXEL_SPRITE(heart_small_data);
static const lv_img_dsc_t img_runner_f0   = MAKE_PIXEL_SPRITE(runner_f0_data);
static const lv_img_dsc_t img_runner_f1   = MAKE_PIXEL_SPRITE(runner_f1_data);
static const lv_img_dsc_t img_runner_f2   = MAKE_PIXEL_SPRITE(runner_f2_data);

static const lv_img_dsc_t *runner_frames[4] = {
    &img_runner_f0,
    &img_runner_f1,
    &img_runner_f2,
    &img_runner_f1
};

static uint16_t current_bpm       = 0;
static bool     user_is_logging    = false;
static bool     heart_expanded    = false;
static uint8_t  runner_step       = 0;

// Bootup screen, only happens when powering on the watch
static lv_obj_t *screen_bootup = NULL; 
static lv_obj_t *label_bootup = NULL;
static lv_obj_t *img_heart_1 = NULL; 

// Selection Screen, Allows user to choose what screen they want
static lv_obj_t *screen_selection = NULL;
static lv_obj_t *label_to_home = NULL;
static lv_obj_t *label_to_active = NULL;
static lv_obj_t *label_to_stopwatch = NULL;
static lv_obj_t *label_to_transmission = NULL;

// Home screen; displays time and date like regular clock
static lv_obj_t *screen_home = NULL; 
static lv_obj_t *label_battery = NULL;
static lv_obj_t *label_date = NULL;
static lv_obj_t *label_time = NULL;

// Stopwatch screen, self-explanatory
static lv_obj_t *screen_stopwatch = NULL; 
static lv_obj_t *label_stopwatch = NULL;
static lv_obj_t *label_start_stop = NULL;

// Active screen, displays HR, active step count, and movement indicator
static lv_obj_t *screen_active = NULL; 
static lv_obj_t *label_bpm = NULL;
static lv_obj_t *label_status = NULL;
static lv_obj_t *img_heart = NULL; 
static lv_obj_t *img_runner = NULL;

// Transmission screen, indicates data is being transferred from watch
static lv_obj_t *screen_transmission = NULL; 
static lv_obj_t *label_transmission = NULL;
static lv_obj_t *label_download = NULL;

static void heart_timer_cb(lv_timer_t *timer) {
    if (screen_active == NULL || lv_scr_act() != screen_active) {
        return;
    }

    if (current_bpm == 0) {
        heart_expanded = false;
        lv_img_set_src(img_heart, &img_heart_small);
        lv_timer_set_period(timer, 500);
        return;
    }

    uint32_t beat_period_ms = 60000 / current_bpm;
    if (beat_period_ms < 250) {
        beat_period_ms = 250; // Clamp at 240 BPM max
    }

    if (!heart_expanded) {
        heart_expanded = true;
        lv_img_set_src(img_heart, &img_heart_large);
        lv_timer_set_period(timer, 150); // Quick systolic beat
    } else {
        heart_expanded = false;
        lv_img_set_src(img_heart, &img_heart_small);
        lv_timer_set_period(timer, beat_period_ms - 150);
    }
}

static void runner_timer_cb(lv_timer_t *timer) {
    if (screen_active == NULL || lv_scr_act() != screen_active) {
        return;
    }

    if (!user_is_logging) {
        runner_step = 1;
        lv_img_set_src(img_runner, &img_runner_f1);
        return;
    }

    runner_step = (runner_step + 1) % 4;
    lv_img_set_src(img_runner, runner_frames[runner_step]);
}




// Initializes LVGL Screens
void setupScreens() {

    // ========== Bootup Screen ==========
    screen_bootup = lv_obj_create(NULL);
    lv_obj_set_style_bg_color(screen_bootup, lv_color_hex(0x0D1117), 0);
  
    label_bootup = lv_label_create(screen_bootup);
    lv_obj_set_style_text_font(label_bootup, &lv_font_montserrat_16, 0);
    lv_obj_align(label_bootup, LV_ALIGN_CENTER, 0, 80);
    lv_obj_set_style_text_color(label_bootup, lv_color_hex(0xFFFFFF), 0);
    lv_label_set_text(label_bootup, "Heartwatch Initializing");

    img_heart_1 = lv_img_create(screen_bootup);
    lv_img_set_src(img_heart_1, &img_heart_large);
    lv_obj_set_style_img_recolor(img_heart_1, lv_color_hex(0xFF2E4C), 0);
    lv_obj_set_style_img_recolor_opa(img_heart_1, LV_OPA_COVER, 0);
    lv_obj_align(img_heart_1, LV_ALIGN_CENTER, 0, 0);



    // ========== Selection Screen ==========
    screen_selection = lv_obj_create(NULL);
    lv_obj_set_style_bg_color(screen_selection, lv_color_hex(0x0D1117), 0);

    label_to_home = lv_label_create(screen_selection);
    lv_obj_set_style_text_font(label_to_home, &lv_font_montserrat_48, 0);
    lv_label_set_text(label_to_home, LV_SYMBOL_HOME);
    lv_obj_set_style_text_color(label_to_home, lv_color_hex(0x3FB950), 0);
    lv_obj_align(label_to_home, LV_ALIGN_CENTER, 0, -80);

    label_to_stopwatch = lv_label_create(screen_selection);
    lv_obj_set_style_text_font(label_to_stopwatch, &lv_font_montserrat_36, 0);
    lv_label_set_text(label_to_stopwatch, LV_SYMBOL_PLAY);
    lv_obj_set_style_text_color(label_to_stopwatch, lv_color_hex(0xa89d9a), 0);
    lv_obj_align(label_to_stopwatch, LV_ALIGN_CENTER, 80, 0);

    label_to_active = lv_label_create(screen_selection);
    lv_obj_set_style_text_font(label_to_active, &lv_font_montserrat_36, 0);
    lv_label_set_text(label_to_active, LV_SYMBOL_LOOP);
    lv_obj_set_style_text_color(label_to_active, lv_color_hex(0xa89d9a), 0);
    lv_obj_align(label_to_active, LV_ALIGN_CENTER, 0, 80);

    label_to_transmission = lv_label_create(screen_selection);
    lv_obj_set_style_text_font(label_to_transmission, &lv_font_montserrat_36, 0);
    lv_label_set_text(label_to_transmission, LV_SYMBOL_DRIVE);
    lv_obj_set_style_text_color(label_to_transmission, lv_color_hex(0xa89d9a), 0);
    lv_obj_align(label_to_transmission, LV_ALIGN_CENTER, -80, 0);



    // ========== Home Screen ==========
    screen_home = lv_obj_create(NULL);
    lv_obj_set_style_bg_color(screen_home, lv_color_hex(0x0D1117), 0);

    label_battery = lv_label_create(screen_home);
    lv_obj_set_style_text_font(label_battery, &lv_font_montserrat_16, 0);
    lv_label_set_text(label_battery, "100% " LV_SYMBOL_BATTERY_FULL);
    lv_obj_set_style_text_color(label_battery, lv_color_hex(0x3FB950), 0);
    lv_obj_align(label_battery, LV_ALIGN_CENTER, 0, -100);

    label_date = lv_label_create(screen_home);
    lv_obj_set_style_text_font(label_date, &lv_font_montserrat_20, 0);
    lv_obj_set_style_text_color(label_date, lv_color_hex(0xFFFFFF), 0);
    lv_obj_set_style_text_letter_space(label_date, 2, 0);
    lv_label_set_text(label_date, "Mon, Nov 26");
    lv_obj_align(label_date, LV_ALIGN_CENTER, 0, -60);

    label_time = lv_label_create(screen_home);
    lv_obj_set_style_text_font(label_time, &lv_font_montserrat_48, 0);
    lv_label_set_text(label_time, "12:00");
    lv_obj_set_style_text_color(label_time, lv_color_hex(0x58A6FF), 0);
    lv_obj_set_style_text_letter_space(label_time, 2, 0);
    lv_obj_align(label_time, LV_ALIGN_CENTER, 0, 0);



    // ========== Stopwatch Screen ==========
    screen_stopwatch = lv_obj_create(NULL);
    lv_obj_set_style_bg_color(screen_stopwatch, lv_color_hex(0x0D1117), 0);

    label_stopwatch = lv_label_create(screen_stopwatch);
    lv_obj_set_style_text_font(label_stopwatch, &lv_font_montserrat_48, 0);
    lv_obj_set_style_text_color(label_stopwatch, lv_color_hex(0xFFFFFF), 0);
    lv_obj_set_style_text_letter_space(label_stopwatch, 2, 0);
    lv_label_set_text(label_stopwatch, "00:00:00");
    lv_obj_align(label_stopwatch, LV_ALIGN_CENTER, 0, 0);

    label_start_stop = lv_label_create(screen_stopwatch);
    lv_obj_set_style_text_font(label_start_stop, &lv_font_montserrat_30, 0);
    lv_label_set_text(label_start_stop, LV_SYMBOL_PLAY);
    lv_obj_set_style_text_color(label_start_stop, lv_color_hex(0x3FB950), 0);
    lv_obj_align(label_start_stop, LV_ALIGN_CENTER, 80, 80);



    // ========== Active Screen ==========
    screen_active = lv_obj_create(NULL);
    lv_obj_set_style_bg_color(screen_active, lv_color_hex(0x0D1117), 0);

    img_heart = lv_img_create(screen_active);
    lv_img_set_src(img_heart, &img_heart_small);
    lv_obj_set_style_img_recolor(img_heart, lv_color_hex(0xFF2E4C), 0);
    lv_obj_set_style_img_recolor_opa(img_heart, LV_OPA_COVER, 0);
    lv_obj_align(img_heart, LV_ALIGN_CENTER, -70, -40);

    label_bpm = lv_label_create(screen_active);
    lv_label_set_text(label_bpm, "-- BPM");
    lv_obj_set_style_text_font(label_bpm, &lv_font_montserrat_36, 0);
    lv_obj_set_style_text_color(label_bpm, lv_color_hex(0xFFFFFF), 0);
    lv_obj_align(label_bpm, LV_ALIGN_CENTER, 30, -40);

    // Bottom-half: Running Stick Figure + IMU Status
    img_runner = lv_img_create(screen_active);
    lv_img_set_src(img_runner, &img_runner_f1);
    lv_obj_set_style_img_recolor(img_runner, lv_color_hex(0x58A6FF), 0);
    lv_obj_set_style_img_recolor_opa(img_runner, LV_OPA_COVER, 0);
    lv_obj_align(img_runner, LV_ALIGN_CENTER, -70, 40);

    label_status = lv_label_create(screen_active);
    lv_label_set_text(label_status, " Activity\n  Ready");
    lv_obj_set_style_text_font(label_status, &lv_font_montserrat_30, 0);
    lv_obj_set_style_text_color(label_status, lv_color_hex(0x8B949E), 0);
    lv_obj_align(label_status, LV_ALIGN_CENTER, 30, 40);

    // Start internal LVGL animation timers
    lv_timer_create(heart_timer_cb, 500, NULL);
    lv_timer_create(runner_timer_cb, 200, NULL);


    // ========== Data Transmission Screen ========== 
    screen_transmission = lv_obj_create(NULL);
    lv_obj_set_style_bg_color(screen_transmission, lv_color_hex(0x0D1117), 0);

    label_transmission = lv_label_create(screen_transmission);
    lv_label_set_text(label_transmission, "  Heartwatch\nReady To Sync");
    lv_obj_set_style_text_font(label_transmission, &lv_font_montserrat_20, 0);
    lv_obj_set_style_text_color(label_transmission, lv_color_hex(0xFFFFFF), 0);
    lv_obj_align(label_transmission, LV_ALIGN_CENTER, 0, -60);
    
    label_download = lv_label_create(screen_transmission);
    lv_obj_set_style_text_font(label_download, &lv_font_montserrat_48, 0);
    lv_label_set_text(label_download, LV_SYMBOL_DOWNLOAD);
    lv_obj_set_style_text_color(label_download, lv_color_hex(0xa89d9a), 0);
    lv_obj_align(label_download, LV_ALIGN_CENTER, 0, 60);

}

void displayBootup() {
    lv_scr_load(screen_bootup);
}

void displaySelection() {
    lv_scr_load(screen_selection);
}

void displayHome() {
    lv_scr_load(screen_home);
}

void displayStopwatch() {
    lv_scr_load(screen_stopwatch);
}

void displayActive() {
    lv_scr_load(screen_active); 
}

void displayTransmission() {
    lv_scr_load(screen_transmission);
    updateTransmissionScreen(false);
}

void updateHomeScreen(const struct tm *timeinfo, uint8_t battery_pct) {
//void updateHomeScreen(const struct tm *timeinfo) {
    if (lv_scr_act() != screen_home) {
        return;
    }

    // Update date and time
    char date_buf[64];
    char time_buf[64];

    strftime(date_buf, sizeof(date_buf), "%a, %b %d", timeinfo);   
    strftime(time_buf, sizeof(time_buf), "%I:%M", timeinfo);

    lv_label_set_text(label_date, date_buf);
    lv_label_set_text(label_time, time_buf);

    // Update battery percentage
    if (battery_pct > 100) {
        battery_pct = 100;
    }

    const char *bat_icon;
    lv_color_t bat_color;

    if (battery_pct > 80) {
        bat_icon = LV_SYMBOL_BATTERY_FULL;
        bat_color = lv_color_hex(0x3FB950);
    } else if (battery_pct > 60) {
        bat_icon = LV_SYMBOL_BATTERY_3;
        bat_color = lv_color_hex(0x3FB950);
    } else if (battery_pct > 40) {
        bat_icon = LV_SYMBOL_BATTERY_2;
        bat_color = lv_color_hex(0x3FB950);
    } else if (battery_pct > 20) {
        bat_icon = LV_SYMBOL_BATTERY_1;
        bat_color = lv_color_hex(0xD29922);
    } else {
        bat_icon = LV_SYMBOL_BATTERY_EMPTY;
        bat_color = lv_color_hex(0xF85149);
    }

    char bat_buf[16];
    snprintf(bat_buf, sizeof(bat_buf), "%u%% %s", battery_pct, bat_icon);
    lv_obj_set_style_text_color(label_battery, bat_color, 0);
    lv_label_set_text(label_battery, bat_buf);

}



void updateSelectionScreen(watch_screen_t current_screen) {
    if (lv_scr_act() != screen_selection) {
        return;
    }

    switch (current_screen) {
        case SCREEN_HOME:
            lv_obj_set_style_text_font(label_to_home, &lv_font_montserrat_48, 0);
            lv_label_set_text(label_to_home, LV_SYMBOL_HOME);
            lv_obj_set_style_text_color(label_to_home, lv_color_hex(0x3FB950), 0);
            lv_obj_align(label_to_home, LV_ALIGN_CENTER, 0, -80);

            lv_obj_set_style_text_font(label_to_stopwatch, &lv_font_montserrat_36, 0);
            lv_label_set_text(label_to_stopwatch, LV_SYMBOL_PLAY);
            lv_obj_set_style_text_color(label_to_stopwatch, lv_color_hex(0xa89d9a), 0);
            lv_obj_align(label_to_stopwatch, LV_ALIGN_CENTER, 80, 0);

            lv_obj_set_style_text_font(label_to_active, &lv_font_montserrat_36, 0);
            lv_label_set_text(label_to_active, LV_SYMBOL_LOOP);
            lv_obj_set_style_text_color(label_to_active, lv_color_hex(0xa89d9a), 0);
            lv_obj_align(label_to_active, LV_ALIGN_CENTER, 0, 80);

            lv_obj_set_style_text_font(label_to_transmission, &lv_font_montserrat_36, 0);
            lv_label_set_text(label_to_transmission, LV_SYMBOL_DRIVE);
            lv_obj_set_style_text_color(label_to_transmission, lv_color_hex(0xa89d9a), 0);
            lv_obj_align(label_to_transmission, LV_ALIGN_CENTER, -80, 0);
            break;
        case SCREEN_STOPWATCH:
            lv_obj_set_style_text_font(label_to_home, &lv_font_montserrat_36, 0);
            lv_label_set_text(label_to_home, LV_SYMBOL_HOME);
            lv_obj_set_style_text_color(label_to_home, lv_color_hex(0xa89d9a), 0);
            lv_obj_align(label_to_home, LV_ALIGN_CENTER, 0, -80);

            lv_obj_set_style_text_font(label_to_stopwatch, &lv_font_montserrat_48, 0);
            lv_label_set_text(label_to_stopwatch, LV_SYMBOL_PLAY);
            lv_obj_set_style_text_color(label_to_stopwatch, lv_color_hex(0x3FB950), 0);
            lv_obj_align(label_to_stopwatch, LV_ALIGN_CENTER, 80, 0);

            lv_obj_set_style_text_font(label_to_active, &lv_font_montserrat_36, 0);
            lv_label_set_text(label_to_active, LV_SYMBOL_LOOP);
            lv_obj_set_style_text_color(label_to_active, lv_color_hex(0xa89d9a), 0);
            lv_obj_align(label_to_active, LV_ALIGN_CENTER, 0, 80);

            lv_obj_set_style_text_font(label_to_transmission, &lv_font_montserrat_36, 0);
            lv_label_set_text(label_to_transmission, LV_SYMBOL_DRIVE);
            lv_obj_set_style_text_color(label_to_transmission, lv_color_hex(0xa89d9a), 0);
            lv_obj_align(label_to_transmission, LV_ALIGN_CENTER, -80, 0);
            break;
        case SCREEN_ACTIVE:
            lv_obj_set_style_text_font(label_to_home, &lv_font_montserrat_36, 0);
            lv_label_set_text(label_to_home, LV_SYMBOL_HOME);
            lv_obj_set_style_text_color(label_to_home, lv_color_hex(0xa89d9a), 0);
            lv_obj_align(label_to_home, LV_ALIGN_CENTER, 0, -80);

            lv_obj_set_style_text_font(label_to_stopwatch, &lv_font_montserrat_36, 0);
            lv_label_set_text(label_to_stopwatch, LV_SYMBOL_PLAY);
            lv_obj_set_style_text_color(label_to_stopwatch, lv_color_hex(0xa89d9a), 0);
            lv_obj_align(label_to_stopwatch, LV_ALIGN_CENTER, 80, 0);

            lv_obj_set_style_text_font(label_to_active, &lv_font_montserrat_48, 0);
            lv_label_set_text(label_to_active, LV_SYMBOL_LOOP);
            lv_obj_set_style_text_color(label_to_active, lv_color_hex(0x3FB950), 0);
            lv_obj_align(label_to_active, LV_ALIGN_CENTER, 0, 80);

            lv_obj_set_style_text_font(label_to_transmission, &lv_font_montserrat_36, 0);
            lv_label_set_text(label_to_transmission, LV_SYMBOL_DRIVE);
            lv_obj_set_style_text_color(label_to_transmission, lv_color_hex(0xa89d9a), 0);
            lv_obj_align(label_to_transmission, LV_ALIGN_CENTER, -80, 0);
            break;
        case SCREEN_TRANSMISSION:
            lv_obj_set_style_text_font(label_to_home, &lv_font_montserrat_36, 0);
            lv_label_set_text(label_to_home, LV_SYMBOL_HOME);
            lv_obj_set_style_text_color(label_to_home, lv_color_hex(0xa89d9a), 0);
            lv_obj_align(label_to_home, LV_ALIGN_CENTER, 0, -80);

            lv_obj_set_style_text_font(label_to_stopwatch, &lv_font_montserrat_36, 0);
            lv_label_set_text(label_to_stopwatch, LV_SYMBOL_PLAY);
            lv_obj_set_style_text_color(label_to_stopwatch, lv_color_hex(0xa89d9a), 0);
            lv_obj_align(label_to_stopwatch, LV_ALIGN_CENTER, 80, 0);

            lv_obj_set_style_text_font(label_to_active, &lv_font_montserrat_36, 0);
            lv_label_set_text(label_to_active, LV_SYMBOL_LOOP);
            lv_obj_set_style_text_color(label_to_active, lv_color_hex(0xa89d9a), 0);
            lv_obj_align(label_to_active, LV_ALIGN_CENTER,0, 80);

            lv_obj_set_style_text_font(label_to_transmission, &lv_font_montserrat_48, 0);
            lv_label_set_text(label_to_transmission, LV_SYMBOL_DRIVE);
            lv_obj_set_style_text_color(label_to_transmission, lv_color_hex(0x3FB950), 0);
            lv_obj_align(label_to_transmission, LV_ALIGN_CENTER, -80, 0);
            break;
        default:
            break;
    }
}

void updateStopwatchScreen(uint32_t elapsed_ms, bool is_running) {
    if (label_stopwatch == NULL) {
        return;
    }

    static uint32_t last_rendered_ms = UINT32_MAX;
    if (elapsed_ms == last_rendered_ms) {
        return;
    }
    last_rendered_ms = elapsed_ms;

    uint32_t total_seconds = elapsed_ms / 1000;
    uint32_t hours         = total_seconds / 3600;
    uint32_t minutes       = (total_seconds / 60) % 60;
    uint32_t seconds       = total_seconds % 60;
    uint32_t centiseconds  = (elapsed_ms % 1000) / 10; 

    char time_buf[16];
    if (hours > 0) {
        // Switch to HH:MM:SS once the stopwatch passes 59:59.99
        snprintf(time_buf, sizeof(time_buf), "%02" PRIu32 ":%02" PRIu32 ":%02" PRIu32,
                 hours, minutes, seconds);
    } else {
        // Standard MM:SS.cc format (rolls seconds at 60, minutes at 60)
        snprintf(time_buf, sizeof(time_buf), "%02" PRIu32 ":%02" PRIu32 ".%02" PRIu32,
                 minutes, seconds, centiseconds);
    }
    if (is_running) {
        lv_label_set_text(label_start_stop, LV_SYMBOL_PAUSE);
    } else {
        lv_label_set_text(label_start_stop, LV_SYMBOL_PLAY);
    }

    lv_label_set_text(label_stopwatch, time_buf);
}

void updateActiveScreen(uint16_t bpm, bool is_logging) {
    current_bpm    = bpm;
    user_is_logging = is_logging;

    if (lv_scr_act() != screen_active) {
        return;
    }

   char bpm_buf[16];
    if (bpm > 0) {
        snprintf(bpm_buf, sizeof(bpm_buf), "%u BPM", bpm);
    } else {
        snprintf(bpm_buf, sizeof(bpm_buf), "-- BPM");
    }
    lv_label_set_text(label_bpm, bpm_buf);

    if (is_logging) {
        lv_label_set_text(label_status, "   Activity\n Tracking!");
        lv_obj_set_style_text_color(label_status, lv_color_hex(0x3FB950), 0);
    } else {
        lv_label_set_text(label_status, " Activity\n  Ready");
        lv_obj_set_style_text_color(label_status, lv_color_hex(0x8B949E), 0);
    }
}


void updateTransmissionScreen(bool is_transmitting) {
    if (is_transmitting) {
        lv_label_set_text(label_transmission, " Heartwatch\n--Syncing--");
        lv_obj_set_style_text_color(label_download, lv_color_hex(0x3FB950), 0); 
    } else {
        lv_label_set_text(label_transmission, " Heartwatch\nReady To Sync");
        lv_obj_set_style_text_color(label_download, lv_color_hex(0xa89d9a), 0); 
    }
}