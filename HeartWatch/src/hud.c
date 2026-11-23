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

// Bootup screen, only happens when powering on the watch
static lv_obj_t *screen_bootup = NULL; 
static lv_obj_t *label_bootup = NULL;
static lv_obj_t *led_bootup = NULL;

// Home screen; displays time and date like regular clock
static lv_obj_t *screen_home = NULL; 
static lv_obj_t *label_battery = NULL;
static lv_obj_t *label_date = NULL;
static lv_obj_t *label_time = NULL;

// Stopwatch screen, self-explanatory
static lv_obj_t *screen_stopwatch = NULL; 
static lv_obj_t *label_stopwatch = NULL;

// Active screen, displays HR, active step count, and movement indicator
static lv_obj_t *screen_active = NULL; 
static lv_obj_t *label_active = NULL;
static lv_obj_t *label_accel = NULL;
static lv_obj_t *label_gyro = NULL;

// Transmission screen, indicates data is being transferred from watch
static lv_obj_t *screen_transmission = NULL; 
static lv_obj_t *label_transmission = NULL;


// Initializes LVGL Screens
void setupScreens() {

    // Bootup Screen
    screen_bootup = lv_obj_create(NULL);
  
    label_bootup = lv_label_create(screen_bootup);
    lv_label_set_text(label_bootup, "Heartwatch\nBootup Screen!\n");
    lv_obj_align(label_bootup, LV_ALIGN_CENTER, 0, 0);

    led_bootup = lv_led_create(screen_bootup);
    lv_led_set_color(led_bootup, lv_color_hex(0xff0080));
    lv_led_set_brightness(led_bootup, 255);
    lv_obj_align(led_bootup, LV_ALIGN_CENTER, 40, 40);


    // Home Screen
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


    // Stopwatch Screen
    screen_stopwatch = lv_obj_create(NULL);

    label_stopwatch = lv_label_create(screen_stopwatch);
    lv_label_set_text(label_stopwatch, "Heartwatch\nHome Screen");
    lv_obj_align(label_stopwatch, LV_ALIGN_CENTER, 0, 0   );

    // Active Screen
    screen_active = lv_obj_create(NULL);

    label_active = lv_label_create(screen_active);
    lv_label_set_text(label_active, "Heartwatch\nActive Screen");
    lv_obj_align(label_active, LV_ALIGN_CENTER, 0, -60);

    label_accel = lv_label_create(screen_active);
    lv_label_set_text(label_accel, "Accel (g):\nX: 0.0  Y: 0.0  Z: 0.0");
    lv_obj_align(label_accel, LV_ALIGN_CENTER, 0, -20);

    label_gyro = lv_label_create(screen_active);
    lv_label_set_text(label_gyro, "Gyro (dps):\nX: 0.0  Y: 0.0  Z: 0.0");
    lv_obj_align(label_gyro, LV_ALIGN_CENTER, 0, 35);


    // Data Transmission Screen
    screen_transmission = lv_obj_create(NULL);

    label_transmission = lv_label_create(screen_transmission);
    lv_label_set_text(label_transmission, "Heartwatch\nTransmitting Data...");
    lv_obj_align(label_transmission, LV_ALIGN_CENTER, 0, -60);


    //lv_obj_t *intro_label = lv_label_create(screen_bootup);

    //lv_scr_load(screen_bootup); // to swap screens, load 
}

void displayBootup() {
    lv_scr_load(screen_bootup);
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



void updateActiveScreen(int16_t ax, int16_t ay, int16_t az, int16_t gx, int16_t gy, int16_t gz) {
    if (lv_scr_act() != screen_active) {
        return;
    }

    char buf[128];

    snprintf(buf, sizeof(buf), "Accel (g):\nX: %d Y: %d Z: %d", ax, ay, az);
    lv_label_set_text(label_accel, buf);

    snprintf(buf, sizeof(buf), "Gyro (dps):\nX: %d  Y: %d  Z: %d", gx, gy, gz);
    lv_label_set_text(label_gyro, buf);
}






// SH1107 Screen Options
// void displayHUD(int selection) {
//     // Clear the screen ONCE instead of fighting 15 individual prints
//     ESP_ERROR_CHECK(sh1107_clear());
//     ESP_ERROR_CHECK(sh1107_set_cursor(0, 0));

//     switch (selection) {
//         case 0: // Bootup Screen
//             ESP_ERROR_CHECK(sh1107_print("****************\n*  Welcome to  *\n*  Heartwatch  *\n*    Carson    *\n*    Zach      *\n*    Bobby     *\n*    Dylan     *\n****************\n"));
//         break;

//         case 1: // Time Telling Screen
//             ESP_ERROR_CHECK(sh1107_print("****************\n* Time Display *\n*              *\n*              *\n****************\n"));
//         break;

//         case 2: // Activity Screen
//             ESP_ERROR_CHECK(sh1107_print("****************\n*  Heart Rate  *\n*              *\n*              *\n****************\n"));
//         break;

//         case 3: // Motion Sensing with IMU
//             ESP_ERROR_CHECK(sh1107_print("****************\n*    Motion    *\n*              *\n*              *\n****************\n"));
//         break;

//         case 4: // Screen Off
//             ESP_ERROR_CHECK(sh1107_print(""));
//         break;

//         default:
//             ESP_ERROR_CHECK(sh1107_print("****************\n*    Ready     *\n****************\n"));
//     }
// }