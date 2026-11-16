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

static lv_obj_t *screen_bootup = NULL;
static lv_obj_t *screen_home = NULL;
static lv_obj_t *screen_active = NULL;
static lv_obj_t *screen_transmission = NULL;
static lv_obj_t *label_bootup = NULL;
static lv_obj_t *label_home = NULL;
static lv_obj_t *label_active = NULL;
static lv_obj_t *label_accel = NULL;
static lv_obj_t *label_gyro = NULL;
static lv_obj_t *label_transmission = NULL;
static lv_obj_t *led_bootup = NULL;


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

    label_home = lv_label_create(screen_home);
    lv_label_set_text(label_home, "Heartwatch\nHome Screen");
    lv_obj_align(label_home, LV_ALIGN_CENTER, 0, 0   );


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

void displayActive() {
    lv_scr_load(screen_active);
}

void displayTransmission() {
    lv_scr_load(screen_transmission);
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