/**
 * Code for the HUD of the SH1107 Display
 */

// Custom headers
#include "sh1107.h"

// Displays the selected screen
void displayHUD(int selection) {
    // Clear the screen ONCE instead of fighting 15 individual prints
    ESP_ERROR_CHECK(sh1107_clear());
    ESP_ERROR_CHECK(sh1107_set_cursor(0, 0));

    switch (selection) {
        case 0: // Bootup Screen
            ESP_ERROR_CHECK(sh1107_print("****************\n*  Welcome to  *\n*  Heartwatch  *\n*    Carson    *\n*    Zach      *\n*    Bobby     *\n*    Dylan     *\n****************\n"));
        break;

        case 1: // Time Telling Screen
            ESP_ERROR_CHECK(sh1107_print("****************\n* Time Display *\n*              *\n*              *\n****************\n"));
        break;

        case 2: // Activity Screen
            ESP_ERROR_CHECK(sh1107_print("****************\n*  Heart Rate  *\n*              *\n*              *\n****************\n"));
        break;

        case 3: // Motion Sensing with IMU
            ESP_ERROR_CHECK(sh1107_print("****************\n*    Motion    *\n*              *\n*              *\n****************\n"));
        break;

        case 4: // Screen Off
            ESP_ERROR_CHECK(sh1107_print(""));
        break;

        default:
            ESP_ERROR_CHECK(sh1107_print("****************\n*    Ready     *\n****************\n"));
    }
}