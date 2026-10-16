/**
 * Code for the HUD of the SH1107 Display
 */

// Custom headers
#include "sh1107.h"

// Displays the selected screen
void displayHUD(int selection) {
    switch (selection) {
        case 0: // Bootup Screen
            ESP_ERROR_CHECK(sh1107_print("****************\n"));
            ESP_ERROR_CHECK(sh1107_print("*              *\n"));
            ESP_ERROR_CHECK(sh1107_print("*  Welcome to  *\n"));
            ESP_ERROR_CHECK(sh1107_print("*     the      *\n"));
            ESP_ERROR_CHECK(sh1107_print("*  Heartwatch  *\n"));
            ESP_ERROR_CHECK(sh1107_print("*              *\n"));
            ESP_ERROR_CHECK(sh1107_print("*     Zach     *\n"));
            ESP_ERROR_CHECK(sh1107_print("*     Bobby    *\n"));
            ESP_ERROR_CHECK(sh1107_print("*     Dylan    *\n"));
            ESP_ERROR_CHECK(sh1107_print("*    Carson    *\n"));
            ESP_ERROR_CHECK(sh1107_print("*              *\n"));
            ESP_ERROR_CHECK(sh1107_print("* Initializing *\n"));
            ESP_ERROR_CHECK(sh1107_print("*     ...      *\n"));
            ESP_ERROR_CHECK(sh1107_print("*              *\n"));
            ESP_ERROR_CHECK(sh1107_print("*              *\n"));
            ESP_ERROR_CHECK(sh1107_print("****************\n"));
        break;

        case 1: // Time Telling Screen
            ESP_ERROR_CHECK(sh1107_print("****************\n"));
            ESP_ERROR_CHECK(sh1107_print("*              *\n"));
            ESP_ERROR_CHECK(sh1107_print("*              *\n"));
            ESP_ERROR_CHECK(sh1107_print("* Time Display *\n"));
            ESP_ERROR_CHECK(sh1107_print("*              *\n"));
            ESP_ERROR_CHECK(sh1107_print("*              *\n"));
            ESP_ERROR_CHECK(sh1107_print("*              *\n"));
            ESP_ERROR_CHECK(sh1107_print("*              *\n"));
            ESP_ERROR_CHECK(sh1107_print("*              *\n"));
            ESP_ERROR_CHECK(sh1107_print("*              *\n"));
            ESP_ERROR_CHECK(sh1107_print("*              *\n"));
            ESP_ERROR_CHECK(sh1107_print("*              *\n"));
            ESP_ERROR_CHECK(sh1107_print("*              *\n"));
            ESP_ERROR_CHECK(sh1107_print("*              *\n"));
            ESP_ERROR_CHECK(sh1107_print("*              *\n"));
            ESP_ERROR_CHECK(sh1107_print("****************\n"));
        break;

        case 2: // Activity Screen
            ESP_ERROR_CHECK(sh1107_print("****************\n"));
            ESP_ERROR_CHECK(sh1107_print("*              *\n"));
            ESP_ERROR_CHECK(sh1107_print("* Heart Rate:  *\n"));
            ESP_ERROR_CHECK(sh1107_print("*              *\n"));
            ESP_ERROR_CHECK(sh1107_print("*              *\n"));
            ESP_ERROR_CHECK(sh1107_print("*              *\n"));
            ESP_ERROR_CHECK(sh1107_print("*              *\n"));
            ESP_ERROR_CHECK(sh1107_print("*              *\n"));
            ESP_ERROR_CHECK(sh1107_print("*              *\n"));
            ESP_ERROR_CHECK(sh1107_print("*              *\n"));
            ESP_ERROR_CHECK(sh1107_print("*              *\n"));
            ESP_ERROR_CHECK(sh1107_print("*              *\n"));
            ESP_ERROR_CHECK(sh1107_print("*              *\n"));
            ESP_ERROR_CHECK(sh1107_print("*              *\n"));
            ESP_ERROR_CHECK(sh1107_print("*              *\n"));
            ESP_ERROR_CHECK(sh1107_print("****************\n"));
        break;
        case 3: // Motion Sensing with IMU
            ESP_ERROR_CHECK(sh1107_print("****************\n"));
            ESP_ERROR_CHECK(sh1107_print("*    Motion    *\n"));
            ESP_ERROR_CHECK(sh1107_print("*    Sensing:  *\n"));
            ESP_ERROR_CHECK(sh1107_print("*              *\n"));
            ESP_ERROR_CHECK(sh1107_print("*              *\n"));
            ESP_ERROR_CHECK(sh1107_print("*              *\n"));
            ESP_ERROR_CHECK(sh1107_print("*              *\n"));
            ESP_ERROR_CHECK(sh1107_print("*              *\n"));
            ESP_ERROR_CHECK(sh1107_print("*              *\n"));
            ESP_ERROR_CHECK(sh1107_print("*              *\n"));
            ESP_ERROR_CHECK(sh1107_print("*              *\n"));
            ESP_ERROR_CHECK(sh1107_print("*              *\n"));
            ESP_ERROR_CHECK(sh1107_print("*              *\n"));
            ESP_ERROR_CHECK(sh1107_print("*              *\n"));
            ESP_ERROR_CHECK(sh1107_print("*              *\n"));
            ESP_ERROR_CHECK(sh1107_print("****************\n"));
        break;
        default:
            ESP_ERROR_CHECK(sh1107_print("****************\n"));
            ESP_ERROR_CHECK(sh1107_print("*   Default    *\n"));
            ESP_ERROR_CHECK(sh1107_print("*              *\n"));
            ESP_ERROR_CHECK(sh1107_print("*              *\n"));
            ESP_ERROR_CHECK(sh1107_print("*              *\n"));
            ESP_ERROR_CHECK(sh1107_print("*              *\n"));
            ESP_ERROR_CHECK(sh1107_print("*              *\n"));
            ESP_ERROR_CHECK(sh1107_print("*              *\n"));
            ESP_ERROR_CHECK(sh1107_print("*              *\n"));
            ESP_ERROR_CHECK(sh1107_print("*              *\n"));
            ESP_ERROR_CHECK(sh1107_print("*              *\n"));
            ESP_ERROR_CHECK(sh1107_print("*              *\n"));
            ESP_ERROR_CHECK(sh1107_print("*              *\n"));
            ESP_ERROR_CHECK(sh1107_print("*              *\n"));
            ESP_ERROR_CHECK(sh1107_print("*              *\n"));
            ESP_ERROR_CHECK(sh1107_print("****************\n"));
    }
}