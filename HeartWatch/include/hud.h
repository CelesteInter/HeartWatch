/**
 * Header code for the HUD of the SH1107 Display
 */

typedef enum {
    SCREEN_BOOTUP = 0,
    SCREEN_HOME = 1,
    SCREEN_ACTIVE = 2,
    SCREEN_TRANSMISSION = 3
} watch_screen_t;

/**
 * Initializes LVGL Screens
 */
void setupScreens();



/**
 * Displays the selected screen
 * 0 = Bootup Screen
 * 1 = Time-telling display
 * 2 = Heartrate Sensing Display
 * 3 = Motion Sensing Screen (IMU)
 */
 //void displayHUD(int selection);


