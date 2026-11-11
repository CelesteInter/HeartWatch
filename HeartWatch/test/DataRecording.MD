# How to record data using screen on a UNIX machine (macOS, Linux)

## Recording on the watch
1. Press the record button (button 2) and start excercising
2. Press the button again when finished with your excercise
3. Connect the watch to Computer via USB
4. Continue to logging on computer

## Logging on Computer
1. Open Terminal
2. Navigatte to ~/.../Heartwatch/HeartWatch/test
3. Find your USB port using `ls /dev/tty.*` - should look like `usbserial-combinationOf#'sandLetters` (`usbmodem101`)
4. Run `screen -L /dev/tty.[yourUSBport]`
5. Press the "record" button on the Heartwatch, hold it for a second or two if nescessary
    - You will know the print was successful if the watch screen goes blank and the logs appear in the terminal
6. Stop logging with `ctrl + a` then `ctrl + |`
7. Select y

Your new log appears in the test folder as `screenlog.0`!
