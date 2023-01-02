import win32gui
import win32ui
import win32con
import win32api
import time
from PIL import Image, ImageChops

def capture_foreground_window():
    hwnd = win32gui.GetForegroundWindow()
    left, top, right, bottom = win32gui.GetWindowRect(hwnd)
    width = right - left
    height = bottom - top

    hwndDC = win32gui.GetWindowDC(hwnd)
    mfcDC  = win32ui.CreateDCFromHandle(hwndDC)
    saveDC = mfcDC.CreateCompatibleDC()

    saveBitMap = win32ui.CreateBitmap()
    saveBitMap.CreateCompatibleBitmap(mfcDC, width, height)

    saveDC.SelectObject(saveBitMap)

    result = win32gui.PrintWindow(hwnd, saveDC, 0)

    bmpinfo = saveBitMap.GetInfo()
    bmpstr = saveBitMap.GetBitmapBits(True)

    im = Image.frombuffer(
        'RGB',
        (bmpinfo['bmWidth'], bmpinfo['bmHeight']),
        bmpstr, 'raw', 'BGRX', 0, 1)

    win32gui.DeleteObject(saveBitMap.GetHandle())
    saveDC.DeleteDC()
    mfcDC.DeleteDC()
    win32gui.ReleaseDC(hwnd, hwndDC)

    if result == 1:
        # PrintWindow Succeeded
        return im
    else:
        # PrintWindow Failed
        return None

# Check if the image contains "play.png" with a little tolerance
def check_image_for_play_button(im):
    try:
        # Convert the captured image to grayscale
        im = im.convert("L")
        # Load the "play.png" image and compare it to the captured image
        play_image = Image.open("play.png").convert("L")
        diff = ImageChops.difference(im, play_image)
        if diff.getbbox() is None:
            # No differences were found, "play.png" image was found in the captured image
            return True
        else:
            # Calculate the total number of different pixels
            diff_pixels = sum(diff.getdata())
            # Calculate the total number of pixels in the image
            total_pixels = im.size[0] * im.size[1]
            # Calculate the difference tolerance
            tolerance = total_pixels * 0.1
            # Check if the number of different pixels is within the tolerance
            if diff_pixels <= tolerance:
                return True
            else:
                return False
    except Exception as e:
        print(e)
        return False

while True:
    # Capture the image of the foreground window
    im = capture_foreground_window()
    if im is not None:
        if check_image_for_play_button(im):
            # Send a mouse click event to the background window
            win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,0,0)
            win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,0,0)
            # Send a long "w" key press event for 2 seconds
            win32api.keybd_event(0x57, 0, 0, 0)
            time.sleep(2)
            win32api.keybd_event(0x57, 0, win32con.KEYEVENTF_KEYUP, 0)
