from skimage.segmentation import clear_border
import pytesseract
import numpy as np
import imutils
import cv2

class PyImageSearchANPR:
    def __init__(self,minAR=4,maxAR=5, debug=False):
        self.minAR = minAR
        self.maxAR = maxAR
        self.debug = debug
        #min and max rectangular aspect ratio for the license plates

    def debug_imshow(self, title, image, waitKey=False):
        #check to see if we are in debug
        if self.debug:
            cv2.imshow(title, image)
            #waits for a keypress on the computer
            if waitKey:
                cv2.waitKey(0)

    def locate_license_plate_candidates(self, gray, keep=5):
        #will let us view dark regions on a light background.
        #These lines of code will allows the computer to read the plates
        rectKern = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKern)
        self.debug_imshow("Blackhat", blackhat)
        #finds areas that are light
        squareKern = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        light = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, squareKern)
        light = cv2.threshold(light, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        self.debug_imshow("Light Regions", light)
        #Uses Scharr gradient method to filter and highlight gradient edges
        gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
        gradX = np.absolute(gradX)
        (minVal, maxVal) = (np.min(gradX), np.max(gradX))
        gradX = 255 * ((gradX - minVal) / (maxVal - minVal))
        gradX = gradX.astype("uint8")
        self.debug_imshow("Scharr", gradX)

        #blurs the gradient
        gradX = cv2.GaussianBlur(gradX, (5, 5), 0)
        gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKern)

        # erosion and dilation to clean the image up for the computer
        thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        self.debug_imshow("Grad Thresh", thresh)
        thresh = cv2.bitwise_and(thresh, thresh, mask=light)
        thresh = cv2.dilate(thresh, None, iterations=2)
        thresh = cv2.erode(thresh, None, iterations=1)
        self.debug_imshow("Final", thresh, waitKey=True)

        #finds contour by size and sorts them keeping only largest
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:keep]
        return cnts



    def locate_license_plate(self, gray, candidates, clearBorder=False):
        #initialize plate contour
        lpCnt = None
        roi = None

        #loop over license plate contours
        for c in candidates:
            #computes license plate aspect ratio using contour
            (x, y, w, h) = cv2.boundingRect(c)
            ar = w / float(h)

            #is aspect ratio rectangular and stores contour
            #extracts licenseplate from image
            if ar >= self.minAR and ar <= self.maxAR:
                lpCnt = c
                licensePlate = gray[y:y + h, x:x + w]
                roi = cv2.threshold(licensePlate, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

                #clear any pixels touching border of image
                if clearBorder:
                    roi = clear_border(roi)

                #displays debug info
                self.debug_imshow("License Plate", licensePlate)
                self.debug_imshow("ROI", roi, waitKey=True)
                break
        return (roi, lpCnt)

    def build_tesseract_options(self, psm=7):
        #tells tesseract to only use english characters
        alphanumeric = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        options = "-c tessedit_char_whitelist={}".format(alphanumeric)
        #tells tesseract to use default settings
        options += " --psm {}".format(psm)
        return options

    def find_and_ocr(self, image, psm=7, clearBorder=False):
        #license plate text
        lpText = None

        #convert to grayscale and locate license plate images
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        candidates = self.locate_license_plate_candidates(gray)
        (lp, lpCnt) = self.locate_license_plate(gray, candidates,
                                                clearBorder=clearBorder)

        #tells computer to run OCR only if there is a viable license plate
        if lp is not None:
            options = self.build_tesseract_options(psm=psm)
            lpText = pytesseract.image_to_string(lp, config=options)
            self.debug_imshow("License Plate", lp)

        return (lpText, lpCnt)




