from PyImageSearch import PyImageSearchANPR
from imutils import paths
import argparse
import imutils
import cv2

def cleanup_text(text):
	#strips non-ASCII text
	return "".join([c if ord(c) < 128 else "" for c in text]).strip()

#parses the arguments/options to run the final code
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help="path to input directory of images")
ap.add_argument("-c", "--clear-border", type=int, default=-1,
	help="whether or to clear border pixels before OCR'ing")
ap.add_argument("-p", "--psm", type=int, default=7,
	help="default PSM mode for OCR'ing license plates")
ap.add_argument("-d", "--debug", type=int, default=-1,
	help="whether or not to show additional visualizations")
args = vars(ap.parse_args())

#initializes the ANPR class
anpr = PyImageSearchANPR(debug=args["debug"] > 0)

#grabs the images that we put as input
imagePaths = sorted(list(paths.list_images(args["input"])))

#loops over the images
for imagePath in imagePaths:
	#resizes the images
	image = cv2.imread(imagePath)
	image = imutils.resize(image, width=600)

	#applys the ANPR
	(lpText, lpCnt) = anpr.find_and_ocr(image, psm=args["psm"],
		clearBorder=args["clear_border"] > 0)

	#continues if ANPR is successful
	if lpText is not None and lpCnt is not None:

		#box around the license plate
		box = cv2.boxPoints(cv2.minAreaRect(lpCnt))
		box = box.astype("int")
		cv2.drawContours(image, [box], -1, (0, 255, 0), 2)

		#finds the box and draws the text onto the image
		(x, y, w, h) = cv2.boundingRect(lpCnt)
		cv2.putText(image, cleanup_text(lpText), (x, y - 15),
			cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

		#shows the final product
		print("[INFO] {}".format(lpText))
		cv2.imshow("Output ANPR", image)
		cv2.waitKey(0)
