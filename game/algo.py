"""

My tracker loses my face every so often, so I want to
redetect. My simple method would just be every ten frames
do detection again. This means I would need a counter, which
is again persistent. But where should I update it, in
process or update?

"""
import numpy as np
import argparse
import cv2
import math, os
import utils
import heapq
cv_version = map(int, cv2.__version__.split('.'))
if cv_version[0] != 2 or cv_version[1] != 4 or cv_version[2] < 9:
    raise ImportError('OpenCV version is %s, must be 2.4.x where 9<=x' % cv2.__version__)

class FaceDetector(object):
    """
    This detector looks for a face in a given image over several frames and returns face candidates
    """
    def detect(self, img, draw_rect=False):
        """
        Pass in an image and return the cordinates of teh face center.
        Also draw a rectangle around the face
        :param img:
        :return: (x_center, y_center), (x, y, width, height)
        """
        #built in statistical model to search for faces fast by scanning with a sliding window
        #of varying size. Then can
        # mearge overlapping windows since multiple windows will find the same face
        face_cascade = cv2.CascadeClassifier(r'..\notebooks\haarcascade_frontalface_default.xml')
        rects = face_cascade.detectMultiScale(img,
                                      minSize=(16,16), # Smallest window
                                      maxSize=(180, 180), # Largest window
                                      scaleFactor=1.1, # Step between windows
                                      minNeighbors=1) # how many neighbors each candidate rectangle should have

        if 0 != len(rects):
            for rect in rects:
                #Shrink the rectangle cuz we only want to capture the face for coloring
                shrink = 0.8 #shrinkage ratio
                x, y = rect[0] + rect[2]/2, rect[1] + rect[3]/2
                rect[2], rect[3] = shrink*rect[2], shrink*rect[3]
                rect[0], rect[1] = x - rect[2]/2, y - rect[3]/2
                #expect only one face
                height, width, x = img.shape #rows, cols, depth
                face_position = (x/float(width), y/float(height))
                #create a face hsv_histogram, normalized, for tracking
                det_face_hsv = self.get_face_hsv(img, rect)
                return face_position, tuple(rect), det_face_hsv
        else:
            print("No face detected")
            return None, None, None

    def get_face_hsv(self, img, face_rect):
        # calcHist is kinda an annoying function. The [0] is cuz the Hue in
        # an HSV matrix (aka image) is at the 0th index along that dimension.
        # [0, 180] is cuz hue is in degrees and 1 byte can only go up to 255
        # not 360, so we use 0-180 and each one is 2 degrees of hue.
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        face_left = face_rect[0]
        face_top = face_rect[1]
        face_width = face_rect[2]
        face_height = face_rect[3]
        img_face_hsv = img_hsv[face_top:face_top+face_height, face_left:face_left+face_width]
        return img_face_hsv


class FaceTracker(object):
    """
    This class represents a face candidate that was detected and is now being tracked
    """

    def track(self, img, center, face_rect, det_face_hsv):
        """
        Uses mean shifting to track the users face. Only useful once
        a face has already been detected. The Process
        1) Convert the image to HSV, since we track with Hue
        2) Pull out the Hue values in the detected region, and develop a histogram
        3) Create a Back Projection of the initial image with values equal to the
            probability of that hue appearing in the facial region
        4) Use mean shifting to determine where the new face is

        :param img: BGR image from webcam
        :param center: tuple giving the normalized center of teh face
        :param face_rect: non-normalized dimensions of face rectangle (x, y, cols, rows)
        :param det_face_hsv: hsv of the most recently detected face
        :return: (new_position, rect)
        """
        # convert the original image to hsv, and pull out the face
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hue_face_hist = cv2.calcHist([det_face_hsv],[0], None, [32], [0, 180])
        cv2.normalize(hue_face_hist, hue_face_hist, 0, 255, cv2.NORM_MINMAX)
        #calculate teh back projection probabilities
        back_proj = cv2.calcBackProject([img_hsv], [0], hue_face_hist, [0, 180], 1)
        #track face using meanshift
        term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
        track_box, rect = cv2.meanShift(back_proj, face_rect, term_crit)
        #return values
        height, width, x = img.shape #rows, cols, depth
        new_position = ((rect[0] + rect[2]/2)/float(width), (rect[1] + rect[3]/2)/float(height))
        cv2.rectangle(img, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), 255)

        return (new_position, rect)


class SkinDetector(object):
    """
    Takes an image and a face, and decides
    """
    def skin_contours(self, img, det_face_hsv, face_rect):
        """
        Super Duper light sensitive
        :param img:
        :param det_face_hsv:
        :param face_rect:
        :return:
        """
        masked_img = self.skin_mask(img, det_face_hsv, face_rect)
        # do a bitwise and with the foreground mask
        blob_img = self.skin_blobs(img, det_face_hsv, face_rect, masked_img)
        contours, hierarchy = cv2.findContours(np.copy(blob_img), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = self.large_contours(contours, 3) #return the n largest contours up to 6
        color_img, angles = self.color_contours(blob_img, contours)

        _ = cv2.drawContours(blob_img, contours, -1, (128), 3)
        return masked_img, blob_img, color_img, angles

    def color_contours(self, blob_img, contours):
        """
        Return a colored image where the regions within certain the contours
        is colored in.
        :param blob_image:
        :return:
        """
        labeled_img = np.zeros(blob_img.shape + (3, ), np.uint8)
        colors = ((0,0,255),(0,255,0),(255,0,0),(0,255,255),(255,0,255), (255, 255, 0))
        pnts_list = []
        mask_list = []
        for ind, contour in enumerate(contours):
            mask = np.zeros(blob_img.shape, np.uint8)
            cv2.drawContours(mask, [contour], 0, 255, -1, 8)
            pixel_points = cv2.findNonZero(mask)#(x,y)

            labeled_img[mask == 255] = colors[ind]
            pnts_list.append(pixel_points)
            mask_list.append(mask)

        k = 0
        angles = []
        for cnt in contours:
            if len(cnt) < 10:
                #don't care about tiny contours
                # this should have already been protected for in the
                # large_contour code, but that is technically area
                angles.append(0)
                continue

            pixel_points = pnts_list[k]
            M = cv2.moments(cnt)#expects to get a contour - uses Green's theorem

            #center of blob
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])

            #ellipsoid outline of blob
            ellipse = cv2.fitEllipse(cnt)

            (x, y), (MA, ma), angle = cv2.fitEllipse(pixel_points)#yet another way to get the angle
            angles.append(angle)

            #line fitting, THIS WAS SLOWING ME DOWN
            #DIST_L1 = 1: |x1-x2| + |y1-y2| */, DIST_L2 = 2: euclidean distance, DIST_C = : max(|x1-x2|,|y1-y2|)
            [vx, vy, x, y] = cv2.fitLine(pixel_points, 1, 0, 0.01, 0.01)
            pt1 = (np.array((x, y)) + 20*np.array((vx, vy))).astype('int32')
            pt2 = (np.array((x, y)) - 20*np.array((vx, vy))).astype('int32')
            cv2.line(labeled_img, tuple(pt1), tuple(pt2), (0, 128, 128), 2, 8)
            k += 1

        return labeled_img, angles

    def major_axes(self, contours):
        """
        Calculate teh Major axes and the angles they are at
        :param contours:
        :return:
        """

    def large_contours(self, contours, n=5):
        """
        Return the n largest contours from my skin detection.
        Don't send back overly small contours cuz it will cause an error
        :param contours:
        :param n:  up to 6
        :return:
        """
        #calculate the area of the contours, and select the 5 largest blobs
        areas = []
        for i, countour in enumerate(contours):
            area = cv2.moments(countour)['m00']
            if area > 10:
                areas.append(area)
        if len(areas) < n:
            # we have less than n contours of the min size
            return [contours[areas.index(i)] for i in heapq.nlargest(len(areas), areas)]
        else:
            return [contours[areas.index(i)] for i in heapq.nlargest(n, areas)]


    def skin_blobs(self, img, det_face_hsv, face_rect, masked_img):
        """
        Do blob morphology stuff on faces. Perform a mask,
        Then dilate and erode to make them into more coherent blobs.

        :param img: BGR image from webcam
        :param det_face_hsv: hsv image of the face from the previous detection
        :param face_rect: non-normalized dimensions of face rectangle (left, top, cols, rows)
        :return: 2D array, black and white image of skin blobs
        """

        #open and close
        # kernel size and shape are more art than science
        # using a small kernel to erode noise and a large on to
        # to dilate since I have more false negatives with skin
        # detection than I do false positives.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
        kernel_small = kernel & np.transpose(kernel) #symmetry
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6, 6))
        kernel_big = kernel & np.transpose(kernel) #symmetry
        blob_img = cv2.erode(masked_img, kernel_small)
        blob_img = cv2.dilate(blob_img, kernel_big)
        blob_img = cv2.erode(blob_img, kernel_small)
        blob_img = cv2.dilate(blob_img, kernel_big)
        return blob_img



    def skin_mask(self, img, det_face_hsv, face_rect):
        """
        Create a mask of the image which returns a binary image (black and white) based
        on whether we thing a section is skin or not. We do this by analyzing the hue and
        saturation from the detected face. From this we can calculate the probability of
        any pixel in the full image occuring in the face image. Then we can filter out
        any values whose probability is below a certain threshold.

        :param img: BGR image from webcam
        :param det_face_hsv: hsv image of the face from the previous detection
        :param face_rect: non-normalized dimensions of face rectangle (left, top, cols, rows)
        :return: 2D array, black and white if pixels are thought to be skin
        """
        #Get the HSV images of the whole thing and the face
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        face_left = face_rect[0]
        face_top = face_rect[1]
        face_width = face_rect[2]
        face_height = face_rect[3]
        #create a Hue-Saturation histogram of the face
        hs_face_hist = cv2.calcHist([det_face_hsv], [0,1], None, [32,32], [0, 180,0, 255])
        cv2.normalize(hs_face_hist, hs_face_hist, 0, 255, cv2.NORM_MINMAX)
        #create a Hue-Saturation BackProjection, and a mask
        #This mask ignores dark pixels < 32, and saturated pixels, <60
        hue_min, sat_min, val_min = 0.0, 32.0, 16.0
        mask = cv2.inRange(img_hsv, np.array((hue_min, sat_min, val_min)), np.array((180., 255., 255.)))
        mask_face = mask[face_top:face_top+face_height, face_left:face_left+face_width]
        masked_hs_hist = cv2.calcHist([det_face_hsv], [0,1], mask_face, [32,32], [0, 180,0, 255])
        cv2.normalize(masked_hs_hist, masked_hs_hist, 0, 255, cv2.NORM_MINMAX)
        masked_hs_prob = cv2.calcBackProject([img_hsv], [0,1], masked_hs_hist, [0, 180,0, 255],1)
        cv2.bitwise_and(masked_hs_prob, mask, dst=masked_hs_prob) #seems to lessen noise???
        thresh = 8.0 #threshold likelihood for being skin, changes a lot based on setting
        _, masked_img = cv2.threshold(masked_hs_prob, thresh, 255, cv2.CV_8U) #throw out below thresh

        return masked_img


class ForegroundDetector(object):
    """
    Extracts foreground from a given image based on previous frames
    Take in the bgr images and print out the black and white
    This guy is good for tracking motion
    """
    # TODO: Implement this

    def __init__(self):
        #need persistent background detector

        self.bs = cv2.BackgroundSubtractorMOG2(history=10, varThreshold=16, bShadowDetection=False)
        self.bs.setDouble('backgroundRatio', 0.7)
        self.bs.setInt('nmixtures', 5)
        self.bs.setDouble('varThresholdGen', 9.)
        self.bs.setDouble('fCT', 0)

    def detect(self, img):
        """
        Take in the live BGR image and do foreground analysis
        :param img:
        :return:
        """
        #could have used learning rate instead of history. higher learning rate means
        # that I forget earlier pictures faster. Sorta always rely on teh first though.
        # since I want a movement tracker I want a high learnign rate even though
        # I fade to background when I don't move

        return self.bs.apply(img, learningRate=.025)

    pass


class ArmDetector(object):
    """
    Detects an (either left or right) arm based on face, skin and foreground detections
    """

    def __init__(self, side):
        self.side = side

    # TODO: Implement this
    pass


class NUIEngine(object):
    show_debug_window = True

    def __init__(self):
        args = self.parse_cmd_arguments()
        self.playback_video = args.playback_video
        self.record_video = args.record_video
        # Outputs
        self.right_degrees = None
        self.left_degrees = None
        self.face_position = None #(y_center, x_center) normalized. top left is (0, 0)
        self.face_rect = None #(x_left, y_top, width, height) abs location of face rects
        self.hue_face_hist = None #normalized hue histogram of last image (assume this is the face)
            # We update this every detection, for the tracker
        self.count = 1 #counter detecting and tracking
        self.fg_det = ForegroundDetector() #need persistent foreground detector
        # Video IO
        self.frame_num = -1
        self.video_writer = None
        if self.playback_video:
            if os.path.isdir(self.playback_video):
                self.video_capture = utils.ImageSequenceReader(self.playback_video, 5)
            else:
                self.video_capture = cv2.VideoCapture(self.playback_video)
        else:
            self.video_capture = cv2.VideoCapture(0)
            if self.record_video:
                self.video_writer = cv2.VideoWriter(self.record_video, 0, 30, (640, 480), isColor=True)

    def read_next_frame(self):
        self.frame_num += 1
        res, img = self.video_capture.read()
        if not res:
            raise IOError('Failed to capture next frame')
        if self.video_writer:
            if self.frame_num < 1000:
                self.video_writer.write(img)
            else:
                self.video_writer.release()
                del self.video_writer
        return img

    def preprocess_image(self, img):
        img = cv2.resize(img, (320, 240), interpolation=cv2.INTER_AREA)
        cv2.flip(img, 1, img)
        return img

    def process_image(self, img):
        """
        The main image processing method that takes the input image and finds the output variable if applicable
        Make sure to normalize units!!!
        :param img:
        :return:
        """
        # TODO: make a queue so that I average the last 5 values for angle and loc
        # cuz right now it's too jumpy
        if self.face_position == None or self.count%25 == 0:
            #If I don't find a face I should probs just track ass opposed to wasting a frame
            self.count = 1
            # We initialize with no face, so we want to detect one
            # after that we will just track
            # This is assuming we never lose the face
            face_det = FaceDetector()
            # algo.py: TypeError: 'NoneType' object is not iterable
            #   usually happens, not always
            # game.py: TypeError: 'NoneType' object is not iterable
            #   Exception pygame.error: 'mixer system not initialized' in <bound method Game.__del__ of <__main__.Game object at 0x0000000003781E48>> ignored
            self.face_position, self.face_rect, self.det_face_hsv = face_det.detect(img)
        else:
            self.count += 1
            # we gunna try to track the face
            face_track = FaceTracker()
            self.face_position, self.face_rect = \
                face_track.track(img, self.face_position, self.face_rect, self.det_face_hsv)

        if self.face_position != None:
            # contours changes the img, so put this first
            fore_img = self.fg_det.detect(img)
            cv2.namedWindow("foreground")
            cv2.imshow("foreground", fore_img)

            #Show the skin back projection in another window
            # puts the contours onto the img, changes it
            skin_det = SkinDetector()
            skin_det, skin_blobs, skin_colored, angles =  skin_det.skin_contours(img, self.det_face_hsv,self.face_rect)
            if len(angles) == 3:
                self.left_degrees = angles[1]
                self.right_degrees = angles[2]
            cv2.namedWindow("skin_det")
            cv2.imshow("skin_det", skin_det)
            cv2.namedWindow("skin_blobs")
            cv2.imshow("skin_blobs", skin_blobs)
            cv2.namedWindow("skin_colored")
            cv2.imshow("skin_colored", skin_colored)



        return

    def show_output_variables(self, img):
        canvas = np.array(img)
        if self.face_position:
            center = np.float32((self.face_position[0]*img.shape[1], self.face_position[1]*img.shape[0]))
            cv2.circle(canvas, tuple(center), 20, (0, 255, 0), thickness=3)
        if self.left_degrees:
            left_vector = np.float32((-np.cos(math.radians(self.left_degrees)), -np.sin(math.radians(self.left_degrees))))
            cv2.line(canvas, tuple(np.uint8(center+left_vector*20)), tuple(np.uint8(center+left_vector*40)), (255, 0, 0), thickness=3)
        if self.right_degrees:
            right_vector = np.float32((np.cos(math.radians(self.right_degrees)), -np.sin(math.radians(self.right_degrees))))
            cv2.line(canvas, tuple(np.uint8(center+right_vector*20)), tuple(np.uint8(center+right_vector*40)), (0, 0, 255), thickness=3)
        cv2.namedWindow("output_img")
        cv2.moveWindow("output_img", 0, 0)
        cv2.imshow("output_img", canvas)
        cv2.waitKey(1)

    def update(self):
        # Reset output variables
        self.right_degrees = None
        self.left_degrees = None
        #self.face_position = None

        img = self.read_next_frame()
        img = self.preprocess_image(img)
        self.process_image(img)
        if NUIEngine.show_debug_window:
            self.show_output_variables(img)

    def parse_cmd_arguments(self):
        parser = argparse.ArgumentParser(description='')
        parser.add_argument('-playback-video', dest='playback_video', action='store', required=False, default=None,
                            help='Full path to the video file to playback')
        parser.add_argument('-record-video', dest='record_video', action='store', required=False, default=None,
                            help='Full path to the video file to record')
        return parser.parse_args()


if __name__ == '__main__':
    nui_engine = NUIEngine()
    try:
        while True:
            nui_engine.update()
    except IOError:
        pass
