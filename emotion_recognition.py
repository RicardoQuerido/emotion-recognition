import cv2 as cv
import numpy as np
from fer import FER
from tkinter import Tk, Canvas, filedialog, Checkbutton, Label, IntVar, Scale, HORIZONTAL
from tkinter.ttk import Combobox, Button, Radiobutton
from PIL import ImageTk, Image


def detectAndDisplay(frame):
    global face_cascade
    global eyes_cascade
    global mouth_cascade

    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_gray = cv.equalizeHist(frame_gray)

    #-- Detect faces
    faces = face_cascade.detectMultiScale(frame_gray)
    eye_threshold_y = 1
    mouth_threshold_y = 0.9
    eyebrows_threshold_y = 0.5

    highlight_frame = frame.copy()

    for (x,y,w,h) in faces:
        end_w, end_h = (x + w, y + h)
        highlight_frame = cv.rectangle(highlight_frame, (x,y), (end_w,end_h), (255, 0, 255), 2)

        faceROI = frame_gray[y:end_h,x:end_w]
        #-- In each face, detect eyes
        eyes = eyes_cascade.detectMultiScale(faceROI)
        mouth_eyebrow = mouth_cascade.detectMultiScale(faceROI)
        #cv.imshow("face", faceROI)

        for (x2,y2,w2,h2) in eyes:
            end_x, end_y = (x + x2 + w2, y + y2 + h2)
            eye_center = (x + x2 + w2//2, y + y2 + h2//2)
            if eye_center[1] < y + eye_threshold_y * h:
                radius = int(round((w2 + h2)*0.25))
                highlight_frame = cv.circle(highlight_frame, eye_center, radius, (255, 0, 0 ), 2)
                #eyeROI = frame_gray[y+y2:end_y, x+x2:end_x]
                #cv.imshow("eye", eyeROI)

        for (x3,y3,w3,h3) in mouth_eyebrow:
            end_x, end_y = (x + x3 + w3, y + y3 + h3)
            if end_y < y + eyebrows_threshold_y * h:
                highlight_frame = cv.rectangle(highlight_frame, (x+x3,y+y3), (x+x3+w3,end_y), (0, 255, 255), 2)
                #eyebrowROI = frame_gray[y+y3:end_y, x+x3:x+x3+w3]
                #cv.imshow("eyebrow", eyebrowROI)

            elif end_y > end_h * mouth_threshold_y:
                highlight_frame = cv.rectangle(highlight_frame, (x+x3,y+y3), (x+x3+w3,end_y), (255, 255, 0), 2)
                #mouthROI = frame_gray[y+y3:end_y, x+x3:x+x3+w3]
                #cv.imshow("mouth_eyebrow", mouthROI)


    # cv.imshow('Live Emotion Recognition', frame)
    return highlight_frame


def blur(frame):
    global current_blur

    blur_type, blur_intensity = current_blur

    blur_frame = frame.copy()
    
    if blur_type == "Blur":
        if blur_intensity == 0:
            blur_intensity += 1 
        blur_frame = cv.blur(blur_frame, (blur_intensity, blur_intensity))
    elif blur_type == "Gaussian Blur":
        if not blur_intensity % 2:
            blur_intensity += 1 
        blur_frame = cv.GaussianBlur(blur_frame, (blur_intensity, blur_intensity), 0)
    elif blur_type == "Median Blur":
        if not blur_intensity % 2:
            blur_intensity += 1 
        blur_frame = cv.medianBlur(blur_frame, blur_intensity)
    elif blur_type == "Bilateral Filtering":
        blur_frame = cv.bilateralFilter(blur_frame,9,blur_intensity*25,blur_intensity*25)

    return blur_frame


def canny(frame):
    global canny_active
    global current_canny

    lower, upper = current_canny
    v = np.median(frame)
    sigma = 0.33

    edged = frame.copy()

    if not lower:
        lower = int(max(0, (1.0 - sigma) * v))
    if not upper:
        upper = int(min(255, (1.0 + sigma) * v))

    edged = cv.Canny(edged, lower, upper)

    return cv.cvtColor(edged, cv.COLOR_GRAY2RGB)

def load_cascades():
    face_cascade_name = 'data/haarcascade_frontalface_alt.xml'
    eyes_cascade_name = 'data/haarcascade_eye.xml'
    mouth_cascade_name = 'data/haarcascade_mcs_mouth.xml'
    face_cascade = cv.CascadeClassifier()
    eyes_cascade = cv.CascadeClassifier()
    mouth_cascade = cv.CascadeClassifier()

    if not face_cascade.load(cv.samples.findFile(face_cascade_name)):
        print('--(!)Error loading face cascade')
        exit(0)

    if not eyes_cascade.load(cv.samples.findFile(eyes_cascade_name)):
        print('--(!)Error loading eyes cascade')
        exit(0)

    if not mouth_cascade.load(cv.samples.findFile(mouth_cascade_name)):
        print('--(!)Error loading mouth_eyebrow cascade')
        exit(0)

    return face_cascade, eyes_cascade, mouth_cascade


def detect_emotions(frame):
    global detector

    result = detector.detect_emotions(frame)
    if result != []:
        emotion, score = detector.top_emotion(frame)
        cv.putText(frame, f"{emotion} - {int(score*100)}%", (10,50), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2, cv.LINE_AA)

    return frame


def create_image(frame):
    global image_tk
    global hightlight_parts
    global canny_active

    frame = blur(frame)

    frame = detect_emotions(frame)

    if canny_active:
        frame = canny(frame)

    if hightlight_parts:
        frame = detectAndDisplay(frame)

    image = Image.fromarray(frame)
    image_tk = ImageTk.PhotoImage(image.resize((400,400)))
    canvas.create_image(10, 10, anchor="nw", image=image_tk)
    

def load_image():
    global current_image

    file = filedialog.askopenfilename(initialdir="./", title="Select an image", filetypes=(("png", "*.png"), ("jpg", "*.jpg"),( "All files", "*")))
    if file:
        current_image = cv.imread(file)
        create_image(current_image)


def show_vid():
    global cap
    global current_image

    _, frame = cap.read()

    if not camera:
        cap.release() 
        return True

    if frame is None:
        print('--(!) No captured frame -- Break!')
        return False

    current_image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    create_image(current_image)
    root.after(10, show_vid)


def camera_switch():
    global camera
    global cap
    global image_path
    global current_image

    camera = not camera
    if camera:
        cap = cv.VideoCapture(0)
        if cap.isOpened:
            show_vid()
    else:
        current_image = cv.imread(image_path)
        create_image(current_image)


def highlight_switch():
    global hightlight_parts
    global current_image

    hightlight_parts = not hightlight_parts

    create_image(current_image)

def canny_switch():
    global canny_active
    global current_image
    global current_canny

    canny_active = not canny_active

    create_image(current_image)


def canny_change_scale(e):
    global scale_upper_canny
    global scale_lower_canny
    global current_image
    global current_canny

    current_canny = (scale_lower_canny.get(), scale_upper_canny.get())

    create_image(current_image)


def identification_selected():
    global current_image
    global identification_type
    global detector

    detector = FER() if identification_type.get() == 2 else FER(mtcnn=True)
    
    create_image(current_image)


def blur_selected(e):
    global combobox_blur
    global scale_blur
    global current_image
    global current_blur

    current_blur = (combobox_blur.get(), scale_blur.get())

    create_image(current_image)


if __name__ == "__main__":
    face_cascade, eyes_cascade, mouth_cascade = load_cascades()

    detector = FER()

    root = Tk(className="Emotion Recognition") 
    root.columnconfigure(3, weight=2)
    root.columnconfigure(2, weight=1)
    root.resizable(False, False)

    image_path = "./lena.jpg"
    current_image = cv.imread(image_path)
    camera = False
    hightlight_parts = False
    canny_active = False
    identification_type = IntVar(value=2)
    current_canny = (0,255)
    current_blur = (None, 0)

    canvas = Canvas(root, width=400, height=400)  
    canvas.grid(column=0,row=0, columnspan=2, rowspan=12)

    button_load_image = Button(root, text ="Load Image", command = load_image, padding=5)
    button_load_image.grid(column = 0, row=13, ipadx=30, pady=10)
    
    button_camera = Button(root, text ="Switch Camera ON/OFF", command = camera_switch, padding=5)
    button_camera.grid(column = 1, row=13, ipadx=30, pady=10)
    
    label_preprocessing = Label(root, text="Identification", font='Helvetica 12 bold')
    label_preprocessing.grid(column = 2, columnspan=2,row=0)

    identification_mtcnn = Radiobutton(root, text='MTCNN network', variable=identification_type, value=1,command=identification_selected)
    identification_opencv = Radiobutton(root, text='OpenCV\'s Haar Cascade classifier', variable=identification_type, value=2,command=identification_selected)

    identification_mtcnn.grid(column = 2, row=1, padx=10)
    identification_opencv.grid(column = 3, row=1, padx=10)

    label_preprocessing = Label(root, text="Preprocessing", font='Helvetica 12 bold')
    label_preprocessing.grid(column = 2, row=2, columnspan=2,padx=10)

    label_blur = Label(root, text="Blurring", font='Helvetica 12')
    label_blur.grid(column = 2, row=3, columnspan=2,padx=10)

    label_blur_type = Label(root, text="Type:", font='Helvetica 10')
    label_blur_type.grid(column = 2, row=4, padx=10, sticky="e")

    blur_opts = [None, "Blur", "Gaussian Blur", "Median Blur", "Bilateral Filtering"]
    combobox_blur = Combobox(root, value=blur_opts)
    combobox_blur.current(0)
    combobox_blur.bind("<<ComboboxSelected>>", blur_selected)
    combobox_blur.grid(column = 3, row=4, padx=10, sticky="w")

    label_blur_scale = Label(root, text="Intensity:", font='Helvetica 10')
    label_blur_scale.grid(column = 2, row=5, padx=10, sticky="e")

    scale_blur = Scale(root, from_=0, to=20, orient=HORIZONTAL, length=200, showvalue=0,tickinterval=5, resolution=1, command=blur_selected)
    scale_blur.set(5)
    scale_blur.grid(column = 3, row=5, padx=10,sticky="sw")

    label_canny = Label(root, text="Canny", font='Helvetica 12')
    label_canny.grid(column = 2, row=6, columnspan=2,padx=10)

    checkbox_canny = Checkbutton(root, text='Activate Canny',variable=IntVar(), command=canny_switch)
    checkbox_canny.grid(column = 2, row=7, columnspan=2, padx=10)

    label_canny_lower_scale = Label(root, text="Lower:", font='Helvetica 10')
    label_canny_lower_scale.grid(column = 2, row=8, padx=10, sticky="e")

    scale_lower_canny = Scale(root, from_=0, to=255, orient=HORIZONTAL, length=200, showvalue=0,tickinterval=128, resolution=1, command=canny_change_scale)
    scale_lower_canny.set(127)
    scale_lower_canny.grid(column = 3, row=8, padx=10,sticky="sw")

    label_canny_upper_scale = Label(root, text="Upper:", font='Helvetica 10')
    label_canny_upper_scale.grid(column = 2, row=9, padx=10, sticky="e")

    scale_upper_canny = Scale(root, from_=0, to=255, orient=HORIZONTAL, length=200, showvalue=0,tickinterval=128, resolution=1, command=canny_change_scale)
    scale_upper_canny.set(255)
    scale_upper_canny.grid(column = 3, row=9, padx=10,sticky="sw")

    label_extra = Label(root, text="Extra", font='Helvetica 12 bold')
    label_extra.grid(column = 2, row=10, columnspan=2,padx=10)

    checkbox_hightlight = Checkbutton(root, text='Highlight parts',variable=IntVar(), command=highlight_switch)
    checkbox_hightlight.grid(column = 2, row=11, columnspan=2,padx=10)

    create_image(current_image)

    root.mainloop() 



    


