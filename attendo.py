from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.button import Button
from kivy.uix.camera import Camera
from kivy.uix.label import Label
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.logger import Logger
from kivy.lang import Builder
from kivy.uix.textinput import TextInput
from kivy.uix.popup import Popup
from kivy.graphics import Rectangle, Color, Line

import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import requests
import time
import threading

class HomeScreen(Screen):
    def __init__(self, **kwargs):
        super(HomeScreen, self).__init__(**kwargs)
        homebox = BoxLayout(orientation='vertical')
        regis_button = Button(text="Face registration", size_hint=(1, .1))
        check_button = Button(text="Check-in", size_hint=(1, .1))
        reset_data_button = Button(text="Reset Face Database", size_hint=(1, .03),background_color=(255,0,0,.8))
        regis_button.bind(on_press=self.registration)
        check_button.bind(on_press=self.checkin)
        reset_data_button.bind(on_press=self.reset_database)
        homebox.add_widget(regis_button)
        homebox.add_widget(check_button)
        homebox.add_widget(reset_data_button)
        self.add_widget(homebox)

    def registration(self, *args):
        self.manager.current = 'registration'

    def checkin(self, *args):
        self.manager.current = 'checkin'

    def reset_database(self, *args):
        content = BoxLayout(orientation='vertical')
        content.add_widget(Label(text='Are you sure you want to reset?'))
        btn_confirm = Button(text='Yes')
        btn_cancel = Button(text='No')
        popup = Popup(title='Reset Database Confirmation', content=content, size_hint=(None, None), size=(300, 200))
        content.add_widget(btn_confirm)
        content.add_widget(btn_cancel)
        btn_confirm.bind(on_press=self.perform_reset_database)
        btn_cancel.bind(on_press=popup.dismiss)
        popup.open()

    def perform_reset_database(self, instance):
        # Delete all files in the FaceData folder
        folder_path = 'FaceData'
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"Error deleting file {filename}: {e}")
        Popup(title='Reset Database', content=Label(text='Face database has been reset.'), size_hint=(None, None),
              size=(300, 200)).open()

class RegistrationScreen(Screen):
    def __init__(self,**kwargs):
        super (RegistrationScreen,self).__init__(**kwargs)
        self.web_cam = Image(size_hint=(1,.8))
        my_box1 = BoxLayout(orientation='vertical')
        home_button = Button(text="back",size_hint=(None, None), size=(100, 50))
        capture_button = Button(text="capture",size_hint=(1,.1))
        home_button.bind(on_press=self.changer)
        capture_button.bind(on_press=lambda instance: self.capture_face(None))
        self.name_input = TextInput(hint_text="Enter your name", size_hint=(1, .1), multiline=False)
        self.message_label = Label(text='', size_hint=(1, .1),color=(1,1, 0))
        my_box1.add_widget(self.web_cam)
        my_box1.add_widget(self.message_label)
        my_box1.add_widget(capture_button)
        my_box1.add_widget(self.name_input)
        my_box1.add_widget(home_button)
        self.add_widget(my_box1)

    def on_enter(self):
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0 / 33.0)
        self.update_message('Press capture to record the face')

    def update_message(self, message):
        self.message_label.text = message

    def update(self, *args):
        ret, frame = self.capture.read()
        if ret:
            self.frame = frame[0:480, 0:640, :]

            # drawn_frame = self.face_recog()
            drawn_frame = self.face_recog()
            if drawn_frame is not None:
                buf = cv2.flip(drawn_frame, 0).tobytes()
                img_texture = Texture.create(size=(drawn_frame.shape[1], drawn_frame.shape[0]), colorfmt='bgr')
                img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
                self.web_cam.texture = img_texture

    def capture_face(self, instance):
        user_provided_name = self.name_input.text.strip()
        save_dir = "FaceData"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if user_provided_name:
            frame = self.capture.read()[1]
            # Check if a face is detected in the captured frame
            face_locations = face_recognition.face_locations(frame)
            
            if len(face_locations) > 0:
                filename = os.path.join(save_dir, f"{user_provided_name}.png")
                try:
                    cv2.imwrite(filename, frame)
                    self.update_message('Image saved.')
                except Exception as e:
                    print('Error exporting image:', e)
                    self.update_message('Error exporting image.')
            else:
                print('No face detected in the captured image. Please try again.')
                self.update_message('No face detected. Please try again.')
        else:
            print('Please enter a file name before capturing.')
            self.update_message('Please enter the name before capturing.')
            
    def on_leave(self):
        self.capture.release()

        
    def changer(self,*args):
        self.manager.current = 'home'

class CheckinScreen(Screen):
    def __init__(self, encodeListKnown, classNames, **kwargs):
        super (CheckinScreen,self).__init__(**kwargs)
        self.encodeListKnown = encodeListKnown
        self.classNames = classNames
        self.web_cam = Image(size_hint=(1,.8))
        my_box1 = BoxLayout(orientation='vertical')
        self.message_label = Label(text='', size_hint=(1, .1),color=(1,1, 0))
        home_button = Button(text="back",size_hint=(None, None), size=(100, 50))
        reset_button = Button(text="Reset-attendance",size_hint=(1,.1))
        home_button.bind(on_press=self.changer)
        reset_button.bind(on_press=self.reset)
        my_box1.add_widget(self.web_cam)
        my_box1.add_widget(self.message_label)
        my_box1.add_widget(reset_button)
        my_box1.add_widget(home_button)
        self.add_widget(my_box1)
        self.frame = None
        self.stop_recog_event = threading.Event()
        self.detected_faces = set()
        self.update_message("Loading Camera . . .")

    def update_message(self, message):
        self.message_label.text = message

    def on_enter(self):
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0 / 120.0)
        path = 'FaceData'
        self.update_message("Syncing face database . . .")
        self.encodeListKnown, self.classNames = load_images_and_encodings(path)
        self.update_message("Face database synced")
        self.face_recog_thread = threading.Thread(target=self.face_recog_threaded)
        self.face_recog_thread.daemon = True  
        self.face_recog_thread.start()
        self.drawn_frame = None

    def update(self, *args):
        ret, frame = self.capture.read()
        if ret:
            self.frame = frame[0:480, 0:640, :]

            # drawn_frame = self.face_recog()
            drawn_frame = self.frame
            if drawn_frame is not None:
                buf = cv2.flip(drawn_frame, 0).tobytes()
                img_texture = Texture.create(size=(drawn_frame.shape[1], drawn_frame.shape[0]), colorfmt='bgr')
                img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
                self.web_cam.texture = img_texture

    def reset(self, instance):
        self.detected_faces.clear()
        print("Detected faces reset.")
        markAttendance("- - -")
        linenotify("reset")
        self.update_message("Detected faces reset.")

    def face_recog(self):
        imgS = cv2.resize(self.frame, (0, 0), None, 0.8, 0.8)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        facesCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)
        num_detected_faces = len(facesCurFrame)

        drawn_frame = self.frame.copy()

        print("Number of detected faces:", num_detected_faces)
        self.update_message("Number of detected faces: " + str(num_detected_faces))

        if num_detected_faces == 1:
            encodeFace = encodesCurFrame[0]
            faceLoc = facesCurFrame[0]

            if not self.encodeListKnown:
                print("No known faces")
                return 

            faceDis = face_recognition.face_distance(self.encodeListKnown, encodeFace)

            if len(faceDis) > 0:  
                min_distance = min(faceDis)
                if min_distance < 0.5: 
                    matchIndex = np.argmin(faceDis)
                    name = self.classNames[matchIndex].upper()
                    namenotify = self.classNames[matchIndex]
                    y1, x2, y2, x1 = faceLoc
                    y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4

                    # Draw rectangle around the face
                    cv2.rectangle(drawn_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Draw filled rectangle for label
                    cv2.rectangle(drawn_frame, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)

                    # Draw text label
                    cv2.putText(drawn_frame, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

                else:
                    name = 'Unknown'
                    namenotify = 'Unknown'
                    self.update_message("Face Detected : " + namenotify)
            else:
                name = 'Unknown'
                namenotify = 'Unknown'
                self.update_message("Face Detected : " + namenotify)

            print(namenotify)
            message = f'{namenotify} checked at {datetime.now().strftime("%H:%M:%S")}'
            if namenotify != 'Unknown' and namenotify not in self.detected_faces:
                self.detected_faces.add(namenotify)
                markAttendance(name)
                linenotify(message)
            else:
                if namenotify != 'Unknown':
                    print("Already checked")
                    self.update_message("Face Detected : " + namenotify)
        elif num_detected_faces > 1:
            names = []
            for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
                faceDis = face_recognition.face_distance(self.encodeListKnown, encodeFace)
                if len(faceDis) > 0:  
                    min_distance = min(faceDis)
                    if min_distance < 0.6: 
                        matchIndex = np.argmin(faceDis)
                        namenotify = self.classNames[matchIndex]
                        name = self.classNames[matchIndex].upper()
                        names.append(name)
                        y1, x2, y2, x1 = faceLoc
                        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4

                        # Draw rectangle around the face
                        cv2.rectangle(drawn_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                        # Draw filled rectangle for label
                        cv2.rectangle(drawn_frame, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)

                        # Draw text label
                        cv2.putText(drawn_frame, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

                    else:
                        names.append('Unknown')
                        namenotify = 'Unknown'
                        self.update_message("Face Detected : " + namenotify)
                else:
                    names.append('Unknown')
            message = f'{namenotify} checked at {datetime.now().strftime("%H:%M:%S")}'
            if namenotify != 'Unknown' and namenotify not in self.detected_faces:
                    self.detected_faces.add(namenotify)
                    markAttendance(name)
                    linenotify(message)
            else:
                if namenotify != 'Unknown':
                    print("Already checked")
                    self.update_message("Face Detected : " + namenotify)
            names_str = ', '.join(names)
            self.update_message(f"{num_detected_faces} faces detected: {names_str}")
        return drawn_frame

    def face_recog_threaded(self):
        while True:
            if self.frame is not None:  
                self.face_recog()
            time.sleep(1) 

    def on_leave(self):
        self.capture.release()
        self.stop_recog_event.set()

    def changer(self,*args):
        self.manager.current = 'home'
        self.stop_recog_event.set()

def linenotify(message):
    url = 'https://notify-api.line.me/api/notify'
    token = 'wCGK3StYoSwjVHl74ToYj2GHQJCgLyzRNGVAl1Vesby' # Line Notify Token   
    data = {'message': message}
    headers = {'Authorization': 'Bearer ' + token}
    session = requests.Session()
    session_post = session.post(url, headers=headers, data=data)
    print(session_post.text)

def markAttendance(name):
    file_path = 'Attendance.csv'
    if not os.path.isfile(file_path):
        with open(file_path, 'w') as f:
            f.write('Name,Time')
    with open(file_path, 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        now = datetime.now()
        dtString = now.strftime('%H:%M:%S')
        f.writelines(f'\n{name},{dtString}')

def findEncodings(images):
    encodeList = [] 
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face_encodings = face_recognition.face_encodings(img)
        if len(face_encodings) > 0:
            encodeList.append(face_encodings[0])
        else:
            print("No face found in one of the images.")
    return encodeList

def load_images_and_encodings(path):
    images = []
    classNames = []
    if not os.path.exists(path):
        os.makedirs(path)
    myList = os.listdir(path)
    for cl in myList:
        curImg = cv2.imread(f'{path}/{cl}')
        images.append(curImg)
        classNames.append(os.path.splitext(cl)[0])
    encodeListKnown = findEncodings(images)
    print(classNames)
    print('Encoding Complete')
    return encodeListKnown, classNames

class FaceAttendanceApp(App):
    def build(self):
        sm = ScreenManager()
        sm.add_widget(HomeScreen(name='home'))
        sm.add_widget(RegistrationScreen(name='registration'))
        checkin_screen = CheckinScreen(encodeListKnown, classNames, name='checkin')
        sm.add_widget(checkin_screen)
        return sm

if __name__ == '__main__':
    path = 'FaceData'
    encodeListKnown, classNames = load_images_and_encodings(path)
    FaceAttendanceApp().run()