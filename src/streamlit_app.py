import os
import streamlit as st
import cv2
import tempfile
import numpy as np
from pytube import YouTube


class FallDetectApp():
    def __init__(self, model, fall_label=0, stream_detection_confidence=60, stream_frame_threshold=5):
        self.model = model
        self.fall_label = fall_label
        
        self.stream_fall_frame_counter = 0
        self.stream_detection_confidence = stream_detection_confidence
        self.stream_frame_threshold = stream_frame_threshold
        
        st.title("Human Fall Detection")
        st.write("This app detects human falls from multiple sources including streams.")
        # Option to select input source
        self.input_source = st.radio("Select input source:", ('image', 'image_URL', 'video', 'YouTube', 'stream', 'multi_stream')).lower()
        self._ask_inputs()
        
    def _ask_inputs(self):
        if self.input_source == 'image':
            uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
            self.image_input(uploaded_image)
            
        elif self.input_source == 'image_url':
            uploaded_url = st.text_input("Enter the image url:")
            self.image_url_input(uploaded_url)
            
        elif self.input_source == 'video':
            uploaded_video = st.file_uploader("Upload a video...", type=["mp4", "avi", "mov"])
            self.video_input(uploaded_video)
            
        elif self.input_source == 'youtube':
            youtube_url = st.text_input("Enter the YouTube video URL:")
            self.youtube_input(youtube_url)
            
        elif self.input_source == 'stream':
            uploaded_stream = st.text_input("Enter the video stream URL (RTSP, RTMP, TCP, etc.):")
            self.stream_input(uploaded_stream)


    def _inference(self, frame, stream=False):
        return self.model(frame) if not stream else self.model(frame, stream=True)
    
    def _stream_detection(self, cls, confidence):
        
        if (cls == self.fall_label) and (confidence > self.stream_detection_confidence):
            self.stream_fall_frame_counter += 1
            self._action_following_detection()
    
    def _action_following_detection(self):
        if self.stream_fall_frame_counter > self.stream_detection_confidence:
            print('DO YOUR ACTION')
       # CALL / SEND MSG / MAIL ?
        


    def _detection(self, frame, stream=False):
        # Perform inference on the frame
        results = self._inference(frame, stream)

        # Draw bounding boxes on the frame
        for result in results:
            initial_frame_to_modify = result.orig_img

            for xyxy in result.boxes.xyxy:
                x1, y1, x2, y2 = map(int, xyxy.tolist())
                confidence = result.boxes.conf[0].item()
                cls = result.boxes.cls[0].item()

                if cls == self.fall_label:  
                    label = f'Fall_detected {confidence:.2f}'
                    cv2.rectangle(initial_frame_to_modify, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(initial_frame_to_modify, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

                if stream:
                    self._stream_detection(cls, confidence)

        return initial_frame_to_modify



    def stream_input(self, video_stream_url):
        # Real-time fall detection from the provided video stream URL
        if video_stream_url:
            st.write(f"Connecting to video stream: {video_stream_url}")
            video_capture = cv2.VideoCapture(video_stream_url)

            if not video_capture.isOpened():
                st.error("Error: Could not open video stream. Please check the URL and try again.")
            # If stream is opened
            else:
                stframe = st.empty()
                # While stream is on
                while video_capture.isOpened():
                    # Read a frame
                    success, frame = video_capture.read()
                    if not success:
                        st.write("Stream ended or failed to capture video")
                        break
                    
                    # Detect falls in the frame
                    result_frame = self._detection(frame, stream=True)
                    # Display the frame with detection boxes
                    stframe.image(result_frame, channels="BGR")
                    
                video_capture.release()
        

    def image_input(self, uploaded_image):
        # Handle Image input
        if uploaded_image:
            stframe = st.empty()
            
            file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)
            
            # Detect falls in the image
            result_image = self._detection(image)
            
            # Display the image with detection boxes
            stframe.image(result_image, channels="BGR")

    def image_url_input(self, image_url):
        # Handle image URL input
        if image_url:
            stframe = st.empty()
            # Run inference on the source
            result_image = self._detection(image_url)
            # Display the image with detection boxes
            stframe.image(result_image, channels="BGR")
            
            
    def video_input(self, video):
        if video:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(video.read())

            video_capture = cv2.VideoCapture(tfile.name)

            stframe = st.empty()

            while video_capture.isOpened():
                success, frame = video_capture.read()
                if not success:
                    break

                # Detect falls in the frame
                result_frame = self._detection(frame)
                # Display the frame with detection boxes
                stframe.image(result_frame, channels="BGR")
                
            video_capture.release()
            tfile.close()
            os.unlink(tfile.name)
        
            
    def youtube_input(self, youtube_url):
        try:
            yt = YouTube(youtube_url)
            stream = yt.streams.filter(file_extension='mp4').first()
            
            tfile = tempfile.NamedTemporaryFile(delete=False)
            
            stream.download(output_path=os.path.dirname(tfile.name), filename=os.path.basename(tfile.name))
            
            video_capture = cv2.VideoCapture(tfile.name)
            stframe = st.empty()
            while video_capture.isOpened():
                success, frame = video_capture.read()
                if not success:
                    break
                
                # Detect falls in the frame
                result_frame = self._detection(frame)
                
                # Display the frame with detection boxes
                stframe.image(frame, channels="BGR")
                
            video_capture.release()
            tfile.close()
            os.unlink(tfile.name)
            
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
