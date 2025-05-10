import os
os.environ["MEDIAPIPE_MODEL_PATH"] = "/tmp/mediapipe"
import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import time
import os
from tensorflow.keras.layers import Layer, InputSpec
import tensorflow.keras.backend as K
from tensorflow.keras.saving import register_keras_serializable
import psutil
import gc
import tensorflow as tf
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
import threading
import queue

# Custom Attention Layer
@register_keras_serializable()
class Attention(Layer):
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                    shape=(input_shape[-1], 1),
                                    initializer='uniform',
                                    trainable=True)
        super(Attention, self).build(input_shape)

    def call(self, x, mask=None):
        e = K.tanh(K.dot(x, self.kernel))
        a = K.softmax(e, axis=1)
        output = x * a
        return K.sum(output, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

    def get_config(self):
        config = super(Attention, self).get_config()
        return config

# Video Processor for WebRTC
class SignLanguageProcessor(VideoProcessorBase):
    def __init__(self):
        self.frame_buffer = []
        self.prediction_buffer = []
        self.buffer_size = 30
        self.smoothing_window = 3
        self.confidence_threshold = 0.4
        self.last_prediction = None
        self.last_confidence = 0
        self.last_update_time = time.time()
        self.frame_count = 0
        self.frame_skip = 2
        self.show_keypoints = True
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3,
            model_complexity=0
        )
        self.model = None
        self.prediction_queue = queue.Queue()
        
    def load_model(self, model_path):
        try:
            self.model = load_model(model_path, custom_objects={'Attention': Attention})
            self.model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'],
                run_eagerly=False
            )
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def extract_keypoints(self, results):
        if not results.pose_landmarks:
            return np.zeros(33 * 4 + 21 * 3 * 2)  # pose + left hand + right hand
        
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten()
        
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() \
            if results.left_hand_landmarks else np.zeros(21 * 3)
        
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() \
            if results.right_hand_landmarks else np.zeros(21 * 3)
        
        return np.concatenate([pose, lh, rh])

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Skip frames
        self.frame_count += 1
        if self.frame_count % self.frame_skip != 0:
            return av.VideoFrame.from_ndarray(img, format="bgr24")
        
        # Process frame
        try:
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = self.holistic.process(rgb_frame)
            
            if results.pose_landmarks:
                # Extract landmarks
                landmarks = self.extract_keypoints(results)
                
                # Add to buffer
                self.frame_buffer.append(landmarks)
                if len(self.frame_buffer) > self.buffer_size:
                    self.frame_buffer.pop(0)
                
                # Make prediction when buffer is full
                if len(self.frame_buffer) == self.buffer_size and self.model is not None:
                    # Prepare sequence
                    sequence = np.array(self.frame_buffer)
                    
                    # Make prediction
                    prediction = self.model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
                    predicted_class = np.argmax(prediction)
                    confidence = float(prediction[predicted_class])
                    
                    # Apply smoothing
                    self.prediction_buffer.append(predicted_class)
                    if len(self.prediction_buffer) > self.smoothing_window:
                        self.prediction_buffer.pop(0)
                    
                    # Get most common prediction
                    if len(self.prediction_buffer) == self.smoothing_window:
                        final_prediction = max(set(self.prediction_buffer), key=self.prediction_buffer.count)
                        final_confidence = confidence
                        
                        # Update prediction if confidence is high enough
                        if final_confidence > self.confidence_threshold:
                            if (final_prediction != self.last_prediction or 
                                time.time() - self.last_update_time > 1.0):
                                self.last_prediction = final_prediction
                                self.last_confidence = final_confidence
                                self.last_update_time = time.time()
                                
                                # Put prediction in queue for main thread
                                self.prediction_queue.put((final_prediction, final_confidence))
                
                # Draw landmarks if enabled
                if self.show_keypoints:
                    mp_drawing.draw_landmarks(
                        img, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
                    if results.left_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            img, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                    if results.right_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            img, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            
        except Exception as e:
            print(f"Error processing frame: {e}")
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Set page config with custom theme
st.set_page_config(
    page_title="TSL Interpreter",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 20px;
        border-radius: 5px;
        border: none;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .prediction-box {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .info-box {
        background-color: #e8f5e9;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Constants
MAX_SEQ_LENGTH = 30
PREDICTION_THRESHOLD = 0.4  # Lower threshold for testing

# Class mapping (update with your actual class mapping)
CLASS_MAP = {
    0: "abla (sister)",  
    1: "acele (hurry)",  
    2: "acikmak (hungry)",  
    3: "afiyet_olsun (enjoy_your_meal)",  
    4: "agabey (brother)",  
    5: "agac (tree)",  
    6: "agir (heavy)",  
    7: "aglamak (cry)",  
    8: "aile (family)",  
    9: "akilli (wise)",  
    10: "akilsiz (unwise)",  
    11: "akraba (kin)",  
    12: "alisveris (shopping)",  
    13: "anahtar (key)",  
    14: "anne (mother)",  
    15: "arkadas (friend)",  
    16: "ataturk (ataturk)",  
    17: "ayakkabi (shoe)",  
    18: "ayna (mirror)",  
    19: "ayni (same)",  
    20: "baba (father)",  
    21: "bahce (garden)",  
    22: "bakmak (look)",  
    23: "bal (honey)",  
    24: "bardak (glass)",  
    25: "bayrak (flag)",  
    26: "bayram (feast)",  
    27: "bebek (baby)",  
    28: "bekar (single)",  
    29: "beklemek (wait)",  
    30: "ben (I)",  
    31: "benzin (petrol)",  
    32: "beraber (together)",  
    33: "bilgi_vermek (inform)",  
    34: "biz (we)",  
    35: "calismak (work)",  
    36: "carsamba (wednesday)",  
    37: "catal (fork)",  
    38: "cay (tea)",  
    39: "caydanlik (teapot)",  
    40: "cekic (hammer)",  
    41: "cirkin (ugly)",  
    42: "cocuk (child)",  
    43: "corba (soup)",  
    44: "cuma (friday)",  
    45: "cumartesi (saturday)",  
    46: "cuzdan (wallet)",  
    47: "dakika (minute)",  
    48: "dede (grandfather)",  
    49: "degistirmek (change)",  
    50: "devirmek (topple)",  
    51: "devlet (government)",  
    52: "doktor (doctor)",  
    53: "dolu (full)",  
    54: "dugun (wedding)",  
    55: "dun (yesterday)",  
    56: "dusman (enemy)",  
    57: "duvar (wall)",  
    58: "eczane (pharmacy)",  
    59: "eldiven (glove)",  
    60: "emek (labor)",  
    61: "emekli (retired)",  
    62: "erkek (male)",  
    63: "et (meal)",  
    64: "ev (house)",  
    65: "evet (yes)",  
    66: "evli (married)",  
    67: "ezberlemek (memorize)",  
    68: "fil (elephant)",  
    69: "fotograf (photograph)",  
    70: "futbol (football)",  
    71: "gecmis (past)",  
    72: "gecmis_olsun (get_well)",  
    73: "getirmek (bring)",  
    74: "gol (lake)",  
    75: "gomlek (shirt)",  
    76: "gormek (see)",  
    77: "gostermek (show)",  
    78: "gulmek (laugh)",  
    79: "hafif (lightweight)",  
    80: "hakli (right)",  
    81: "hali (carpet)",  
    82: "hasta (ill)",  
    83: "hastane (hospital)",  
    84: "hata (fault)",  
    85: "havlu (towel)",  
    86: "hayir (no)",  
    87: "hayirli_olsun (congratulations)",  
    88: "hayvan (animal)",  
    89: "hediye (gift)",  
    90: "helal (halal)",  
    91: "hep (always)",  
    92: "hic (never)",  
    93: "hoscakal (goodbye)",  
    94: "icmek (drink)",  
    95: "igne (needle)",  
    96: "ilac (medicine)",  
    97: "ilgilenmemek (not_interested)",  
    98: "isik (light)",  
    99: "itmek (push)",  
    100: "iyi (good)",  
    101: "kacmak (escape)",  
    102: "kahvalti (breakfast)",  
    103: "kalem (pencil)",  
    104: "kalorifer (radiator)",  
    105: "kapi (door)",  
    106: "kardes (sibling)",  
    107: "kavsak (crossroads)",  
    108: "kaza (accident)",  
    109: "kemer (belt)",  
    110: "keske (if_only)",  
    111: "kim (who)",  
    112: "kimlik (identity)",  
    113: "kira (rent)",  
    114: "kitap (book)",  
    115: "kiyma (mince)",  
    116: "kiz (female)",  
    117: "koku (smell)",  
    118: "kolonya (cologne)",  
    119: "komur (coal)",  
    120: "kopek (dog)",  
    121: "kopru (bridge)",  
    122: "kotu (bad)",  
    123: "kucak (lap)",  
    124: "leke (stain)",  
    125: "maas (salary)",  
    126: "makas (scissors)",  
    127: "masa (tongs)",  
    128: "masallah (god_preserve)",  
    129: "melek (angel)",  
    130: "memnun_olmak (be_pleased)",  
    131: "mendil (napkin)",  
    132: "merdiven (stairs)",  
    133: "misafir (guest)",  
    134: "mudur (manager)",  
    135: "musluk (tap)",  
    136: "nasil (how)",  
    137: "neden (why)",  
    138: "nerede (where)",  
    139: "nine (grandmother)",  
    140: "ocak (oven)",  
    141: "oda (room)",  
    142: "odun (wood)",  
    143: "ogretmen (teacher)",  
    144: "okul (school)",  
    145: "olimpiyat (olympiad)",  
    146: "olmaz (nope)",  
    147: "olur (allright)",  
    148: "onlar (they)",  
    149: "orman (forest)",  
    150: "oruc (fasting)",  
    151: "ozur_dilemek (apologize)",  
    152: "pamuk (cotton)",  
    153: "pantolon (trousers)",  
    154: "para (money)",  
    155: "pastirma (pastrami)",  
    156: "patates (potato)",  
    157: "pazar (sunday)",  
    158: "pazartesi (monday)",  
    159: "pencere (window)",  
    160: "persembe (thursday)",  
    161: "piknik (picnic)",  
    162: "polis (police)",  
    163: "psikoloji (psychology)",  
    164: "rica_etmek (request)",  
    165: "saat (hour)",  
    166: "sabun (soap)",  
    167: "salca (sauce)",  
    168: "sali (tuesday)",  
    169: "sampiyon (champion)",  
    170: "sapka (hat)",  
    171: "savas (war)",  
    172: "seker (sugar)",  
    173: "selam (hi)",  
    174: "semsiye (umbrella)",  
    175: "sen (you)",  
    176: "senet (bill)",  
    177: "serbest (free)",  
    178: "ses (voice)",  
    179: "sevmek (love)",  
    180: "seytan (evil)",  
    181: "sinir (border)",  
    182: "siz (you)",  
    183: "soylemek (say)",  
    184: "soz (promise)",  
    185: "sut (milk)",  
    186: "tamam (okay)",  
    187: "tarak (comb)",  
    188: "tarih (date)",  
    189: "tatil (holiday)",  
    190: "tatli (sweet)",  
    191: "tavan (ceiling)",  
    192: "tehlike (danger)",  
    193: "telefon (telephone)",  
    194: "terazi (scales)",  
    195: "terzi (tailor)",  
    196: "tesekkur (thanks)",  
    197: "tornavida (screwdriver)",  
    198: "turkiye (turkey)",  
    199: "turuncu (orange)",  
    200: "tuvalet (toilet)",  
    201: "un (flour)",  
    202: "uzak (far)",  
    203: "uzgun (sad)",  
    204: "var (existing)",  
    205: "vergi (tax)",  
    206: "yakin (near)",  
    207: "yalniz (alone)",  
    208: "yanlis (wrong)",  
    209: "yapmak (do)",  
    210: "yarabandi (band-aid)",  
    211: "yardim (help)",  
    212: "yarin (tomorrow)",  
    213: "yasak (forbidden)",  
    214: "yastik (pillow)",  
    215: "yatak (bed)",  
    216: "yavas (slow)",  
    217: "yemek (eat)",  
    218: "yemek_pisirmek (cook)",  
    219: "yildiz (star)",  
    220: "yok (absent)",  
    221: "yol (road)",  
    222: "yorgun (tired)",  
    223: "yumurta (egg)",  
    224: "zaman (time)",  
    225: "zor (difficult)"  
    }

# App function
def main():
    st.title("🤟 Turkish Sign Language Interpreter")
    
    # Add memory monitoring in sidebar
    st.sidebar.header("System Monitor")
    memory_placeholder = st.sidebar.empty()
    
    # Add model selection in sidebar
    st.sidebar.header("Settings")
    model_option = st.sidebar.selectbox(
        "Model to use:",
        ["Simple LSTM Model", "Advanced Model"],
        index=0
    )
    
    # Add performance mode with memory-optimized defaults
    performance_mode = st.sidebar.selectbox(
        "Performance Mode",
        ["Low Memory", "Balanced", "High Accuracy"],
        index=0  # Default to Low Memory
    )
    
    # Adjust settings based on performance mode
    if performance_mode == "High Accuracy":
        frame_skip = 1
        smoothing_window = 4
        confidence_threshold = 0.35
        buffer_size = 30
    elif performance_mode == "Low Memory":
        frame_skip = 3
        smoothing_window = 2
        confidence_threshold = 0.4
        buffer_size = 20
    else:  # Balanced
        frame_skip = 2
        smoothing_window = 3
        confidence_threshold = 0.35
        buffer_size = 25
    
    # Display options
    st.sidebar.markdown("---")
    st.sidebar.subheader("Display Options")
    show_keypoints = st.sidebar.checkbox("Show landmarks", value=True)
    show_debug = st.sidebar.checkbox("Show debug info", value=False)
    
    # Create layout
    col1, col2 = st.columns([3, 1])
    
    # Camera column
    with col1:
        st.markdown("### 📸 Camera Feed")
        
        # Initialize WebRTC
        webrtc_ctx = webrtc_streamer(
            key="sign-language",
            video_processor_factory=SignLanguageProcessor,
            rtc_configuration=RTCConfiguration(
                {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
            ),
            media_stream_constraints={"video": True, "audio": False},
        )
        
        # Add info box with enhanced tips
        st.markdown("""
        <div class="info-box">
        <h4>💡 Tips for Better Recognition:</h4>
        <ul>
            <li>Ensure good lighting and clear background</li>
            <li>Position your hands clearly in view</li>
            <li>Perform signs at a moderate, consistent speed</li>
            <li>Hold each sign for a moment before transitioning</li>
            <li>Keep your hands within the camera frame</li>
            <li>Adjust the confidence threshold if needed</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Prediction column
    with col2:
        st.markdown("### 🎯 Recognition")
        prediction_text = st.empty()
        confidence_bar = st.empty()
        
        # Buffer display
        st.markdown("### 📊 Sequence Buffer")
        buffer_progress = st.progress(0)
        buffer_counter = st.empty()
        
        # Add debug info
        if show_debug:
            st.markdown("### 🔍 Debug Info")
            debug_text = st.empty()
    
    # Load model
    model_path = 'tsl_simple_model_v14.keras'
    if not os.path.exists(model_path):
        st.sidebar.error(f"Model file {model_path} not found!")
        return
    
    # Initialize video processor and load model
    if webrtc_ctx.video_processor:
        webrtc_ctx.video_processor.buffer_size = buffer_size
        webrtc_ctx.video_processor.smoothing_window = smoothing_window
        webrtc_ctx.video_processor.confidence_threshold = confidence_threshold
        webrtc_ctx.video_processor.frame_skip = frame_skip
        webrtc_ctx.video_processor.show_keypoints = show_keypoints
        
        if webrtc_ctx.video_processor.load_model(model_path):
            st.sidebar.success("Model loaded successfully!")
        else:
            st.sidebar.error("Failed to load model!")
            return
    
    # Update UI with predictions
    if webrtc_ctx.video_processor:
        try:
            while True:
                if not webrtc_ctx.video_processor.prediction_queue.empty():
                    prediction, confidence = webrtc_ctx.video_processor.prediction_queue.get()
                    prediction_text.markdown(f"### {CLASS_MAP.get(prediction, f'Class {prediction}')}")
                    confidence_bar.progress(confidence)
                    
                    if show_debug:
                        debug_text.text(f"""
                        Prediction: {prediction}
                        Confidence: {confidence:.2f}
                        Buffer Size: {len(webrtc_ctx.video_processor.frame_buffer)}
                        Frame Skip: {frame_skip}
                        """)
                
                # Update buffer display
                if webrtc_ctx.video_processor.frame_buffer:
                    buffer_progress.progress(len(webrtc_ctx.video_processor.frame_buffer) / buffer_size)
                    buffer_counter.text(f"Frames: {len(webrtc_ctx.video_processor.frame_buffer)}/{buffer_size}")
                
                # Update memory usage
                memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
                memory_placeholder.metric("Memory Usage", f"{memory_usage:.1f} MB")
                
                time.sleep(0.1)
        except Exception as e:
            st.error(f"Error in main loop: {e}")

if __name__ == "__main__":
    main()