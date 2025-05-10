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

# Custom Attention Layer (keeping for reference but not using)
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
        # Apply attention mechanism
        e = K.tanh(K.dot(x, self.kernel))
        a = K.softmax(e, axis=1)
        output = x * a
        return K.sum(output, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

    def get_config(self):
        config = super(Attention, self).get_config()
        return config

# Load model with custom_objects
@st.cache_resource
def load_model_for_inference():
    try:
        # Try loading with custom_objects
        model = load_model('tsl_simple_model_v8.keras', 
                          custom_objects={'Attention': Attention},
                          compile=False)
        # Recompile the model
        model.compile(optimizer='adam',
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Ensure we use the same normalization as training
def normalize_sequence(sequence):
    # Take a single sequence and normalize it
    norm_seq = np.zeros_like(sequence)
    
    # Find non-zero values
    mask = sequence != 0
    if np.any(mask):
        # Get stats only from non-zero values
        mean = np.mean(sequence[mask])
        std = np.std(sequence[mask])
        # Normalize with small epsilon to avoid division by zero
        if std > 1e-6:
            norm_seq[mask] = (sequence[mask] - mean) / std
        else:
            norm_seq[mask] = sequence[mask] - mean
    
    return norm_seq

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

# MediaPipe setup
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Function to extract keypoints (exact same as in training)
def extract_keypoints(results):
    # Extract pose landmarks
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() \
        if results.pose_landmarks else np.zeros(33 * 4)
    
    # Extract hand landmarks
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() \
        if results.left_hand_landmarks else np.zeros(21 * 3)
    
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() \
        if results.right_hand_landmarks else np.zeros(21 * 3)
    
    # Concatenate
    return np.concatenate([pose, lh, rh])

# Add these new functions after the existing imports
def smooth_predictions(predictions, window_size=3):
    """Apply temporal smoothing to predictions"""
    if len(predictions) < window_size:
        return predictions
    smoothed = []
    for i in range(len(predictions)):
        start_idx = max(0, i - window_size + 1)
        window = predictions[start_idx:i+1]
        smoothed.append(np.mean(window, axis=0))
    return np.array(smoothed)

def get_ensemble_prediction(predictions, threshold=0.4):
    """Get ensemble prediction from multiple frames"""
    # Average predictions
    avg_pred = np.mean(predictions, axis=0)
    # Get top 3 predictions
    top_3_idx = np.argsort(avg_pred)[-3:][::-1]
    top_3_conf = avg_pred[top_3_idx]
    
    # If top prediction is confident enough
    if top_3_conf[0] >= threshold:
        return top_3_idx[0], top_3_conf[0]
    
    # If top 2 predictions are close and confident
    if top_3_conf[0] >= 0.3 and (top_3_conf[0] - top_3_conf[1]) < 0.1:
        return top_3_idx[0], top_3_conf[0]
    
    return None, 0.0

# Add this function after the imports
def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process()
    memory_info = process.memory_info()
    return memory_info.rss / 1024 / 1024  # Convert to MB

# Add this function after get_memory_usage
def optimize_tf_memory():
    """Optimize TensorFlow memory usage"""
    # Disable GPU if not needed
    tf.config.set_visible_devices([], 'GPU')
    
    # Limit thread usage
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    
    # Clear any existing sessions
    tf.keras.backend.clear_session()

# App function
def main():
    # Optimize TensorFlow memory usage
    optimize_tf_memory()
    
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
        buffer_size = 30  # MAX_SEQ_LENGTH
    elif performance_mode == "Low Memory":
        frame_skip = 3
        smoothing_window = 2
        confidence_threshold = 0.4
        buffer_size = 20  # Reduced buffer size
    else:  # Balanced
        frame_skip = 2
        smoothing_window = 3
        confidence_threshold = 0.35
        buffer_size = 25  # Medium buffer size
    
    model_path = 'tsl_simple_model_v14.keras'
    if not os.path.exists(model_path):
        st.sidebar.error(f"Model file {model_path} not found!")
        return
    
    # Load model with memory optimization
    try:
        # Clear any existing models from memory
        gc.collect()
        tf.keras.backend.clear_session()
        
        # Load model with memory optimization
        model = load_model(model_path, custom_objects={'Attention': Attention})
        
        # Optimize model for inference
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy'],
            run_eagerly=False
        )
        
        st.sidebar.success(f"Model loaded: {model_path}")
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return
    
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
        cam_placeholder = st.empty()
        
        # Camera button with enhanced styling
        start_button = st.button("🎥 Start Camera", key="start_camera")
        
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
    
    # Run camera if button pressed
    if start_button:
        frame_buffer = []
        prediction_buffer = []
        
        try:
            # Initialize webcam with lower resolution for memory efficiency
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("⚠️ Camera not available. Running in demo mode with sample data.")
                # Create a sample frame for demo
                demo_frame = np.zeros((180, 240, 3), dtype=np.uint8)
                cv2.putText(demo_frame, "Demo Mode", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cam_placeholder.image(demo_frame, channels="BGR", use_container_width=True)
                return
            
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 240)  # Reduced resolution
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 180)
            
            # Initialize holistic model with lowest complexity
            try:
                with mp_holistic.Holistic(
                    min_detection_confidence=0.3,
                    min_tracking_confidence=0.3,
                    model_complexity=0
                ) as holistic:
                    # Status variables
                    frame_count = 0
                    last_prediction = None
                    last_confidence = 0
                    last_update_time = time.time()
                    
                    while True:
                        # Memory monitoring
                        if frame_count % 30 == 0:  # Update every 30 frames
                            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
                            memory_placeholder.metric("Memory Usage", f"{memory_usage:.1f} MB")
                        
                        # Read frame
                        ret, frame = cap.read()
                        if not ret:
                            st.error("Failed to read from camera")
                            break
                        
                        # Skip frames based on performance mode
                        frame_count += 1
                        if frame_count % frame_skip != 0:
                            continue
                        
                        # Process frame
                        try:
                            # Convert to RGB for MediaPipe
                            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            
                            # Process with MediaPipe
                            results = holistic.process(rgb_frame)
                            
                            if results.pose_landmarks:
                                # Extract landmarks
                                landmarks = extract_keypoints(results)
                                
                                # Add to buffer
                                frame_buffer.append(landmarks)
                                if len(frame_buffer) > buffer_size:
                                    frame_buffer.pop(0)
                                
                                # Update buffer display
                                buffer_progress.progress(len(frame_buffer) / buffer_size)
                                buffer_counter.text(f"Frames: {len(frame_buffer)}/{buffer_size}")
                                
                                # Make prediction when buffer is full
                                if len(frame_buffer) == buffer_size:
                                    # Prepare sequence
                                    sequence = np.array(frame_buffer)
                                    
                                    # Make prediction
                                    prediction = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
                                    predicted_class = np.argmax(prediction)
                                    confidence = float(prediction[predicted_class])
                                    
                                    # Apply smoothing
                                    prediction_buffer.append(predicted_class)
                                    if len(prediction_buffer) > smoothing_window:
                                        prediction_buffer.pop(0)
                                    
                                    # Get most common prediction
                                    if len(prediction_buffer) == smoothing_window:
                                        final_prediction = max(set(prediction_buffer), key=prediction_buffer.count)
                                        final_confidence = confidence
                                        
                                        # Update prediction if confidence is high enough
                                        if final_confidence > confidence_threshold:
                                            if (final_prediction != last_prediction or 
                                                time.time() - last_update_time > 1.0):
                                                last_prediction = final_prediction
                                                last_confidence = final_confidence
                                                last_update_time = time.time()
                                                
                                                # Update prediction display
                                                prediction_text.markdown(f"### {CLASS_MAP.get(final_prediction, f'Class {final_prediction}')}")
                                                confidence_bar.progress(final_confidence)
                                                
                                                # Show debug info
                                                if show_debug:
                                                    debug_text.text(f"""
                                                    Raw Prediction: {predicted_class}
                                                    Confidence: {confidence:.2f}
                                                    Buffer Size: {len(frame_buffer)}
                                                    Frame Skip: {frame_skip}
                                                    """)
                            
                            # Draw landmarks if enabled
                            if show_keypoints and results.pose_landmarks:
                                mp_drawing.draw_landmarks(
                                    frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
                            
                            # Display frame
                            cam_placeholder.image(frame, channels="BGR", use_container_width=True)
                            
                        except Exception as e:
                            st.error(f"Error processing frame: {e}")
                            continue
                            
            except PermissionError as e:
                st.error("⚠️ Permission error accessing MediaPipe models. Running in limited mode.")
                st.info("The app will continue with basic functionality, but some features may be limited.")
                return
                
        except Exception as e:
            st.error(f"⚠️ Error initializing camera: {str(e)}")
            st.info("Running in demo mode with sample data.")
            # Create a sample frame for demo
            demo_frame = np.zeros((180, 240, 3), dtype=np.uint8)
            cv2.putText(demo_frame, "Demo Mode", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cam_placeholder.image(demo_frame, channels="BGR", use_container_width=True)
            return
        finally:
            if 'cap' in locals():
                cap.release()
            # Clear memory
            gc.collect()
            tf.keras.backend.clear_session()
    
    else:
        # Display placeholder when camera is not running
        cam_placeholder.image("https://via.placeholder.com/640x480.png?text=Camera+Off", use_container_width=True)
        
        # Enhanced instructions
        st.markdown("""
        ## Instructions
        1. Click "Start Camera" to begin
        2. Position yourself in the camera view
        3. Perform Turkish Sign Language signs
        4. The model will attempt to recognize your signs
        
        ### Tips for better recognition:
        - Ensure good lighting and clear background
        - Position your hands clearly in view
        - Perform signs at a moderate, consistent speed
        - Hold each sign for a moment before transitioning
        - Keep your hands within the camera frame
        - Adjust the confidence threshold if needed
        """)

if __name__ == "__main__":
    main()