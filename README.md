# Turkish Sign Language Interpreter

A real-time Turkish Sign Language recognition system using TensorFlow and Streamlit.

## Features

- Real-time sign language recognition using webcam
- Support for 226 Turkish sign language gestures
- Memory-optimized performance modes
- Interactive UI with confidence scores
- Debug mode for detailed predictions

## Setup

1. Clone the repository:
```bash
git clone <your-repo-url>
cd <repo-name>
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run streamlit_app.py
```

## Usage

1. Click "Start Camera" to begin
2. Position yourself in the camera view
3. Perform Turkish Sign Language signs
4. The model will attempt to recognize your signs

## Tips for Better Recognition

- Ensure good lighting and clear background
- Position your hands clearly in view
- Perform signs at a moderate, consistent speed
- Hold each sign for a moment before transitioning
- Keep your hands within the camera frame
- Adjust the confidence threshold if needed

## Model Training

The model can be trained using `simple_model_trainer.py`. This script includes:
- Data augmentation
- Advanced model architecture with attention mechanism
- Memory optimization
- Performance monitoring

## License

MIT License 