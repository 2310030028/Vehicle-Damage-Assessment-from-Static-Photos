# 🚗 Car Damage Assessment AI

A computer vision system for automatic car damage detection and assessment using YOLOv8 and Streamlit.

## ✨ Features

- **Damage Detection**: Identifies various types of vehicle damage (scratches, dents, cracks, etc.)
- **Damage Assessment**: Provides severity assessment and location of damage
- **Interactive UI**: User-friendly web interface for easy interaction
- **Report Generation**: Generates detailed damage assessment reports
- **Real-time Processing**: Processes images in real-time for quick assessments

## 🛠 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/SrikartikMateti/Car-Damage-Assessment-AI.git
   cd Car-Damage-Assessment-AI
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Quick Start

1. **Run the application**
   ```bash
   streamlit run app.py
   ```

2. **Open your browser** to `http://localhost:8501`

3. **Upload an image** of a damaged vehicle using the file uploader

4. **View the results** including detected damage, severity assessment, and a detailed report

## 🏗 Project Structure

```
Car-Damage-Assessment-AI/
├── app.py                 # Main Streamlit application
├── car_damage_detector.py # Core detection logic with YOLOv8
├── utils.py              # Utility functions for processing
├── requirements.txt      # Python dependencies
└── README.md            # Project documentation
```

## 📂 Sample Images

You can use the following sample images for testing:

1. `samples/front_damage.jpg` - Front-end collision damage
2. `samples/side_damage.jpg` - Side impact damage
3. `samples/rear_damage.jpg` - Rear-end collision damage

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact

For any questions or feedback, please open an issue on GitHub.
