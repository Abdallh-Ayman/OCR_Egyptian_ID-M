# **Egyptian ID Card Recognition System – AI-Powered OCR & Fraud Detection**

**A Python-based AI application for detecting, processing, and verifying Egyptian ID cards using YOLO and EasyOCR.**

## **Key Features**

🔹 **AI-Powered ID Detection** – Automatically detects and crops Egyptian ID cards from images.  
🔹 **Advanced OCR (Optical Character Recognition)** – Extracts Arabic and English text from ID cards using EasyOCR.  
🔹 **Field Extraction & Data Processing** – Captures essential details, including:

- Full Name
- Address
- National ID Number
- Birth Date
- Governorate
- Gender
- Birth Place
- Location
- Nationality  

🔹 **Fraud Detection System** – Detects fake IDs by verifying the authenticity of the ID photo and personal details.  
🔹 **Web Interface with Streamlit** – Provides a user-friendly dashboard for seamless ID card processing.

## **How It Works**

1️⃣ **Upload an Image** – Use the web interface to upload an Egyptian ID card.  
2️⃣ **AI-Powered Detection** – The system detects and extracts ID information.  
3️⃣ **ID Decoding & Verification** – Deciphers ID numbers and flags potential fraudulent documents.  
4️⃣ **Results Displayed** – View structured data, extracted text, and fraud detection status.

## **Installation Guide**

1. **Clone the repository**:
   ```bash
   git clone https://github.com/NASO7Y/ocr_egyptian_ID.git
   ```
2. **Navigate to the project directory**:
   ```bash
   cd ocr_egyptian_ID
   ```
3. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scriptsctivate
   ```
4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
5. **Run the application**:
   ```bash
   streamlit run APP.py
   ```

## **Model Training**

- **YOLO Object Detection** – Trained for Egyptian ID card detection.  
- **EasyOCR** – Used for high-accuracy text recognition in Arabic and English.

## **Why Choose This System?**

✅ **High Accuracy** – Advanced deep learning models ensure precise ID recognition.  
✅ **Fraud Detection** – Protects against fake IDs by verifying images and personal details.  
✅ **Fast & Automated** – AI speeds up document processing with minimal human effort.  
✅ **User-Friendly Web Interface** – Easy-to-use Streamlit dashboard for seamless operation.

## **Acknowledgments**

This project utilizes:

- [YOLO](https://github.com/ultralytics/yolov5) for object detection.  
- [EasyOCR](https://github.com/JaidedAI/EasyOCR) for text recognition.  
- [Streamlit](https://streamlit.io/) for the web interface.

## **Contribute to the Project**

Contributions are welcome! Fork the repository and submit a pull request with improvements. Make sure your code meets project standards and includes tests.

## **Contact & Support**

For questions or feedback, feel free to open an issue or reach out to [NASO7Y](https://github.com/NASO7Y).

Email: ahmed.noshy2004@gmail.com

LinkedIn: [LinkedIn](https://www.linkedin.com/in/nos7y/)

