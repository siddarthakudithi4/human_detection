## Person Detection with OpenVINO

This project demonstrates person detection using OpenVINO's pre-trained model `person-detection-retail-0013`. The detection results are visualized with bounding boxes on static images.

## 🚀 Features

- Person detection using OpenVINO inference engine
- Real-time image processing with OpenCV
- Result visualization using Matplotlib

---

## 📂 Folder Structure

```
human_detection-main/
├── openvino_detector_2022_3/
│   └── model_2022_3/
│       ├── person-detection-retail-0013.xml
│       └── person-detection-retail-0013.bin
└── videos/
    └── image5.jpg
```

---

## ⚙️ Requirements

Ensure you have Python **3.6 - 3.10** installed.

### Install Dependencies

```bash
pip install -r requirements.txt
```

### `requirements.txt` Includes:

- `opencv-python`
- `numpy`
- `matplotlib`
- `openvino==2022.3.0`
- `ipython`

If `pip` is outdated:

```bash
pip install --upgrade pip
```

---

## 💡 Setup OpenVINO Toolkit

1. **Download OpenVINO Toolkit:** [Intel OpenVINO Toolkit](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/download.html)
2. **Activate OpenVINO:**\
   **Linux:**
   ```bash
   source /opt/intel/openvino/setupvars.sh
   ```
   **Windows:**
   ```bash
   "C:\Program Files (x86)\Intel\openvino\bin\setupvars.bat"
   ```

---

## 🧠 Model Download (if missing)

```bash
pip install openvino-dev
omz_downloader --name person-detection-retail-0013
```

---

## 📸 Run the Code

```bash
python person_detection.py
```

Make sure the paths for the model and image are correct:

```python
model = "human_detection-main/openvino_detector_2022_3/model_2022_3/person-detection-retail-0013.xml"
image_path = "human_detection-main/openvino_detector_2022_3/videos/image5.jpg"
```

---

## 🖼️ Example Output

The output image will display bounding boxes around detected persons.
(https://github.com/user-attachments/assets/34051632-56b2-4d86-b5e2-ee295e8903e7)


---

## 🙏 Acknowledgements

- OpenVINO Toolkit by Intel
- Pre-trained models from Open Model Zoo

Feel free to contribute or suggest improvements!

