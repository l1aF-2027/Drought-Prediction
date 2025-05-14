# 🌵 Drought Prediction

A powerful and interactive visualization platform for **drought prediction** and **analysis** across **U.S. counties** and global regions. This tool provides predictive insights using weather, soil, and spatial data.

## 🔍 Features

* 🗺️ **Geospatial visualization**:

  * County-level predictions across the U.S. using FIPS codes.
  * 5km resolution grid for global coverage.
* 🔎 **Smart place search** with suggestions and autocomplete.
* 📆 **Multi-week forecast** of drought levels.
* 📊 **Interactive charts** of historical weather, soil moisture, and climate variables.
* ⚡ **FastAPI backend** with integrated ML model serving.
* 🌐 **Next.js frontend** with responsive maps and dashboards.

---

## 🚀 Getting Started

Follow these steps to run the app on your local machine:

### 1️⃣ Clone the repository

```bash
git clone https://github.com/l1aF-2027/drought-prediction.git
cd drought-prediction
```

### 2️⃣ Set up the **frontend**

```bash
cd drought-prediction
npm install
npm run dev
```

> The frontend will start at `http://localhost:3000`.

### 3️⃣ Set up the **backend**

```bash
python -3.10 -m venv myvenv
myvenv/Scripts/activate     # On Windows
source myvenv/bin/activate  # On macOS/Linux

pip install -r requirements.txt
```

### ▶️ Run the ML Models (Optional for prediction tasks)

If you're using VSCode :

1. Press `Ctrl + Shift + P`
2. Type: `Tasks: Run Task`
3. Select: `Run All Models`

Or, run them manually if defined in `tasks.json`.

---

## 📸 Screenshots

> Example county-level drought forecast

![App Screenshot](https://github.com/user-attachments/assets/a0142f91-e223-4915-9bc3-d8973cdb60e6)

---

## 📁 Project Structure

```bash
drought-prediction/
├── drought-api/               # FastAPI server + ML models
├── drought-prediction/        # Next.js app for visualization
└── README.md
```

---

## 🧠 Technologies Used

* **Frontend**: Next.js, React, Tailwind CSS, Leaflet.js
* **Backend**: FastAPI, PyTorch, scikit-learn, pandas
* **ML Models**: LSTM, Transformers, GRU, LSTM + Attention
* **Maps**: GeoJSON, Shapely, US FIPS mapping

---

## 📬 Contact

* 📧 Email: \[[ha.huy.hoang.tk@example.com](mailto:ha.huy.hoang.tk@gmail.com)]
* 🌐 Website: \[[My Site](https://l1af.vercel.app/)]

---

## 📄 License

This project is licensed under the **MIT License**.

