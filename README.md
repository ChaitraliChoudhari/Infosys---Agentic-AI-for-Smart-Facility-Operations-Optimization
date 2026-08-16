# ⚡ Agentic FacilityOps AI Platform

An AI-powered facility management platform designed to monitor, analyse, and optimise energy consumption using intelligent analytics and agent-based recommendations.

This repository contains **Milestone 1: Energy Intelligence & Monitoring**, developed as part of the Infosys Springboard project.

---

## 📌 Project Overview

The Agentic FacilityOps AI Platform helps facility managers monitor energy consumption, identify abnormal usage, and receive intelligent recommendations to improve energy efficiency.

The system analyses historical energy data, generates alerts, and provides actionable insights through an interactive dashboard.

---

## 🚀 Milestone 1 Features

- 📊 Energy consumption analytics
- 📈 Interactive dashboard using Streamlit
- 🤖 Rule-based Energy Agent
- 🚨 High energy consumption alerts
- 💡 AI-generated energy-saving recommendations
- 🔌 FastAPI REST APIs
- 📉 Data visualisation using Plotly
- 🧹 Data cleaning and preprocessing

---

## 🛠️ Technologies Used

- Python
- FastAPI
- Streamlit
- Pandas
- Plotly
- Matplotlib
- Requests

---

## 📂 Project Structure

```
Agentic_FacilityOps_AI/
│
├── analytics/
├── backend/
├── dashboard/
├── database/
├── dataset/
├── energy_agent/
├── frontend/
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/your-username/Agentic-FacilityOps-AI.git

cd Agentic-FacilityOps-AI
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Backend

Navigate to the backend folder:

```bash
cd backend
```

Start FastAPI:

```bash
uvicorn main:app --reload
```

Open:

```
http://127.0.0.1:8000/docs
```

to access the API documentation.

---

## ▶️ Run the Dashboard

Navigate to the dashboard folder:

```bash
cd dashboard
```

Run Streamlit:

```bash
streamlit run app.py
```

The dashboard will open automatically in your browser