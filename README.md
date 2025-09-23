
# 🎓 Dropout Risk Dashboard

A Streamlit-based dashboard to monitor students and detect early signs of dropout risk.
It integrates **attendance, scores, fees, and daily activity data** to provide real-time alerts for educators, mentors, students and parents.

⚡ This is a prototype developed for the internal hackathon round of Smart India Hackathon (SIH) at Gauhati University, by Team Hacksa.

🔗 **Live Demo**: [ai-dropout-dashboard-v2.streamlit.app](https://ai-dropout-dashboard-v2.streamlit.app/)

---

## 🚀 Features

* **Student Risk Evaluation**: Calculates Red / Amber / Green labels based on configurable thresholds.
* **Daily Activity Alerts**: Detects consecutive absences, missed assignments, or low scores.
* **Email Notifications**: Sends automated alerts to students, mentors, or parents via SMTP.
* **Interactive Attendance Heatmap**: Visualizes presence/absence trends with color coding.
* **Data Upload Support**: Upload custom CSVs for attendance, scores, fees, and daily activity.
* **Custom Thresholds**: Configure rules (attendance %, scores, fee overdue days, consecutive days) directly in the sidebar.
* **Downloadable Reports**: Export flagged students and alerts as CSVs.

---

## 📦 Installation

1. Clone the repository:

```bash
git clone https://github.com/your-username/dropout-risk-dashboard.git
cd dropout-risk-dashboard
```

2. Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

3. Install requirements:

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

Run the dashboard locally:

```bash
streamlit run dashboard.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

Or try the **hosted version** here:
👉 [https://ai-dropout-dashboard-v2.streamlit.app/](https://ai-dropout-dashboard-v2.streamlit.app/)

---

## 📂 Sample Data

If no CSVs are uploaded, the app **auto-generates sample data**.
You can also upload your own CSVs:

* **Attendance CSV**: `student_id, name, course, mentor, email, attendance_percent`
* **Scores CSV**: `student_id, assessment_1, assessment_2, assessment_3, failed_attempts, avg_score`
* **Fees CSV**: `student_id, total_fee_due, last_payment_date`
* **Daily Activity CSV**: `student_id, date, attended, assignment_submitted, score`
* **Heatmap CSV**: `ID, Name, YYYY-MM-DD ...` (dates as columns with `P` / `A` values)

---

## 🛠️ Tech Stack

* **Frontend & Dashboard**: [Streamlit](https://streamlit.io/)
* **Data Handling**: Pandas, NumPy
* **Visualization**: Matplotlib, Seaborn
* **Email Alerts**: Python `smtplib` + SMTP configuration

---

## 📧 Email Notifications

To enable email alerts:

1. Enter SMTP server details (e.g., `smtp.gmail.com`, port `587`).
2. Enter sender email and password.
3. Select students from the Alerts panel and send notifications.

---

## 🏗️ Project Structure

```
.
├── dashboard.py           # Main Streamlit app
├── requirements.txt       # Python dependencies
├── sample_data/           # Example CSVs (optional)
└── README.md              # Project documentation
```

---

## 🤝 Contributing

This project was built for hackathon prototyping. Contributions are welcome!

* Fork the repo
* Create a feature branch (`git checkout -b feature-name`)
* Commit changes and push
* Open a Pull Request

---

## 📜 License

MIT License – free to use and modify.

---
