# dashboard.py
# Streamlit dashboard for Drop-out Prediction and Counseling
# Run: streamlit run dashboard.py

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

st.set_page_config(page_title="Dropout Risk Dashboard", layout="wide")

# -----------------------
# Utility: generate sample data if CSVs are missing
# -----------------------
def generate_sample_data(n=150, seed=42):
    np.random.seed(seed)
    student_ids = [f"S{1000+i}" for i in range(n)]
    names = [f"Student_{i}" for i in range(n)]
    mentors = [f"mentor_{i%6}" for i in range(n)]
    courses = [f"CE-{(i%4)+1}" for i in range(n)]
    emails = [f"student{i}@example.com" for i in range(n)]

    attendance = np.clip(np.random.normal(80, 12, n).round(1), 30, 100)
    failed_attempts = np.random.choice([0,0,0,1,1,2], n, p=[0.5,0.2,0.1,0.1,0.06,0.04])

    base_date = datetime.now()
    last_payment_days_ago = np.random.choice(range(0,120), n)
    last_payment_date = [(base_date - timedelta(days=int(d))).date() for d in last_payment_days_ago]

    attendance_df = pd.DataFrame({
        "student_id": student_ids,
        "name": names,
        "course": courses,
        "mentor": mentors,
        "attendance_percent": attendance,
        "email": emails
    })

    scores_df = pd.DataFrame({
        "student_id": student_ids,
        "assessment_1": np.clip(np.random.normal(60, 22, n).round(1), 0, 100),
        "assessment_2": np.clip(np.random.normal(62, 20, n).round(1), 0, 100),
        "assessment_3": np.clip(np.random.normal(58, 25, n).round(1), 0, 100),
        "failed_attempts": failed_attempts
    })
    scores_df["avg_score"] = scores_df[["assessment_1","assessment_2","assessment_3"]].mean(axis=1).round(1)

    fees_df = pd.DataFrame({
        "student_id": student_ids,
        "total_fee_due": np.random.choice([0,5000,10000], n, p=[0.6,0.25,0.15]),
        "last_payment_date": last_payment_date
    })

    return attendance_df, scores_df, fees_df

# -----------------------
# Rule engine
# -----------------------
def evaluate_risk(df, thresholds):
    labels, flags, risk_scores = [], [], []
    score_map = {"green":0,"amber":1,"red":2}

    for _, r in df.iterrows():
        f = {}
        if r["attendance_percent"] < thresholds["attendance_red"]:
            f["attendance"] = "red"
        elif r["attendance_percent"] < thresholds["attendance_amber"]:
            f["attendance"] = "amber"
        else:
            f["attendance"] = "green"

        if r["avg_score"] < thresholds["score_red"]:
            f["score"] = "red"
        elif r["avg_score"] < thresholds["score_amber"]:
            f["score"] = "amber"
        else:
            f["score"] = "green"

        if r["failed_attempts"] >= thresholds["failed_attempts_red"]:
            f["attempts"] = "red"
        elif r["failed_attempts"] >= thresholds["failed_attempts_amber"]:
            f["attempts"] = "amber"
        else:
            f["attempts"] = "green"

        fod = r.get("fees_overdue_days", 999)
        if fod >= thresholds["fees_overdue_days_red"]:
            f["fees"] = "red"
        elif fod >= thresholds["fees_overdue_days_amber"]:
            f["fees"] = "amber"
        else:
            f["fees"] = "green"

        if "red" in f.values():
            label = "Red"
        elif "amber" in f.values():
            label = "Amber"
        else:
            label = "Green"

        rs = sum(score_map[v] for v in f.values())
        labels.append(label)
        flags.append(f)
        risk_scores.append(rs)

    df = df.copy()
    df["rule_label"] = labels
    df["rule_flags"] = flags
    df["rule_risk_score"] = risk_scores
    return df

# -----------------------
# Data loading / merging
# -----------------------
@st.cache_data
def load_and_merge(att_path=None, scores_path=None, fees_path=None):
    if att_path and scores_path and fees_path:
        try:
            att = pd.read_csv(att_path)
            sc = pd.read_csv(scores_path)
            fees = pd.read_csv(fees_path)
            if "last_payment_date" in fees.columns:
                fees["last_payment_date"] = pd.to_datetime(fees["last_payment_date"], errors="coerce")
        except Exception as e:
            st.warning("Could not read provided CSVs. Falling back to sample data. Error: " + str(e))
            att, sc, fees = generate_sample_data()
    else:
        att, sc, fees = generate_sample_data()

    df = att.merge(sc, on="student_id", how="outer")
    df = df.merge(fees, on="student_id", how="outer")

    today = pd.to_datetime(datetime.now().date())
    df["attendance_percent"] = pd.to_numeric(df.get("attendance_percent", 0), errors="coerce").fillna(0)
    df["avg_score"] = pd.to_numeric(df.get("avg_score", 0), errors="coerce").fillna(0)
    df["failed_attempts"] = pd.to_numeric(df.get("failed_attempts", 0), errors="coerce").fillna(0).astype(int)
    df["last_payment_date"] = pd.to_datetime(df.get("last_payment_date"), errors="coerce")
    df["fees_overdue_days"] = (today - df["last_payment_date"]).dt.days.fillna(999)

    cols = ["student_id","name","course","mentor","email","attendance_percent","avg_score","failed_attempts","fees_overdue_days"]
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    return df[cols]

# -----------------------
# Alerts processing
# -----------------------
def detect_streaks(series, threshold):
    streaks, count = [], 0
    for val in series:
        count = count + 1 if val else 0
        streaks.append(count)
    return max(streaks) >= threshold

def process_daily_activity(activity_df, master_df, thresholds):
    activity = activity_df.copy()
    activity.columns = [c.strip() for c in activity.columns]
    activity["date"] = pd.to_datetime(activity["date"], errors="coerce")
    for col in ["attended", "assignment_submitted"]:
        if col not in activity.columns:
            activity[col] = 1
    activity["attended"] = pd.to_numeric(activity["attended"], errors="coerce").fillna(1).astype(int)
    activity["assignment_submitted"] = pd.to_numeric(activity["assignment_submitted"], errors="coerce").fillna(1).astype(int)
    if "score" in activity.columns:
        activity["score"] = pd.to_numeric(activity["score"], errors="coerce")

    alerts = []
    grouped = activity.sort_values("date").groupby("student_id")
    for sid, g in grouped:
        g = g.sort_values("date")

        if detect_streaks(g["attended"].eq(0), thresholds["attendance_days"]):
            alerts.append((sid, "consecutive_absences", f"Absent streak >= {thresholds['attendance_days']}"))
        if detect_streaks(g["assignment_submitted"].eq(0), thresholds["assignment_days"]):
            alerts.append((sid, "consecutive_assignment_misses", f"Missed assignments streak >= {thresholds['assignment_days']}"))
        if "score" in g.columns:
            low_scores = g["score"] < thresholds["score_cutoff"]
            if detect_streaks(low_scores, thresholds["score_days"]):
                alerts.append((sid, "consecutive_low_scores", f"Low scores streak >= {thresholds['score_days']}"))

    alerts_df = pd.DataFrame(alerts, columns=["student_id","alert_type","details"])
    if alerts_df.empty:
        return alerts_df
    md = master_df[["student_id","name","mentor","email"]].drop_duplicates(subset=["student_id"])
    alerts_df = alerts_df.merge(md, on="student_id", how="left")
    return alerts_df[["student_id","name","mentor","email","alert_type","details"]]

# -----------------------
# Sidebar: file upload + thresholds
# -----------------------
st.sidebar.header("Data Input")
uploaded_att = st.sidebar.file_uploader("Upload attendance CSV", type=["csv"])
uploaded_scores = st.sidebar.file_uploader("Upload scores CSV", type=["csv"])
uploaded_fees = st.sidebar.file_uploader("Upload fees CSV", type=["csv"])
uploaded_activity = st.sidebar.file_uploader("Upload daily activity CSV", type=["csv"])

use_uploaded = uploaded_att and uploaded_scores and uploaded_fees
process_now = st.sidebar.button("Process Uploaded Files")
process_activity_now = st.sidebar.button("Process Daily Activity")

st.sidebar.header("Risk Thresholds")
attendance_red = st.sidebar.number_input("Attendance red (<)", 0, 100, 65)
attendance_amber = st.sidebar.number_input("Attendance amber (<)", 0, 100, 75)
score_red = st.sidebar.number_input("Score red (<)", 0, 100, 35)
score_amber = st.sidebar.number_input("Score amber (<)", 0, 100, 50)
failed_red = st.sidebar.number_input("Failed attempts red (>=)", 0, 10, 2)
failed_amber = st.sidebar.number_input("Failed attempts amber (>=)", 0, 10, 1)
fees_red = st.sidebar.number_input("Fees overdue red (>= days)", 0, 365, 30)
fees_amber = st.sidebar.number_input("Fees overdue amber (>= days)", 0, 365, 7)

thresholds = {
    "attendance_red": attendance_red,
    "attendance_amber": attendance_amber,
    "score_red": score_red,
    "score_amber": score_amber,
    "failed_attempts_red": failed_red,
    "failed_attempts_amber": failed_amber,
    "fees_overdue_days_red": fees_red,
    "fees_overdue_days_amber": fees_amber
}

st.sidebar.header("Alert Rule Configuration")
alert_thresholds = {
    "attendance_days": st.sidebar.number_input("Consecutive absences (>=)", 1, 30, 3),
    "score_days": st.sidebar.number_input("Consecutive low scores (>=)", 1, 30, 3),
    "assignment_days": st.sidebar.number_input("Consecutive assignment misses (>=)", 1, 30, 3),
    "score_cutoff": 40
}

# -----------------------
# Load and process data
# -----------------------
df = load_and_merge(uploaded_att, uploaded_scores, uploaded_fees) if (use_uploaded and process_now) else load_and_merge()
df = evaluate_risk(df, thresholds)

alerts_df = pd.DataFrame(columns=["student_id","name","mentor","email","alert_type","details"])
if uploaded_activity and process_activity_now:
    try:
        activity_df = pd.read_csv(uploaded_activity)
        alerts_df = process_daily_activity(activity_df, df, alert_thresholds)
        st.sidebar.success(f"{len(alerts_df)} alert(s) detected.") if not alerts_df.empty else st.sidebar.success("No alerts detected.")
    except Exception as e:
        st.sidebar.error("Error processing daily activity: " + str(e))

# -----------------------
# Dashboard display
# -----------------------
st.title("Dropout Risk Dashboard")
st.markdown("Monitor students and catch risks early.")

# metrics
total = len(df)
counts = df["rule_label"].value_counts().reindex(["Red","Amber","Green"]).fillna(0).astype(int)
c1, c2, c3, c4 = st.columns([1,1,1,2])
c1.metric("Total students", total)
c2.metric("Red", counts["Red"])
c3.metric("Amber", counts["Amber"])
c4.metric("Green", counts["Green"])

# -----------------------
# Alerts panel
# -----------------------
st.subheader("Alerts from daily activity")
if alerts_df.empty:
    st.info("No alerts. Upload daily activity and click Process Daily Activity.")
else:
    st.table(alerts_df)
    selected_ids = st.multiselect("Select students to notify", alerts_df["student_id"].tolist())
    smtp_server = st.text_input("SMTP server", "smtp.gmail.com")
    smtp_port = st.number_input("SMTP port", 1, 9999, 587)
    sender_email = st.text_input("Sender email")
    sender_pass = st.text_input("Sender password", type="password")
    if st.button("Send selected notifications"):
        import smtplib
        from email.mime.text import MIMEText
        sel_alerts = alerts_df[alerts_df["student_id"].isin(selected_ids)] if selected_ids else alerts_df
        for _, row in sel_alerts.iterrows():
            subject = f"Alert: {row['alert_type']} for {row['student_id']}"
            body = f"Student: {row['name']} ({row['student_id']})\nMentor: {row['mentor']}\nAlert: {row['alert_type']}\nDetails: {row['details']}"
            try:
                msg = MIMEText(body)
                msg["Subject"] = subject
                msg["From"] = sender_email
                msg["To"] = row["email"]
                with smtplib.SMTP(smtp_server, smtp_port) as server:
                    server.starttls()
                    server.login(sender_email, sender_pass)
                    server.sendmail(sender_email, [row["email"]], msg.as_string())
                st.success(f"Email sent to {row['email']} for {row['student_id']}")
            except Exception as e:
                st.error(f"Failed for {row['student_id']}: {e}")

# -----------------------
# Students table + details + charts + actions (same as old code)
# -----------------------
# ... keep the same students table, student details, charts, flag counts, and actions code from your old script here ...
