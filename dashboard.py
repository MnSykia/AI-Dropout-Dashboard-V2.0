# dashboard.py
# Streamlit dashboard for Drop-out Prediction and Counseling with Attendance Heatmap
# Run: streamlit run dashboard.py

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import io
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors

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
# Generate sample daily attendance data (for heatmap)
# -----------------------
def generate_sample_daily_attendance(student_ids, n_days=30, seed=42):
    """Generate daily attendance data for heatmap visualization"""
    if seed:
        np.random.seed(seed)
    
    base_date = datetime.now().date()
    dates = [(base_date - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(n_days-1, -1, -1)]
    
    # Create attendance matrix
    attendance_data = {
        "ID": student_ids,
        "Name": [f"Student_{sid.replace('S', '')}" for sid in student_ids]
    }
    
    for date in dates:
        # Generate attendance with some patterns (weekends slightly lower attendance)
        date_obj = datetime.strptime(date, '%Y-%m-%d')
        if date_obj.weekday() >= 5:  # Weekend
            prob_present = 0.75
        else:  # Weekday
            prob_present = 0.85
        
        prob_absent = 1.0 - prob_present
        attendance_data[date] = np.random.choice(['P', 'A'], len(student_ids), 
                                                p=[prob_present, prob_absent])
    
    return pd.DataFrame(attendance_data)

# -----------------------
# Generate sample daily activity data
# -----------------------
def generate_sample_daily_activity(student_ids, date, seed=None):
    if seed:
        np.random.seed(seed)
    n = len(student_ids)
    return pd.DataFrame({
        "student_id": student_ids,
        "date": date,
        "attended": np.random.choice([0,1], n, p=[0.15, 0.85]),
        "assignment_submitted": np.random.choice([0,1], n, p=[0.2, 0.8]),
        "score": np.clip(np.random.normal(65, 20, n).round(1), 0, 100)
    })

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

        fod = r["fees_overdue_days"] if "fees_overdue_days" in r else 999
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
    return max(streaks) >= threshold if streaks else False

def process_daily_activity(activity_df, master_df, thresholds):
    activity = activity_df.copy()
    activity.columns = [c.strip() for c in activity.columns]
    activity["date"] = pd.to_datetime(activity["date"], errors="coerce")
    
    # Ensure required columns exist
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

        # Check attendance streaks
        if detect_streaks(g["attended"].eq(0), thresholds["attendance_days"]):
            alerts.append((sid, "consecutive_absences", f"Absent streak >= {thresholds['attendance_days']} days"))
        
        # Check assignment submission streaks
        if detect_streaks(g["assignment_submitted"].eq(0), thresholds["assignment_days"]):
            alerts.append((sid, "consecutive_assignment_misses", f"Missed assignments streak >= {thresholds['assignment_days']} days"))
        
        # Check score streaks if score column exists
        if "score" in g.columns and not g["score"].isna().all():
            low_scores = g["score"] < thresholds["score_cutoff"]
            if detect_streaks(low_scores, thresholds["score_days"]):
                alerts.append((sid, "consecutive_low_scores", f"Low scores streak >= {thresholds['score_days']} days"))

    alerts_df = pd.DataFrame(alerts, columns=["student_id","alert_type","details"])
    
    if alerts_df.empty:
        return pd.DataFrame(columns=["student_id","name","mentor","email","alert_type","details"])
    
    # Merge with master data to get student details
    md = master_df[["student_id","name","mentor","email"]].drop_duplicates(subset=["student_id"])
    alerts_df = alerts_df.merge(md, on="student_id", how="left")
    return alerts_df[["student_id","name","mentor","email","alert_type","details"]]

# Initialize session state for storing activity data
if "activity_data_storage" not in st.session_state:
    st.session_state["activity_data_storage"] = {}

if "alerts_df" not in st.session_state:
    st.session_state["alerts_df"] = pd.DataFrame(columns=["student_id","name","mentor","email","alert_type","details"])

# -----------------------
# Sidebar: Daily activity upload section
# -----------------------
st.sidebar.header("📅 Daily Activity Upload")

# Calendar date picker
activity_date = st.sidebar.date_input("Select activity date", datetime.now().date(), key="activity_date_picker")

# File uploader for that date's activity CSV
daily_activity_file = st.sidebar.file_uploader(
    f"Upload activity CSV for {activity_date}",
    type=["csv"],
    key=f"activity_upload_{activity_date}"
)

# Buttons for daily activity management
col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("Add Activity", key="add_activity_btn"):
        if daily_activity_file is not None:
            try:
                # Read the uploaded file
                activity_df = pd.read_csv(daily_activity_file)
                
                # Store in session state with date as key
                st.session_state["activity_data_storage"][str(activity_date)] = {
                    "data": activity_df.to_csv(index=False),
                    "filename": daily_activity_file.name,
                    "rows": len(activity_df)
                }
                st.sidebar.success(f"✅ Added activity for {activity_date} ({len(activity_df)} records)")
            except Exception as e:
                st.sidebar.error(f"❌ Error reading file: {e}")
        else:
            st.sidebar.warning("⚠️ Please upload a file first")

with col2:
    if st.button("Clear All", key="clear_all_btn"):
        st.session_state["activity_data_storage"] = {}
        st.session_state["alerts_df"] = pd.DataFrame(columns=["student_id","name","mentor","email","alert_type","details"])
        st.sidebar.success("🗑️ Cleared all stored activity data")

# Display stored files
if st.session_state["activity_data_storage"]:
    st.sidebar.markdown("**📋 Stored Activity Files:**")
    for date_str, file_info in sorted(st.session_state["activity_data_storage"].items()):
        st.sidebar.text(f"• {date_str}: {file_info['rows']} rows")

if st.sidebar.button("Generate Sample Activity", key="gen_sample_activity_main"):
    # Load main data to get student IDs - use temporary variables or fallback to sample
    df_main = load_and_merge()  # Always use sample data for generation
    sample_activity = generate_sample_daily_activity(df_main["student_id"].tolist(), activity_date)
    
    # Store the generated sample
    st.session_state["activity_data_storage"][str(activity_date)] = {
        "data": sample_activity.to_csv(index=False),
        "filename": f"sample_activity_{activity_date}.csv",
        "rows": len(sample_activity)
    }
    st.sidebar.success(f"✅ Generated sample activity for {activity_date}")

# Process all stored activity files
process_activity_now = st.sidebar.button("🔄 Process All Activity Files", key="process_activity_btn", type="primary")

st.sidebar.divider()

# Process all stored activity files
st.sidebar.header("🚨 Alert Rule Configuration")
alert_thresholds = {
    "attendance_days": st.sidebar.number_input("Consecutive absences (>=)", 1, 30, 3, key="alert_att"),
    "score_days": st.sidebar.number_input("Consecutive low scores (>=)", 1, 30, 3, key="alert_score"),
    "assignment_days": st.sidebar.number_input("Consecutive assignment misses (>=)", 1, 30, 3, key="alert_assign"),
    "score_cutoff": st.sidebar.number_input("Low score threshold (<)", 0, 100, 40, key="score_cutoff")
}

st.sidebar.divider()
st.sidebar.divider()

# -----------------------
# Sidebar: Data Input
# -----------------------
st.sidebar.header("📁 Data Input")

# Main data files
uploaded_att = st.sidebar.file_uploader("Upload attendance CSV", type=["csv"], key="attendance_upload")
uploaded_scores = st.sidebar.file_uploader("Upload scores CSV", type=["csv"], key="scores_upload")
uploaded_fees = st.sidebar.file_uploader("Upload fees CSV", type=["csv"], key="fees_upload")

use_uploaded = uploaded_att and uploaded_scores and uploaded_fees
process_now = st.sidebar.button("Process Uploaded Files", key="process_main_files")

st.sidebar.divider()

# -----------------------
# Sidebar: Attendance Heatmap Data Upload
# -----------------------
st.sidebar.header("📊 Attendance Heatmap Data")

# File uploader for daily attendance CSV
attendance_heatmap_file = st.sidebar.file_uploader(
    "Upload daily attendance CSV for heatmap",
    type=["csv"],
    key="attendance_heatmap_upload",
    help="CSV should have ID, Name columns followed by date columns with P/A values"
)

col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("Load Heatmap Data", key="load_heatmap_btn"):
        if attendance_heatmap_file is not None:
            try:
                heatmap_df = pd.read_csv(attendance_heatmap_file)
                st.session_state["attendance_heatmap_data"] = heatmap_df
                st.sidebar.success(f"✅ Loaded heatmap data ({len(heatmap_df)} students)")
            except Exception as e:
                st.sidebar.error(f"❌ Error reading heatmap file: {e}")
        else:
            st.sidebar.warning("⚠️ Please upload a file first")

with col2:
    if st.button("Generate Sample Heatmap", key="gen_sample_heatmap"):
        df_main = load_and_merge()
        sample_heatmap = generate_sample_daily_attendance(df_main["student_id"].tolist())
        st.session_state["attendance_heatmap_data"] = sample_heatmap
        st.sidebar.success("✅ Generated sample heatmap data")

st.sidebar.divider()

# -----------------------
# Risk Thresholds
# -----------------------
st.sidebar.header("⚙️ Risk Thresholds")
attendance_red = st.sidebar.number_input("Attendance red (<)", 0, 100, 65, key="att_red")
attendance_amber = st.sidebar.number_input("Attendance amber (<)", 0, 100, 75, key="att_amber")
score_red = st.sidebar.number_input("Score red (<)", 0, 100, 35, key="score_red")
score_amber = st.sidebar.number_input("Score amber (<)", 0, 100, 50, key="score_amber")
failed_red = st.sidebar.number_input("Failed attempts red (>=)", 0, 10, 2, key="failed_red")
failed_amber = st.sidebar.number_input("Failed attempts amber (>=)", 0, 10, 1, key="failed_amber")
fees_red = st.sidebar.number_input("Fees overdue red (>= days)", 0, 365, 30, key="fees_red")
fees_amber = st.sidebar.number_input("Fees overdue amber (>= days)", 0, 365, 7, key="fees_amber")

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

# -----------------------
# Load and process main data
# -----------------------
if st.session_state.get("force_new_data"):
    df = load_and_merge() 
    st.session_state["force_new_data"] = False
else:
    df = load_and_merge(uploaded_att, uploaded_scores, uploaded_fees) if (use_uploaded and process_now) else load_and_merge()

df = evaluate_risk(df, thresholds)

# -----------------------
# Process daily activity files when button is clicked
# -----------------------
if process_activity_now:
    if st.session_state["activity_data_storage"]:
        combined_activity = []
        
        for date_str, file_info in st.session_state["activity_data_storage"].items():
            try:
                # Read the stored CSV data
                act_df = pd.read_csv(io.StringIO(file_info["data"]))
                
                # Ensure date column exists and is properly set
                if "date" not in act_df.columns:
                    act_df["date"] = date_str
                else:
                    act_df["date"] = pd.to_datetime(act_df["date"], errors="coerce").fillna(pd.to_datetime(date_str))
                
                combined_activity.append(act_df)
            except Exception as e:
                st.sidebar.error(f"Error processing {date_str}: {e}")
        
        if combined_activity:
            # Combine all activity data
            all_activity_df = pd.concat(combined_activity, ignore_index=True)
            
            # Process for alerts - use the already loaded df
            alerts_df = process_daily_activity(all_activity_df, df, alert_thresholds)
            st.session_state["alerts_df"] = alerts_df
            
            if not alerts_df.empty:
                st.sidebar.success(f"🎯 {len(alerts_df)} alert(s) detected from activity data")
            else:
                st.sidebar.info("ℹ️ No alerts detected from activity data")
    else:
        st.sidebar.warning("⚠️ No activity files stored. Please add activity files first.")

# Load alerts from session state
alerts_df = st.session_state["alerts_df"]

# -----------------------
# Dashboard display
# -----------------------
st.title("🎓 Dropout Risk Dashboard")
st.markdown("Monitor students and catch risks early to prevent dropouts.")

# -----------------------
# Alerts panel
# -----------------------
st.subheader("🚨 Alerts from Daily Activity")

if alerts_df.empty:
    st.info("ℹ️ No alerts detected. Upload daily activity files and click 'Process All Activity Files' to generate alerts.")
else:
    # Display alerts with better formatting
    st.dataframe(
        alerts_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "student_id": st.column_config.TextColumn("Student ID", width="small"),
            "name": st.column_config.TextColumn("Name", width="medium"),
            "mentor": st.column_config.TextColumn("Mentor", width="small"),
            "email": st.column_config.TextColumn("Email", width="medium"),
            "alert_type": st.column_config.TextColumn("Alert Type", width="medium"),
            "details": st.column_config.TextColumn("Details", width="large")
        }
    )
    
    # Email notification section
    with st.expander("📧 Send Email Notifications"):
        selected_ids = st.multiselect(
            "Select students to notify",
            alerts_df["student_id"].tolist(),
            help="Select one or more students to send email notifications"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            smtp_server = st.text_input("SMTP server", "smtp.gmail.com")
            sender_email = st.text_input("Sender email")
        with col2:
            smtp_port = st.number_input("SMTP port", 1, 9999, 587)
            sender_pass = st.text_input("Sender password", type="password")
        
        if st.button("📤 Send Selected Notifications", type="primary"):
            if not sender_email or not sender_pass:
                st.error("Please provide sender email and password")
            else:
                import smtplib
                from email.mime.text import MIMEText
                
                sel_alerts = alerts_df[alerts_df["student_id"].isin(selected_ids)] if selected_ids else alerts_df
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for idx, (_, row) in enumerate(sel_alerts.iterrows()):
                    subject = f"Alert: {row['alert_type']} for {row['student_id']}"
                    body = f"""
Dear {row['name']},

This is an automated alert from the Student Monitoring System.

Student ID: {row['student_id']}
Mentor: {row['mentor']}
Alert Type: {row['alert_type']}
Details: {row['details']}

Please take necessary action to address this issue.

Best regards,
Student Monitoring System
                    """
                    
                    try:
                        msg = MIMEText(body)
                        msg["Subject"] = subject
                        msg["From"] = sender_email
                        msg["To"] = row["email"]
                        
                        with smtplib.SMTP(smtp_server, smtp_port) as server:
                            server.starttls()
                            server.login(sender_email, sender_pass)
                            server.sendmail(sender_email, [row["email"]], msg.as_string())
                        
                        status_text.text(f"✅ Email sent to {row['email']} for {row['student_id']}")
                    except Exception as e:
                        st.error(f"❌ Failed for {row['student_id']}: {e}")
                    
                    progress_bar.progress((idx + 1) / len(sel_alerts))
                
                status_text.text("✅ All emails processed!")

# -----------------------
# Students table
# -----------------------
st.markdown("---")

st.subheader("📊 Students Table")
st.markdown("Click on column headers to sort. Risk labels are color-coded for easy identification.")

# Metrics
total = len(df)
counts = df["rule_label"].value_counts().reindex(["Red","Amber","Green"]).fillna(0).astype(int)
c1, c2, c3, c4 = st.columns([1,1,1,2])
c1.metric("Total Students", total, help="Total number of students in the system")
c2.metric("🔴 Red", counts["Red"], help="High risk students requiring immediate attention")
c3.metric("🟡 Amber", counts["Amber"], help="Medium risk students to monitor closely")
c4.metric("🟢 Green", counts["Green"], help="Low risk students")

# -----------------------
# Filters
# -----------------------
with st.expander("🔍 Filters"):
    col1, col2, col3, col4 = st.columns(4)

    # Risk label filter
    with col1:
        label_sel = st.multiselect(
            "Risk Label",
            options=["Red", "Amber", "Green"],
            default=[]
        )

    # Mentor filter
    with col2:
        mentors_sel = st.multiselect(
            "Mentor",
            options=["All"] + sorted(df["mentor"].dropna().unique().tolist()),
            default=["All"]
        )

    # Course filter
    with col3:
        course_sel = st.multiselect(
            "Course",
            options=["All"] + sorted(df["course"].dropna().unique().tolist()),
            default=["All"]
        )

    # Fee status filter (derived from fees_overdue_days)
    with col4:
        fee_status = pd.cut(
            df["fees_overdue_days"],
            bins=[-1, 0, 7, 30, float("inf")],
            labels=["Paid", "Due <7d", "Due <30d", "Overdue"]
        )
        df["fees_status"] = fee_status.astype(str)

        fee_sel = st.multiselect(
            "Fee Status",
            options=["All"] + sorted(df["fees_status"].unique().tolist()),
            default=["All"]
        )

# -----------------------
# Apply filters
# -----------------------
df_view = df.copy()

if label_sel:
    df_view = df_view[df_view["rule_label"].isin(label_sel)]

if "All" not in mentors_sel:
    df_view = df_view[df_view["mentor"].isin(mentors_sel)]

if "All" not in course_sel:
    df_view = df_view[df_view["course"].isin(course_sel)]

if "All" not in fee_sel:
    df_view = df_view[df_view["fees_status"].isin(fee_sel)]

# Display with better formatting
display_cols = ["student_id","name","mentor","course","attendance_percent","avg_score","failed_attempts","fees_overdue_days","rule_label"]

def highlight_risk(row):
    if row["rule_label"] == "Red":
        return ["background-color: #ffcccb; color: black;"] * len(row)
    elif row["rule_label"] == "Amber":
        return ["background-color: #ffe4b5; color: black;"] * len(row)
    else:
        return ["background-color: #90ee90; color: black;"] * len(row)

styled_df = df_view[display_cols].style.apply(highlight_risk, axis=1)
st.dataframe(
    styled_df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "student_id": st.column_config.TextColumn("Student ID"),
        "name": st.column_config.TextColumn("Name"),
        "mentor": st.column_config.TextColumn("Mentor"),
        "course": st.column_config.TextColumn("Course"),
        "attendance_percent": st.column_config.NumberColumn("Attendance %", format="%.1f%%"),
        "avg_score": st.column_config.NumberColumn("Avg Score", format="%.1f"),
        "failed_attempts": st.column_config.NumberColumn("Failed Attempts"),
        "fees_overdue_days": st.column_config.NumberColumn("Fees Overdue (days)"),
        "rule_label": st.column_config.TextColumn("Risk Label")
    }
)

# -----------------------
# Student details
# -----------------------
st.subheader("🔎 Student Details")
selected = st.text_input("Enter student ID to see detailed information (e.g., S1000)", value="")

if selected:
    sel = df[df["student_id"] == selected]
    if sel.empty:
        st.warning("⚠️ Student ID not found in current dataset.")
    else:
        r = sel.iloc[0]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Basic Info**")
            st.write(f"ID: {r['student_id']}")
            st.write(f"Name: {r['name']}")
            st.write(f"Course: {r['course']}")
            st.write(f"Mentor: {r['mentor']}")
        
        with col2:
            st.markdown("**Academic Performance**")
            st.write(f"Attendance: {r['attendance_percent']:.1f}%")
            st.write(f"Avg Score: {r['avg_score']:.1f}")
            st.write(f"Failed Attempts: {r['failed_attempts']}")
        
        with col3:
            st.markdown("**Risk Assessment**")
            risk_color = {"Red": "🔴", "Amber": "🟡", "Green": "🟢"}
            st.write(f"Risk Label: {risk_color.get(r['rule_label'], '')} {r['rule_label']}")
            st.write(f"Fees Overdue: {r['fees_overdue_days']:.0f} days")
        
        st.markdown("**Risk Flags:**")
        flags_df = pd.DataFrame([
            {"Category": k, "Status": v} for k, v in r["rule_flags"].items()
        ])
        st.dataframe(flags_df, hide_index=True, use_container_width=False)

# -----------------------
# Charts and Heatmap
# -----------------------
st.subheader("📈 Distributions and Insights")

# First row: Attendance vs Risk and Scores vs Risk
col_a, col_b = st.columns(2)

with col_a:
    st.caption("Attendance vs Risk")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(
        x="rule_label",
        y="attendance_percent",
        data=df,
        ax=ax,
        palette={"Red":"#ff4b4b","Amber":"#ffb84d","Green":"#66c266"},
        order=["Red", "Amber", "Green"]
    )
    ax.set_xlabel("Risk Label")
    ax.set_ylabel("Attendance %")
    ax.set_title("Attendance Distribution by Risk Level")
    st.pyplot(fig)

with col_b:
    st.caption("Scores vs Risk")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(
        x="rule_label",
        y="avg_score",
        data=df,
        ax=ax,
        palette={"Red":"#ff4b4b","Amber":"#ffb84d","Green":"#66c266"},
        order=["Red", "Amber", "Green"]
    )
    ax.set_xlabel("Risk Label")
    ax.set_ylabel("Average Score")
    ax.set_title("Score Distribution by Risk Level")
    st.pyplot(fig)

# Mentor risk distribution
st.caption("Risk Distribution Across Mentors")
mentor_risk_counts = df.groupby(["mentor","rule_label"]).size().unstack(fill_value=0)
fig, ax = plt.subplots(figsize=(10, 5))
mentor_risk_counts.plot(
    kind="bar",
    stacked=True,
    color={"Red":"#ff4b4b","Amber":"#ffb84d","Green":"#66c266"},
    ax=ax
)
ax.set_ylabel("Number of Students")
ax.set_xlabel("Mentor")
ax.set_title("Mentor-wise Risk Level Distribution")
ax.legend(title="Risk Level", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig)

# -----------------------
# Attendance Heatmap Section
# -----------------------
st.markdown("---")
st.subheader("🔥 Interactive Attendance Heatmap")

# Ensure session state variables are initialized
if "attendance_heatmap_data" not in st.session_state:
    st.session_state["attendance_heatmap_data"] = None
if "heatmap_student_offset" not in st.session_state:
    st.session_state["heatmap_student_offset"] = 0
if "heatmap_date_offset" not in st.session_state:
    st.session_state["heatmap_date_offset"] = 0

if st.session_state.get("attendance_heatmap_data") is not None:
    heatmap_df = st.session_state["attendance_heatmap_data"]
    
    # Extract date columns (assuming first 2 columns are ID and Name)
    if len(heatmap_df.columns) > 2:
        attendance_cols = heatmap_df.columns[2:]
        
        # Parameters for windowing
        window_students = 10
        window_dates = 7
        
        # Get min/max ID values for navigation
        if "ID" in heatmap_df.columns:
            id_col = "ID"
            min_id_idx = 0
            max_id_idx = len(heatmap_df) - 1
        else:
            id_col = heatmap_df.columns[0]  # First column as ID
            min_id_idx = 0
            max_id_idx = len(heatmap_df) - 1
        
        total_dates = len(attendance_cols)
        
        # Navigation functions
        def move_students_up():
            if "heatmap_student_offset" not in st.session_state:
                st.session_state["heatmap_student_offset"] = 0
            st.session_state["heatmap_student_offset"] = max(0, st.session_state["heatmap_student_offset"] - 1)
        
        def move_students_down():
            if "heatmap_student_offset" not in st.session_state:
                st.session_state["heatmap_student_offset"] = 0
            st.session_state["heatmap_student_offset"] = min(max_id_idx - window_students + 1, st.session_state["heatmap_student_offset"] + 1)
        
        def move_dates_left():
            if "heatmap_date_offset" not in st.session_state:
                st.session_state["heatmap_date_offset"] = 0
            st.session_state["heatmap_date_offset"] = max(0, st.session_state["heatmap_date_offset"] - 1)
        
        def move_dates_right():
            if "heatmap_date_offset" not in st.session_state:
                st.session_state["heatmap_date_offset"] = 0
            st.session_state["heatmap_date_offset"] = min(total_dates - window_dates, st.session_state["heatmap_date_offset"] + 1)
        
        # Navigation buttons
        col1, col2, col3 = st.columns([2, 1, 2])
        
        with col1:
            st.markdown("**Student Navigation:**")
            subcol1, subcol2 = st.columns(2)
            with subcol1:
                if st.button("⬆️ Previous Students", key="heatmap_students_up"):
                    move_students_up()
            with subcol2:
                if st.button("⬇️ Next Students", key="heatmap_students_down"):
                    move_students_down()
        
        with col3:
            st.markdown("**Date Navigation:**")
            subcol1, subcol2 = st.columns(2)
            with subcol1:
                if st.button("⬅️ Previous Dates", key="heatmap_dates_left"):
                    move_dates_left()
            with subcol2:
                if st.button("➡️ Next Dates", key="heatmap_dates_right"):
                    move_dates_right()
        
        # Calculate current window
        start_student_idx = st.session_state.get("heatmap_student_offset", 0)
        end_student_idx = min(start_student_idx + window_students - 1, max_id_idx)
        
        start_date_idx = st.session_state.get("heatmap_date_offset", 0)
        end_date_idx = min(start_date_idx + window_dates - 1, total_dates - 1)
        
        # Get subset of data
        selected_dates = attendance_cols[start_date_idx:end_date_idx + 1]
        df_subset = heatmap_df.iloc[start_student_idx:end_student_idx + 1]
        
        # Convert attendance to numeric: P=1, A=0
        attendance_matrix = df_subset[selected_dates].replace({"P": 1, "A": 0}).fillna(1)
        
        # Apply last 3-day absence shading
        shade_matrix = attendance_matrix.copy()
        for idx in range(len(attendance_matrix)):
            row = attendance_matrix.iloc[idx]
            for col_idx, col in enumerate(attendance_matrix.columns):
                # last 3 days including current
                last3_idx = max(0, col_idx - 2)
                last3_values = row.iloc[last3_idx:col_idx+1]
                absences = (last3_values == 0).sum()
                
                if row.iloc[col_idx] == 1:
                    shade_matrix.iloc[idx, col_idx] = 0  # Present = green
                else:
                    if absences == 1:
                        shade_matrix.iloc[idx, col_idx] = 1  # light red
                    elif absences == 2:
                        shade_matrix.iloc[idx, col_idx] = 2  # medium red
                    else:
                        shade_matrix.iloc[idx, col_idx] = 3  # dark red
        
        # Use names if available
        if "Name" in df_subset.columns:
            shade_matrix.index = df_subset["Name"]
        else:
            shade_matrix.index = df_subset[id_col]
        
        # Colormap: Green + 3 reds
        cmap = mcolors.ListedColormap(["green", "#ff9999", "#ff4d4d", "#cc0000"])
        bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
        norm = mcolors.BoundaryNorm(bounds, cmap.N)
        
        # Current window info
        start_student_name = df_subset.iloc[0]["Name"] if "Name" in df_subset.columns else df_subset.iloc[0][id_col]
        end_student_name = df_subset.iloc[-1]["Name"] if "Name" in df_subset.columns else df_subset.iloc[-1][id_col]
        
        st.info(f"📊 Showing students {start_student_name} to {end_student_name}, dates {selected_dates[0]} to {selected_dates[-1]}")
        
        # Plot heatmap
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.heatmap(
            shade_matrix[selected_dates],
            cmap=cmap,
            norm=norm,
            cbar=True,
            ax=ax,
            linewidths=0.3,
            linecolor="black",
            square=True,
            annot=False,
            cbar_kws={'label': 'Attendance Status'}
        )
        ax.set_ylabel("Students", fontsize=10)
        ax.set_xlabel("Date", fontsize=10)
        ax.set_title("Daily Attendance Heatmap\n(Green=Present, Red shades=Absences in last 3 days)", fontsize=12)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=9)
        
        # Add custom legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', label='Present'),
            Patch(facecolor='#ff9999', label='1 absence (last 3 days)'),
            Patch(facecolor='#ff4d4d', label='2 absences (last 3 days)'),
            Patch(facecolor='#cc0000', label='3+ absences (last 3 days)')
        ]
        ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Heatmap statistics
        total_students_shown = len(df_subset)
        total_days_shown = len(selected_dates)
        total_present = (attendance_matrix == 1).sum().sum()
        total_absent = (attendance_matrix == 0).sum().sum()
        attendance_rate = (total_present / (total_present + total_absent)) * 100 if (total_present + total_absent) > 0 else 0
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Students Shown", total_students_shown)
        col2.metric("Days Shown", total_days_shown)
        col3.metric("Overall Attendance", f"{attendance_rate:.1f}%")
        col4.metric("Total Records", total_students_shown * total_days_shown)
        
    else:
        st.warning("⚠️ Loaded heatmap data doesn't have enough columns. Expected: ID, Name, followed by date columns.")
else:
    st.info("ℹ️ No attendance heatmap data loaded. Please upload a daily attendance CSV or generate sample data from the sidebar.")
st.markdown("---")

# Flag summary
flag_rows = []
for _, row in df.iterrows():
    f = row["rule_flags"]
    for k, v in f.items():
        flag_rows.append({"student_id": row["student_id"], "reason": k, "level": v})

if flag_rows:
    flag_df = pd.DataFrame(flag_rows)
    pivot = flag_df.groupby(["reason","level"]).size().unstack(fill_value=0)
    
    st.subheader("📋 Flag Summary")
    st.dataframe(pivot, use_container_width=True)

# -----------------------
# Actions
# -----------------------
st.subheader("⚡ Actions")

col1, col2 = st.columns(2)

with col1:
    # Download red list
    red_df = df[df["rule_label"] == "Red"]
    if not red_df.empty:
        csv = red_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download Red List (CSV)",
            csv,
            file_name=f"red_list_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            help="Download list of all high-risk students"
        )
    else:
        st.info("No red-flagged students")

with col2:
    # Download all alerts
    if not alerts_df.empty:
        alerts_csv = alerts_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download Alerts (CSV)",
            alerts_csv,
            file_name=f"alerts_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            help="Download all current alerts"
        )
    else:
        st.info("No alerts to download")


# -----------------------
# Summary Statistics
# -----------------------
with st.expander("📊 Summary Statistics"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Overall Statistics**")
        st.write(f"• Average Attendance: {df['attendance_percent'].mean():.1f}%")
        st.write(f"• Average Score: {df['avg_score'].mean():.1f}")
        st.write(f"• Students with Failed Attempts: {(df['failed_attempts'] > 0).sum()}")
        st.write(f"• Students with Fee Overdue: {(df['fees_overdue_days'] < 999).sum()}")
    
    with col2:
        st.markdown("**Risk Distribution**")
        risk_pct = (counts / total * 100).round(1)
        st.write(f"• High Risk (Red): {counts['Red']} ({risk_pct['Red']}%)")
        st.write(f"• Medium Risk (Amber): {counts['Amber']} ({risk_pct['Amber']}%)")
        st.write(f"• Low Risk (Green): {counts['Green']} ({risk_pct['Green']}%)")

# -----------------------
# Footer
# -----------------------
st.divider()
st.markdown(
    """
    <div style='text-align: center; color: gray; padding: 20px;'>
    <p>💡 <b>Tips:</b> Adjust thresholds in the sidebar to fine-tune risk detection | 
    Upload daily activity CSVs for multiple dates to track patterns | 
    Use filters to focus on specific groups | Load attendance heatmap data to visualize daily patterns</p>
    <p>🎓 Dropout Risk Detection System v2.1 with Interactive Attendance Heatmap</p>
    </div>
    """,
    unsafe_allow_html=True
)
