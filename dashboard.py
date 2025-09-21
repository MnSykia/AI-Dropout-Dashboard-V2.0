# dashboard.py
# Streamlit dashboard for Drop-out Prediction and Counseling
# Run: streamlit run dashboard.py

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import io

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

# Generate sample daily activity button
if st.sidebar.button("Generate Sample Activity", key="gen_sample_activity"):
    # Load main data to get student IDs
    df_main = load_and_merge(uploaded_att, uploaded_scores, uploaded_fees) if (use_uploaded and process_now) else load_and_merge()
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

st.sidebar.header("🚨 Alert Rule Configuration")
alert_thresholds = {
    "attendance_days": st.sidebar.number_input("Consecutive absences (>=)", 1, 30, 3, key="alert_att"),
    "score_days": st.sidebar.number_input("Consecutive low scores (>=)", 1, 30, 3, key="alert_score"),
    "assignment_days": st.sidebar.number_input("Consecutive assignment misses (>=)", 1, 30, 3, key="alert_assign"),
    "score_cutoff": st.sidebar.number_input("Low score threshold (<)", 0, 100, 40, key="score_cutoff")
}

# -----------------------
# Sidebar: file upload + thresholds
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
# Thresholds
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
            
            # Process for alerts
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
# Filters
# -----------------------
st.markdown("---")
with st.expander("🔍 Filters"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        mentors = sorted(df["mentor"].dropna().unique().tolist())
        mentors_sel = st.multiselect("Mentor", ["All"] + mentors, default=["All"])
    
    with col2:
        courses = sorted(df["course"].dropna().unique().tolist())
        course_sel = st.multiselect("Course", ["All"] + courses, default=["All"])
    
    with col3:
        label_sel = st.multiselect("Risk Label", ["Red","Amber","Green"], default=["Red","Amber","Green"])

# Apply filters
df_view = df.copy()
if "All" not in mentors_sel:
    df_view = df_view[df_view["mentor"].isin(mentors_sel)]
if "All" not in course_sel:
    df_view = df_view[df_view["course"].isin(course_sel)]
df_view = df_view[df_view["rule_label"].isin(label_sel)]

# Metrics
total = len(df)
counts = df["rule_label"].value_counts().reindex(["Red","Amber","Green"]).fillna(0).astype(int)
c1, c2, c3, c4 = st.columns([1,1,1,2])
c1.metric("Total Students", total, help="Total number of students in the system")
c2.metric("🔴 Red", counts["Red"], help="High risk students requiring immediate attention")
c3.metric("🟡 Amber", counts["Amber"], help="Medium risk students to monitor closely")
c4.metric("🟢 Green", counts["Green"], help="Low risk students")

# -----------------------
# Students table
# -----------------------
st.subheader("📊 Students Table")
st.markdown("Click on column headers to sort. Risk labels are color-coded for easy identification.")

# Display with better formatting
display_cols = ["student_id","name","mentor","course","attendance_percent","avg_score","failed_attempts","fees_overdue_days","rule_label"]

def highlight_risk(row):
    if row["rule_label"] == "Red":
        return ["background-color: #ffcccb"] * len(row)
    elif row["rule_label"] == "Amber":
        return ["background-color: #ffe4b5"] * len(row)
    else:
        return ["background-color: #90ee90"] * len(row)

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
# Charts
# -----------------------
import matplotlib.pyplot as plt
import seaborn as sns

st.subheader("📈 Distributions and Insights")

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

col1, col2, col3 = st.columns(3)

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

with col3:
    # Generate new sample data
    if st.button("🔄 Generate New Sample Data"):
        # Clear session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.success("✅ Generated new sample data. Page will refresh.")
        st.rerun()

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
    Use filters to focus on specific groups</p>
    <p>🎓 Dropout Risk Detection System v2.0</p>
    </div>
    """,
    unsafe_allow_html=True
)
