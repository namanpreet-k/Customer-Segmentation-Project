import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Page configuration
st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="🛍️",
    layout="wide"
)

# Custom dashboard styling
st.markdown("""
<style>
    .stApp {
        background-color: #F8F9FC;
    }

    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    h1 {
        color: #243B53;
        font-size: 2.4rem !important;
        font-weight: 700 !important;
    }

    h2, h3 {
        color: #334E68;
    }

    p {
        color: #486581;
        line-height: 1.7;
    }

    div[data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #E4E7EC;
        padding: 20px;
        border-radius: 14px;
        box-shadow: 0 3px 12px rgba(36, 59, 83, 0.05);
    }

    div[data-testid="stMetricLabel"] {
        color: #627D98;
    }

    div[data-testid="stMetricValue"] {
        color: #243B53;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid #E4E7EC;
        border-radius: 12px;
        overflow: hidden;
    }
div.stButton > button {
    background-color: #245B52 !important;
    color: #FFFFFF !important;
    border-radius: 9px;
    border: none;
    padding: 0.6rem 1.2rem;
    font-weight: 700;
}

div.stButton > button p {
    color: #FFFFFF !important;
    font-weight: 700;
}

div.stButton > button:hover {
    background-color: #17463F !important;
    color: #FFFFFF !important;
    border: none;
}

div.stButton > button:hover p {
    color: #FFFFFF !important;
}

    border-radius: 9px;
    border: none;
    padding: 0.6rem 1.2rem;
    font-weight: 600;
}

div.stButton > button:hover {
    background-color: #17463F;
    color: #FFFFFF;
    border: none;
}
```
        border-radius: 9px;
        border: none;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
    }

    div.stButton > button:hover {
        background-color: #2D5F56;
        color: white;
        border: none;
    }

    hr {
        border-color: #E4E7EC;
    }
</style>
""", unsafe_allow_html=True)

st.title("🛍️ Get to Know Your Customers")

st.markdown("""
Welcome to your customer insights dashboard!

Every customer has different shopping habits. Some spend more,
some prefer to save, and others fall somewhere in between.

This project uses machine learning to discover patterns in customer
income and spending behaviour, helping us understand different
customer groups.

### What you can explore
- **Discover patterns:** See how customers differ in their spending.
- **Explore customer groups:** Compare income and spending habits.
- **Try it yourself:** Enter customer details and discover which
  cluster the model assigns them to.
""")

st.divider()

st.markdown("""
### Machine Learning Project using K-Means Clustering

This application analyses customer purchasing behaviour by grouping
customers according to their annual income and spending scores.

**Project objectives:**
- Identify customer groups with similar spending patterns.
- Visualise customer segments using machine learning.
- Analyse the characteristics of each customer group.
- Assign a new customer to a cluster.
""")

st.divider()

# Load the dataset
@st.cache_data
def load_data():
    return pd.read_csv("Mall_Customers.csv")


try:
    df = load_data()
except FileNotFoundError:
    st.error("Mall_Customers.csv not found. Keep it in the project folder.")
    st.stop()

# Check dataset columns
required_columns = ["Annual Income (k$)", "Spending Score (1-100)"]

if not all(column in df.columns for column in required_columns):
    st.error("The dataset columns do not match the expected format.")
    st.stop()

# Sidebar settings
st.sidebar.header("Clustering Settings")

k = st.sidebar.slider(
    "Select number of customer groups",
    min_value=2,
    max_value=10,
    value=5
)

# Prepare data
features = df[required_columns].copy()

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Apply K-Means
model = KMeans(n_clusters=k, random_state=42, n_init=10)
df["Cluster"] = model.fit_predict(scaled_features)

# Evaluate the clustering model
from sklearn.metrics import silhouette_score

score = silhouette_score(scaled_features, df["Cluster"])

st.subheader("Model Evaluation")
st.metric("Silhouette Score", f"{score:.3f}")

st.write(
    "The Silhouette Score ranges from -1 to 1. "
    "A higher score generally indicates better-separated clusters. "
    "It should be interpreted alongside the cluster graph and customer data."
)

# Elbow Method
st.subheader("Finding the Optimal Number of Clusters")
st.write(
    "The Elbow Method helps us choose a suitable number "
    "of customer groups."
)

inertia_values = []

for i in range(1, 11):
    elbow_model = KMeans(
        n_clusters=i,
        random_state=42,
        n_init=10
    )
    elbow_model.fit(scaled_features)
    inertia_values.append(elbow_model.inertia_)

fig_elbow, ax_elbow = plt.subplots(figsize=(9, 5))

ax_elbow.plot(
    range(1, 11),
    inertia_values,
    marker="o"
)

ax_elbow.set_xlabel("Number of Clusters (K)")
ax_elbow.set_ylabel("Inertia")
ax_elbow.set_title("Elbow Method for Optimal K")
ax_elbow.set_xticks(range(1, 11))
ax_elbow.grid(alpha=0.3)

st.pyplot(fig_elbow)
plt.close(fig_elbow)

# Display summary metrics
st.subheader("Customer Overview")

col1, col2, col3 = st.columns(3)

col1.metric("Total Customers", len(df))
col2.metric("Number of Groups", k)
col3.metric(
    "Average Spending Score",
    round(df["Spending Score (1-100)"].mean(), 1)
)

# Visualize clusters
st.subheader("Customer Segmentation Graph")

fig, ax = plt.subplots(figsize=(10, 5))

scatter = ax.scatter(
    df["Annual Income (k$)"],
    df["Spending Score (1-100)"],
    c=df["Cluster"],
    cmap="viridis",
    s=70
)

ax.set_xlabel("Annual Income (thousands)")
ax.set_ylabel("Spending Score (1-100)")
ax.set_title("Customer Groups using K-Means Clustering")
ax.grid(alpha=0.2)

fig.colorbar(scatter, ax=ax, label="Cluster Number")

st.pyplot(fig)
plt.close(fig)

# Analyze customer groups
st.subheader("Customer Group Analysis")

summary = df.groupby("Cluster").agg(
    Number_of_Customers=("Cluster", "size"),
    Average_Age=("Age", "mean"),
    Average_Income=("Annual Income (k$)", "mean"),
    Average_Spending=("Spending Score (1-100)", "mean")
).round(1)

st.dataframe(summary, use_container_width=True)

# Customer Group Interpretation
st.subheader("Understanding Customer Groups")

for cluster_id, row in summary.iterrows():
    avg_income = row["Average_Income"]
    avg_spending = row["Average_Spending"]

    if avg_income < 50 and avg_spending < 50:
        group_name = "Low Income, Low Spending"
    elif avg_income < 50 and avg_spending >= 50:
        group_name = "Low Income, High Spending"
    elif avg_income >= 50 and avg_spending < 50:
        group_name = "High Income, Low Spending"
    else:
        group_name = "High Income, High Spending"

    st.write(
        f"**Cluster {cluster_id}: {group_name}** — "
        f"Average income: {avg_income}k, "
        f"Average spending score: {avg_spending}"
    )

# Display dataset
st.subheader("Customer Dataset")
st.dataframe(df, use_container_width=True)

# Assign a new customer to a cluster
st.subheader("Find a Customer's Group")

income = st.number_input(
    "Annual Income (thousands)",
    min_value=0.0,
    max_value=200.0,
    value=50.0
)

spending = st.slider(
    "Spending Score",
    min_value=1,
    max_value=100,
    value=50
)

if st.button("Find Customer Group"):
    new_customer = pd.DataFrame(
        [[income, spending]],
        columns=required_columns
    )

    new_scaled = scaler.transform(new_customer)
    cluster = model.predict(new_scaled)[0]

    st.success(f"This customer belongs to Cluster {cluster}.")

# Download results
st.subheader("Download Results")

csv_data = df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Customer Data with Clusters",
    data=csv_data,
    file_name="customer_segments.csv",
    mime="text/csv"
)