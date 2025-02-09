import streamlit as st
import random
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# This data is purely random and sample data! Not the real data from BBS
np.random.seed(45)
num_listings = 100

df = pd.DataFrame({
    "Listing ID": range(1, num_listings + 1),
    "Completeness Percentage": np.random.uniform(50, 100, num_listings),  # Completeness 50% to 100%
    "Text Quality Score": np.random.uniform(0, 1, num_listings),  # Text quality (0-1 scale)
    "Image Quality Score": np.random.uniform(0, 1, num_listings),  # Image quality (0-1 scale)
    "Days Active": np.random.randint(10, 365, num_listings),  # Days the listing has been active
    "Num Attachments": np.random.randint(0, 5, num_listings),  # Number of documents uploaded
    "Contact Info": np.random.choice([0, 1], num_listings),  # Binary: 0 or 1
    "Location Data": np.random.choice([0, 1], num_listings),  # Binary: 0 or 1
    "Category": np.random.choice(["Restaurant", "Retail", "E-commerce", "Medical", "Franchise"], num_listings)
})

# Make a copy of the df before normalization
df_original = df.copy()

# Normalize only for LQS calculation (original values remain unchanged in df_original)
# This is the formula discussed in the proposal (minus profitability)
# LQS=(0.25×completeness)+(0.20×text quality)+(0.20×image quality)+(0.15×days active)+(0.10×num attachments)+(0.10×contact info)
scaler = MinMaxScaler(feature_range=(0.3, 1))
features_to_normalize = ["Completeness Percentage", "Text Quality Score", "Image Quality Score", "Days Active", "Num Attachments"]
df[features_to_normalize] = scaler.fit_transform(df[features_to_normalize])

# Compute LQS using the provided formula
df["LQS"] = (
    (df["Completeness Percentage"] * 0.25) +
    (df["Text Quality Score"] * 0.20) +
    (df["Image Quality Score"] * 0.20) +
    (df["Days Active"] * 0.15) +
    (df["Num Attachments"] * 0.10) +
    ((df["Contact Info"] ** 1.2) * 0.10)
) * 100  # Scale LQS to 0-100 range



# Exponential Scaling
df["LQS"] = df["LQS"].apply(lambda x: x * 1.1 if x <= 100 else x)

def assign_badge(score):
    if score >= 85:
        return "🏆 High"
    elif score >= 70:
        return "🥈 Medium"
    elif score >= 50:
        return "🥉 Low"
    else:
        return "⚠️ Needs Improvement"



# Add the newly calculated LQS to the original dataframe
df_original["LQS"] = df["LQS"]
df_original["Badge"] = df_original["LQS"].apply(assign_badge)

headline_templates = {
    "Restaurant": [
        "Turnkey Italian Restaurant with High Foot Traffic",
        "Profitable Sushi Bar in Downtown",
        "Cozy Café with Loyal Customer Base",
        "Award-Winning Steakhouse for Sale",
        "High-Volume Pizzeria with Delivery Service"
    ],
    "Retail": [
        "Established Boutique in Prime Shopping District",
        "Profitable Electronics Store with Online Presence",
        "Trendy Clothing Store - Great Location!",
        "Specialty Bookstore with Strong Local Following",
        "Home Decor Retail Store with Steady Sales"
    ],
    "E-commerce": [
        "Dropshipping Store with High-Profit Margins",
        "Branded Online Clothing Store with Loyal Customers",
        "Subscription Box Business - Automated Revenue",
        "Amazon FBA Business with Positive Growth",
        "Niche Electronics E-commerce Store for Sale"
    ],
    "Medical": [
        "Well-Established Dental Practice for Sale",
        "Profitable Pharmacy with High Repeat Customers",
        "Chiropractic Clinic with Modern Equipment",
        "Physical Therapy Clinic in Growing Area",
        "Dermatology Practice with State-of-the-Art Facilities"
    ],
    "Franchise": [
        "Popular Fast-Food Franchise in Busy Location",
        "Well-Known Gym Franchise with Steady Memberships",
        "Growing Coffee Franchise with Great Support",
        "Top-Rated Home Services Franchise Opportunity",
        "Automotive Repair Franchise with Loyal Clientele"
    ]
}

# Function to generate random headlines based on category
def generate_headline(category):
    return random.choice(headline_templates.get(category, ["Great Business Opportunity!"]))

# Generate headlines for each listing
df_original["Headline"] = df_original["Category"].apply(generate_headline)


st.title("📊 Listing Quality Score (LQS) Dashboard")
st.caption("This data is purely randomly generated for demo purposes!")
st.sidebar.header("Filter Options")

# Filter by Category
category_filter = st.sidebar.multiselect("Select Category", df_original["Category"].unique(), default=df_original["Category"].unique())

# Filter by Minimum LQS Score
min_lqs = st.sidebar.slider("Minimum LQS Score", int(df_original["LQS"].min()), int(df_original["LQS"].max()), 50)

# Filter by Days Active
days_active_range = st.sidebar.slider("Days Active Range", int(df_original["Days Active"].min()), int(df_original["Days Active"].max()), (30, 180))

# Apply Filters to the Original Data
filtered_df = df_original[
    (df_original["Category"].isin(category_filter)) &
    (df_original["LQS"] >= min_lqs) &
    (df_original["Days Active"].between(days_active_range[0], days_active_range[1]))
]

# Display Data Table
st.subheader("📋 LQS Rankings")
st.caption("The LQS is calculated based on the formula defined in the proposal.")
st.dataframe(filtered_df[[
    "Listing ID", "Headline", "Category", "Completeness Percentage",
    "Text Quality Score", "Image Quality Score", "Days Active",
    "Num Attachments", "Contact Info", "Location Data", "Badge","LQS"
]])

# Interactive Plot: LQS Histogram
st.subheader("📊 LQS Distribution")
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(df_original["LQS"], bins=20, color="blue", alpha=0.7, edgecolor="black")
ax.set_xlabel("Listing Quality Score (LQS)")
ax.set_ylabel("Frequency")
ax.set_title("Distribution of LQS Scores")
st.pyplot(fig)

# Find the top 10 listings based on LQS
st.subheader("🏆 Top 10 High-Quality Listings")
top_listings = df_original.sort_values(by="LQS", ascending=False).head(10)
st.dataframe(top_listings)

# LQS vs Completeness Scatterplot --> this can be modified to see other things
st.subheader("📈 LQS vs Completeness Percentage")
fig = px.scatter(filtered_df, x="LQS", y="Completeness Percentage", color="Category",
                 size="Num Attachments", hover_data=["Listing ID"],
                 title="LQS vs Completeness Percentage for Listings")
st.plotly_chart(fig)

st.markdown("🔹 **Higher Completeness + LQS = More Buyer Interest!**")
