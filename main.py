import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import json
import os
from transformers import pipeline
from functools import lru_cache


# Setup classifier once
@st.cache_resource
def load_classifier():
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")


classifier = load_classifier()

CATEGORY_OVERRIDES_FILE = "./category_overrides.json"
CUSTOM_CATEGORIES_FILE = "./custom_categories.json"

# Define default categories
DEFAULT_EXPENSE_LABELS = [
    "Shopping", "Food & Dining", "Transport", "Bills & Utilities",
    "Entertainment", "Healthcare", "Travel", "Electronics", "Mobile Phones",
    "Retail", "Groceries", "EMI", "Insurance", "Rent", "Credit Cards"
]

DEFAULT_INCOME_LABELS = [
    "Salary", "Business Income", "Investment Income", "Loan Received", "Refunds", "Others"
]


# Load and save functions for overrides
def load_overrides():
    if os.path.exists(CATEGORY_OVERRIDES_FILE):
        try:
            with open(CATEGORY_OVERRIDES_FILE, "r") as f:
                return json.load(f)
        except:
            return {}
    return {}


def save_override(narration, category):
    overrides = load_overrides()
    norm_narration = normalize_narration(narration)
    overrides[norm_narration] = category
    with open(CATEGORY_OVERRIDES_FILE, "w") as f:
        json.dump(overrides, f)


def load_custom_categories():
    # If file doesn't exist, create it with defaults
    if not os.path.exists(CUSTOM_CATEGORIES_FILE):
        return reset_categories()

    try:
        with open(CUSTOM_CATEGORIES_FILE, "r") as f:
            data = json.load(f)
            expense_cats = data.get("expense", DEFAULT_EXPENSE_LABELS)
            income_cats = data.get("income", DEFAULT_INCOME_LABELS)
            return expense_cats, income_cats
    except:
        # If any error, return defaults
        return DEFAULT_EXPENSE_LABELS.copy(), DEFAULT_INCOME_LABELS.copy()


def save_custom_categories(expense_cats, income_cats):
    with open(CUSTOM_CATEGORIES_FILE, "w") as f:
        json.dump({"expense": expense_cats, "income": income_cats}, f)


def reset_categories():
    expense_cats = DEFAULT_EXPENSE_LABELS.copy()
    income_cats = DEFAULT_INCOME_LABELS.copy()

    with open(CUSTOM_CATEGORIES_FILE, "w") as f:
        json.dump({"expense": expense_cats, "income": income_cats}, f)

    return expense_cats, income_cats


def normalize_narration(narration: str) -> str:
    return narration.strip().lower()


# Function to collect all used categories from overrides
def get_all_used_categories():
    overrides = load_overrides()
    used_categories = set(overrides.values())
    return list(used_categories)


# Function to sync categories from overrides with custom categories
def sync_categories_from_overrides():
    expense_cats, income_cats = load_custom_categories()
    used_categories = get_all_used_categories()

    # Keep track of if we made changes
    changes_made = False

    # Add any categories found in overrides to the appropriate list
    for category in used_categories:
        if category not in expense_cats and category not in income_cats:
            # For simplicity, add unknown categories to expense list by default
            expense_cats.append(category)
            changes_made = True

    if changes_made:
        save_custom_categories(expense_cats, income_cats)

    return expense_cats, income_cats, changes_made


# Categorization function
@lru_cache(maxsize=10000)
def categorize_transaction(narration, amount, txn_type):
    norm_narration = normalize_narration(narration)
    narration = narration.lower()

    # Check overrides first
    overrides = load_overrides()
    if norm_narration in overrides:
        return overrides[norm_narration]

    # Credit Card Detection
    if any(keyword in narration for keyword in ['credit card', 'creditcard', 'cc ', 'card payment']):
        return 'Credit Cards'

    # Other existing detection logic...
    if 'salary' in narration or ('cred' in narration and txn_type == 'credit' and amount > 2000):
        return 'Salary'

    if 'interest' in narration and txn_type == 'credit':
        return 'Investment Income'

    if any(keyword in narration for keyword in
           ['swiggy', 'zomato', 'foods', 'dosa', 'hotel', 'restaurant']) and amount < 1000:
        return 'Food & Dining'

    if any(keyword in narration for keyword in ['electricity', 'airtel', 'vodafone', 'bsnl', 'bill', 'recharge']):
        return 'Bills & Utilities'

    if 'ach d' in narration and any(x in narration for x in ['life', 'sbi life', 'insurance']):
        return 'Insurance'

    if ('emi' in narration or 'loan' in narration or
            ('ach d' in narration and 'hdfc bank' in narration) or
            (amount >= 5000 and txn_type == 'debit')):
        return 'EMI'

    if any(keyword in narration for keyword in ['amazon', 'flipkart', 'myntra', 'shop']):
        return 'Shopping'

    if 'upi' in narration and amount < 2000:
        return 'Personal Transfer'

    return 'Others'


# Data transformation
def transform_narration(df, all_categories):
    df = df[df["Narration"].notnull()].copy()
    df["Withdrawal Amt."] = pd.to_numeric(df["Withdrawal Amt."], errors="coerce")
    df["Deposit Amt."] = pd.to_numeric(df["Deposit Amt."], errors="coerce")
    df["Closing Balance"] = pd.to_numeric(df["Closing Balance"], errors="coerce")

    df["txn_type"] = np.where(df["Deposit Amt."].notnull(), "credit",
                              np.where(df["Withdrawal Amt."].notnull(), "debit", "unknown"))

    df["Category"] = df.apply(
        lambda row: categorize_transaction(
            row["Narration"],
            row["Withdrawal Amt."] if row["txn_type"] == "debit" else row["Deposit Amt."],
            row["txn_type"]
        ), axis=1
    )

    # Make sure all categories are included in our category lists
    for cat in df["Category"].unique():
        if cat not in all_categories and cat != "Others":
            # Add new categories to expense by default
            st.session_state.custom_expense_labels.append(cat)
            all_categories.append(cat)

    # Save the updated categories
    save_custom_categories(st.session_state.custom_expense_labels, st.session_state.custom_income_labels)

    return df


# File loading
def load_transactions(upload_file, all_categories):
    try:
        df = pd.read_csv(upload_file)
        df.columns = [col.strip() for col in df.columns]
        df = transform_narration(df, all_categories)
        df["Closing Balance"] = df["Closing Balance"].astype(str).str.replace(",", "").astype(float)
        df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%y")
        st.success("File Processed Successfully!")
        return df
    except Exception as e:
        st.error(f"Error Processing file: {str(e)}")
        return None


def main():
    st.set_page_config(page_title="Budget Consultant", page_icon="💰", layout="wide")

    # Initialize session state
    if "initialization_done" not in st.session_state:
        # Sync categories from overrides first
        expense_cats, income_cats, _ = sync_categories_from_overrides()
        st.session_state.custom_expense_labels = expense_cats
        st.session_state.custom_income_labels = income_cats
        st.session_state.initialization_done = True
    else:
        # Load custom categories from file
        expense_cats, income_cats = load_custom_categories()
        st.session_state.custom_expense_labels = expense_cats
        st.session_state.custom_income_labels = income_cats

    st.title("Bank Statement Categorizer")

    # Debug section
    with st.expander("Categories Info"):
        st.write("Available Categories:")
        st.write(st.session_state.custom_expense_labels)
        st.write("Credit Cards in list:", "Credit Cards" in st.session_state.custom_expense_labels)

        # Add button to sync categories from overrides
        if st.button("Sync Categories from Overrides"):
            expense_cats, income_cats, changes_made = sync_categories_from_overrides()
            st.session_state.custom_expense_labels = expense_cats
            st.session_state.custom_income_labels = income_cats
            if changes_made:
                st.success("Categories synced from overrides!")
            else:
                st.info("No new categories found in overrides.")
            st.rerun()

        if st.button("Force Reset Categories"):
            expense_cats, income_cats = reset_categories()
            st.session_state.custom_expense_labels = expense_cats
            st.session_state.custom_income_labels = income_cats
            st.success("Categories have been reset!")
            st.rerun()

    # Combine all categories for dropdown
    all_categories = st.session_state.custom_expense_labels + st.session_state.custom_income_labels

    # Main application continues
    upload_file = st.file_uploader("Upload your Bank Statement (CSV)", type=["csv"])

    if upload_file is not None:
        # Before loading, sync categories from overrides
        expense_cats, income_cats, _ = sync_categories_from_overrides()
        st.session_state.custom_expense_labels = expense_cats
        st.session_state.custom_income_labels = income_cats
        all_categories = expense_cats + income_cats

        df = load_transactions(upload_file, all_categories)

        if df is not None:
            df_debits = df[df["Withdrawal Amt."].notnull()].copy()
            df_credits = df[df["Deposit Amt."].notnull()].copy()

            df_debits["Amount"] = df_debits["Withdrawal Amt."]
            df_credits["Amount"] = df_credits["Deposit Amt."]

            st.session_state.df_debits = df_debits.copy()
            st.session_state.df_credits = df_credits.copy()

            df_debits_display = df_debits[["Date", "Narration", "Amount", "Category"]].copy()
            df_credits_display = df_credits[["Date", "Narration", "Amount", "Category"]].copy()

            st.sidebar.header("🛠️ Manage Categories")
            cat_type = st.sidebar.radio("Category Type", ["Expense", "Income"], horizontal=True)
            new_cat = st.sidebar.text_input("New Category Name")

            if st.sidebar.button("➕ Add Category"):
                if new_cat:
                    if cat_type == "Expense" and new_cat not in st.session_state.custom_expense_labels:
                        st.session_state.custom_expense_labels.append(new_cat)
                        save_custom_categories(st.session_state.custom_expense_labels,
                                               st.session_state.custom_income_labels)
                        st.sidebar.success(f"Added '{new_cat}' to Expense Categories.")
                        st.rerun()
                    elif cat_type == "Income" and new_cat not in st.session_state.custom_income_labels:
                        st.session_state.custom_income_labels.append(new_cat)
                        save_custom_categories(st.session_state.custom_expense_labels,
                                               st.session_state.custom_income_labels)
                        st.sidebar.success(f"Added '{new_cat}' to Income Categories.")
                        st.rerun()
                    else:
                        st.sidebar.warning(f"'{new_cat}' already exists.")

            # Show current categories
            st.sidebar.subheader("Current Categories")
            if cat_type == "Expense":
                st.sidebar.write(", ".join(st.session_state.custom_expense_labels))
                if st.session_state.custom_expense_labels:
                    remove_cat = st.sidebar.selectbox("Select category to remove",
                                                      st.session_state.custom_expense_labels)
                    if st.sidebar.button("❌ Remove Category"):
                        if remove_cat != "Credit Cards":  # Prevent removal of Credit Cards
                            st.session_state.custom_expense_labels.remove(remove_cat)
                            save_custom_categories(st.session_state.custom_expense_labels,
                                                   st.session_state.custom_income_labels)
                            st.sidebar.success(f"Removed '{remove_cat}'")
                            st.rerun()
                        else:
                            st.sidebar.error("Cannot remove Credit Cards category")
            else:
                st.sidebar.write(", ".join(st.session_state.custom_income_labels))
                if st.session_state.custom_income_labels:
                    remove_cat = st.sidebar.selectbox("Select category to remove",
                                                      st.session_state.custom_income_labels)
                    if st.sidebar.button("❌ Remove Category"):
                        st.session_state.custom_income_labels.remove(remove_cat)
                        save_custom_categories(st.session_state.custom_expense_labels,
                                               st.session_state.custom_income_labels)
                        st.sidebar.success(f"Removed '{remove_cat}'")
                        st.rerun()

            # Update all_categories to include any that might have been added
            all_categories = st.session_state.custom_expense_labels + st.session_state.custom_income_labels

            # Make sure Credit Cards is always included
            if "Credit Cards" not in all_categories:
                all_categories.append("Credit Cards")

            tab1, tab2, tab3 = st.tabs(["Expense (Debits)", "Payments (Credits)", "Analysis"])

            with tab1:
                st.subheader("Edit Categories - Expenses")

                edited_debits = st.data_editor(
                    df_debits_display,
                    column_config={
                        "Category": st.column_config.SelectboxColumn(
                            "Category",
                            options=all_categories,
                            required=True
                        ),
                        "Narration": st.column_config.TextColumn("Details"),
                        "Amount": st.column_config.NumberColumn("Amount", format="%.2f")
                    },
                    key="edit_debits"
                )
                if st.button("💾 Submit Changes (Debits)"):
                    for idx in df_debits_display.index:
                        orig_cat = df_debits_display.at[idx, "Category"]
                        edited_cat = edited_debits.at[idx, "Category"]
                        if orig_cat != edited_cat:
                            narration = df_debits_display.at[idx, "Narration"]
                            save_override(narration, edited_cat)
                            st.success(f"Saved override: {narration[:40]} → {edited_cat}")

                    # Sync categories after changes
                    _, _, changes = sync_categories_from_overrides()
                    if changes:
                        st.info("New categories detected and added to your category list.")
                        st.rerun()

            with tab2:
                st.subheader("Edit Categories - Payments")
                edited_credits = st.data_editor(
                    df_credits_display,
                    column_config={
                        "Category": st.column_config.SelectboxColumn(
                            "Category",
                            options=all_categories,
                            required=True
                        ),
                        "Narration": st.column_config.TextColumn("Details"),
                        "Amount": st.column_config.NumberColumn("Amount", format="%.2f")
                    },
                    key="edit_credits"
                )
                if st.button("💾 Submit Changes (Credits)"):
                    for idx in df_credits_display.index:
                        orig_cat = df_credits_display.at[idx, "Category"]
                        edited_cat = edited_credits.at[idx, "Category"]
                        if orig_cat != edited_cat:
                            narration = df_credits_display.at[idx, "Narration"]
                            save_override(narration, edited_cat)
                            st.success(f"Saved override: {narration[:40]} → {edited_cat}")

                    # Sync categories after changes
                    _, _, changes = sync_categories_from_overrides()
                    if changes:
                        st.info("New categories detected and added to your category list.")
                        st.rerun()

            with tab3:
                st.subheader("Expense Analysis")

                # Calculate total expenses by category
                category_total = st.session_state.df_debits.groupby("Category")["Amount"].sum().reset_index()
                category_total = category_total.sort_values("Amount", ascending=False)

                # Calculate total expense amount for percentage calculation
                total_expense = category_total["Amount"].sum()

                # Add percentage column
                if total_expense > 0:
                    category_total["Percentage"] = (category_total["Amount"] / total_expense * 100).round(2)
                else:
                    category_total["Percentage"] = 0

                # Display as table
                st.dataframe(
                    category_total,
                    column_config={
                        "Amount": st.column_config.NumberColumn("Amount", format="%.2f INR"),
                        "Percentage": st.column_config.NumberColumn("% of Total", format="%.2f%%")
                    },
                    use_container_width=True,
                    hide_index=True
                )

                # Create two columns for charts
                col1, col2 = st.columns(2)

                with col1:
                    # Create the pie chart with Plotly Express
                    if not category_total.empty and total_expense > 0:
                        st.subheader("Expense Distribution")
                        fig = px.pie(
                            category_total,
                            values="Amount",
                            names="Category",
                            title="Expense Distribution by Category",
                            hole=0.4,  # Creates a donut chart
                            color_discrete_sequence=px.colors.qualitative.Pastel,  # Use a nice color scheme
                        )
                        fig.update_traces(textposition='inside', textinfo='percent+label')
                        fig.update_layout(
                            legend=dict(orientation="h", yanchor="bottom", y=0, xanchor="center", x=0.5),
                            height=500
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No expense data available for visualization")

                with col2:
                    # Create a bar chart for top expenses
                    if not category_total.empty and total_expense > 0:
                        st.subheader("Top Expenses")
                        # Get top 5 categories
                        top_categories = category_total.head(5)

                        fig = px.bar(
                            top_categories,
                            x="Category",
                            y="Amount",
                            title="Top 5 Expense Categories",
                            color="Category",
                            color_discrete_sequence=px.colors.qualitative.Pastel
                        )
                        fig.update_layout(height=500)
                        st.plotly_chart(fig, use_container_width=True)

                # Monthly trend analysis if date data is available
                if not df_debits.empty and "Date" in df_debits.columns:
                    st.subheader("Monthly Expense Trend")

                    # Create a copy with month extracted
                    df_monthly = df_debits.copy()
                    df_monthly["Month"] = df_monthly["Date"].dt.strftime("%Y-%m")

                    # Group by month and sum amounts
                    monthly_totals = df_monthly.groupby("Month")["Amount"].sum().reset_index()
                    monthly_totals = monthly_totals.sort_values("Month")

                    # Create line chart
                    fig = px.line(
                        monthly_totals,
                        x="Month",
                        y="Amount",
                        title="Monthly Expense Trend",
                        markers=True
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()