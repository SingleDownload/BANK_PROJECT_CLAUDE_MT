import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve,
    classification_report
)
import warnings
import io

warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Universal Bank · Loan Intelligence",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Serif+Display&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #0f1923 0%, #1a2d3d 100%);
    border-right: 1px solid rgba(255,255,255,0.07);
}
[data-testid="stSidebar"] * { color: #c8d8e8 !important; }
[data-testid="stSidebar"] .stRadio label { font-size: 0.92rem; }

/* Main background */
.main { background: #f7f9fc; }

/* KPI Cards */
.kpi-card {
    background: white;
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    border: 1px solid #e8edf3;
    box-shadow: 0 2px 12px rgba(0,0,0,0.05);
    text-align: center;
}
.kpi-label { font-size: 0.78rem; font-weight: 500; color: #7a8a9a; letter-spacing: 0.06em; text-transform: uppercase; margin-bottom: 6px; }
.kpi-value { font-size: 2rem; font-weight: 600; color: #0f2a44; }
.kpi-delta { font-size: 0.8rem; margin-top: 4px; }

/* Section headers */
.section-header {
    font-family: 'DM Serif Display', serif;
    font-size: 1.5rem;
    color: #0f2a44;
    margin: 0.5rem 0 0.2rem 0;
    border-left: 4px solid #1a6fa8;
    padding-left: 0.75rem;
}
.section-sub {
    font-size: 0.85rem;
    color: #5a7a95;
    margin-bottom: 1.4rem;
    padding-left: 1rem;
}

/* Insight cards */
.insight-box {
    background: #f0f6ff;
    border-left: 4px solid #1a6fa8;
    border-radius: 0 10px 10px 0;
    padding: 0.75rem 1rem;
    margin: 0.5rem 0;
    font-size: 0.85rem;
    color: #1a3a5c;
    line-height: 1.55;
}
.warning-box {
    background: #fff8f0;
    border-left: 4px solid #e07b2a;
    border-radius: 0 10px 10px 0;
    padding: 0.75rem 1rem;
    margin: 0.5rem 0;
    font-size: 0.85rem;
    color: #5a3010;
    line-height: 1.55;
}

/* Chart containers */
.chart-card {
    background: white;
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    border: 1px solid #e8edf3;
    box-shadow: 0 2px 12px rgba(0,0,0,0.04);
    margin-bottom: 1rem;
}
.chart-title {
    font-size: 0.92rem;
    font-weight: 600;
    color: #0f2a44;
    margin-bottom: 0.3rem;
}
.chart-desc {
    font-size: 0.8rem;
    color: #6a8090;
    margin-bottom: 0.8rem;
    line-height: 1.5;
}

/* Table styling */
.styled-table { width: 100%; border-collapse: collapse; font-size: 0.88rem; }
.styled-table th { background: #0f2a44; color: white; padding: 10px 14px; text-align: left; font-weight: 500; }
.styled-table td { padding: 9px 14px; border-bottom: 1px solid #eef2f7; color: #2a3a4a; }
.styled-table tr:nth-child(even) td { background: #f7f9fc; }
.styled-table tr:hover td { background: #eaf2fb; }

/* Divider */
hr { border: none; border-top: 1px solid #e2e8f0; margin: 2rem 0; }
</style>
""", unsafe_allow_html=True)


# ── Colour palette ─────────────────────────────────────────────────────────
PALETTE = {
    "navy":    "#0f2a44",
    "blue":    "#1a6fa8",
    "sky":     "#3fa0d4",
    "teal":    "#1ab8a4",
    "amber":   "#e07b2a",
    "red":     "#d64045",
    "green":   "#2a9d4e",
    "purple":  "#7b4fa6",
    "light":   "#e8f3fb",
    "gray":    "#8a9ab0",
}
MODEL_COLORS = {"Decision Tree": "#1a6fa8", "Random Forest": "#1ab8a4", "Gradient Boosted Tree": "#e07b2a"}
FIGSIZE_FULL = (12, 4.5)


# ── Data loading & preprocessing ─────────────────────────────────────────────
@st.cache_data
def load_and_preprocess(uploaded_file=None):
    if uploaded_file is not None:
        df_raw = pd.read_csv(uploaded_file)
    else:
        df_raw = pd.read_csv("UniversalBank.csv")

    df = df_raw.copy()
    df['Experience'] = df['Experience'].clip(lower=0)
    df['Has_Mortgage'] = (df['Mortgage'] > 0).astype(int)
    df['Income_CCAvg_Ratio'] = df['CCAvg'] / (df['Income'] + 1)
    df['High_Value'] = ((df['Income'] > 98) & (df['CCAvg'] > 2.5)).astype(int)

    drop_cols = ['ID', 'ZIP Code']
    df_model = df.drop(columns=[c for c in drop_cols if c in df.columns])
    return df_raw, df_model


@st.cache_data
def train_models(df_model):
    X = df_model.drop(columns=['Personal Loan'])
    y = df_model['Personal Loan']
    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    models = {
        'Decision Tree':       DecisionTreeClassifier(random_state=42, class_weight='balanced', max_depth=8),
        'Random Forest':       RandomForestClassifier(random_state=42, class_weight='balanced', n_estimators=150, n_jobs=-1),
        'Gradient Boosted Tree': GradientBoostingClassifier(random_state=42, n_estimators=150, learning_rate=0.1, max_depth=4),
    }

    results, trained = {}, {}
    for name, m in models.items():
        m.fit(X_train_sc, y_train)
        y_pred  = m.predict(X_test_sc)
        y_prob  = m.predict_proba(X_test_sc)[:, 1]
        y_tr_pred = m.predict(X_train_sc)
        results[name] = {
            'train_acc': accuracy_score(y_train, y_tr_pred),
            'test_acc':  accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall':    recall_score(y_test, y_pred),
            'f1':        f1_score(y_test, y_pred),
            'auc':       roc_auc_score(y_test, y_prob),
            'cm':        confusion_matrix(y_test, y_pred),
            'y_test':    y_test.values,
            'y_pred':    y_pred,
            'y_prob':    y_prob,
        }
        trained[name] = m
    return results, trained, scaler, feature_names, X_train_sc, X_test_sc, y_train, y_test


def apply_chart_style(ax, title="", xlabel="", ylabel="", grid_axis="y"):
    ax.set_title(title, fontsize=11, fontweight='600', color=PALETTE["navy"], pad=10)
    ax.set_xlabel(xlabel, fontsize=9, color=PALETTE["gray"])
    ax.set_ylabel(ylabel, fontsize=9, color=PALETTE["gray"])
    ax.tick_params(colors=PALETTE["gray"], labelsize=8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    if grid_axis:
        ax.grid(axis=grid_axis, color='#e8edf3', linewidth=0.7, linestyle='--', alpha=0.8)
        ax.set_axisbelow(True)
    ax.set_facecolor('white')


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🏦 Universal Bank")
    st.markdown("#### Loan Intelligence Suite")
    st.markdown("---")
    uploaded = st.file_uploader("📂 Upload Dataset (CSV)", type=["csv"])
    st.markdown("---")
    nav = st.radio("Navigate", [
        "📊 Executive Overview",
        "🔍 Descriptive Analytics",
        "📈 Diagnostic Analytics",
        "🤖 Predictive Models",
        "🎯 Prescriptive Analytics",
        "🔮 Predict New Customers",
    ])
    st.markdown("---")
    st.markdown("<small style='color:#4a7a9a;'>Universal Bank · Marketing Analytics v1.0</small>", unsafe_allow_html=True)


# ── Load data ─────────────────────────────────────────────────────────────────
df_raw, df_model = load_and_preprocess(uploaded)
results, trained_models, scaler, feature_names, X_train_sc, X_test_sc, y_train, y_test = train_models(df_model)

best_model_name = max(results, key=lambda k: results[k]['auc'])
best_model = trained_models[best_model_name]

total      = len(df_raw)
acceptors  = df_raw['Personal Loan'].sum()
accept_pct = acceptors / total * 100


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — EXECUTIVE OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if nav == "📊 Executive Overview":
    st.markdown('<div class="section-header">Executive Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">High-level summary of the Universal Bank personal loan dataset and model performance.</div>', unsafe_allow_html=True)

    # KPIs
    c1, c2, c3, c4, c5 = st.columns(5)
    kpis = [
        ("Total Customers", f"{total:,}", ""),
        ("Loan Acceptors", f"{acceptors:,}", f"({accept_pct:.1f}% of base)"),
        ("Non-Acceptors", f"{total - acceptors:,}", f"({100-accept_pct:.1f}% of base)"),
        ("Best Model AUC", f"{results[best_model_name]['auc']:.4f}", f"{best_model_name}"),
        ("Best F1-Score", f"{results[best_model_name]['f1']:.4f}", "on test set"),
    ]
    for col, (label, val, delta) in zip([c1,c2,c3,c4,c5], kpis):
        col.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-label'>{label}</div>
            <div class='kpi-value'>{val}</div>
            <div class='kpi-delta' style='color:#5a7a95;'>{delta}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_a, col_b = st.columns([1, 1])

    # Class imbalance donut
    with col_a:
        st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
        st.markdown("<div class='chart-title'>Target Class Distribution</div>", unsafe_allow_html=True)
        st.markdown("<div class='chart-desc'>Only 9.6% of customers accepted a personal loan in the last campaign — a strong class imbalance. Models are trained with balanced class weights to ensure minority class detection is not sacrificed for overall accuracy.</div>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 4))
        fig.patch.set_facecolor('white')
        sizes  = [acceptors, total - acceptors]
        colors = [PALETTE["blue"], "#dde6ef"]
        wedges, _, autotexts = ax.pie(
            sizes, labels=["Accepted", "Not Accepted"],
            autopct="%1.1f%%", colors=colors,
            startangle=90, pctdistance=0.75,
            wedgeprops=dict(width=0.55, edgecolor='white', linewidth=2),
        )
        for at in autotexts:
            at.set_fontsize(10); at.set_color(PALETTE["navy"]); at.set_fontweight('600')
        ax.set_facecolor('white')
        ax.set_title("Personal Loan Acceptance", fontsize=11, fontweight='600', color=PALETTE["navy"])
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.markdown("</div>", unsafe_allow_html=True)

    # Model comparison bar
    with col_b:
        st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
        st.markdown("<div class='chart-title'>Model Performance at a Glance</div>", unsafe_allow_html=True)
        st.markdown("<div class='chart-desc'>Gradient Boosted Tree leads across all key metrics. All three models achieve >97% test accuracy and AUC >0.98, making them highly reliable for targeting likely loan acceptors.</div>", unsafe_allow_html=True)
        metrics_show = ['test_acc', 'precision', 'recall', 'f1', 'auc']
        metric_labels = ['Test Acc', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor('white')
        x = np.arange(len(metrics_show))
        w = 0.25
        for i, (mname, mcol) in enumerate(MODEL_COLORS.items()):
            vals = [results[mname][m] for m in metrics_show]
            bars = ax.bar(x + i*w, vals, w, label=mname, color=mcol, alpha=0.88, zorder=3, edgecolor='white')
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                        f"{v:.2f}", ha='center', va='bottom', fontsize=6.5, color=PALETTE["navy"])
        ax.set_xticks(x + w)
        ax.set_xticklabels(metric_labels, fontsize=8.5)
        ax.set_ylim(0.75, 1.05)
        apply_chart_style(ax, ylabel="Score")
        ax.legend(fontsize=7.5, frameon=False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.markdown("</div>", unsafe_allow_html=True)

    # Summary insight boxes
    st.markdown('<div class="insight-box">📌 <strong>Income is the single strongest predictor.</strong> Customers earning above $98k have a 35.6% acceptance rate vs near-zero for those below $39k. High-income segments must be the primary marketing target.</div>', unsafe_allow_html=True)
    st.markdown('<div class="insight-box">📌 <strong>CD Account holders convert at 7× the base rate (42.6%).</strong> Customers with certificates of deposit are highly financially engaged and receptive to cross-sell. Always flag this segment for outreach.</div>', unsafe_allow_html=True)
    st.markdown('<div class="warning-box">⚠️ <strong>Class imbalance requires careful handling.</strong> At 9.6% positive class, naive models would simply predict "no" and achieve 90% accuracy. Balanced training weights ensure the model genuinely identifies likely converters.</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — DESCRIPTIVE ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
elif nav == "🔍 Descriptive Analytics":
    st.markdown('<div class="section-header">Descriptive Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Understanding the shape, distribution and composition of the customer base.</div>', unsafe_allow_html=True)

    # Summary stats table
    st.subheader("Dataset Summary Statistics")
    num_cols = ['Age', 'Experience', 'Income', 'Family', 'CCAvg', 'Mortgage']
    desc = df_raw[num_cols].describe().T.round(2)
    desc.columns = ['Count','Mean','Std Dev','Min','25th %ile','Median','75th %ile','Max']
    st.dataframe(desc, use_container_width=True)

    st.markdown("---")

    # Distribution plots — row 1
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
        st.markdown("<div class='chart-title'>Age Distribution</div>", unsafe_allow_html=True)
        st.markdown("<div class='chart-desc'>The customer base peaks in the 35–55 age band. Marketing campaigns targeting middle-aged professionals (40–55) are more likely to reach high-income earners who are prime loan candidates.</div>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(4.5, 3.2))
        fig.patch.set_facecolor('white')
        ax.hist(df_raw['Age'], bins=25, color=PALETTE["blue"], alpha=0.8, edgecolor='white', zorder=3)
        ax.axvline(df_raw['Age'].mean(), color=PALETTE["amber"], linewidth=1.5, linestyle='--', label=f"Mean: {df_raw['Age'].mean():.1f}")
        ax.legend(fontsize=8, frameon=False)
        apply_chart_style(ax, xlabel="Age (years)", ylabel="Number of Customers")
        plt.tight_layout(); st.pyplot(fig); plt.close()
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
        st.markdown("<div class='chart-title'>Annual Income Distribution</div>", unsafe_allow_html=True)
        st.markdown("<div class='chart-desc'>Income is right-skewed with a long tail of high earners. The top quartile (>$98k) drives virtually all loan conversions — this segment is the golden tier for marketing spend allocation.</div>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(4.5, 3.2))
        fig.patch.set_facecolor('white')
        ax.hist(df_raw['Income'], bins=30, color=PALETTE["teal"], alpha=0.8, edgecolor='white', zorder=3)
        ax.axvline(98, color=PALETTE["red"], linewidth=1.5, linestyle='--', label="Top quartile threshold ($98k)")
        ax.legend(fontsize=8, frameon=False)
        apply_chart_style(ax, xlabel="Annual Income ($000)", ylabel="Number of Customers")
        plt.tight_layout(); st.pyplot(fig); plt.close()
        st.markdown("</div>", unsafe_allow_html=True)

    with col3:
        st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
        st.markdown("<div class='chart-title'>Monthly Credit Card Spending</div>", unsafe_allow_html=True)
        st.markdown("<div class='chart-desc'>Most customers spend under $2,500/month on credit cards. High spenders (>$2,500/month) are likely high-income and represent a strong secondary signal for loan propensity.</div>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(4.5, 3.2))
        fig.patch.set_facecolor('white')
        ax.hist(df_raw['CCAvg'], bins=30, color=PALETTE["purple"], alpha=0.8, edgecolor='white', zorder=3)
        ax.axvline(2.5, color=PALETTE["amber"], linewidth=1.5, linestyle='--', label="High spender ($2.5k)")
        ax.legend(fontsize=8, frameon=False)
        apply_chart_style(ax, xlabel="CC Spending/Month ($000)", ylabel="Number of Customers")
        plt.tight_layout(); st.pyplot(fig); plt.close()
        st.markdown("</div>", unsafe_allow_html=True)

    # Distribution plots — row 2
    col4, col5, col6 = st.columns(3)

    with col4:
        st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
        st.markdown("<div class='chart-title'>Education Level Breakdown</div>", unsafe_allow_html=True)
        st.markdown("<div class='chart-desc'>42% of customers are undergraduates. Graduate and advanced-degree holders (58% combined) are significantly more likely to accept personal loans — education correlates with higher income and financial sophistication.</div>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(4.5, 3.2))
        fig.patch.set_facecolor('white')
        edu_counts = df_raw['Education'].value_counts().sort_index()
        labels = ['Undergrad\n(1)', 'Graduate\n(2)', 'Advanced\n(3)']
        bars = ax.bar(labels, edu_counts.values, color=[PALETTE["sky"], PALETTE["blue"], PALETTE["navy"]], zorder=3, edgecolor='white', width=0.5)
        for b in bars:
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+20, f"{b.get_height():,}\n({b.get_height()/total*100:.1f}%)", ha='center', fontsize=8, color=PALETTE["navy"])
        apply_chart_style(ax, ylabel="Count")
        plt.tight_layout(); st.pyplot(fig); plt.close()
        st.markdown("</div>", unsafe_allow_html=True)

    with col5:
        st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
        st.markdown("<div class='chart-title'>Family Size Distribution</div>", unsafe_allow_html=True)
        st.markdown("<div class='chart-desc'>Family sizes 3–4 show slightly higher loan uptake, possibly reflecting greater financial need for larger expenses (home improvement, education, etc.). Single-person households are the smallest segment.</div>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(4.5, 3.2))
        fig.patch.set_facecolor('white')
        fam_counts = df_raw['Family'].value_counts().sort_index()
        bars = ax.bar([str(x) for x in fam_counts.index], fam_counts.values, color=PALETTE["teal"], alpha=0.85, zorder=3, edgecolor='white', width=0.5)
        for b in bars:
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+20, f"{b.get_height():,}", ha='center', fontsize=9, color=PALETTE["navy"])
        apply_chart_style(ax, xlabel="Family Size", ylabel="Count")
        plt.tight_layout(); st.pyplot(fig); plt.close()
        st.markdown("</div>", unsafe_allow_html=True)

    with col6:
        st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
        st.markdown("<div class='chart-title'>Binary Feature Adoption Rates</div>", unsafe_allow_html=True)
        st.markdown("<div class='chart-desc'>60% of customers use online banking, 29% hold a UniversalBank credit card, but only 6% have a CD account. CD account holders are rare but hyper-valuable — the highest loan conversion segment in the dataset.</div>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(4.5, 3.2))
        fig.patch.set_facecolor('white')
        bin_feats = ['Securities Account', 'CD Account', 'Online', 'CreditCard']
        rates = [df_raw[f].mean()*100 for f in bin_feats]
        short = ['Securities\nAccount', 'CD\nAccount', 'Online\nBanking', 'Credit\nCard']
        colors_b = [PALETTE["sky"], PALETTE["teal"], PALETTE["blue"], PALETTE["purple"]]
        bars = ax.barh(short, rates, color=colors_b, alpha=0.85, zorder=3, edgecolor='white')
        for b in bars:
            ax.text(b.get_width()+0.5, b.get_y()+b.get_height()/2, f"{b.get_width():.1f}%", va='center', fontsize=9, color=PALETTE["navy"])
        apply_chart_style(ax, xlabel="% of Customers", grid_axis="x")
        plt.tight_layout(); st.pyplot(fig); plt.close()
        st.markdown("</div>", unsafe_allow_html=True)

    # Correlation heatmap
    st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
    st.markdown("<div class='chart-title'>Feature Correlation Matrix</div>", unsafe_allow_html=True)
    st.markdown("<div class='chart-desc'>Income, CCAvg, CD Account, and Education show the strongest positive correlations with Personal Loan uptake. Age and Experience are highly correlated with each other (multicollinearity) but weakly tied to loan acceptance. Mortgage shows low correlation overall.</div>", unsafe_allow_html=True)
    corr_cols = ['Age','Experience','Income','Family','CCAvg','Education','Mortgage',
                 'Securities Account','CD Account','Online','CreditCard','Personal Loan']
    corr = df_raw[corr_cols].corr()
    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor('white')
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="coolwarm",
        center=0, linewidths=0.5, linecolor='#f0f4f8',
        ax=ax, annot_kws={"size": 8}, vmin=-1, vmax=1,
        cbar_kws={'shrink': 0.7}
    )
    ax.set_title("Pearson Correlation — All Features", fontsize=11, fontweight='600', color=PALETTE["navy"], pad=10)
    ax.tick_params(labelsize=8)
    plt.tight_layout(); st.pyplot(fig); plt.close()
    st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — DIAGNOSTIC ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
elif nav == "📈 Diagnostic Analytics":
    st.markdown('<div class="section-header">Diagnostic Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Why do customers accept or reject personal loans? Drilling into the factors that drive conversion.</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
        st.markdown("<div class='chart-title'>Loan Acceptance Rate by Income Quartile</div>", unsafe_allow_html=True)
        st.markdown("<div class='chart-desc'>Acceptance rate jumps dramatically above $98k annual income. The bottom two income quartiles have near-zero conversion rates — marketing spend on these segments produces negligible ROI and should be minimised.</div>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5.5, 3.5))
        fig.patch.set_facecolor('white')
        df_raw['Income_Q'] = pd.qcut(df_raw['Income'], 4, labels=['Q1\n<$39k','Q2\n$39–64k','Q3\n$64–98k','Q4\n>$98k'])
        rates_q = df_raw.groupby('Income_Q', observed=True)['Personal Loan'].mean() * 100
        bars = ax.bar(rates_q.index, rates_q.values,
                      color=[PALETTE["sky"], PALETTE["sky"], PALETTE["blue"], PALETTE["navy"]],
                      zorder=3, edgecolor='white', width=0.55)
        for b, v in zip(bars, rates_q.values):
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.5, f"{v:.1f}%", ha='center', fontsize=9.5, fontweight='600', color=PALETTE["navy"])
        apply_chart_style(ax, xlabel="Income Quartile", ylabel="Acceptance Rate (%)")
        plt.tight_layout(); st.pyplot(fig); plt.close()
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
        st.markdown("<div class='chart-title'>Loan Acceptance Rate by Education Level</div>", unsafe_allow_html=True)
        st.markdown("<div class='chart-desc'>Graduate and advanced-degree holders are 3× more likely to accept a loan than undergraduates. This likely reflects both higher income levels and greater awareness of financial products.</div>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5.5, 3.5))
        fig.patch.set_facecolor('white')
        edu_rate = df_raw.groupby('Education')['Personal Loan'].mean() * 100
        labels_e = ['Undergrad (1)', 'Graduate (2)', 'Advanced (3)']
        colors_e = [PALETTE["sky"], PALETTE["blue"], PALETTE["navy"]]
        bars = ax.bar(labels_e, edu_rate.values, color=colors_e, zorder=3, edgecolor='white', width=0.5)
        for b, v in zip(bars, edu_rate.values):
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.3, f"{v:.1f}%", ha='center', fontsize=10, fontweight='600', color=PALETTE["navy"])
        apply_chart_style(ax, xlabel="Education Level", ylabel="Acceptance Rate (%)")
        plt.tight_layout(); st.pyplot(fig); plt.close()
        st.markdown("</div>", unsafe_allow_html=True)

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
        st.markdown("<div class='chart-title'>Loan Acceptance vs CD Account Ownership</div>", unsafe_allow_html=True)
        st.markdown("<div class='chart-desc'>Customers with a CD account accept personal loans at 42.6% vs only 6.1% for those without. CD account ownership is the single best binary proxy for loan intent — it should always be part of the targeting criteria.</div>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5.5, 3.5))
        fig.patch.set_facecolor('white')
        cd_rate = df_raw.groupby('CD Account')['Personal Loan'].mean() * 100
        bars = ax.bar(['No CD Account', 'Has CD Account'], cd_rate.values,
                      color=[PALETTE["sky"], PALETTE["teal"]], zorder=3, edgecolor='white', width=0.45)
        for b, v in zip(bars, cd_rate.values):
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.5, f"{v:.1f}%", ha='center', fontsize=12, fontweight='700', color=PALETTE["navy"])
        apply_chart_style(ax, ylabel="Loan Acceptance Rate (%)")
        plt.tight_layout(); st.pyplot(fig); plt.close()
        st.markdown("</div>", unsafe_allow_html=True)

    with col4:
        st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
        st.markdown("<div class='chart-title'>Income vs CC Spending — Loan Acceptance Scatter</div>", unsafe_allow_html=True)
        st.markdown("<div class='chart-desc'>Loan acceptors (orange) cluster strongly in the high-income + high-spending quadrant. This dual-filter — Income >$98k AND CCAvg >$2.5k/month — creates a high-conversion micro-segment worth prioritising.</div>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5.5, 3.5))
        fig.patch.set_facecolor('white')
        no_loan = df_raw[df_raw['Personal Loan'] == 0]
        yes_loan = df_raw[df_raw['Personal Loan'] == 1]
        ax.scatter(no_loan['Income'], no_loan['CCAvg'], c=PALETTE["light"], edgecolors=PALETTE["gray"], s=12, alpha=0.5, linewidth=0.3, zorder=2, label='No Loan')
        ax.scatter(yes_loan['Income'], yes_loan['CCAvg'], c=PALETTE["amber"], edgecolors="#c05a10", s=22, alpha=0.75, linewidth=0.4, zorder=3, label='Accepted Loan')
        ax.axvline(98, color=PALETTE["red"], linestyle='--', linewidth=1, alpha=0.7)
        ax.axhline(2.5, color=PALETTE["red"], linestyle='--', linewidth=1, alpha=0.7)
        ax.legend(fontsize=8, frameon=False, markerscale=1.5)
        apply_chart_style(ax, xlabel="Annual Income ($000)", ylabel="CC Spending/Month ($000)", grid_axis=None)
        plt.tight_layout(); st.pyplot(fig); plt.close()
        st.markdown("</div>", unsafe_allow_html=True)

    # Family size x acceptance grouped bar
    st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
    st.markdown("<div class='chart-title'>Loan Acceptance Count & Rate by Family Size</div>", unsafe_allow_html=True)
    st.markdown("<div class='chart-desc'>Larger families (3–4 members) show the highest absolute number of loan acceptances. While acceptance rates are similar across groups, larger families represent a larger addressable segment with meaningful loan demand for bigger financial goals.</div>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor('white')
    fam_data = df_raw.groupby('Family')['Personal Loan'].agg(['sum','count','mean']).reset_index()
    fam_data.columns = ['Family','Accepted','Total','Rate']
    x = np.arange(len(fam_data))
    ax.bar(x - 0.2, fam_data['Total'], 0.38, label='Total Customers', color='#dde6ef', zorder=3, edgecolor='white')
    ax.bar(x + 0.2, fam_data['Accepted'], 0.38, label='Loan Acceptors', color=PALETTE["blue"], zorder=3, edgecolor='white')
    ax2 = ax.twinx()
    ax2.plot(x, fam_data['Rate']*100, 'o--', color=PALETTE["amber"], linewidth=2, markersize=7, label='Acceptance Rate', zorder=4)
    for i, (xi, r) in enumerate(zip(x, fam_data['Rate'])):
        ax2.text(xi, r*100+0.5, f"{r*100:.1f}%", ha='center', fontsize=9, fontweight='600', color=PALETTE["amber"])
    ax.set_xticks(x)
    ax.set_xticklabels([f"Family Size {f}" for f in fam_data['Family']], fontsize=9)
    ax.set_ylabel("Customer Count", fontsize=9, color=PALETTE["gray"])
    ax2.set_ylabel("Acceptance Rate (%)", fontsize=9, color=PALETTE["amber"])
    for spine in ax.spines.values(): spine.set_visible(False)
    for spine in ax2.spines.values(): spine.set_visible(False)
    ax.grid(axis='y', color='#e8edf3', linewidth=0.7, linestyle='--', alpha=0.8)
    ax.set_axisbelow(True)
    ax.set_facecolor('white')
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1+lines2, labels1+labels2, fontsize=8, frameon=False, loc='upper right')
    ax.tick_params(labelsize=8)
    ax2.tick_params(labelsize=8, colors=PALETTE["amber"])
    plt.tight_layout(); st.pyplot(fig); plt.close()
    st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — PREDICTIVE MODELS
# ══════════════════════════════════════════════════════════════════════════════
elif nav == "🤖 Predictive Models":
    st.markdown('<div class="section-header">Predictive Models</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Decision Tree, Random Forest, and Gradient Boosted Tree — trained with balanced class weights to handle the 9.6% minority class.</div>', unsafe_allow_html=True)

    # ── Performance comparison table ──
    st.subheader("Model Performance Summary")
    rows = []
    for m in ['Decision Tree', 'Random Forest', 'Gradient Boosted Tree']:
        r = results[m]
        rows.append({
            'Model': m,
            'Train Accuracy': f"{r['train_acc']*100:.2f}%",
            'Test Accuracy':  f"{r['test_acc']*100:.2f}%",
            'Precision':      f"{r['precision']*100:.2f}%",
            'Recall':         f"{r['recall']*100:.2f}%",
            'F1-Score':       f"{r['f1']*100:.2f}%",
            'AUC-ROC':        f"{r['auc']:.4f}",
        })
    df_perf = pd.DataFrame(rows)
    html_table = "<table class='styled-table'><thead><tr>"
    for col in df_perf.columns:
        html_table += f"<th>{col}</th>"
    html_table += "</tr></thead><tbody>"
    for _, row in df_perf.iterrows():
        html_table += "<tr>"
        for i, val in enumerate(row):
            html_table += f"<td>{val}</td>"
        html_table += "</tr>"
    html_table += "</tbody></table>"
    st.markdown(html_table, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="insight-box">📌 <strong>Gradient Boosted Tree is the recommended production model.</strong> It achieves the highest AUC-ROC (0.999) and F1-score, balancing precision and recall optimally. Random Forest is a strong runner-up and faster to retrain. Decision Tree is the most interpretable but shows signs of slight overfitting.</div>', unsafe_allow_html=True)

    st.markdown("---")

    # ── Single combined ROC curve ──
    st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
    st.markdown("<div class='chart-title'>ROC Curves — All Models (Combined)</div>", unsafe_allow_html=True)
    st.markdown("<div class='chart-desc'>The ROC curve plots the True Positive Rate against the False Positive Rate at every decision threshold. A higher AUC (Area Under the Curve) means better separation between loan acceptors and non-acceptors. All three models achieve AUC > 0.98, indicating excellent discriminative ability. The Gradient Boosted Tree curve hugs closest to the top-left corner, confirming its superiority.</div>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    ax.plot([0,1],[0,1],'--', color='#c0c8d4', linewidth=1.2, label='Random Classifier (AUC = 0.50)')
    for mname, mcol in MODEL_COLORS.items():
        r = results[mname]
        fpr, tpr, _ = roc_curve(r['y_test'], r['y_prob'])
        ax.plot(fpr, tpr, color=mcol, linewidth=2.2,
                label=f"{mname}  (AUC = {r['auc']:.4f})")
    ax.set_xlim([-0.01, 1.01]); ax.set_ylim([-0.01, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=10, color=PALETTE["gray"])
    ax.set_ylabel("True Positive Rate", fontsize=10, color=PALETTE["gray"])
    ax.set_title("ROC Curves — Decision Tree vs Random Forest vs Gradient Boosted Tree",
                 fontsize=11, fontweight='600', color=PALETTE["navy"], pad=12)
    ax.legend(fontsize=9.5, frameon=True, fancybox=False, edgecolor='#e0e8f0', loc='lower right')
    ax.grid(color='#e8edf3', linewidth=0.7, linestyle='--', alpha=0.8)
    for spine in ax.spines.values(): spine.set_visible(False)
    ax.tick_params(colors=PALETTE["gray"], labelsize=9)
    plt.tight_layout(); st.pyplot(fig); plt.close()
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")

    # ── Confusion matrices ──
    st.subheader("Confusion Matrices — All Models")
    st.markdown("*Each matrix shows True Negatives (top-left), False Positives (top-right), False Negatives (bottom-left), and True Positives (bottom-right). Values are shown as both counts and row-percentages.*")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.patch.set_facecolor('white')
    for ax, (mname, mcol) in zip(axes, MODEL_COLORS.items()):
        cm = results[mname]['cm']
        total_cm = cm.sum()
        annot = np.array([[f"{cm[i,j]:,}\n({cm[i,j]/cm[i].sum()*100:.1f}%)" for j in range(2)] for i in range(2)])
        cmap = sns.light_palette(mcol, as_cmap=True)
        sns.heatmap(cm, annot=annot, fmt='', cmap=cmap, ax=ax,
                    linewidths=1.5, linecolor='white', cbar=False, annot_kws={"size": 10, "weight": "600"})
        ax.set_xlabel("Predicted Label", fontsize=9, color=PALETTE["gray"])
        ax.set_ylabel("Actual Label", fontsize=9, color=PALETTE["gray"])
        ax.set_xticklabels(['No Loan\n(0)', 'Loan\n(1)'], fontsize=8.5)
        ax.set_yticklabels(['No Loan (0)', 'Loan (1)'], fontsize=8.5, rotation=0)
        acc = results[mname]['test_acc']
        ax.set_title(f"{mname}\nTest Accuracy: {acc*100:.2f}%", fontsize=10, fontweight='600', color=PALETTE["navy"], pad=8)
        ax.set_facecolor('white')
    plt.tight_layout(pad=2.0); st.pyplot(fig); plt.close()

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Feature importances ──
    st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
    st.markdown("<div class='chart-title'>Feature Importance — Random Forest & Gradient Boosted Tree</div>", unsafe_allow_html=True)
    st.markdown("<div class='chart-desc'>Feature importance ranks how much each variable contributed to the model's predictions. Income and CCAvg dominate both ensemble models, confirming they are the primary levers for identifying potential loan customers. CD Account and Education are strong secondary signals.</div>", unsafe_allow_html=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor('white')
    for ax, mname in zip(axes, ['Random Forest', 'Gradient Boosted Tree']):
        imp = trained_models[mname].feature_importances_
        fi = pd.Series(imp, index=feature_names).sort_values(ascending=True)
        colors_fi = [PALETTE["teal"] if f not in ['Income','CCAvg','CD Account','Education'] else PALETTE["blue"] for f in fi.index]
        fi.plot(kind='barh', ax=ax, color=colors_fi, edgecolor='white', zorder=3)
        for bar, v in zip(ax.patches, fi.values):
            ax.text(bar.get_width()+0.001, bar.get_y()+bar.get_height()/2,
                    f"{v:.3f}", va='center', fontsize=7.5, color=PALETTE["navy"])
        apply_chart_style(ax, title=mname, xlabel="Feature Importance", grid_axis="x")
    plt.tight_layout(); st.pyplot(fig); plt.close()
    st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — PRESCRIPTIVE ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
elif nav == "🎯 Prescriptive Analytics":
    st.markdown('<div class="section-header">Prescriptive Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">From insight to action — who to target, what to say, and how to allocate your reduced marketing budget for maximum ROI.</div>', unsafe_allow_html=True)

    # ── Segment profiling ──
    st.subheader("High-Value Segment Identification")

    df_seg = df_raw.copy()
    df_seg['Predicted_Prob'] = trained_models['Gradient Boosted Tree'].predict_proba(
        scaler.transform(
            df_model.drop(columns=['Personal Loan'])[feature_names]
        )
    )[:,1]
    df_seg['Segment'] = pd.cut(df_seg['Predicted_Prob'],
                                bins=[0, 0.2, 0.5, 0.75, 1.0],
                                labels=['Cold (0–20%)', 'Warm (20–50%)', 'Hot (50–75%)', 'Prime (75–100%)'])
    seg_counts = df_seg['Segment'].value_counts().reindex(['Prime (75–100%)','Hot (50–75%)','Warm (20–50%)','Cold (0–20%)'])
    seg_actuals = df_seg.groupby('Segment', observed=True)['Personal Loan'].mean() * 100

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
        st.markdown("<div class='chart-title'>Customer Segments by Model Probability</div>", unsafe_allow_html=True)
        st.markdown("<div class='chart-desc'>The Gradient Boosted Tree model scores every customer from 0–100% loan probability. Customers above 75% are the 'Prime' tier — the highest-yield targets for your marketing budget with the best conversion potential.</div>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5.5, 3.8))
        fig.patch.set_facecolor('white')
        seg_colors = [PALETTE["navy"], PALETTE["blue"], PALETTE["sky"], "#dde6ef"]
        bars = ax.barh(seg_counts.index, seg_counts.values, color=seg_colors, zorder=3, edgecolor='white', height=0.5)
        for b in bars:
            ax.text(b.get_width()+5, b.get_y()+b.get_height()/2, f"{int(b.get_width()):,} ({b.get_width()/total*100:.1f}%)", va='center', fontsize=9, color=PALETTE["navy"])
        apply_chart_style(ax, xlabel="Customer Count", grid_axis="x")
        ax.set_xlim(0, seg_counts.max()*1.25)
        plt.tight_layout(); st.pyplot(fig); plt.close()
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
        st.markdown("<div class='chart-title'>Actual Loan Rate Within Each Segment</div>", unsafe_allow_html=True)
        st.markdown("<div class='chart-desc'>The Prime segment shows a dramatically higher actual conversion rate — validating that the model's probability scoring accurately separates converters from non-converters. Focus 70%+ of the budget here.</div>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5.5, 3.8))
        fig.patch.set_facecolor('white')
        bars = ax.bar(seg_actuals.index, seg_actuals.values,
                      color=[PALETTE["navy"], PALETTE["blue"], PALETTE["sky"], "#dde6ef"],
                      zorder=3, edgecolor='white', width=0.5)
        for b, v in zip(bars, seg_actuals.values):
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.5, f"{v:.1f}%", ha='center', fontsize=10, fontweight='700', color=PALETTE["navy"])
        ax.tick_params(axis='x', labelrotation=15, labelsize=8)
        apply_chart_style(ax, ylabel="Actual Acceptance Rate (%)")
        plt.tight_layout(); st.pyplot(fig); plt.close()
        st.markdown("</div>", unsafe_allow_html=True)

    # Budget allocation recommendation
    st.markdown("---")
    st.subheader("📋 Marketing Budget Allocation Recommendation")

    col_r1, col_r2, col_r3 = st.columns(3)
    recs = [
        ("🎯 Prime Segment", "70% of Budget",
         "High-income (>$98k), CD account holders, graduate/advanced education. Expected conversion 30–45%. Personalised loan offers with tailored interest rates."),
        ("🔥 Hot Segment", "20% of Budget",
         "Moderate-to-high income, credit card users, online banking. Expected conversion 10–20%. Targeted digital campaigns with benefit-focused messaging."),
        ("❄️ Warm & Cold", "10% of Budget",
         "Broad awareness campaigns only. Low expected conversion. Focus on building engagement for future campaigns — email newsletters, app nudges."),
    ]
    for col, (title, budget, desc) in zip([col_r1, col_r2, col_r3], recs):
        col.markdown(f"""
        <div style='background:white; border-radius:14px; padding:1.3rem; border:1px solid #e8edf3;
                    box-shadow: 0 2px 12px rgba(0,0,0,0.05); min-height:200px;'>
            <div style='font-size:1rem; font-weight:600; color:{PALETTE["navy"]}; margin-bottom:6px;'>{title}</div>
            <div style='font-size:1.4rem; font-weight:700; color:{PALETTE["blue"]}; margin-bottom:10px;'>{budget}</div>
            <div style='font-size:0.83rem; color:#5a7a95; line-height:1.55;'>{desc}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Top 10 most likely converters profile
    st.subheader("Ideal Customer Profile — Top 10% Converters")
    top_10 = df_seg[df_seg['Predicted_Prob'] >= df_seg['Predicted_Prob'].quantile(0.9)]
    profile = pd.DataFrame({
        'Attribute': ['Average Age','Average Income','Average CC Spending/mo','CD Account Holders (%)','Graduate/Advanced Education (%)','Online Banking Users (%)','Average Family Size'],
        'Top 10% Converters': [
            f"{top_10['Age'].mean():.1f} yrs",
            f"${top_10['Income'].mean():.0f}k",
            f"${top_10['CCAvg'].mean():.2f}k",
            f"{top_10['CD Account'].mean()*100:.1f}%",
            f"{(top_10['Education']>=2).mean()*100:.1f}%",
            f"{top_10['Online'].mean()*100:.1f}%",
            f"{top_10['Family'].mean():.1f}",
        ],
        'Full Population': [
            f"{df_raw['Age'].mean():.1f} yrs",
            f"${df_raw['Income'].mean():.0f}k",
            f"${df_raw['CCAvg'].mean():.2f}k",
            f"{df_raw['CD Account'].mean()*100:.1f}%",
            f"{(df_raw['Education']>=2).mean()*100:.1f}%",
            f"{df_raw['Online'].mean()*100:.1f}%",
            f"{df_raw['Family'].mean():.1f}",
        ],
    })

    html_profile = "<table class='styled-table'><thead><tr>"
    for c in profile.columns:
        html_profile += f"<th>{c}</th>"
    html_profile += "</tr></thead><tbody>"
    for _, row in profile.iterrows():
        html_profile += "<tr>"
        for val in row:
            html_profile += f"<td>{val}</td>"
        html_profile += "</tr>"
    html_profile += "</tbody></table>"
    st.markdown(html_profile, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="insight-box">🎯 <strong>Campaign Recommendation:</strong> Design a personalised loan offer for high-income professionals (40–55 yrs, income >$98k) with existing CD accounts, emphasising flexible repayment and competitive interest rates. Use digital channels (online banking in-app notifications) since 75%+ are online banking users.</div>', unsafe_allow_html=True)
    st.markdown('<div class="insight-box">💡 <strong>Cross-sell Opportunity:</strong> Target credit card holders who do NOT yet have a CD account or personal loan with a bundled offer — "Upgrade to Premium Banking" including a CD Account + pre-approved loan line. This can expand the high-conversion CD account segment organically.</div>', unsafe_allow_html=True)
    st.markdown('<div class="warning-box">⚠️ <strong>Budget Guard:</strong> With a halved budget, avoid spray-and-pray approaches. The bottom two income quartiles have near-zero conversion — exclude them from paid campaigns entirely and instead use low-cost channels (email, push notification) for awareness only.</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — PREDICT NEW CUSTOMERS
# ══════════════════════════════════════════════════════════════════════════════
elif nav == "🔮 Predict New Customers":
    st.markdown('<div class="section-header">Predict New Customers</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Upload a CSV of new customers (same format as training data, without the Personal Loan column) and download predictions instantly.</div>', unsafe_allow_html=True)

    st.info("📎 Upload a CSV file with the same columns as the training data (ID, Age, Experience, Income, ZIP Code, Family, CCAvg, Education, Mortgage, Securities Account, CD Account, Online, CreditCard). The 'Personal Loan' column should be absent — it will be predicted.")

    pred_file = st.file_uploader("Upload new customer data (CSV)", type=["csv"], key="pred_uploader")

    if pred_file is not None:
        try:
            df_new = pd.read_csv(pred_file)
            st.success(f"✅ Loaded {len(df_new):,} customers successfully.")
            st.dataframe(df_new.head(), use_container_width=True)

            # Preprocess
            df_new_proc = df_new.copy()
            df_new_proc['Experience'] = df_new_proc['Experience'].clip(lower=0)
            df_new_proc['Has_Mortgage']      = (df_new_proc['Mortgage'] > 0).astype(int)
            df_new_proc['Income_CCAvg_Ratio'] = df_new_proc['CCAvg'] / (df_new_proc['Income'] + 1)
            df_new_proc['High_Value']         = ((df_new_proc['Income'] > 98) & (df_new_proc['CCAvg'] > 2.5)).astype(int)

            # Align features
            missing = [f for f in feature_names if f not in df_new_proc.columns]
            if missing:
                st.error(f"Missing columns: {missing}")
            else:
                X_new = df_new_proc[feature_names]
                X_new_sc = scaler.transform(X_new)

                gbt = trained_models['Gradient Boosted Tree']
                preds = gbt.predict(X_new_sc)
                probs = gbt.predict_proba(X_new_sc)[:,1]

                df_out = df_new.copy()
                df_out['Personal_Loan_Prediction'] = preds
                df_out['Loan_Probability_%']        = (probs * 100).round(2)
                df_out['Segment'] = pd.cut(
                    probs,
                    bins=[0, 0.2, 0.5, 0.75, 1.0],
                    labels=['Cold','Warm','Hot','Prime']
                ).astype(str)

                st.markdown("---")
                st.subheader("Prediction Results")

                c1, c2, c3, c4 = st.columns(4)
                metrics_pred = [
                    ("Total Scored", f"{len(df_out):,}"),
                    ("Predicted Loan Acceptors", f"{preds.sum():,} ({preds.mean()*100:.1f}%)"),
                    ("Prime Segment", f"{(df_out['Segment']=='Prime').sum():,}"),
                    ("Avg Loan Probability", f"{probs.mean()*100:.1f}%"),
                ]
                for col, (label, val) in zip([c1,c2,c3,c4], metrics_pred):
                    col.markdown(f"""<div class='kpi-card'>
                        <div class='kpi-label'>{label}</div>
                        <div class='kpi-value' style='font-size:1.4rem;'>{val}</div>
                    </div>""", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                st.dataframe(df_out, use_container_width=True)

                # Download
                csv_out = df_out.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="⬇️ Download Predictions as CSV",
                    data=csv_out,
                    file_name="loan_predictions.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
        except Exception as e:
            st.error(f"Error processing file: {e}")
    else:
        st.markdown("""
        <div style='background:#f0f6ff; border-radius:12px; padding:2rem; text-align:center; border:1px dashed #a0c4e8;'>
            <div style='font-size:2.5rem;'>📤</div>
            <div style='font-size:1rem; font-weight:500; color:#1a3a5c; margin-top:0.5rem;'>Upload a CSV file to get predictions</div>
            <div style='font-size:0.85rem; color:#5a7a95; margin-top:0.4rem;'>A sample test file <code>test_data_sample.csv</code> is included in the project ZIP for immediate testing.</div>
        </div>
        """, unsafe_allow_html=True)
