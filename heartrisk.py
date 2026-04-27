"""
Heart Disease Risk Prediction Model
Uses: Age, Cholesterol, Blood Pressure + medical features
Algorithm: Logistic Regression (better for binary classification like risk prediction)
"""
 
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, accuracy_score)
import warnings
warnings.filterwarnings('ignore')
 
np.random.seed(42)
 
# ─────────────────────────────────────────────
# 1. SYNTHETIC DATASET GENERATION
# ─────────────────────────────────────────────
n = 1000
 
age         = np.random.randint(30, 80, n)
cholesterol = np.random.randint(150, 350, n)     # mg/dL
bp_systolic = np.random.randint(90, 180, n)      # mmHg
bp_diastolic= np.random.randint(60, 110, n)      # mmHg
max_hr      = 220 - age + np.random.randint(-20, 20, n)  # max heart rate
resting_hr  = np.random.randint(50, 100, n)      # BPM
blood_sugar = np.random.randint(70, 200, n)      # mg/dL (fasting)
bmi         = np.round(np.random.uniform(18, 42, n), 1)
smoking     = np.random.choice([0, 1], n, p=[0.65, 0.35])
diabetes    = np.random.choice([0, 1], n, p=[0.80, 0.20])
family_hist = np.random.choice([0, 1], n, p=[0.70, 0.30])
exercise    = np.random.choice([0, 1, 2], n, p=[0.30, 0.45, 0.25])  # 0=none,1=moderate,2=active
chest_pain  = np.random.choice([0, 1, 2, 3], n)  # 0=none to 3=severe
ecg_result  = np.random.choice([0, 1, 2], n, p=[0.55, 0.30, 0.15])  # 0=normal,1=ST-T,2=LVH
 
# Risk score based on medical logic
risk_score = (
    0.04 * age +
    0.02 * (cholesterol - 200) +
    0.03 * (bp_systolic - 120) +
    0.01 * (bp_diastolic - 80) +
    0.02 * (blood_sugar - 100) +
    0.05 * bmi +
    0.8  * smoking +
    1.2  * diabetes +
    0.6  * family_hist +
    -0.5 * exercise +
    0.9  * chest_pain +
    0.5  * ecg_result +
    np.random.normal(0, 1, n)
)
 
# Binary target: 1 = high risk, 0 = low risk
threshold = np.percentile(risk_score, 55)
heart_risk = (risk_score > threshold).astype(int)
 
df = pd.DataFrame({
    'Age':              age,
    'Cholesterol':      cholesterol,
    'BP_Systolic':      bp_systolic,
    'BP_Diastolic':     bp_diastolic,
    'Max_HeartRate':    max_hr,
    'Resting_HeartRate':resting_hr,
    'Blood_Sugar':      blood_sugar,
    'BMI':              bmi,
    'Smoking':          smoking,
    'Diabetes':         diabetes,
    'Family_History':   family_hist,
    'Exercise_Level':   exercise,
    'Chest_Pain':       chest_pain,
    'ECG_Result':       ecg_result,
    'Heart_Risk':       heart_risk
})
 
print("=" * 55)
print("   HEART RISK PREDICTION — MODEL REPORT")
print("=" * 55)
print(f"\nDataset: {n} patients  |  Features: {df.shape[1]-1}")
print(f"High Risk: {heart_risk.sum()} ({heart_risk.mean()*100:.1f}%)  |  Low Risk: {(1-heart_risk).sum()} ({(1-heart_risk).mean()*100:.1f}%)")
 
# ─────────────────────────────────────────────
# 2. TRAIN / TEST SPLIT & SCALING
# ─────────────────────────────────────────────
X = df.drop('Heart_Risk', axis=1)
y = df['Heart_Risk']
 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
 
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)
 
# ─────────────────────────────────────────────
# 3. LOGISTIC REGRESSION MODEL
# ─────────────────────────────────────────────
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_s, y_train)
 
y_pred      = model.predict(X_test_s)
y_prob      = model.predict_proba(X_test_s)[:, 1]
accuracy    = accuracy_score(y_test, y_pred)
auc         = roc_auc_score(y_test, y_prob)
 
print(f"\nModel: Logistic Regression")
print(f"Accuracy  : {accuracy*100:.1f}%")
print(f"AUC-ROC   : {auc:.3f}")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Low Risk', 'High Risk']))
 
# Feature importance
coef_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_[0]
}).sort_values('Coefficient', key=abs, ascending=False)
 
# ─────────────────────────────────────────────
# 4. VISUALIZATIONS
# ─────────────────────────────────────────────
fig = plt.figure(figsize=(18, 14))
fig.patch.set_facecolor('#0f0f1a')
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.42, wspace=0.38)
 
DARK    = '#0f0f1a'
CARD    = '#1a1a2e'
ACCENT1 = '#e94560'
ACCENT2 = '#0f3460'
ACCENT3 = '#16213e'
TEXT    = '#e0e0e0'
GREEN   = '#00d4aa'
YELLOW  = '#f5a623'
 
plt.rcParams.update({
    'text.color': TEXT,
    'axes.labelcolor': TEXT,
    'xtick.color': TEXT,
    'ytick.color': TEXT,
    'axes.facecolor': CARD,
    'figure.facecolor': DARK,
    'grid.color': '#2a2a3e',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.edgecolor': '#2a2a3e',
})
 
# ── Title ──────────────────────────────────────
ax_title = fig.add_subplot(gs[0, :])
ax_title.set_facecolor(CARD)
ax_title.text(0.5, 0.65, '❤  Heart Disease Risk Prediction', ha='center', va='center',
              fontsize=22, fontweight='bold', color=ACCENT1, transform=ax_title.transAxes)
ax_title.text(0.5, 0.22,
              f'Logistic Regression  ·  {n} patients  ·  {X.shape[1]} features  ·  Accuracy {accuracy*100:.1f}%  ·  AUC {auc:.3f}',
              ha='center', va='center', fontsize=11, color=TEXT, alpha=0.7, transform=ax_title.transAxes)
ax_title.axis('off')
 
# ── 1. Feature Importance ─────────────────────
ax1 = fig.add_subplot(gs[1, 0])
colors = [ACCENT1 if c > 0 else GREEN for c in coef_df['Coefficient']]
bars = ax1.barh(coef_df['Feature'], coef_df['Coefficient'], color=colors, edgecolor='none', height=0.6)
ax1.axvline(0, color=TEXT, linewidth=0.5, alpha=0.4)
ax1.set_title('Feature Coefficients', color=TEXT, fontsize=11, pad=10)
ax1.set_xlabel('Coefficient Value', fontsize=9)
ax1.tick_params(labelsize=8)
 
# ── 2. Confusion Matrix ───────────────────────
ax2 = fig.add_subplot(gs[1, 1])
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', ax=ax2,
            cmap=sns.color_palette(['#16213e', '#0f3460', '#e94560', '#ff6b81'], as_cmap=True),
            xticklabels=['Low Risk', 'High Risk'],
            yticklabels=['Low Risk', 'High Risk'],
            linewidths=2, linecolor=DARK, cbar=False,
            annot_kws={'size': 14, 'weight': 'bold', 'color': 'white'})
ax2.set_title('Confusion Matrix', color=TEXT, fontsize=11, pad=10)
ax2.set_xlabel('Predicted', fontsize=9)
ax2.set_ylabel('Actual', fontsize=9)
ax2.tick_params(labelsize=9)
 
# ── 3. ROC Curve ─────────────────────────────
ax3 = fig.add_subplot(gs[1, 2])
fpr, tpr, _ = roc_curve(y_test, y_prob)
ax3.plot(fpr, tpr, color=ACCENT1, lw=2.5, label=f'AUC = {auc:.3f}')
ax3.fill_between(fpr, tpr, alpha=0.12, color=ACCENT1)
ax3.plot([0,1],[0,1], '--', color=TEXT, alpha=0.3, lw=1)
ax3.set_title('ROC Curve', color=TEXT, fontsize=11, pad=10)
ax3.set_xlabel('False Positive Rate', fontsize=9)
ax3.set_ylabel('True Positive Rate', fontsize=9)
ax3.legend(fontsize=9, facecolor=CARD, edgecolor='none', labelcolor=TEXT)
ax3.tick_params(labelsize=8)
ax3.grid(True, alpha=0.2)
 
# ── 4. Age vs Cholesterol scatter ─────────────
ax4 = fig.add_subplot(gs[2, 0])
low  = df[df['Heart_Risk'] == 0]
high = df[df['Heart_Risk'] == 1]
ax4.scatter(low['Age'],  low['Cholesterol'],  alpha=0.35, s=12, color=GREEN,   label='Low Risk', edgecolors='none')
ax4.scatter(high['Age'], high['Cholesterol'], alpha=0.45, s=12, color=ACCENT1, label='High Risk', edgecolors='none')
ax4.set_title('Age vs Cholesterol', color=TEXT, fontsize=11, pad=10)
ax4.set_xlabel('Age', fontsize=9)
ax4.set_ylabel('Cholesterol (mg/dL)', fontsize=9)
ax4.legend(fontsize=8, facecolor=CARD, edgecolor='none', labelcolor=TEXT)
ax4.tick_params(labelsize=8)
ax4.grid(True, alpha=0.15)
 
# ── 5. BP Systolic distribution ───────────────
ax5 = fig.add_subplot(gs[2, 1])
ax5.hist(low['BP_Systolic'],  bins=30, alpha=0.6, color=GREEN,   label='Low Risk',  edgecolor='none', density=True)
ax5.hist(high['BP_Systolic'], bins=30, alpha=0.6, color=ACCENT1, label='High Risk', edgecolor='none', density=True)
ax5.set_title('Blood Pressure Distribution', color=TEXT, fontsize=11, pad=10)
ax5.set_xlabel('Systolic BP (mmHg)', fontsize=9)
ax5.set_ylabel('Density', fontsize=9)
ax5.legend(fontsize=8, facecolor=CARD, edgecolor='none', labelcolor=TEXT)
ax5.tick_params(labelsize=8)
ax5.grid(True, alpha=0.15)
 
# ── 6. Risk Probability histogram ─────────────
ax6 = fig.add_subplot(gs[2, 2])
ax6.hist(y_prob[y_test==0], bins=25, alpha=0.65, color=GREEN,   label='Low Risk',  density=True, edgecolor='none')
ax6.hist(y_prob[y_test==1], bins=25, alpha=0.65, color=ACCENT1, label='High Risk', density=True, edgecolor='none')
ax6.axvline(0.5, color=YELLOW, lw=1.5, linestyle='--', alpha=0.8, label='Threshold 0.5')
ax6.set_title('Predicted Risk Probability', color=TEXT, fontsize=11, pad=10)
ax6.set_xlabel('Probability of Heart Risk', fontsize=9)
ax6.set_ylabel('Density', fontsize=9)
ax6.legend(fontsize=8, facecolor=CARD, edgecolor='none', labelcolor=TEXT)
ax6.tick_params(labelsize=8)
ax6.grid(True, alpha=0.15)
 
plt.savefig('/home/claude/heart_risk_report.png', dpi=150, bbox_inches='tight',
            facecolor=DARK, edgecolor='none')
plt.close()
print("\n✓ Visualization saved.")
 
# ─────────────────────────────────────────────
# 5. PREDICTION FUNCTION DEMO
# ─────────────────────────────────────────────
def predict_risk(age, cholesterol, bp_sys, bp_dia, max_hr, resting_hr,
                 blood_sugar, bmi, smoking, diabetes, family_hist,
                 exercise, chest_pain, ecg):
    """Predict heart disease risk for a new patient."""
    data = np.array([[age, cholesterol, bp_sys, bp_dia, max_hr, resting_hr,
                      blood_sugar, bmi, smoking, diabetes, family_hist,
                      exercise, chest_pain, ecg]])
    data_s = scaler.transform(data)
    prob   = model.predict_proba(data_s)[0][1]
    label  = "⚠ HIGH RISK" if prob >= 0.5 else "✓ LOW RISK"
    return prob, label
 
print("\n" + "─" * 55)
print("EXAMPLE PREDICTIONS")
print("─" * 55)
 
examples = [
    ("Patient A (high-risk profile)",
     62, 280, 165, 100, 145, 85, 160, 32, 1, 1, 1, 0, 3, 2),
    ("Patient B (low-risk profile)",
     35, 180, 115, 75, 170, 65, 90, 22, 0, 0, 0, 2, 0, 0),
    ("Patient C (borderline)",
     50, 220, 138, 88, 155, 78, 115, 27, 1, 0, 1, 1, 1, 1),
]
 
for name, *params in examples:
    prob, label = predict_risk(*params)
    bar_len = int(prob * 30)
    bar = '█' * bar_len + '░' * (30 - bar_len)
    print(f"\n{name}")
    print(f"  Risk: {label}  ({prob*100:.1f}%)")
    print(f"  [{bar}]")
 
print("\n" + "=" * 55)
print("Model training complete!")
print("=" * 55)