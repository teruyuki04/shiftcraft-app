
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, brier_score_loss
import io

st.set_page_config(page_title="Shiftcraft: Success Probability App", layout="centered")
st.title("Shiftcraft｜課題仮説×実装×スケール化 → 成功確率アプリ")

with st.expander("このアプリについて（クリックで展開）", expanded=False):
    st.markdown("""
    - 入力：**H=課題仮説(0–30)**、**I=初期実装(0–5)**、**S=スケール化(0–5)**  
    - 学習：CSVをアップロードすると、モデル（ロジスティック回帰＋確率校正）を学習します。  
    - 予測：H/I/Sから**成功確率**を返します。  
    - ラベル：成功= A+B≥8（A=財務0–5, B=新市場0–5）を1、それ以外は0として学習。  
    """)

st.header("① データを用意（CSVアップロード or サンプル使用）")
st.markdown("必要列：company, H, I, S, A, B  （A+B>=8 を成功=1として学習します）")

sample_btn = st.button("サンプルデータを読み込む（Octopus / Udacity / Airbnb）")
uploaded = st.file_uploader("CSVをアップロード（UTF-8）", type=["csv"])

if sample_btn and not uploaded:
    sample = pd.DataFrame([
        {"company":"Octopus Energy","H":30,"I":5,"S":5,"A":5,"B":5},
        {"company":"Udacity","H":23,"I":3,"S":2,"A":3,"B":3},
        {"company":"Airbnb","H":30,"I":4,"S":5,"A":5,"B":5},
    ])
    df = sample.copy()
elif uploaded is not None:
    df = pd.read_csv(uploaded)
else:
    df = None

if df is not None:
    st.dataframe(df)

    # Build label (success=1 if A+B>=8)
    if not {'company','H','I','S','A','B'}.issubset(df.columns):
        st.error("必要列が不足しています。company, H, I, S, A, B を含めてください。")
        st.stop()

    data = df.copy()
    data['label_success'] = ((data['A'] + data['B']) >= 8).astype(int)

   # data: アップロードCSVのコピー（既にあります）
# ラベル（成功=1）も既に作成済みの想定: data['label_success']

# 正規化した特徴量（学習用）
data["h"]   = data["H"] / 30.0
data["i"]   = data["I"] / 5.0
data["s"]   = data["S"] / 5.0
data["h_i"] = data["h"] * data["i"]
data["i_s"] = data["i"] * data["s"]

X = data[["h","i","s","h_i","i_s"]].copy()
y = data["label_success"].astype(int)


    # Because dataset may be tiny, fall back to simple fit without split
    if y.nunique() < 2:
        st.warning("成功/非成功の両方のデータが必要です。現状は片側のみのため、評価はスキップして学習のみ行います。")

 # ===== 学習（ここから） =====
# モデル定義（未定義なら定義） ← try: の直前に置く
if 'model' not in locals():
    model = LogisticRegression(solver="liblinear", max_iter=1000, random_state=42)

try:
    model.fit(X[['h','i','s','h_i','i_s']], y)

    # 校正（小規模データなので prefit）
    calibrated = CalibratedClassifierCV(model, cv="prefit", method="isotonic")
    calibrated.fit(X[['h','i','s','h_i','i_s']], y)

    # セッションに保存＆完了表示
    st.session_state["calibrated"] = calibrated
    st.success("モデル学習＋確率校正 完了")

except Exception as e:
    st.error(f"学習中にエラー: {e}")
    st.stop()
# ===== 学習（ここまで） =====

# 参考メトリクス（任意：失敗しても落とさない）
if y.nunique() == 2 and len(y) >= 4:
    try:
        prob = calibrated.predict_proba(X[['h','i','s','h_i','i_s']])[:,1]
        auc = roc_auc_score(y, prob)
        brier = brier_score_loss(y, prob)
        st.write(f"AUC: {auc:.3f} | Brier: {brier:.3f}")
    except Exception:
        pass



st.header("② 予測（H/I/Sを入力）")

# 入力
H_in = st.number_input("H（課題仮説: 0-30）", min_value=0, max_value=30, value=24, step=1)
I_in = st.number_input("I（初期実装: 0-5）", min_value=0, max_value=5, value=3, step=1)
S_in = st.number_input("S（スケール化: 0-5）", min_value=0, max_value=5, value=0, step=1)

# 正規化（先に変数に入れておくと NameError を回避しやすい）
Hn = H_in / 30.0
In_ = I_in / 5.0
Sn = S_in / 5.0

# 予測用の特徴量（学習時と同じ列名: 'h','i','s','h_i','i_s'）
Xq = pd.DataFrame([{
    "h": Hn,
    "i": In_,
    "s": Sn,
    "h_i": Hn * In_,
    "i_s": In_ * Sn,
}])[["h", "i", "s", "h_i", "i_s"]]


# 学習済みモデルを session_state から取得
calibrated = st.session_state.get("calibrated", None)

if calibrated is None:
    st.warning("まだモデルが学習されていません。ページ上部でCSVをアップロードして学習を実行してください。")
else:
    # ← ここだけ列名を明示
    P = float(calibrated.predict_proba(Xq[["h", "i", "s", "h_i", "i_s"]])[0, 1])  # 0.0〜1.0
    # （以下は上流モード補正 → st.metric 表示、の流れでOK）


    use_early_mode = st.sidebar.checkbox("上流モード（H重視の安全補正を有効化）", value=True)

    if use_early_mode:
        H_GATE = 12
        H_HARD_FLOOR = 8
        CAP_STRONG_IS = 0.50
        CAP_ZERO_H   = 0.35
        explain_rules = []

        if H_in == 0:
            old = P; P = min(P, CAP_ZERO_H)
            if P < old: explain_rules.append(f"H=0のため {old*100:.1f}%→{P*100:.1f}%に補正")

        if H_in < H_GATE and I_in >= 4 and S_in >= 4:
            old = P; P = min(P, CAP_STRONG_IS)
            if P < old: explain_rules.append(f"H<{H_GATE} かつ I/S高スコアのため {old*100:.1f}%→{P*100:.1f}%に補正")

        if H_in < H_HARD_FLOOR:
            old = P; P = min(P, CAP_ZERO_H)
            if P < old: explain_rules.append(f"H<{H_HARD_FLOOR} のため {old*100:.1f}%→{P*100:.1f}%に補正")

    st.metric(label="成功確率（校正後）", value=f"{P*100:.1f}%")

    if use_early_mode and 'explain_rules' in locals():
        with st.expander("補正ルールの適用理由（クリックで表示）"):
            if explain_rules:
                for r in explain_rules: st.write("・" + r)
            else:
                st.write("補正は適用されていません。")

