
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

# ここまでの分岐で df が決まる（サンプル or アップロード or None）

# ---- データ未選択のときはここで止める ----
if df is None:
    st.info("CSVをアップロードするか、上のボタンでサンプルデータを読み込んでください。")
    st.stop()

# ---- データ表示 ----
st.dataframe(df)
# === ベンチマーク統計を保存（H / I / 合成） ===
import numpy as np

# 正規化（学習と同じスケール）
h_samples = (df["H"].astype(float) / 30.0).to_numpy()
i_samples = (df["I"].astype(float) / 5.0).to_numpy()

# NaN / Inf を除去
mask = np.isfinite(h_samples) & np.isfinite(i_samples)
h_samples = h_samples[mask]
i_samples  = i_samples[mask]

# 平均・分散（極小分散対策の eps 付与）
eps = 1e-6
mu_h = float(h_samples.mean()); sd_h = float(h_samples.std() + eps)
mu_i = float(i_samples.mean());  sd_i = float(i_samples.std()  + eps)

# z 配列と H+I 合成 z 配列（平均）
z_h = (h_samples - mu_h) / sd_h
z_i = (i_samples - mu_i) / sd_i
z_hi_samples = (z_h + z_i) / 2.0

# セッションに保存（ベンチマーク比較タブで使う）
st.session_state["bench"] = {
    "h_samples": h_samples,
    "i_samples": i_samples,
    "z_hi_samples": z_hi_samples,
    "mu_h": mu_h, "sd_h": sd_h,
    "mu_i": mu_i, "sd_i": sd_i,
}

# ---- 必要列チェック ----
required_cols = {'company', 'H', 'I', 'S', 'A', 'B'}
if not required_cols.issubset(df.columns):
    st.error("必要列が不足しています。company, H, I, S, A, B を含めてください。")
    st.stop()

# ---- ここから下は df が必ずある前提で学習処理 ----
data = df.copy()
data['label_success'] = ((data['A'] + data['B']) >= 8).astype(int)

# 正規化した特徴量（学習用・小文字で統一）
data["h"]   = data["H"] / 30.0
data["i"]   = data["I"] / 5.0
data["s"]   = data["S"] / 5.0
data["h_i"] = data["h"] * data["i"]
data["i_s"] = data["i"] * data["s"]

X = data[["h","i","s","h_i","i_s"]].copy()
y = data["label_success"].astype(int)

# 両クラスが無い場合は終了
if y.nunique() < 2:
    st.warning("成功/非成功の両方のサンプルが必要です。現状は片側のみのため、学習はスキップします。")
    st.stop()
# ---- ベンチマーク統計（成功=1）を作って保存 ----  ← ここを68行目の直後に入れる
success_df = data[data["label_success"] == 1].copy()

if len(success_df) >= 5:  # 最低件数ガード
    h_vals = success_df["h"].to_numpy()  # 0-1 正規化済み
    i_vals = success_df["i"].to_numpy()

    eps = 1e-6
    mu_h = float(h_vals.mean());  sd_h = float(h_vals.std(ddof=1) or eps)
    mu_i = float(i_vals.mean());  sd_i = float(i_vals.std(ddof=1) or eps)

    # H+I 複合のサンプル（zスコア平均）
    z_hi_samples = ((h_vals - mu_h) / sd_h + (i_vals - mu_i) / sd_i) / 2.0

    st.session_state["bench"] = {
        "h_samples": h_vals.tolist(),
        "mu_h": mu_h, "sd_h": sd_h,
        "mu_i": mu_i, "sd_i": sd_i,
        "z_hi_samples": z_hi_samples.tolist(),
    }
else:
    st.session_state.pop("bench", None)

# ==== 学習（ここから）====
if 'model' not in locals():
    model = LogisticRegression(solver="liblinear", max_iter=1000, random_state=42)

try:
    model.fit(X[['h','i','s','h_i','i_s']], y)
    calibrated = CalibratedClassifierCV(model, cv="prefit", method="isotonic")
    calibrated.fit(X[['h','i','s','h_i','i_s']], y)
    st.session_state["calibrated"] = calibrated
    st.success("モデル学習＋確率校正 完了")
except Exception as e:
    st.error(f"学習中にエラー: {e}")
    st.stop()
# ==== 学習（ここまで）====


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
tab_bench, tab_prob = st.tabs(["📊 ベンチマーク比較", "🎯 成功確率予測"])

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
# タブ定義（3つのタブを作る）
tab_data, tab_bench, tab_prob = st.tabs(
    ["📂 データ準備", "📊 ベンチマーク比較", "📈 成功確率予測"]
)
# ---- ベンチマーク比較モード ----  ← 118行目の直後に入れる
with tab_bench:
    # ===== ベンチマーク比較（成功企業に対する相対位置） =====

    # 1) ヘルパー：NaN/Inf 排除 + 上位％(大きいほど上位)
    def percentile_rank(samples, x):
        import numpy as np
        samples = np.asarray(samples, dtype=float)
        samples = samples[np.isfinite(samples)]
        if samples.size == 0:
            return None
        return float((samples > x).sum()) / samples.size * 100.0

    bench = st.session_state.get("bench")

    if not bench:
        st.warning("まだベンチマーク統計がありません。上部でCSV学習（成功企業データ）を実行してください。")
    else:
        # 2) H 単独
        h_pct = percentile_rank(bench["h_samples"], Hn)
        if h_pct is None:
            st.warning("Hの比較に必要なデータが不足しています。")

        # 3) H+I 複合（z平均）
        # H と I の重み
        w = st.slider("H と I の重み（H をどれだけ重視するか）", min_value=0.0, max_value=1.0, value=0.60)

        # 安定化用の下限（極小分散対策）
        eps = 1e-6
        sd_h = bench.get("sd_h", None)
        sd_i = bench.get("sd_i", None)
        sd_h = sd_h if (sd_h is not None and sd_h > eps) else eps
        sd_i = sd_i if (sd_i is not None and sd_i > eps) else eps

        # あなたの現在値の z
        z_h = (Hn - bench["mu_h"]) / sd_h
        z_i = (In_ - bench["mu_i"]) / sd_i

        # 母集団の z 配列も同じ条件で再計算（NaN/Inf除去）
        import numpy as np
        h_samples = np.asarray(bench["h_samples"], dtype=float)
        i_samples = np.asarray(bench["i_samples"], dtype=float)
        z_h_samples = (h_samples - bench["mu_h"]) / sd_h
        z_i_samples = (i_samples - bench["mu_i"]) / sd_i
        z_h_samples = z_h_samples[np.isfinite(z_h_samples)]
        z_i_samples = z_i_samples[np.isfinite(z_i_samples)]

        # 合成 z（あなた & 配列）
        z_hi = w * z_h + (1.0 - w) * z_i
        z_hi_samples = w * z_h_samples + (1.0 - w) * z_i_samples
        z_hi_samples = z_hi_samples[np.isfinite(z_hi_samples)]

        # パーセンタイル
        hi_pct = percentile_rank(z_hi_samples, z_hi)

        # 表示
        st.subheader("📊 ベンチマーク比較（成功企業に対する相対位置）")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("H（課題仮説）の位置", f"上位 { (h_pct if h_pct is not None else 0):.1f}%")
            st.caption(f"H 正規化値: {Hn:.3f} | 成功企業の平均: {bench['mu_h']:.3f} (±{bench['sd_h']:.3f})")
        with col2:
            st.metric("H+I（複合）の位置", f"上位 { (hi_pct if hi_pct is not None else 0):.1f}%")
            st.caption(f"z_h={z_h:.2f}, z_i={z_i:.2f}, 重み w={w:.2f} → 合成 z={z_hi:.2f}")

        # 任意の可視化（進捗バー）
        st.write("H（課題仮説）の位置")
        st.progress(int(max(0, min(100, h_pct or 0))))
        st.write("H+I（複合）の位置")
        st.progress(int(max(0, min(100, hi_pct or 0))))




# 学習済みモデルを session_state から取得
with tab_prob:
    calibrated = st.session_state.get("calibrated", None)
    if calibrated is None:
        st.warning("まだモデルが学習されていません。ページ上部でCSVをアップロードして学習を実行してください。")
    else:
        P = float(calibrated.predict_proba(Xq[["h","i","s","h_i","i_s"]])[0, 1])

        # 既存の上流モード補正（use_early_mode チェックボックス〜補正ルール）をここにそのまま
        use_early_mode = st.sidebar.checkbox("上流モード（H重視の安全補正を有効化）", value=True)
        if use_early_mode:
            H_GATE = 12; H_HARD_FLOOR = 8; CAP_STRONG_IS = 0.50; CAP_ZERO_H = 0.35
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

        st.metric("成功確率（校正後）", f"{P*100:.1f}%")
        if use_early_mode and 'explain_rules' in locals():
            with st.expander("補正ルールの適用理由（クリックで表示）"):
                if explain_rules:
                    for r in explain_rules:
                        st.write("・" + r)
                else:
                    st.write("補正は適用されていません。")

