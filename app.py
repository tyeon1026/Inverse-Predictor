from flask import Flask, render_template, request, session, jsonify
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import uuid
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from all_functions import VAE, Inverse_Model

# 모델 및 스케일러 로드
descriptors = pd.read_csv('./data/descriptors.csv', encoding="utf-8-sig")
ds_names = np.array(descriptors.columns)

scaler_ds = StandardScaler()
scaler_ds.fit(np.array(descriptors))

vae = load_model("./models/vae.keras", custom_objects={"VAE": VAE})
model = load_model("./models/inverse_model.keras", custom_objects={"Inverse_Model": Inverse_Model})
model.set_vae(vae)

app = Flask(__name__)
app.secret_key = "secret_for_session"

cache = {}  # 서버 메모리 캐시

def generate_heatmap(x_reshaped, ds_names_list, desc_name):
    desc_idx = ds_names_list.index(desc_name)
    map_data = x_reshaped[..., desc_idx]
    fig, ax = plt.subplots()
    im = ax.imshow(map_data, cmap='inferno')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(f"{desc_name}")
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode()

@app.route("/", methods=["GET", "POST"])
def index():
    result_table, error_message, shape_info, index_columns = None, None, None, []

    if request.method == "POST" and "data_file" in request.files:
        file = request.files.get("data_file")
        try:
            if file.filename.endswith(".npy"):
                y_input = np.load(file)
            elif file.filename.endswith(".csv"):
                y_input = np.loadtxt(file, delimiter=",")
            else:
                return "❌ Only .npy or .csv files are supported", 400

            # ✅ (2048,) 벡터인 경우 자동으로 (1, 2048)로 확장
            if y_input.ndim == 1 and y_input.shape[0] == 2048:
                y_input = y_input[np.newaxis, :]  # (1, 2048)

            if y_input.shape[-1] != 2048:
                return "❌ Last dimension must be 2048", 400

            original_shape = y_input.shape[:-1]
            y_flat = y_input.reshape(-1, 2048)

            # 모델 예측
            z_pred = model.predict(y_flat)
            x_pred_s = vae.decoder_ds(z_pred)
            x_pred = scaler_ds.inverse_transform(x_pred_s)
            x_reshaped = x_pred.reshape(*original_shape, 125)
            shape_info = x_reshaped.shape

            # 캐시에 저장
            cache_id = str(uuid.uuid4())
            cache[cache_id] = {"x_reshaped": x_reshaped, "ds_names": ds_names.tolist()}
            session["cache_id"] = cache_id

            # 테이블 데이터
            idx = np.indices(x_reshaped.shape[:-1]).reshape(len(x_reshaped.shape)-1, -1).T
            vals = x_reshaped.reshape(-1, x_reshaped.shape[-1])
            index_columns = [f"Dim{i}" for i in range(idx.shape[1])]
            result_table = [(list(i), dict(zip(ds_names, v))) for i, v in zip(idx, vals)]

        except Exception as e:
            error_message = f"❌ 오류 발생: {e}"

    return render_template(
        "index.html",
        result_table=result_table,
        ds_names=ds_names,
        error_message=error_message,
        shape_info=shape_info,
        index_columns=index_columns
    )

@app.route("/heatmap", methods=["POST"])
def heatmap():
    desc_name = request.json.get("descriptor")
    cache_id = session.get("cache_id")

    if not cache_id or cache_id not in cache:
        return jsonify({"error": "먼저 데이터를 업로드하고 예측하세요"}), 400

    x_reshaped = cache[cache_id]["x_reshaped"]
    ds_names_list = cache[cache_id]["ds_names"]

    if desc_name not in ds_names_list:
        return jsonify({"error": f"'{desc_name}' 디스크립터를 찾을 수 없습니다"}), 400

    heatmap_url = generate_heatmap(x_reshaped, ds_names_list, desc_name)
    return jsonify({"descriptor": desc_name, "heatmap": heatmap_url})

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
