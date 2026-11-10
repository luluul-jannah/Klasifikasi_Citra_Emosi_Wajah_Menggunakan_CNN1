import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import json
import tempfile
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

from PIL import Image, ImageOps

CV2_ERROR = None
try:
    import cv2
except Exception as _e:
    cv2 = None
    CV2_ERROR = f"{type(_e).__name__}: {_e}"

# True hanya jika OpenCV benar-benar lengkap (punya CascadeClassifier)
CV2_OK = cv2 is not None and hasattr(cv2, "CascadeClassifier")


# ----------------------------------------------------------------------------
# KONFIGURASI
# ----------------------------------------------------------------------------

st.set_page_config(
    page_title="Dashboard Emosi Wajah (CNN)",
    page_icon="🙂",
    layout="wide"
)

APP_DIR = Path(__file__).resolve().parent

EMOJI = {
    "angry": "😠",
    "disgust": "🤢",
    "fear": "😨",
    "happy": "😄",
    "neutral": "😐",
    "sad": "😢",
    "surprise": "😲",
}

LOW_CONFIDENCE = 0.50

MARKER_FILES = ("class_names.json", "dataset_summary.json", "history.json")


def _looks_like_output(d: Path) -> bool:
    """Folder dianggap folder output jika berisi file hasil notebook."""
    if not d.is_dir():
        return False
    if any((d / m).is_file() for m in MARKER_FILES):
        return True
    return any(d.glob("*.keras"))


def find_output_dir() -> str:
    """Cari folder hasil notebook secara otomatis."""
    env = os.environ.get("EMOTION_OUT_DIR")
    candidates = []
    if env:
        candidates.append(Path(env))

    cwd = Path.cwd()
    candidates += [
        APP_DIR / "output",
        APP_DIR,
        cwd / "output",
        cwd,
        APP_DIR.parent / "output",
    ]

    for c in candidates:
        if _looks_like_output(c):
            return str(c)

    # Cari ke subfolder (maks. 3 tingkat) dari folder app
    for marker in MARKER_FILES:
        for p in APP_DIR.rglob(marker):
            if len(p.relative_to(APP_DIR).parts) <= 4:
                return str(p.parent)

    return str(APP_DIR)


DEFAULT_OUT_DIR = find_output_dir()


# ----------------------------------------------------------------------------
# LOADER DATA DAN MODEL
# ----------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _load_json(path: str, mtime: float):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource(show_spinner="Memuat model CNN...")
def _load_model(path: str, mtime: float):
    model_path = Path(path)

    # Jika model berbentuk folder
    if model_path.is_dir():
        required_files = ["config.json", "metadata.json", "model.weights.h5"]
        missing_files = [
            n for n in required_files if not (model_path / n).is_file()
        ]
        if missing_files:
            raise FileNotFoundError(
                "File model tidak lengkap. "
                f"File yang tidak ditemukan: {missing_files}"
            )

        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                suffix=".keras", delete=False
            ) as temp_file:
                temp_path = Path(temp_file.name)

            with zipfile.ZipFile(
                temp_path, "w", compression=zipfile.ZIP_DEFLATED
            ) as archive:
                for item in model_path.rglob("*"):
                    if item.is_file():
                        archive.write(
                            item,
                            arcname=item.relative_to(model_path).as_posix()
                        )

            return tf.keras.models.load_model(str(temp_path), compile=False)

        finally:
            if temp_path is not None and temp_path.exists():
                temp_path.unlink()

    # Jika model berupa file .keras biasa
    return tf.keras.models.load_model(str(model_path), compile=False)


def read_json(out_dir: Path, name: str):
    p = out_dir / name
    if p.exists() and p.is_file():
        return _load_json(str(p), p.stat().st_mtime)
    return None


def show_png(out_dir: Path, name: str, caption: str):
    p = out_dir / name
    if p.exists() and p.is_file():
        st.image(str(p), caption=caption)
    else:
        st.info(
            f"`{name}` belum ada di folder output. "
            "Jalankan sel notebook yang menghasilkannya."
        )


def fmt_pct(x):
    return f"{x * 100:.2f}%"


# ----------------------------------------------------------------------------
# PRA-PEMROSESAN DAN PREDIKSI
# ----------------------------------------------------------------------------

def _find_cascade():
    """Cari file haarcascade di beberapa lokasi yang mungkin."""
    name = "haarcascade_frontalface_default.xml"
    paths = []
    try:
        paths.append(Path(cv2.data.haarcascades) / name)
    except AttributeError:
        pass
    try:
        paths.append(Path(cv2.__file__).parent / "data" / name)
    except Exception:
        pass
    paths.append(APP_DIR / name)
    for p in paths:
        if p.is_file():
            return str(p)
    return None


def crop_face(img: Image.Image):
    """Memotong wajah terbesar dengan Haar Cascade.
    Aman: jika OpenCV bermasalah, gambar utuh dipakai tanpa error."""
    if not CV2_OK:
        return img, False

    try:
        path = _find_cascade()
        if path is None:
            return img, False

        cascade = cv2.CascadeClassifier(path)
        if cascade.empty():
            return img, False

        gray = np.array(img.convert("L"))
        faces = cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
        )
    except Exception:
        return img, False

    if len(faces) == 0:
        return img, False

    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    pad = int(0.10 * max(w, h))
    x0 = max(x - pad, 0)
    y0 = max(y - pad, 0)
    x1 = min(x + w + pad, img.width)
    y1 = min(y + h + pad, img.height)
    return img.crop((x0, y0, x1, y1)), True


def preprocess(img: Image.Image, size):
    gray = np.asarray(img.convert("L"), dtype=np.float32)[..., None]
    resized = tf.image.resize(gray, size).numpy()
    return resized[None, ...]


def predict(model, x):
    return model(tf.constant(x), training=False).numpy()[0]


# ----------------------------------------------------------------------------
# GRAD-CAM
# ----------------------------------------------------------------------------

def find_last_conv(model):
    try:
        model.get_layer("last_conv")
        return "last_conv"
    except ValueError:
        pass

    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    return None


def make_gradcam(model, x, layer_name):
    grad_model = tf.keras.models.Model(
        model.inputs,
        [model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(tf.constant(x), training=False)
        pred_index = tf.argmax(preds[0])
        class_score = preds[:, pred_index]

    grads = tape.gradient(class_score, conv_out)
    weights = tf.reduce_mean(grads, axis=(1, 2))
    cam = tf.reduce_sum(conv_out * weights[:, None, None, :], axis=-1)[0]
    cam = tf.nn.relu(cam)
    cam = cam / (tf.reduce_max(cam) + 1e-8)
    return cam.numpy()


def overlay_cam(gray_img: np.ndarray, cam: np.ndarray, alpha=0.45, out=256):
    base = (
        Image.fromarray(np.clip(gray_img, 0, 255).astype("uint8"))
        .convert("RGB")
        .resize((out, out))
    )
    cam_img = Image.fromarray(np.uint8(255 * cam)).resize(
        (out, out), Image.Resampling.BILINEAR
    )
    heat = matplotlib.colormaps["jet"](np.asarray(cam_img) / 255.0)[..., :3]
    blended = (
        (1 - alpha) * np.asarray(base, dtype=np.float32) + alpha * heat * 255
    )
    return Image.fromarray(blended.astype("uint8"))


# ----------------------------------------------------------------------------
# PLOT
# ----------------------------------------------------------------------------

def plot_confusion(cm, names):
    cm = np.asarray(cm)
    row_sum = cm.sum(axis=1, keepdims=True)
    pct = np.divide(
        cm, row_sum, out=np.zeros(cm.shape, dtype=float), where=row_sum != 0
    )

    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    im = ax.imshow(pct, cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, label="Proporsi per kelas sebenarnya")

    ticks = range(len(names))
    ax.set_xticks(list(ticks))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticks(list(ticks))
    ax.set_yticklabels(names)

    for i in ticks:
        for j in ticks:
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if pct[i, j] > 0.5 else "black",
                fontsize=9
            )

    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Kelas sebenarnya")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    return fig


def plot_history(history):
    best_epoch = int(np.argmax(history["val_accuracy"])) + 1
    epochs = range(1, len(history["loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, (key, title) in zip(
        axes, [("accuracy", "Accuracy"), ("loss", "Loss")]
    ):
        ax.plot(epochs, history[key], "o-", label=f"Train {title}")
        ax.plot(epochs, history[f"val_{key}"], "s-", label=f"Val {title}")
        ax.axvline(
            best_epoch, color="gray", linestyle="--",
            label=f"Epoch terbaik ({best_epoch})"
        )
        ax.set_title(f"Training vs Validation {title}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(title)
        ax.grid(alpha=0.3)
        ax.legend()

    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------------------------

st.sidebar.title("⚙️ Pengaturan")

out_dir = Path(
    st.sidebar.text_input(
        "Folder output notebook (terdeteksi otomatis)",
        DEFAULT_OUT_DIR
    )
).expanduser()

if not out_dir.is_dir():
    st.title("Dashboard Klasifikasi Emosi Wajah (CNN)")
    st.error(f"Folder output tidak ditemukan: `{out_dir}`")
    st.info(
        "Pastikan file hasil notebook (class_names.json, history.json, "
        "model .keras, dll.) sudah ada di repository GitHub, "
        "lalu reboot aplikasi."
    )
    st.stop()

# Cari semua model .keras di folder output
model_files = sorted(p.name for p in out_dir.glob("*.keras"))

model_choice = (
    st.sidebar.selectbox(
        "Model untuk prediksi",
        model_files,
        help="Pilih file model .keras yang akan dipakai untuk prediksi."
    )
    if model_files
    else None
)

st.sidebar.markdown("---")
if CV2_OK:
    st.sidebar.caption(f"✅ OpenCV aktif (v{cv2.__version__})")
elif cv2 is None:
    st.sidebar.warning(f"OpenCV gagal di-import:\n\n{CV2_ERROR}")
else:
    st.sidebar.warning(
        "OpenCV terpasang tetapi tidak lengkap "
        f"(v{getattr(cv2, '__version__', '?')}, "
        "tidak ada CascadeClassifier)."
    )
st.sidebar.caption(
    "Dashboard ini membaca artefak dari notebook "
    "(model, riwayat training, metrik, confusion matrix, dll.)."
)


# ----------------------------------------------------------------------------
# MEMBACA DATA JSON
# ----------------------------------------------------------------------------

summary = read_json(out_dir, "dataset_summary.json")
class_names = (
    read_json(out_dir, "class_names.json")
    or (summary or {}).get("class_names")
)
config = read_json(out_dir, "training_config.json")
history = read_json(out_dir, "history.json")
metrics = read_json(out_dir, "eval_metrics.json")
cm_data = read_json(out_dir, "confusion_matrix.json")


# ----------------------------------------------------------------------------
# HEADER DASHBOARD
# ----------------------------------------------------------------------------

st.title("🙂 Dashboard Klasifikasi Emosi Wajah (CNN)")
st.caption(
    "Klasifikasi ekspresi wajah menjadi beberapa kelas emosi "
    "menggunakan Convolutional Neural Network."
)

tab_data, tab_eda, tab_train, tab_eval, tab_pred = st.tabs(
    [
        "📦 Dataset",
        "🔎 Eksplorasi Data",
        "🏋️ Pelatihan",
        "📊 Evaluasi",
        "🔮 Prediksi"
    ]
)


# ----------------------------------------------------------------------------
# TAB 1 — DATASET
# ----------------------------------------------------------------------------

with tab_data:
    st.subheader("Ringkasan Dataset")

    if not summary:
        st.warning(
            "`dataset_summary.json` tidak ditemukan. "
            "Jalankan Langkah 3 pada notebook."
        )
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Jumlah kelas", summary["num_classes"])
        c2.metric("Total gambar", f"{summary['total_images']:,}")
        c3.metric(
            "Data training", f"{summary['train_images']:,}",
            fmt_pct(summary["train_ratio"]), delta_color="off"
        )
        c4.metric(
            "Data validasi/test", f"{summary['val_images']:,}",
            fmt_pct(summary["val_ratio"]), delta_color="off"
        )

        st.write(
            f"**Ukuran input:** "
            f"{summary['image_size'][0]}×{summary['image_size'][1]} piksel "
            f"({summary['color_mode']}) "
            f"&nbsp;|&nbsp; **Kelas:** {', '.join(summary['class_names'])}"
        )

        df_cls = pd.DataFrame({
            "Training": summary["class_counts_train"],
            "Validasi": summary["class_counts_val"]
        })
        df_cls["Total"] = df_cls["Training"] + df_cls["Validasi"]
        df_cls["Persen (%)"] = (
            df_cls["Total"] / df_cls["Total"].sum() * 100
        ).round(2)

        left, right = st.columns([3, 2])
        with left:
            st.markdown("**Jumlah gambar per kelas**")
            st.bar_chart(df_cls[["Training", "Validasi"]])
        with right:
            st.markdown("**Tabel distribusi**")
            st.dataframe(df_cls)

        ratio = summary["imbalance_ratio"]
        if ratio > 1.5:
            st.warning(
                "Terdapat ketidakseimbangan kelas "
                f"(rasio terbanyak/tersedikit = {ratio:.2f}×)."
            )
        else:
            st.success(
                "Distribusi kelas relatif seimbang "
                f"(rasio terbanyak/tersedikit = {ratio:.2f}×)."
            )


# ----------------------------------------------------------------------------
# TAB 2 — EKSPLORASI DATA
# ----------------------------------------------------------------------------

with tab_eda:
    st.subheader("Eksplorasi Data (EDA)")
    show_png(out_dir, "1_class_distribution.png",
             "Distribusi jumlah gambar per kelas")
    show_png(out_dir, "2_sample_per_class.png",
             "Contoh gambar dari setiap kelas")
    show_png(out_dir, "3_image_quality.png",
             "Distribusi kecerahan dan kontras gambar")
    show_png(out_dir, "4_sample_training.png",
             "Contoh data training setelah pra-pemrosesan")


# ----------------------------------------------------------------------------
# TAB 3 — PELATIHAN
# ----------------------------------------------------------------------------

with tab_train:
    st.subheader("Konfigurasi & Hasil Pelatihan")

    if config:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(
            "Epoch dijalankan",
            f"{config['epoch_dijalankan']} / {config['epoch_maksimum']}"
        )
        c2.metric("Epoch terbaik", config["epoch_terbaik"])
        c3.metric("Total parameter", f"{config['total_parameter']:,}")
        c4.metric(
            "Waktu training", f"{config['waktu_training_menit']:.1f} menit"
        )

        with st.expander("Detail konfigurasi", expanded=False):
            cfg_df = pd.DataFrame({
                "Parameter": list(config.keys()),
                "Nilai": [str(v) for v in config.values()]
            })
            st.dataframe(cfg_df, hide_index=True)
    else:
        st.warning("`training_config.json` tidak ditemukan.")

    if history:
        st.pyplot(plot_history(history))
        plt.close("all")

        best = int(np.argmax(history["val_accuracy"]))
        c1, c2, c3 = st.columns(3)
        c1.metric(
            "Train accuracy (epoch terbaik)", fmt_pct(history["accuracy"][best])
        )
        c2.metric(
            "Val accuracy (epoch terbaik)",
            fmt_pct(history["val_accuracy"][best])
        )
        gap = history["accuracy"][best] - history["val_accuracy"][best]
        c3.metric(
            "Selisih train − val", f"{gap * 100:.2f} poin", delta_color="off"
        )

        if gap > 0.10:
            st.warning(
                "Selisih train dan validasi cukup besar: "
                "indikasi overfitting."
            )
        elif history["accuracy"][best] < 0.60:
            st.info(
                "Akurasi training masih rendah: "
                "indikasi underfitting / model belum konvergen."
            )
        else:
            st.success("Kurva train dan validasi relatif sejalan.")

        if "learning_rate" in history:
            with st.expander("Learning rate per epoch"):
                st.line_chart(
                    pd.DataFrame(
                        {"learning_rate": history["learning_rate"]},
                        index=range(1, len(history["learning_rate"]) + 1)
                    )
                )
    else:
        st.warning("`history.json` tidak ditemukan.")


# ----------------------------------------------------------------------------
# TAB 4 — EVALUASI
# ----------------------------------------------------------------------------

with tab_eval:
    st.subheader("Evaluasi Model pada Data Uji")

    if not metrics:
        st.warning(
            "`eval_metrics.json` tidak ditemukan. "
            "Jalankan Langkah 5 pada notebook."
        )
    else:
        report = {
            k: v for k, v in metrics["report"].items() if isinstance(v, dict)
        }
        macro = report["macro avg"]

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Accuracy", fmt_pct(metrics["accuracy"]))
        c2.metric("Loss", f"{metrics['loss']:.4f}")
        c3.metric("Precision (macro)", f"{macro['precision']:.3f}")
        c4.metric("Recall (macro)", f"{macro['recall']:.3f}")
        c5.metric("F1-Score (macro)", f"{macro['f1-score']:.3f}")

        names = metrics["class_names"]
        df_rep = pd.DataFrame(report).T.loc[
            names + ["macro avg", "weighted avg"]
        ]
        df_rep = df_rep.rename(columns={
            "precision": "Precision",
            "recall": "Recall",
            "f1-score": "F1-Score",
            "support": "Jumlah data"
        })
        df_rep["Jumlah data"] = df_rep["Jumlah data"].astype(int)

        left, right = st.columns([3, 2])
        with left:
            st.markdown("**Tabel metrik per kelas**")
            st.dataframe(df_rep.round(3))
        with right:
            st.markdown("**F1-Score per kelas**")
            st.bar_chart(df_rep.loc[names, ["F1-Score"]])

        f1 = df_rep.loc[names, "F1-Score"]
        st.markdown(
            f"**Interpretasi:** akurasi {metrics['accuracy'] * 100:.1f}% "
            f"berarti sekitar {metrics['accuracy'] * 100:.0f} dari 100 gambar "
            f"uji dikenali dengan benar. "
            f"Kelas paling mudah dikenali: **{f1.idxmax()}** "
            f"(F1 = {f1.max():.3f}); "
            f"paling sulit: **{f1.idxmin()}** (F1 = {f1.min():.3f})."
        )

        st.markdown("---")
        st.markdown("#### Confusion Matrix")

        if cm_data:
            cm = np.array(cm_data["matrix"])
            left, right = st.columns([3, 2])

            with left:
                st.pyplot(plot_confusion(cm, cm_data["class_names"]))
                plt.close("all")

            with right:
                st.markdown("**Kesalahan klasifikasi terbanyak**")
                pairs = [
                    {
                        "Kelas asli": cm_data["class_names"][i],
                        "Diprediksi sebagai": cm_data["class_names"][j],
                        "Jumlah": int(cm[i, j])
                    }
                    for i in range(len(cm))
                    for j in range(len(cm))
                    if i != j and cm[i, j] > 0
                ]
                pairs.sort(key=lambda d: -d["Jumlah"])
                st.dataframe(pd.DataFrame(pairs[:8]), hide_index=True)
        else:
            st.warning("`confusion_matrix.json` tidak ditemukan.")

        st.markdown("---")
        st.markdown("#### Visualisasi Tambahan")
        show_png(
            out_dir, "7_sample_predictions.png",
            "Contoh prediksi pada data uji (hijau = benar, merah = salah)"
        )
        show_png(
            out_dir, "8_gradcam.png",
            "Grad-CAM: area yang paling memengaruhi keputusan model"
        )


# ----------------------------------------------------------------------------
# TAB 5 — PREDIKSI
# ----------------------------------------------------------------------------

with tab_pred:
    st.subheader("Prediksi Emosi dari Gambar")

    if model_choice is None or not class_names:
        st.warning(
            "Model (file `.keras`) atau `class_names.json` "
            "tidak ditemukan di folder output."
        )
    else:
        model_path = out_dir / model_choice

        try:
            model = _load_model(str(model_path), model_path.stat().st_mtime)
        except Exception as e:
            st.error(f"Model gagal dimuat: {e}")
            st.info(
                "Periksa kembali file model dan pastikan versi "
                "TensorFlow/Keras di requirements.txt sama dengan "
                "yang dipakai saat training."
            )
            st.stop()

        input_size = tuple(int(s) for s in model.input_shape[1:3])
        n_out = int(model.output_shape[-1])

        if n_out != len(class_names):
            st.error(
                f"Jumlah kelas model ({n_out}) tidak sama "
                f"dengan `class_names.json` ({len(class_names)})."
            )
            st.stop()

        st.caption(
            f"Model: `{model_choice}` · "
            f"input {input_size[0]}×{input_size[1]} grayscale · "
            f"{n_out} kelas"
        )

        source = st.radio(
            "Sumber gambar",
            ["Unggah gambar", "Ambil dari kamera"],
            horizontal=True
        )

        if source == "Unggah gambar":
            file = st.file_uploader(
                "Pilih gambar wajah", type=["jpg", "jpeg", "png", "bmp"]
            )
        else:
            file = st.camera_input("Ambil foto")

        opt1, opt2 = st.columns(2)

        use_face = opt1.checkbox(
            "Deteksi & potong wajah otomatis",
            value=CV2_OK,
            disabled=not CV2_OK,
            help=(
                "Model dilatih pada gambar wajah yang sudah dipotong. "
                "Foto utuh sebaiknya dipotong ke area wajah. "
                "Butuh opencv-python-headless."
                if CV2_OK
                else "OpenCV tidak tersedia/rusak di server. Tambahkan "
                     "opencv-python-headless ke requirements.txt."
            )
        )

        show_cam = opt2.checkbox("Tampilkan Grad-CAM", value=True)

        if file is None:
            st.info(
                "Unggah atau ambil gambar wajah untuk memulai prediksi."
            )
        else:
            try:
                img = ImageOps.exif_transpose(Image.open(file))
            except Exception as e:
                st.error(f"Gambar tidak dapat dibaca: {e}")
                st.stop()

            face_found = False
            work_img = img

            if use_face:
                work_img, face_found = crop_face(img)
                if not face_found:
                    st.warning(
                        "Wajah tidak terdeteksi, "
                        "seluruh gambar dipakai apa adanya."
                    )

            x = preprocess(work_img, input_size)
            probs = predict(model, x)
            order = np.argsort(-probs)
            top = int(order[0])
            conf = float(probs[top])

            col_img, col_res = st.columns([1, 1])

            with col_img:
                st.markdown("**Gambar masukan**")
                st.image(img, caption="Gambar asli")
                st.image(
                    Image.fromarray(
                        x[0].squeeze().astype("uint8")
                    ).resize((192, 192), Image.Resampling.NEAREST),
                    caption=(
                        f"Setelah pra-pemrosesan "
                        f"({input_size[0]}×{input_size[1]}, grayscale)"
                        + (" · wajah terpotong" if face_found else "")
                    )
                )

            with col_res:
                st.markdown("**Hasil prediksi**")
                label = class_names[top]

                st.metric(
                    "Emosi terdeteksi",
                    f"{EMOJI.get(label, '')} {label}".strip(),
                    fmt_pct(conf),
                    delta_color="off"
                )

                if conf < LOW_CONFIDENCE:
                    st.warning(
                        "Keyakinan model rendah. "
                        "Coba gambar wajah yang lebih jelas, "
                        "menghadap depan, dan terang."
                    )

                df_prob = pd.DataFrame({
                    "Emosi": [class_names[i] for i in order],
                    "Probabilitas": [float(probs[i]) for i in order]
                }).set_index("Emosi")

                st.bar_chart(df_prob)
                st.dataframe(
                    df_prob.assign(
                        **{
                            "Probabilitas (%)": (
                                df_prob["Probabilitas"] * 100
                            ).round(2)
                        }
                    )[["Probabilitas (%)"]]
                )

            if show_cam:
                layer = find_last_conv(model)

                if layer is None:
                    st.info(
                        "Layer konvolusi terakhir tidak ditemukan; "
                        "Grad-CAM dilewati."
                    )
                else:
                    try:
                        cam = make_gradcam(model, x, layer)

                        st.markdown(
                            "**Grad-CAM** — area merah/kuning "
                            "paling berpengaruh pada keputusan model"
                        )

                        a, b = st.columns(2)
                        gray = x[0].squeeze()

                        a.image(
                            Image.fromarray(gray.astype("uint8")).resize(
                                (256, 256), Image.Resampling.NEAREST
                            ),
                            caption="Gambar"
                        )
                        b.image(
                            overlay_cam(gray, cam),
                            caption=(
                                f"Overlay Grad-CAM "
                                f"(prediksi: {class_names[top]})"
                            )
                        )
                    except Exception as e:
                        st.info(
                            f"Grad-CAM tidak dapat dibuat "
                            f"untuk model ini: {e}"
                        )
