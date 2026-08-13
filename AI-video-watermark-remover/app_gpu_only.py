import os
import cv2
import uuid
import torch
import tempfile
import subprocess
import numpy as np
import gradio as gr

from PIL import Image
from simple_lama_inpainting import SimpleLama


# ============================================================
# GPU-ONLY CUDA CONFIGURATION
# ============================================================

# This application intentionally runs AI inference on NVIDIA CUDA only.
# `python app.py` is the only supported launch command.
#
# There is NO CPU fallback and no -cpu/-gpu command-line switch.

if not torch.cuda.is_available():
    print()
    print("=" * 70)
    print("ERROR: NVIDIA CUDA GPU NOT AVAILABLE")
    print("=" * 70)
    print("This application is GPU-only.")
    print()
    print("Check CUDA with:")
    print("python -c \"import torch; print(torch.cuda.is_available())\"")
    print()
    print("Check the GPU with:")
    print("python -c \"import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA GPU')\"")
    print()
    raise SystemExit(1)

# Always use CUDA device 0.
DEVICE = torch.device("cuda:0")
DEVICE_NAME = torch.cuda.get_device_name(0)
MODE = "GPU ONLY → CUDA"

# CUDA performance settings.
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Use reduced precision where supported by the GPU for faster inference.
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

print()
print("=" * 70)
print("             SMART AI WATERMARK REMOVER")
print("=" * 70)
print("Mode   :", MODE)
print("Device :", DEVICE_NAME)
print("CUDA   :", torch.version.cuda)
print("=" * 70)
print()

# ============================================================
# WORK DIRECTORY
# ============================================================

WORK_DIR = os.path.join(
    tempfile.gettempdir(),
    "smart_watermark_remover"
)

os.makedirs(
    WORK_DIR,
    exist_ok=True
)


# ============================================================
# GLOBAL LAMA MODEL
# ============================================================

LAMA = None


def get_lama():

    global LAMA

    if LAMA is None:

        print()
        print("Loading LaMa AI...")
        print("Device:", DEVICE_NAME)
        print("CUDA inference: ENABLED")

        LAMA = SimpleLama(
            device=DEVICE
        )

        print("LaMa ready.")
        print()

    return LAMA


# ============================================================
# VIDEO PATH
# ============================================================

def get_video_path(value):

    if value is None:
        return None

    if isinstance(value, str):
        return value

    if isinstance(value, dict):
        return value.get("path")

    return None


# ============================================================
# LOAD FRAME 1
# ============================================================

def load_frame(video):

    path = get_video_path(video)

    if not path:
        raise gr.Error(
            "Upload a video first."
        )

    cap = cv2.VideoCapture(path)

    if not cap.isOpened():
        raise gr.Error(
            "Could not open video."
        )

    ok, frame = cap.read()

    cap.release()

    if not ok:
        raise gr.Error(
            "Could not read Frame 1."
        )

    return cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


# ============================================================
# GET ROUGH USER MASK
# ============================================================

def get_rough_mask(editor):

    if editor is None:
        raise gr.Error(
            "Load Frame 1 first."
        )

    if not isinstance(editor, dict):
        raise gr.Error(
            "Invalid editor data."
        )

    layers = editor.get(
        "layers",
        []
    )

    if not layers:
        raise gr.Error(
            "Paint roughly over the watermark."
        )

    for layer in reversed(layers):

        if layer is None:
            continue

        arr = np.asarray(layer)

        if arr.ndim != 3:
            continue

        if arr.shape[2] != 4:
            continue

        alpha = arr[:, :, 3]

        mask = np.where(
            alpha > 10,
            255,
            0
        ).astype(np.uint8)

        if cv2.countNonZero(mask) > 0:
            return mask

    raise gr.Error(
        "Paint roughly over the watermark."
    )


# ============================================================
# SHOW ROUGH MASK
# ============================================================

def show_mask(editor):

    try:

        mask = get_rough_mask(editor)

        output = np.zeros(
            (
                mask.shape[0],
                mask.shape[1],
                3
            ),
            dtype=np.uint8
        )

        output[:, :, 0] = mask
        output[:, :, 1] = mask
        output[:, :, 2] = mask

        return output

    except Exception:

        return None


# ============================================================
# SMART MASK REFINEMENT
#
# User only needs to roughly paint the watermark.
#
# The algorithm:
#
# 1. Finds the rough selection.
# 2. Creates a local region around it.
# 3. Uses color/edge information.
# 4. Uses GrabCut to refine the rough selection.
# 5. Cleans small disconnected regions.
# 6. Keeps the mask conservative.
#
# This is NOT blind dilation.
# ============================================================

def smart_refine_mask(
    frame_bgr,
    rough_mask,
    strength=50
):

    h, w = frame_bgr.shape[:2]

    # --------------------------------------------------------
    # Resize rough mask to actual video/frame dimensions.
    # --------------------------------------------------------

    rough = cv2.resize(
        rough_mask,
        (
            w,
            h
        ),
        interpolation=cv2.INTER_NEAREST
    )

    rough = np.where(
        rough > 10,
        255,
        0
    ).astype(np.uint8)

    if cv2.countNonZero(rough) == 0:
        raise gr.Error(
            "The rough selection is empty."
        )

    # --------------------------------------------------------
    # Find selection bounds.
    # --------------------------------------------------------

    ys, xs = np.where(
        rough > 0
    )

    x1 = int(xs.min())
    y1 = int(ys.min())
    x2 = int(xs.max()) + 1
    y2 = int(ys.max()) + 1

    rw = x2 - x1
    rh = y2 - y1

    # Context around rough selection.
    padding = max(
        20,
        int(
            max(rw, rh) * 0.35
        )
    )

    cx1 = max(
        0,
        x1 - padding
    )

    cy1 = max(
        0,
        y1 - padding
    )

    cx2 = min(
        w,
        x2 + padding
    )

    cy2 = min(
        h,
        y2 + padding
    )

    crop = frame_bgr[
        cy1:cy2,
        cx1:cx2
    ].copy()

    crop_rough = rough[
        cy1:cy2,
        cx1:cx2
    ].copy()

    ch, cw = crop.shape[:2]

    # --------------------------------------------------------
    # GrabCut mask.
    #
    # Outside rough selection = definite background.
    #
    # Rough selection = probable foreground.
    #
    # A smaller inner region = definite foreground.
    # --------------------------------------------------------

    grab = np.full(
        (
            ch,
            cw
        ),
        cv2.GC_BGD,
        dtype=np.uint8
    )

    # Soft/probable foreground around rough selection.
    grab[
        crop_rough > 0
    ] = cv2.GC_PR_FGD

    # --------------------------------------------------------
    # Determine definite foreground core.
    #
    # We intentionally don't make the entire rough mask
    # "definite foreground" because that can make GrabCut
    # consume the background/character.
    # --------------------------------------------------------

    kernel_size = max(
        3,
        int(
            min(cw, ch) * 0.03
        )
    )

    if kernel_size % 2 == 0:
        kernel_size += 1

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            kernel_size,
            kernel_size
        )
    )

    core = cv2.erode(
        crop_rough,
        kernel,
        iterations=1
    )

    if cv2.countNonZero(core) > 20:

        grab[
            core > 0
        ] = cv2.GC_FGD

    # --------------------------------------------------------
    # Protect the outer border as background.
    # --------------------------------------------------------

    border = max(
        3,
        min(cw, ch) // 50
    )

    grab[
        :border,
        :
    ] = cv2.GC_BGD

    grab[
        -border:,
        :
    ] = cv2.GC_BGD

    grab[
        :,
        :border
    ] = cv2.GC_BGD

    grab[
        :,
        -border:
    ] = cv2.GC_BGD

    # --------------------------------------------------------
    # GrabCut.
    # --------------------------------------------------------

    bgd_model = np.zeros(
        (1, 65),
        np.float64
    )

    fgd_model = np.zeros(
        (1, 65),
        np.float64
    )

    try:

        cv2.grabCut(
            crop,
            grab,
            None,
            bgd_model,
            fgd_model,
            5,
            cv2.GC_INIT_WITH_MASK
        )

    except cv2.error:

        # If GrabCut cannot classify the region,
        # retain the user's selection instead.
        refined_crop = crop_rough.copy()

    else:

        refined_crop = np.where(
            (
                (grab == cv2.GC_FGD)
                |
                (grab == cv2.GC_PR_FGD)
            ),
            255,
            0
        ).astype(np.uint8)

    # --------------------------------------------------------
    # Keep refinement conservative.
    #
    # Never allow the algorithm to select outside the
    # user's rough region plus a small safety margin.
    # --------------------------------------------------------

    safety = max(
        2,
        int(
            strength / 25
        )
    )

    safety_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            safety * 2 + 1,
            safety * 2 + 1
        )
    )

    allowed = cv2.dilate(
        crop_rough,
        safety_kernel,
        iterations=1
    )

    refined_crop = cv2.bitwise_and(
        refined_crop,
        allowed
    )

    # --------------------------------------------------------
    # Morphological cleanup.
    # --------------------------------------------------------

    clean_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            3,
            3
        )
    )

    refined_crop = cv2.morphologyEx(
        refined_crop,
        cv2.MORPH_OPEN,
        clean_kernel,
        iterations=1
    )

    refined_crop = cv2.morphologyEx(
        refined_crop,
        cv2.MORPH_CLOSE,
        clean_kernel,
        iterations=1
    )

    # --------------------------------------------------------
    # Remove tiny connected components.
    # --------------------------------------------------------

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        refined_crop,
        8
    )

    cleaned = np.zeros_like(
        refined_crop
    )

    min_area = max(
        10,
        int(
            cv2.countNonZero(
                crop_rough
            ) * 0.002
        )
    )

    for i in range(
        1,
        num_labels
    ):

        area = stats[
            i,
            cv2.CC_STAT_AREA
        ]

        if area >= min_area:

            cleaned[
                labels == i
            ] = 255

    refined_crop = cleaned

    # --------------------------------------------------------
    # Put local mask back into full frame.
    # --------------------------------------------------------

    refined = np.zeros(
        (
            h,
            w
        ),
        dtype=np.uint8
    )

    refined[
        cy1:cy2,
        cx1:cx2
    ] = refined_crop

    # --------------------------------------------------------
    # Safety:
    #
    # If automatic refinement accidentally removes too much
    # of the rough selection, use the rough selection rather
    # than producing an obviously incomplete mask.
    # --------------------------------------------------------

    rough_pixels = cv2.countNonZero(
        rough
    )

    refined_pixels = cv2.countNonZero(
        refined
    )

    if refined_pixels < (
        rough_pixels * 0.20
    ):

        refined = rough.copy()

    return refined


# ============================================================
# ANALYZE + AUTO SELECT
# ============================================================

def analyze_and_select(
    video,
    editor,
    strength
):

    path = get_video_path(video)

    if not path:
        raise gr.Error(
            "Upload a video first."
        )

    rough_mask = get_rough_mask(
        editor
    )

    cap = cv2.VideoCapture(
        path
    )

    if not cap.isOpened():
        raise gr.Error(
            "Could not open video."
        )

    ok, frame = cap.read()

    cap.release()

    if not ok:
        raise gr.Error(
            "Could not read Frame 1."
        )

    refined = smart_refine_mask(
        frame,
        rough_mask,
        int(strength)
    )

    count = cv2.countNonZero(
        refined
    )

    if count == 0:
        raise gr.Error(
            "Automatic selection failed. "
            "Try painting a little more around the watermark."
        )

    # --------------------------------------------------------
    # Detection statistics.
    # --------------------------------------------------------

    h, w = frame.shape[:2]

    ys, xs = np.where(
        refined > 0
    )

    x1 = int(xs.min())
    y1 = int(ys.min())
    x2 = int(xs.max()) + 1
    y2 = int(ys.max()) + 1

    bw = x2 - x1
    bh = y2 - y1

    ratio = (
        count
        /
        float(w * h)
    ) * 100

    # --------------------------------------------------------
    # Analyze local visual characteristics.
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    roi = gray[
        y1:y2,
        x1:x2
    ]

    local_mask = refined[
        y1:y2,
        x1:x2
    ]

    masked_values = roi[
        local_mask > 0
    ]

    if len(masked_values):

        brightness = float(
            np.mean(
                masked_values
            )
        )

        contrast = float(
            np.std(
                masked_values
            )
        )

    else:

        brightness = 0
        contrast = 0

    edges = cv2.Canny(
        roi,
        50,
        150
    )

    edge_density = (
        np.count_nonzero(edges)
        /
        max(
            edges.size,
            1
        )
    )

    # --------------------------------------------------------
    # Simple classification.
    #
    # This is a recommendation, not a claim that AI can
    # perfectly identify the original watermark type.
    # --------------------------------------------------------

    if bw < w * 0.40 and bh < h * 0.25:

        if edge_density > 0.12:
            kind = "Graphic / detailed logo"
        else:
            kind = "Text / logo watermark"

    elif ratio < 15:
        kind = "Overlay watermark"

    else:
        kind = "Large / complex watermark"

    # --------------------------------------------------------
    # Generate preview.
    # --------------------------------------------------------

    preview = frame.copy()

    # Green = automatically detected mask.
    green = np.zeros_like(
        preview
    )

    green[:, :, 1] = 255

    alpha = (
        refined.astype(
            np.float32
        )
        /
        255.0
    )

    alpha = (
        alpha[:, :, None]
        *
        0.45
    )

    preview = (
        preview.astype(
            np.float32
        )
        *
        (
            1.0 - alpha
        )
        +
        green.astype(
            np.float32
        )
        *
        alpha
    ).astype(
        np.uint8
    )

    # Bounding rectangle.
    cv2.rectangle(
        preview,
        (
            x1,
            y1
        ),
        (
            x2,
            y2
        ),
        (
            0,
            255,
            0
        ),
        2
    )

    preview = cv2.cvtColor(
        preview,
        cv2.COLOR_BGR2RGB
    )

    report = f"""
## 🔍 Analysis Complete

### Detected

**Type:** `{kind}`

**Auto-selected area:** `{bw} × {bh}`

**Mask coverage:** `{ratio:.2f}%`

**Edge complexity:** `{edge_density:.3f}`

### Green area

The **green overlay is the mask the remover will use**.

The original Frame 1 remains unchanged.

### Next

If the green selection correctly covers the watermark:

**Click `REMOVE WATERMARK`.**

If the selection is too small, increase **Auto Selection
Strength** and analyze again.

The system intentionally avoids blindly expanding the mask,
because expanding over a person's face, hair, clothing, or
background can make AI reconstruction worse.
"""

    return (
        preview,
        refined,
        report
    )


# ============================================================
# LAMA
# ============================================================

def lama_inpaint(
    image,
    mask
):

    model = get_lama()

    h, w = image.shape[:2]

    rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    pil_image = Image.fromarray(
        rgb
    )

    pil_mask = Image.fromarray(
        mask,
        mode="L"
    )

    # Keep inference on CUDA and disable autograd to reduce GPU memory
    # usage and inference overhead.
    with torch.inference_mode():
        result = model(
            pil_image,
            pil_mask
        )

    if isinstance(
        result,
        Image.Image
    ):

        result = np.asarray(
            result
        )

    else:

        result = np.asarray(
            result
        )

    result = np.clip(
        result,
        0,
        255
    ).astype(
        np.uint8
    )

    if (
        result.shape[0] != h
        or
        result.shape[1] != w
    ):

        result = cv2.resize(
            result,
            (
                w,
                h
            ),
            interpolation=cv2.INTER_LANCZOS4
        )

    return cv2.cvtColor(
        result,
        cv2.COLOR_RGB2BGR
    )


# ============================================================
# PROCESS VIDEO
# ============================================================

def remove_watermark(
    video,
    refined_mask,
    context_padding
):

    path = get_video_path(video)

    if not path:
        raise gr.Error(
            "Upload a video first."
        )

    if refined_mask is None:
        raise gr.Error(
            "Analyze the watermark first."
        )

    mask = np.asarray(
        refined_mask
    )

    if mask.ndim == 3:
        mask = mask[:, :, 0]

    mask = np.where(
        mask > 10,
        255,
        0
    ).astype(
        np.uint8
    )

    if cv2.countNonZero(mask) == 0:
        raise gr.Error(
            "Automatic mask is empty."
        )

    cap = cv2.VideoCapture(
        path
    )

    if not cap.isOpened():
        raise gr.Error(
            "Could not open video."
        )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if not fps or fps <= 0:
        fps = 30.0

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    total = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    # --------------------------------------------------------
    # Make mask match actual video.
    # --------------------------------------------------------

    if (
        mask.shape[1] != width
        or
        mask.shape[0] != height
    ):

        mask = cv2.resize(
            mask,
            (
                width,
                height
            ),
            interpolation=cv2.INTER_NEAREST
        )

    # --------------------------------------------------------
    # Context region.
    # --------------------------------------------------------

    ys, xs = np.where(
        mask > 0
    )

    x1 = max(
        0,
        int(xs.min()) - int(context_padding)
    )

    y1 = max(
        0,
        int(ys.min()) - int(context_padding)
    )

    x2 = min(
        width,
        int(xs.max()) + int(context_padding) + 1
    )

    y2 = min(
        height,
        int(ys.max()) + int(context_padding) + 1
    )

    local_mask = mask[
        y1:y2,
        x1:x2
    ]

    crop_h, crop_w = local_mask.shape

    print()
    print("=" * 70)
    print("AUTOMATIC WATERMARK REMOVAL")
    print("=" * 70)
    print("Video       :", f"{width}x{height}")
    print("FPS         :", f"{fps:.2f}")
    print("Frames      :", total)
    print(
        "AI region   :",
        f"{crop_w}x{crop_h}"
    )
    print(
        "Context     :",
        f"{context_padding}px"
    )
    print("Device      :", DEVICE_NAME)
    print("=" * 70)
    print()

    # --------------------------------------------------------
    # Files.
    # --------------------------------------------------------

    job = uuid.uuid4().hex

    silent = os.path.join(
        WORK_DIR,
        f"{job}_silent.mp4"
    )

    output = os.path.join(
        WORK_DIR,
        f"{job}_removed.mp4"
    )

    # --------------------------------------------------------
    # Writer.
    # --------------------------------------------------------

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        silent,
        fourcc,
        fps,
        (
            width,
            height
        )
    )

    if not writer.isOpened():

        cap.release()

        raise gr.Error(
            "Could not create output video."
        )

    processed = 0

    try:

        while True:

            ok, frame = cap.read()

            if not ok:
                break

            original = frame.copy()

            crop = original[
                y1:y2,
                x1:x2
            ].copy()

            if (
                crop.shape[:2]
                != local_mask.shape[:2]
            ):

                raise RuntimeError(
                    "Video crop and mask dimensions differ."
                )

            # ------------------------------------------------
            # AI CONTENT-AWARE RECONSTRUCTION
            # ------------------------------------------------

            reconstructed = lama_inpaint(
                crop,
                local_mask
            )

            # ------------------------------------------------
            # ONLY REPLACE AUTO-DETECTED MASK
            # ------------------------------------------------

            selected = (
                local_mask > 0
            )

            clean_crop = crop.copy()

            clean_crop[
                selected
            ] = reconstructed[
                selected
            ]

            result = original.copy()

            result[
                y1:y2,
                x1:x2
            ] = clean_crop

            writer.write(
                result
            )

            processed += 1

            if processed % 5 == 0:

                percent = (
                    processed
                    /
                    max(
                        total,
                        1
                    )
                ) * 100

                print(
                    f"\rProcessing "
                    f"{processed}/{total} "
                    f"({percent:.1f}%)",
                    end="",
                    flush=True
                )

    except Exception as error:

        cap.release()
        writer.release()

        if os.path.exists(silent):
            os.remove(silent)

        raise gr.Error(
            "Removal failed:\n\n"
            + str(error)
        )

    cap.release()
    writer.release()

    print()

    if processed == 0:

        if os.path.exists(silent):
            os.remove(silent)

        raise gr.Error(
            "No frames were processed."
        )

    # ========================================================
    # RESTORE AUDIO
    # ========================================================

    command = [
        "ffmpeg",
        "-y",

        "-i",
        silent,

        "-i",
        path,

        "-map",
        "0:v:0",

        "-map",
        "1:a?",

        "-c:v",
        "libx264",

        "-preset",
        "medium",

        "-crf",
        "18",

        "-pix_fmt",
        "yuv420p",

        "-c:a",
        "aac",

        "-b:a",
        "192k",

        "-movflags",
        "+faststart",

        output
    ]

    try:

        ff = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

    except FileNotFoundError:

        if os.path.exists(silent):
            os.remove(silent)

        raise gr.Error(
            "FFmpeg is not installed.\n\n"
            "Fedora:\n"
            "sudo dnf install ffmpeg"
        )

    if ff.returncode != 0:

        if os.path.exists(silent):
            os.remove(silent)

        raise gr.Error(
            "FFmpeg error:\n\n"
            + ff.stderr[-4000:]
        )

    if not os.path.exists(output):

        if os.path.exists(silent):
            os.remove(silent)

        raise gr.Error(
            "Output video was not created."
        )

    try:
        os.remove(silent)
    except OSError:
        pass

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)
    print(output)
    print()

    return output


# ============================================================
# GRADIO APP
# ============================================================

with gr.Blocks(
    title="Smart AI Watermark Remover"
) as demo:

    gr.Markdown(
        f"""
# 🎬 Smart AI Watermark Remover

### GPU-Only • Rough Select → Analyze → Auto Select → Remove

**AI:** LaMa Content-Aware Inpainting  
**Device:** `{DEVICE_NAME}`  
**Mode:** GPU ONLY (CUDA)

You do **not** need to paint the watermark perfectly.

Roughly paint around the watermark on **Frame 1**.
The analyzer will attempt to refine your selection automatically.
"""
    )

    # ========================================================
    # VIDEO
    # ========================================================

    video = gr.Video(
        label="1️⃣ Upload Video",
        sources=["upload"],
        format="mp4"
    )

    load_button = gr.Button(
        "📸 LOAD FRAME 1",
        variant="secondary",
        size="lg"
    )

    # ========================================================
    # FRAME 1 EDITOR
    # ========================================================

    gr.Markdown(
        """
## 2️⃣ Roughly Select the Watermark

You don't need to trace the watermark perfectly.

Just paint/circle the watermark area.
"""
    )

    with gr.Row():

        with gr.Column(
            scale=3
        ):

            editor = gr.ImageEditor(
                label="Frame 1",
                type="numpy",
                layers=True,

                brush=gr.Brush(
                    default_size=35,
                    colors=[
                        "#ff0000"
                    ],
                    default_color="#ff0000",
                    color_mode="fixed"
                ),

                eraser=gr.Eraser(
                    default_size=35
                ),

                transforms=(),

                canvas_size=(
                    1400,
                    900
                ),

                height=650,

                interactive=True
            )

        with gr.Column(
            scale=1
        ):

            rough_preview = gr.Image(
                label="Rough Selection",
                type="numpy",
                height=650
            )

    load_button.click(
        fn=load_frame,
        inputs=video,
        outputs=editor
    )

    editor.change(
        fn=show_mask,
        inputs=editor,
        outputs=rough_preview
    )

    # ========================================================
    # AUTO ANALYSIS
    # ========================================================

    gr.Markdown(
        """
## 3️⃣ Analyze & Auto-Select

The analyzer will refine the rough selection using the actual
Frame 1 image.

It does **not** simply make your mask bigger.
"""
    )

    with gr.Row():

        with gr.Column(
            scale=3
        ):

            analyze_button = gr.Button(
                "🔍 ANALYZE & AUTO-SELECT",
                variant="primary",
                size="lg"
            )

        with gr.Column(
            scale=1
        ):

            strength = gr.Slider(
                minimum=20,
                maximum=100,
                value=50,
                step=5,
                label="Selection Strength"
            )

    auto_preview = gr.Image(
        label="🟢 Automatically Detected Watermark",
        type="numpy",
        height=600
    )

    analysis_text = gr.Markdown(
        "Waiting for analysis..."
    )

    refined_mask = gr.State(
        None
    )

    analyze_button.click(
        fn=analyze_and_select,
        inputs=[
            video,
            editor,
            strength
        ],
        outputs=[
            auto_preview,
            refined_mask,
            analysis_text
        ]
    )

    # ========================================================
    # CONTEXT
    # ========================================================

    with gr.Accordion(
        "⚙️ Advanced AI Context",
        open=False
    ):

        context = gr.Slider(
            minimum=50,
            maximum=500,
            value=180,
            step=10,
            label="AI Context",
            info=(
                "Extra surrounding image supplied to LaMa."
            )
        )

    # ========================================================
    # REMOVE
    # ========================================================

    gr.Markdown(
        """
## 4️⃣ Remove

After the green mask looks correct, click Remove.
"""
    )

    remove_button = gr.Button(
        "✨ REMOVE WATERMARK",
        variant="primary",
        size="lg"
    )

    output = gr.Video(
        label="✅ Final Video",
        format="mp4"
    )

    remove_button.click(
        fn=remove_watermark,
        inputs=[
            video,
            refined_mask,
            context
        ],
        outputs=output
    )

    # ========================================================
    # HELP
    # ========================================================

    gr.Markdown(
        """
---

### Best practice

For a logo/text watermark:

**Roughly paint → Analyze → inspect green mask → Remove**

If the green mask includes part of the character, reduce the
rough selection and analyze again.

If the green mask misses part of the watermark, increase
**Selection Strength** slightly.

### Important

Content-aware AI reconstructs pixels that are hidden by the
watermark. It cannot know the exact original pixels if the
watermark completely covers them.

This application therefore preserves the original video
everywhere outside the automatically detected mask.
"""
    )


# ============================================================
# SERVER
# ============================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "GRADIO_SERVER_PORT",
            os.environ.get(
                "PORT",
                "7861"
            )
        )
    )

    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False
    )