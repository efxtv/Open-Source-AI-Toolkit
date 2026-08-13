import os
import cv2
import uuid
import torch
import argparse
import tempfile
import subprocess
import numpy as np
import gradio as gr

from PIL import Image
from simple_lama_inpainting import SimpleLama


# ============================================================
# COMMAND LINE
#
# python app.py
# python app.py -cpu
# python app.py -gpu
# ============================================================

parser = argparse.ArgumentParser(
    description="Automatic AI Video Watermark Remover"
)

device_group = parser.add_mutually_exclusive_group()

device_group.add_argument(
    "-cpu",
    action="store_true",
    help="Force CPU"
)

device_group.add_argument(
    "-gpu",
    action="store_true",
    help="Force NVIDIA CUDA GPU"
)

args = parser.parse_args()


# ============================================================
# DEVICE
# ============================================================

cuda_available = torch.cuda.is_available()

if args.gpu:

    if not cuda_available:
        print()
        print("ERROR: CUDA is not available.")
        print("Use: python app.py -cpu")
        print()
        raise SystemExit(1)

    DEVICE = torch.device("cuda")
    DEVICE_LABEL = torch.cuda.get_device_name(0)
    MODE = "GPU"

elif args.cpu:

    DEVICE = torch.device("cpu")
    DEVICE_LABEL = "CPU"
    MODE = "CPU"

else:

    if cuda_available:
        DEVICE = torch.device("cuda")
        DEVICE_LABEL = torch.cuda.get_device_name(0)
        MODE = "AUTO → GPU"
    else:
        DEVICE = torch.device("cpu")
        DEVICE_LABEL = "CPU"
        MODE = "AUTO → CPU"


print()
print("=" * 70)
print("          AUTOMATIC AI VIDEO WATERMARK REMOVER")
print("=" * 70)
print("Mode   :", MODE)
print("Device :", DEVICE_LABEL)
print("=" * 70)
print()


# ============================================================
# WORK DIRECTORY
# ============================================================

WORK_DIR = os.path.join(
    tempfile.gettempdir(),
    "automatic_watermark_remover"
)

os.makedirs(
    WORK_DIR,
    exist_ok=True
)


# ============================================================
# MODEL
# ============================================================

LAMA_MODEL = None


def get_lama():

    global LAMA_MODEL

    if LAMA_MODEL is None:

        print()
        print("Loading LaMa...")
        print("Device:", DEVICE_LABEL)

        LAMA_MODEL = SimpleLama(
            device=DEVICE
        )

        print("LaMa loaded.")
        print()

    return LAMA_MODEL


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

def load_frame_1(video):

    path = get_video_path(video)

    if not path:

        raise gr.Error(
            "Please upload a video first."
        )

    cap = cv2.VideoCapture(path)

    if not cap.isOpened():

        raise gr.Error(
            "Could not open the video."
        )

    ok, frame = cap.read()

    cap.release()

    if not ok:

        raise gr.Error(
            "Could not read frame 1."
        )

    frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    return frame


# ============================================================
# EXTRACT USER MASK
# ============================================================

def extract_mask(editor):

    if editor is None:

        raise gr.Error(
            "Load Frame 1 and paint over the watermark."
        )

    if not isinstance(editor, dict):

        raise gr.Error(
            "Invalid Frame 1 editor data."
        )

    layers = editor.get(
        "layers",
        []
    )

    if not layers:

        raise gr.Error(
            "No selection found."
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
            alpha > 15,
            255,
            0
        ).astype(np.uint8)

        if cv2.countNonZero(mask) > 0:

            return mask

    raise gr.Error(
        "Paint over the watermark on Frame 1."
    )


# ============================================================
# MASK PREVIEW
# ============================================================

def preview_mask(editor):

    try:

        mask = extract_mask(editor)

        preview = np.zeros(
            (
                mask.shape[0],
                mask.shape[1],
                3
            ),
            dtype=np.uint8
        )

        preview[:, :, 0] = mask
        preview[:, :, 1] = mask
        preview[:, :, 2] = mask

        return preview

    except Exception:

        return None


# ============================================================
# ANALYZE WATERMARK
# ============================================================

def analyze_watermark(
    video,
    editor
):

    path = get_video_path(video)

    if not path:

        raise gr.Error(
            "Upload a video first."
        )

    user_mask = extract_mask(
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

    height, width = frame.shape[:2]

    # --------------------------------------------------------
    # Convert mask to actual video dimensions.
    # --------------------------------------------------------

    mask = cv2.resize(
        user_mask,
        (
            width,
            height
        ),
        interpolation=cv2.INTER_NEAREST
    )

    mask = np.where(
        mask > 0,
        255,
        0
    ).astype(np.uint8)

    pixels = cv2.countNonZero(
        mask
    )

    if pixels == 0:

        raise gr.Error(
            "Watermark selection is empty."
        )

    # --------------------------------------------------------
    # Bounding box.
    # --------------------------------------------------------

    ys, xs = np.where(
        mask > 0
    )

    x1 = int(xs.min())
    y1 = int(ys.min())
    x2 = int(xs.max()) + 1
    y2 = int(ys.max()) + 1

    box_width = x2 - x1
    box_height = y2 - y1

    video_area = width * height
    mask_ratio = pixels / video_area

    # --------------------------------------------------------
    # Analyze visual characteristics.
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    roi = gray[
        y1:y2,
        x1:x2
    ]

    roi_mask = mask[
        y1:y2,
        x1:x2
    ]

    if roi.size:

        masked_pixels = roi[
            roi_mask > 0
        ]

        if len(masked_pixels):

            brightness = float(
                np.mean(
                    masked_pixels
                )
            )

            contrast = float(
                np.std(
                    masked_pixels
                )
            )

        else:

            brightness = 0
            contrast = 0

    else:

        brightness = 0
        contrast = 0

    # --------------------------------------------------------
    # Edge analysis.
    # --------------------------------------------------------

    edges = cv2.Canny(
        roi,
        50,
        150
    )

    edge_density = (
        np.count_nonzero(edges)
        /
        max(edges.size, 1)
    )

    # --------------------------------------------------------
    # Heuristic classification.
    #
    # This is intentionally conservative. The analyzer does
    # not pretend to know the exact hidden pixels.
    # --------------------------------------------------------

    if box_width < width * 0.35 and box_height < height * 0.20:

        if edge_density > 0.10:

            watermark_type = (
                "Complex logo / graphic"
            )

            recommendation = (
                "AI Content-Aware"
            )

        else:

            watermark_type = (
                "Text / simple logo"
            )

            recommendation = (
                "AI Content-Aware"
            )

    elif mask_ratio < 0.12:

        watermark_type = (
            "Medium overlay / logo"
        )

        recommendation = (
            "AI Content-Aware"
        )

    else:

        watermark_type = (
            "Large or complex overlay"
        )

        recommendation = (
            "AI Content-Aware"
        )

    # --------------------------------------------------------
    # Confidence.
    # --------------------------------------------------------

    if mask_ratio < 0.05:
        confidence = "High"

    elif mask_ratio < 0.15:
        confidence = "Medium"

    else:
        confidence = "Low"

    # --------------------------------------------------------
    # Save analysis information.
    # --------------------------------------------------------

    analysis = {
        "width": width,
        "height": height,
        "mask": mask,
        "bbox": (
            x1,
            y1,
            x2,
            y2
        ),
        "type": watermark_type,
        "recommendation": recommendation,
        "confidence": confidence
    }

    # Store temporarily by process.
    analysis_id = uuid.uuid4().hex

    analysis_path = os.path.join(
        WORK_DIR,
        f"{analysis_id}.npz"
    )

    np.savez_compressed(
        analysis_path,
        mask=mask,
        bbox=np.array(
            [x1, y1, x2, y2]
        )
    )

    report = f"""
## 🔍 Watermark Analysis Complete

**Detected type:** `{watermark_type}`

**Recommended engine:** `{recommendation}`

**Analysis confidence:** `{confidence}`

### Selection

Video resolution:

`{width} × {height}`

Watermark region:

`{box_width} × {box_height}`

Selected area:

`{mask_ratio * 100:.2f}%` of the frame

### AI strategy

The application will:

1. Use your **Frame 1 selection** as the master watermark mask.
2. Automatically create a context region around it.
3. Run content-aware reconstruction only on that region.
4. Replace only the selected watermark pixels.
5. Preserve the rest of every original frame.
6. Keep the original audio.

**Important:** AI reconstruction can estimate pixels hidden by
the watermark, but it cannot recover pixels that are completely
absent from the source video.
"""

    return report, analysis_path


# ============================================================
# LOAD ANALYSIS
# ============================================================

def load_analysis(
    analysis_path
):

    if not analysis_path:

        raise gr.Error(
            "Analyze the watermark first."
        )

    if not os.path.exists(
        analysis_path
    ):

        raise gr.Error(
            "Analysis data is missing. "
            "Please analyze the watermark again."
        )

    data = np.load(
        analysis_path
    )

    mask = data["mask"]

    bbox = data["bbox"]

    x1, y1, x2, y2 = [
        int(v)
        for v in bbox
    ]

    return (
        mask,
        x1,
        y1,
        x2,
        y2
    )


# ============================================================
# LAMA RECONSTRUCTION
# ============================================================

def lama_inpaint(
    image_bgr,
    mask
):

    model = get_lama()

    h, w = image_bgr.shape[:2]

    rgb = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB
    )

    image = Image.fromarray(
        rgb
    )

    mask_image = Image.fromarray(
        mask,
        mode="L"
    )

    result = model(
        image,
        mask_image
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
    ).astype(np.uint8)

    # --------------------------------------------------------
    # SimpleLama can pad the image.
    # Always restore exact dimensions.
    # --------------------------------------------------------

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
# REMOVE
# ============================================================

def remove_watermark(
    video,
    analysis_path,
    context_padding
):

    path = get_video_path(video)

    if not path:

        raise gr.Error(
            "Upload the video."
        )

    mask, x1, y1, x2, y2 = load_analysis(
        analysis_path
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

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    # --------------------------------------------------------
    # Make sure Frame 1 mask matches actual video.
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
    # Add context around the user's selection.
    # --------------------------------------------------------

    padding = int(
        context_padding
    )

    ys, xs = np.where(
        mask > 0
    )

    if len(xs) == 0:

        cap.release()

        raise gr.Error(
            "Watermark mask is empty."
        )

    cx1 = max(
        0,
        int(xs.min()) - padding
    )

    cy1 = max(
        0,
        int(ys.min()) - padding
    )

    cx2 = min(
        width,
        int(xs.max()) + padding + 1
    )

    cy2 = min(
        height,
        int(ys.max()) + padding + 1
    )

    context_mask = mask[
        cy1:cy2,
        cx1:cx2
    ]

    crop_height, crop_width = (
        context_mask.shape
    )

    print()
    print("=" * 65)
    print("AUTOMATIC WATERMARK REMOVAL")
    print("=" * 65)
    print("Video      :", f"{width}x{height}")
    print("FPS        :", fps)
    print("Frames     :", total_frames)
    print(
        "AI region  :",
        f"{crop_width}x{crop_height}"
    )
    print(
        "Context    :",
        f"{padding}px"
    )
    print("=" * 65)
    print()

    # --------------------------------------------------------
    # Output files.
    # --------------------------------------------------------

    job = uuid.uuid4().hex

    silent_file = os.path.join(
        WORK_DIR,
        f"{job}_silent.mp4"
    )

    final_file = os.path.join(
        WORK_DIR,
        f"{job}_final.mp4"
    )

    # --------------------------------------------------------
    # Video writer.
    # --------------------------------------------------------

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        silent_file,
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

    # --------------------------------------------------------
    # FRAME LOOP
    # --------------------------------------------------------

    try:

        while True:

            ok, frame = cap.read()

            if not ok:
                break

            original = frame.copy()

            crop = original[
                cy1:cy2,
                cx1:cx2
            ].copy()

            # Safety.
            if (
                crop.shape[:2]
                != context_mask.shape[:2]
            ):

                raise RuntimeError(
                    "Context crop and mask dimensions differ."
                )

            # ------------------------------------------------
            # AI RECONSTRUCTION
            # ------------------------------------------------

            reconstructed = lama_inpaint(
                crop,
                context_mask
            )

            # ------------------------------------------------
            # CRITICAL:
            #
            # Only selected watermark pixels are replaced.
            # The context pixels remain ORIGINAL.
            # ------------------------------------------------

            local_mask = (
                context_mask > 0
            )

            clean_crop = crop.copy()

            clean_crop[
                local_mask
            ] = reconstructed[
                local_mask
            ]

            # ------------------------------------------------
            # Put processed context back.
            # ------------------------------------------------

            result_frame = original.copy()

            result_frame[
                cy1:cy2,
                cx1:cx2
            ] = clean_crop

            writer.write(
                result_frame
            )

            processed += 1

            if processed % 5 == 0:

                percent = (
                    processed
                    /
                    max(
                        total_frames,
                        1
                    )
                ) * 100

                print(
                    f"\rProcessing "
                    f"{processed}/{total_frames} "
                    f"({percent:.1f}%)",
                    end="",
                    flush=True
                )

    except Exception as error:

        cap.release()
        writer.release()

        if os.path.exists(
            silent_file
        ):
            os.remove(
                silent_file
            )

        raise gr.Error(
            "Watermark removal failed:\n\n"
            + str(error)
        )

    cap.release()
    writer.release()

    print()

    if processed == 0:

        if os.path.exists(
            silent_file
        ):
            os.remove(
                silent_file
            )

        raise gr.Error(
            "No frames were processed."
        )

    # ========================================================
    # RESTORE ORIGINAL AUDIO
    # ========================================================

    ffmpeg = [
        "ffmpeg",
        "-y",

        "-i",
        silent_file,

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

        final_file
    ]

    try:

        result = subprocess.run(
            ffmpeg,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

    except FileNotFoundError:

        if os.path.exists(
            silent_file
        ):
            os.remove(
                silent_file
            )

        raise gr.Error(
            "FFmpeg is not installed.\n\n"
            "Fedora:\n"
            "sudo dnf install ffmpeg"
        )

    if result.returncode != 0:

        if os.path.exists(
            silent_file
        ):
            os.remove(
                silent_file
            )

        raise gr.Error(
            "FFmpeg failed:\n\n"
            + result.stderr[-4000:]
        )

    if not os.path.exists(
        final_file
    ):

        raise gr.Error(
            "Final video was not created."
        )

    # --------------------------------------------------------
    # Cleanup silent intermediate.
    # --------------------------------------------------------

    try:
        os.remove(
            silent_file
        )
    except OSError:
        pass

    print()
    print("=" * 65)
    print("WATERMARK REMOVAL COMPLETE")
    print("=" * 65)
    print(final_file)
    print()

    return final_file


# ============================================================
# GRADIO APPLICATION
# ============================================================

with gr.Blocks(
    title="Automatic AI Watermark Remover"
) as demo:

    gr.Markdown(
        f"""
# 🎬 Automatic AI Watermark Remover

### Frame 1 → Analyze → Remove

**Engine:** AI Content-Aware / LaMa  
**Device:** `{DEVICE_LABEL}`

The watermark is selected **once on Frame 1**.
The application then uses that selection throughout the video.

### Workflow

**1. Upload video**

**2. Load Frame 1**

**3. Paint the watermark**

**4. Analyze Watermark**

**5. Remove Watermark**
"""
    )

    # ========================================================
    # STEP 1
    # ========================================================

    video = gr.Video(
        label="1️⃣ Upload Video",
        sources=["upload"],
        format="mp4"
    )

    load_frame_button = gr.Button(
        "📸 LOAD FRAME 1",
        variant="secondary",
        size="lg"
    )

    # ========================================================
    # STEP 2
    # ========================================================

    gr.Markdown(
        """
## 2️⃣ Select the Watermark

Paint **only the watermark**.

Do not paint the entire person, background, or surrounding
scene. The AI will reconstruct the selected pixels using
the surrounding context.
"""
    )

    with gr.Row():

        with gr.Column(
            scale=3
        ):

            editor = gr.ImageEditor(
                label="Frame 1 — Paint Watermark",
                type="numpy",
                layers=True,

                brush=gr.Brush(
                    default_size=25,
                    colors=[
                        "#ff0000"
                    ],
                    default_color="#ff0000",
                    color_mode="fixed"
                ),

                eraser=gr.Eraser(
                    default_size=25
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

            mask_preview = gr.Image(
                label="Selection Preview",
                type="numpy",
                height=650
            )

    # ========================================================
    # LOAD FRAME
    # ========================================================

    load_frame_button.click(
        fn=load_frame_1,
        inputs=video,
        outputs=editor
    )

    # ========================================================
    # LIVE MASK PREVIEW
    # ========================================================

    editor.change(
        fn=preview_mask,
        inputs=editor,
        outputs=mask_preview
    )

    # ========================================================
    # STEP 3
    # ========================================================

    gr.Markdown(
        """
## 3️⃣ Analyze Watermark

The analyzer examines the selected region and determines
the recommended content-aware reconstruction strategy.
"""
    )

    analyze_button = gr.Button(
        "🔍 ANALYZE WATERMARK",
        variant="secondary",
        size="lg"
    )

    analysis_result = gr.Markdown(
        value="Waiting for watermark selection..."
    )

    analysis_file = gr.State(
        None
    )

    analyze_button.click(
        fn=analyze_watermark,
        inputs=[
            video,
            editor
        ],
        outputs=[
            analysis_result,
            analysis_file
        ]
    )

    # ========================================================
    # ADVANCED SETTINGS
    # ========================================================

    with gr.Accordion(
        "⚙️ Advanced Context Settings",
        open=False
    ):

        context_padding = gr.Slider(
            minimum=50,
            maximum=500,
            value=180,
            step=10,
            label="AI Context",
            info=(
                "Additional surrounding image supplied to "
                "the AI. 150–250 is a good starting range."
            )
        )

        gr.Markdown(
            """
**Recommended:** 150–250.

More context can improve reconstruction when the watermark
covers a character or detailed background, but it also
increases processing time.
"""
        )

    # ========================================================
    # STEP 4
    # ========================================================

    gr.Markdown(
        """
## 4️⃣ Remove Watermark

After analysis, press the button below.
"""
    )

    remove_button = gr.Button(
        "✨ REMOVE WATERMARK",
        variant="primary",
        size="lg"
    )

    # ========================================================
    # OUTPUT
    # ========================================================

    output = gr.Video(
        label="✅ Watermark Removed",
        format="mp4"
    )

    # ========================================================
    # REMOVE EVENT
    # ========================================================

    remove_button.click(
        fn=remove_watermark,
        inputs=[
            video,
            analysis_file,
            context_padding
        ],
        outputs=output
    )

    # ========================================================
    # FOOTER
    # ========================================================

    gr.Markdown(
        """
---

### 💡 Important

The AI reconstructs pixels hidden by the watermark from
surrounding visual information. If the watermark completely
covers an object, face, hand, clothing, etc., the exact
original pixels may not be recoverable from a single frame.

For best results, keep the Frame 1 mask as tight as possible.
"""
    )


# ============================================================
# START
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
