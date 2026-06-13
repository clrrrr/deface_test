import gradio as gr
import cv2
from scrfd import SCRFD

detector = SCRFD()

def detect_faces(image, threshold):
    if image is None:
        return None, "请上传图片"

    img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    dets, _ = detector.detect(img, threshold=threshold)

    for x1, y1, x2, y2, score in dets:
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(img, f'{score:.2f}', (int(x1), int(y1)-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    result = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return result, f"检测到 {len(dets)} 张人脸"

with gr.Blocks(title="SCRFD人脸检测") as demo:
    gr.Markdown("# SCRFD 人脸检测")
    with gr.Row():
        with gr.Column():
            img_in = gr.Image(label="上传图片")
            threshold = gr.Slider(0.1, 0.9, value=0.5, step=0.05, label="检测阈值")
            btn = gr.Button("检测", variant="primary")
        with gr.Column():
            img_out = gr.Image(label="结果")
            info = gr.Textbox(label="信息")

    btn.click(detect_faces, [img_in, threshold], [img_out, info])

demo.launch(server_name="0.0.0.0", server_port=7860)
