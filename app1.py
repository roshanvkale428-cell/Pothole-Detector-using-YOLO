import gradio as gr
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from PIL import Image

# 1. Automatically download a pre-trained YOLOv8 pothole model
model_path = hf_hub_download(
    repo_id="keremberke/yolov8m-pothole-segmentation", 
    filename="best.pt"
)
model = YOLO(model_path)

def detect_and_scale(image):
    if image is None:
        return None, "Please upload an image."
    
    # Run YOLO inference
    results = model.predict(image, conf=0.25)[0]
    
    # Generate the annotated image (converted from BGR to RGB for Gradio)
    annotated_img = Image.fromarray(results.plot()[..., ::-1])
    
    # Calculate total image area in pixels
    img_width, img_height = image.size
    total_area = img_width * img_height
    
    report = []
    boxes = results.boxes
    
    if len(boxes) == 0:
        return annotated_img, "No potholes detected."

    # Loop through each detected pothole and calculate its relative scale
    for i, box in enumerate(boxes):
        # xywh returns [center_x, center_y, width, height]
        _, _, w, h = box.xywh[0].tolist()
        pothole_area = w * h
        coverage_pct = (pothole_area / total_area) * 100
        confidence = box.conf[0].item() * 100
        
        # Categorize severity based on percentage of image covered
        if coverage_pct < 3.0:
            scale = "Small (Low Severity)"
        elif coverage_pct < 10.0:
            scale = "Medium (Moderate Severity)"
        else:
            scale = "Large (High Severity)"
            
        report.append(
            f"Pothole #{i+1}: {scale} | "
            f"Screen Coverage: {coverage_pct:.2f}% | "
            f"Confidence: {confidence:.1f}%"
        )
        
    return annotated_img, "\n".join(report)

# 3. Build the simple Gradio web interface
demo = gr.Interface(
    fn=detect_and_scale,
    inputs=gr.Image(type="pil", label="Upload Road Image"),
    outputs=[
        gr.Image(type="pil", label="Detected Potholes"),
        gr.Textbox(label="Pothole Scale & Severity Report", lines=5)
    ],
    title="Road Pothole Detector & Scaler",
    description="Upload a photo of a road to detect potholes and estimate their size."
)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    demo.launch(server_name="0.0.0.0", server_port=port)

