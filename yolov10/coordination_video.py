import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLOv10 as YOLO

def calculate_overlap_area(box1, box2):
    # Tính tọa độ chồng chéo
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # Tính diện tích chồng chéo
    if x1 < x2 and y1 < y2:
        overlap_area = (x2 - x1) * (y2 - y1)
    else:
        overlap_area = 0

    return overlap_area

def process_frame(frame, model, class_names):
    results = model(frame)
    boxes = results[0].boxes.xyxy.cpu().numpy()  # Tọa độ bounding box
    scores = results[0].boxes.conf.cpu().numpy()  # Độ tin cậy
    class_ids = results[0].boxes.cls.cpu().numpy()  # ID lớp

    img_width = frame.shape[1]
    img_height = frame.shape[0]

    # Tạo danh sách để lưu thông tin
    data = []

    for i, (box, score, class_id) in enumerate(zip(boxes, scores, class_ids)):
        xmin, ymin, xmax, ymax = map(int, box)
        class_id = int(class_id)
        object_name = class_names.get(class_id, 'Unknown')

        # Tính diện tích của bounding box
        width = xmax - xmin
        height = ymax - ymin
        area = width * height

        # Vẽ bounding box
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

        # Tính diện tích chồng chéo với các bounding box khác
        overlap_area = 0
        for j in range(len(boxes)):
            if i != j:
                other_box = boxes[j]
                overlap_area += calculate_overlap_area(box, other_box)

        # Tính tỷ lệ phần trăm diện tích chồng chéo
        percentage = (overlap_area / area * 100) if area > 0 else 0

        # Thay đổi kích thước phông chữ và độ dày của đường viền
        font_scale = 0.6
        font_thickness = 1

        # Thêm bóng cho văn bản
        text_color = (0, 255, 0)
        shadow_color = (0, 0, 0)

        # Vẽ tọa độ của bounding box ở trên
        text_top = f"{object_name}: ({xmin}, {ymin}), ({xmax}, {ymax})"
        text_size_top, _ = cv2.getTextSize(text_top, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        text_x_top, text_y_top = xmin, ymin - 10
        shadow_offset = 1
        cv2.putText(frame, text_top, (text_x_top + shadow_offset, text_y_top + shadow_offset), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, shadow_color, font_thickness + 1)
        cv2.putText(frame, text_top, (text_x_top, text_y_top), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness)

        # Vẽ diện tích và tỷ lệ phần trăm ở dưới
        text_bottom = f"Area: {area} | Overlap: {overlap_area} | Percent: {percentage:.2f}%"
        text_size_bottom, _ = cv2.getTextSize(text_bottom, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        text_x_bottom, text_y_bottom = xmin, ymax + text_size_bottom[1] + 20
        cv2.putText(frame, text_bottom, (text_x_bottom, text_y_bottom), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness)

        # Thêm thông tin vào danh sách
        data.append({
            'Object Name': object_name,
            'Bounding Box': f"({xmin}, {ymin}), ({xmax}, {ymax})",
            'Area': area,
            'Overlap Area': overlap_area,
            'Percentage': percentage
        })

    return frame, data

def process_video(video_path, model, class_names, output_path, csv_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open video at {video_path}")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    all_data = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame, data = process_frame(frame, model, class_names)
        out.write(processed_frame)

        # Thêm dữ liệu vào danh sách tổng hợp
        all_data.extend(data)

    cap.release()
    out.release()
    print(f"Processed video saved to {output_path}")

    # Lưu dữ liệu vào file CSV
    df = pd.DataFrame(all_data)
    df.to_csv(csv_path, index=False)
    print(f"Data saved to {csv_path}")

def main(video_path, model, class_names, output_path, csv_path):
    process_video(video_path, model, class_names, output_path, csv_path)

if __name__ == "__main__":
    video_path = 'vdtest/testvd.mp4'  # Đường dẫn tới video đầu vào
    output_path = 'processed_video/processed_video13.avi'  # Đường dẫn tới video đầu ra
    csv_path = 'processed_video/processed_data.csv'  # Đường dẫn tới file CSV đầu ra
    model_path = 'runs/detect/train/weights/best.pt'

    class_names = {0: 'Bia_Truc_Bach', 1: 'Coca_Cola', 2: 'Fanta', 3: 'Heiniken'}  # Thay thế bằng danh sách các tên lớp của bạn

    model = YOLO(model_path)

    main(video_path, model, class_names, output_path, csv_path)
