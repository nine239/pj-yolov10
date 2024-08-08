import cv2
import os
from ultralytics import YOLOv10 as YOLO


def get_bounding_box_coordinates(image_width, image_height, x_center, y_center, width, height):
    x_center_pixel = x_center * image_width
    y_center_pixel = y_center * image_height
    half_width = width * image_width / 2
    half_height = height * image_height / 2

    xmin = int(x_center_pixel - half_width)
    ymin = int(y_center_pixel - half_height)
    xmax = int(x_center_pixel + half_width)
    ymax = int(y_center_pixel + half_height)

    return xmin, ymin, xmax, ymax


def process_image(image_path, model, class_names, output_dir):
    # Đọc ảnh dự đoán
    image = cv2.imread(image_path)

    if image is None:
        print(f"Could not load image at {image_path}")
        return

    # Dự đoán với mô hình YOLO
    results = model(image)

    # Lấy kết quả dự đoán
    boxes = results[0].boxes.xyxy.cpu().numpy()  # Tọa độ bounding box
    scores = results[0].boxes.conf.cpu().numpy()  # Độ tin cậy
    class_ids = results[0].boxes.cls.cpu().numpy()  # ID lớp

    # Lấy kích thước của ảnh
    img_width = image.shape[1]
    img_height = image.shape[0]

    for box, score, class_id in zip(boxes, scores, class_ids):
        xmin, ymin, xmax, ymax = map(int, box)
        class_id = int(class_id)
        object_name = class_names.get(class_id, 'Unknown')

        # Tính diện tích của bounding box
        width = xmax - xmin
        height = ymax - ymin
        area = width * height

        # Vẽ bounding box và tên đối tượng lên ảnh
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

        # Thay đổi kích thước phông chữ và độ dày của đường viền
        font_scale = 0.5
        font_thickness = 2

        # Thêm bóng cho văn bản
        text_color = (0, 255, 0)
        shadow_color = (0, 0, 0)

        # Vẽ bóng cho văn bản
        text = f"{object_name} ({xmin}, {ymin}), ({xmax}, {ymax}) - Area: {area})"
        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        text_x, text_y = xmin, ymin - 10
        shadow_offset = 2
        cv2.putText(image, text, (text_x + shadow_offset, text_y + shadow_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    shadow_color, font_thickness + 1)
        cv2.putText(image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness)

    # Lưu ảnh đã xử lý vào tệp
    base_name = os.path.basename(image_path)
    output_path = os.path.join(output_dir, base_name)
    cv2.imwrite(output_path, image)
    print(f"Image saved to {output_path}")


def main(image_dir, model, class_names, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)  # Tạo thư mục nếu chưa tồn tại

    image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.png')]

    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        process_image(image_path, model, class_names, output_dir)


if __name__ == "__main__":
    image_dir = 'custom_dataset/dataset/train/images'
    output_dir = 'processed_images'
    model_path = 'runs/detect/train/weights/best.pt'

    class_names = {0: 'Bia_Truc_Bach', 1: 'Coca_Cola', 2: 'Fanta',
                   3: 'Heiniken'}  # Thay thế bằng danh sách các tên lớp của bạn

    model = YOLO(model_path)

    main(image_dir, model, class_names, output_dir)
