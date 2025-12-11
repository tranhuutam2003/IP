import os
import glob
import cv2

# Import các hàm cần thiết từ package lpr
from lpr.pipeline import detect_plate_manual
from lpr.utils import imread_color


def main():
    # 1. THIẾT LẬP THƯ MỤC (thay đường dẫn của bạn)
    input_folder = "E:/Kztech/dataset/dataset_kztek/20250427/vehicle/test/"
    output_folder = "E:/Kztech/dataset/dataset_kztek/20250427/vehicle/test-out-3/"

    os.makedirs(output_folder, exist_ok=True)

    image_extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff')
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(input_folder, ext)))

    print(f"Tìm thấy {len(image_paths)} ảnh trong: {input_folder}")
    if not image_paths:
        print("Không tìm thấy ảnh nào. Bro kiểm tra lại đường dẫn input_folder.")
        return

    for image_path in image_paths:
        print(f"\n--- Đang xử lý: {image_path} ---")

        src = imread_color(image_path)
        if src is None:
            continue

        vis, plate = detect_plate_manual(src)

        base_name = os.path.basename(image_path)
        filename_no_ext, _ = os.path.splitext(base_name)
        output_crop_path = os.path.join(output_folder, f"{filename_no_ext}_crop.jpg")

        if plate is not None and plate.size > 0:
            cv2.imwrite(output_crop_path, plate)
            print(f"Đã lưu kết quả vào: {output_folder}")
        else:
            print(f"Không tìm thấy biển số cho ảnh: {base_name}")

    print("\n--- HOÀN TẤT XỬ LÝ TẤT CẢ ẢNH ---")


if __name__ == "__main__":
    main()