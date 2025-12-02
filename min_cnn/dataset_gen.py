import cv2
import numpy as np
import math
import os
import csv

# ==========================================
# [설정 구역]
VIDEO_PATH = "../drive_data.mp4"  # 사용할 영상 파일
SAVE_FOLDER = "dataset"
STRIDE = 1                     # 1이면 모든 프레임 저장 (데이터 최대화)
# ==========================================

def process_video():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"에러: {VIDEO_PATH} 파일을 찾을 수 없습니다.")
        return

    # 동영상 정보 출력 (참고용)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"영상 정보: FPS={fps:.2f}, 총 프레임 수={total_frames}")

    img_save_path = os.path.join(SAVE_FOLDER, "resize")
    os.makedirs(img_save_path, exist_ok=True)

    csv_path = os.path.join(SAVE_FOLDER, "labels.csv")
    
    # 파일 열기 ('w'모드: 기존 내용 지우고 새로 씀)
    f = open(csv_path, 'w', newline='') 
    wr = csv.writer(f)
    wr.writerow(["filename", "steering"])

    print("전체 영상 처리 시작...")
    saved_count = 0
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # STRIDE에 맞춰 샘플링
        if frame_count % STRIDE == 0:
            # 1. 전처리 (리사이즈 -> 흑백 -> 하단 자르기 -> 차선 검출)
            frame_resize = cv2.resize(frame, (320, 180))
            gray = cv2.cvtColor(frame_resize, cv2.COLOR_BGR2GRAY)
            height, width = frame_resize.shape[:2]
            roi_height = height // 10
            roi_y = height - roi_height
            
            line_row = gray[roi_y:roi_y + 1, :]
            _, thresholded = cv2.threshold(line_row, 100, 255, cv2.THRESH_BINARY_INV)
            nonzero_x = np.nonzero(thresholded)[1]

            if len(nonzero_x) > 0:
                left_x = nonzero_x[0]
                right_x = nonzero_x[-1]
                center_x = (left_x + right_x) // 2

                # 2. 각도 계산
                angle = math.degrees(math.atan(((320.0 - 2 * center_x) * 0.65) / 280)) * 3
                angle_deg = int(90 - angle)
                angle_deg = max(45, min(135, angle_deg))  # 범위 제한

                # --- [저장: 원본 한 장만] ---
                fname = f"frame_{frame_count:06d}.jpg"
                cv2.imwrite(os.path.join(img_save_path, fname), frame_resize)
                wr.writerow([fname, angle_deg])
                saved_count += 1

        frame_count += 1

    cap.release()
    f.close()
    print(f"총 {saved_count}장의 데이터 생성")

if __name__ == '__main__':
    process_video()