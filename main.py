from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import mediapipe as mp

app = FastAPI(title="Sprint Pose Analysis API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

mp_pose = mp.solutions.pose


def calculate_angle(a, b, c):
    """
    Calculate angle at point b.
    a, b, c are 2D points: (x, y)
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    cosine = np.clip(cosine, -1.0, 1.0)
    angle = np.degrees(np.arccos(cosine))

    return round(float(angle), 2)


def midpoint(p1, p2):
    return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)


def get_point(landmarks, idx):
    lm = landmarks[idx]
    return (lm.x, lm.y, lm.visibility)


def visibility_ok(*points, threshold=0.35):
    return all(p[2] >= threshold for p in points)


@app.get("/")
def health_check():
    return {
        "status": "ok",
        "message": "Sprint Pose Analysis API is running."
    }


@app.post("/analyze_sprint_image")
async def analyze_sprint_image(request: Request):
    """
    Receive image bytes from Dify HTTP Request node.
    Content-Type: application/octet-stream
    Return landmarks, joint angles, technical issues, and potential risk flags.
    """

    image_bytes = await request.body()

    if not image_bytes:
        return {
            "success": False,
            "error": "No image bytes received.",
            "visual_summary": "未接收到图片数据。"
        }

    image_array = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        return {
            "success": False,
            "error": "Failed to decode image.",
            "visual_summary": "图片解码失败，请确认上传的是jpg或png格式图片。"
        }

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]

    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=False,
        min_detection_confidence=0.5
    ) as pose:
        result = pose.process(image_rgb)

    if not result.pose_landmarks:
        return {
            "success": False,
            "error": "No human pose detected.",
            "visual_summary": "未检测到清晰人体姿态，建议上传侧面、清晰、完整身体的短跑图片。"
        }

    landmarks = result.pose_landmarks.landmark

    # MediaPipe Pose landmark indexes
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32

    # Extract landmarks
    ls = get_point(landmarks, LEFT_SHOULDER)
    rs = get_point(landmarks, RIGHT_SHOULDER)
    lh = get_point(landmarks, LEFT_HIP)
    rh = get_point(landmarks, RIGHT_HIP)
    lk = get_point(landmarks, LEFT_KNEE)
    rk = get_point(landmarks, RIGHT_KNEE)
    la = get_point(landmarks, LEFT_ANKLE)
    ra = get_point(landmarks, RIGHT_ANKLE)
    lf = get_point(landmarks, LEFT_FOOT_INDEX)
    rf = get_point(landmarks, RIGHT_FOOT_INDEX)

    joint_angles = {}

    # Left lower limb angles
    if visibility_ok(ls, lh, lk):
        joint_angles["left_hip_angle"] = calculate_angle(
            (ls[0], ls[1]), (lh[0], lh[1]), (lk[0], lk[1])
        )
    else:
        joint_angles["left_hip_angle"] = None

    if visibility_ok(lh, lk, la):
        joint_angles["left_knee_angle"] = calculate_angle(
            (lh[0], lh[1]), (lk[0], lk[1]), (la[0], la[1])
        )
    else:
        joint_angles["left_knee_angle"] = None

    if visibility_ok(lk, la, lf):
        joint_angles["left_ankle_angle"] = calculate_angle(
            (lk[0], lk[1]), (la[0], la[1]), (lf[0], lf[1])
        )
    else:
        joint_angles["left_ankle_angle"] = None

    # Right lower limb angles
    if visibility_ok(rs, rh, rk):
        joint_angles["right_hip_angle"] = calculate_angle(
            (rs[0], rs[1]), (rh[0], rh[1]), (rk[0], rk[1])
        )
    else:
        joint_angles["right_hip_angle"] = None

    if visibility_ok(rh, rk, ra):
        joint_angles["right_knee_angle"] = calculate_angle(
            (rh[0], rh[1]), (rk[0], rk[1]), (ra[0], ra[1])
        )
    else:
        joint_angles["right_knee_angle"] = None

    if visibility_ok(rk, ra, rf):
        joint_angles["right_ankle_angle"] = calculate_angle(
            (rk[0], rk[1]), (ra[0], ra[1]), (rf[0], rf[1])
        )
    else:
        joint_angles["right_ankle_angle"] = None

    # Trunk lean angle relative to vertical
    shoulder_mid = midpoint((ls[0], ls[1]), (rs[0], rs[1]))
    hip_mid = midpoint((lh[0], lh[1]), (rh[0], rh[1]))

    trunk_vector = np.array([
        shoulder_mid[0] - hip_mid[0],
        shoulder_mid[1] - hip_mid[1]
    ])
    vertical_vector = np.array([0, -1])

    cos_val = np.dot(trunk_vector, vertical_vector) / (
        np.linalg.norm(trunk_vector) * np.linalg.norm(vertical_vector) + 1e-8
    )
    cos_val = np.clip(cos_val, -1.0, 1.0)
    trunk_lean_angle = round(float(np.degrees(np.arccos(cos_val))), 2)

    joint_angles["trunk_lean_angle"] = trunk_lean_angle

    technical_issues = []
    risk_flags = []

    # Basic qualitative rules
    if trunk_lean_angle < 5:
        technical_issues.append("躯干前倾不足，可能影响加速阶段水平推进。")
    elif trunk_lean_angle > 35:
        technical_issues.append("躯干前倾角较大，需结合动作阶段判断是否合理。")

    left_knee = joint_angles.get("left_knee_angle")
    right_knee = joint_angles.get("right_knee_angle")

    if left_knee is not None and left_knee < 130:
        technical_issues.append("左膝屈曲角度较大，可能存在支撑腿伸展不足。")

    if right_knee is not None and right_knee < 130:
        technical_issues.append("右膝屈曲角度较大，可能存在支撑腿伸展不足。")

    # Simple asymmetry check
    if left_knee is not None and right_knee is not None:
        if abs(left_knee - right_knee) > 25:
            risk_flags.append("左右膝关节角度差异较大，提示动作对称性需要复核。")

    # Rough foot-position cue
    if visibility_ok(lh, la):
        if abs(la[0] - lh[0]) > 0.22:
            risk_flags.append("足部与髋部水平距离较大，可能存在触地点过前或步幅控制问题。")

    if visibility_ok(rh, ra):
        if abs(ra[0] - rh[0]) > 0.22:
            risk_flags.append("足部与髋部水平距离较大，可能存在触地点过前或步幅控制问题。")

    if len(technical_issues) == 0:
        technical_issues.append("未发现明显技术异常，但仍需结合连续视频进一步判断。")

    if len(risk_flags) == 0:
        risk_flags.append("单张图片中暂未发现明显动作风险，建议结合连续视频进一步复核。")

    selected_landmarks = {}
    landmark_names = {
        "left_shoulder": LEFT_SHOULDER,
        "right_shoulder": RIGHT_SHOULDER,
        "left_hip": LEFT_HIP,
        "right_hip": RIGHT_HIP,
        "left_knee": LEFT_KNEE,
        "right_knee": RIGHT_KNEE,
        "left_ankle": LEFT_ANKLE,
        "right_ankle": RIGHT_ANKLE,
        "left_foot_index": LEFT_FOOT_INDEX,
        "right_foot_index": RIGHT_FOOT_INDEX
    }

    for name, idx in landmark_names.items():
        lm = landmarks[idx]
        selected_landmarks[name] = {
            "x_norm": round(float(lm.x), 4),
            "y_norm": round(float(lm.y), 4),
            "x_px": round(float(lm.x * width), 2),
            "y_px": round(float(lm.y * height), 2),
            "visibility": round(float(lm.visibility), 3)
        }

    return {
        "success": True,
        "image_size": {
            "width": width,
            "height": height
        },
        "joint_angles": joint_angles,
        "selected_landmarks": selected_landmarks,
        "main_technical_issues": technical_issues,
        "potential_risk_flags": risk_flags,
        "visual_summary": "已完成短跑图片的人体关键点识别与基础运动学指标计算。结果适合用于教学和训练反馈；若需科研级精度，建议结合标定视频、动作捕捉或多帧姿态估计。"
    }
