#!/usr/bin/env python3
"""VLM 复杂度路由与仲裁节点。"""

import base64
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import cv2
import requests
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseArray
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String


@dataclass
class RequestState:
    request_id: str
    created_at: float
    deadline: float
    qwen_done: bool = False
    qwen_complex: Optional[bool] = None
    qwen_confidence: Optional[float] = None
    qwen_objects: List[str] = field(default_factory=list)
    qwen_scene_summary: str = ""
    qwen_raw_answer: str = ""
    qwen_error: str = ""
    simple_waypoints: Optional[PoseArray] = None
    complex_waypoints: Optional[PoseArray] = None
    resolved: bool = False


class VLMComplexityRouterNode(Node):
    def __init__(self):
        super().__init__("vlm_complexity_router")

        self.declare_parameter("process_pic_topic", "/car/process_pic")
        self.declare_parameter("simple_waypoint_topic", "/goal_point_simple")
        self.declare_parameter("complex_waypoint_topic", "/goal_point_complex")
        self.declare_parameter("output_waypoint_topic", "/goal_point")
        self.declare_parameter("output_text_topic", "/car/model_text")
        self.declare_parameter("qwen_result_topic", "/car/qwen_result")
        self.declare_parameter("qwen_api_url", "http://localhost:8002/v1/chat/completions")
        self.declare_parameter("qwen_timeout", 10.0)
        self.declare_parameter("compression_quality", 30)
        self.declare_parameter("max_pending_frames", 2)
        self.declare_parameter("request_ttl", 20.0)

        self.process_pic_topic = self.get_parameter("process_pic_topic").get_parameter_value().string_value
        self.simple_waypoint_topic = self.get_parameter("simple_waypoint_topic").get_parameter_value().string_value
        self.complex_waypoint_topic = self.get_parameter("complex_waypoint_topic").get_parameter_value().string_value
        self.output_waypoint_topic = self.get_parameter("output_waypoint_topic").get_parameter_value().string_value
        self.output_text_topic = self.get_parameter("output_text_topic").get_parameter_value().string_value
        self.qwen_result_topic = self.get_parameter("qwen_result_topic").get_parameter_value().string_value
        self.qwen_api_url = self.get_parameter("qwen_api_url").get_parameter_value().string_value
        self.qwen_timeout = self.get_parameter("qwen_timeout").get_parameter_value().double_value
        self.compression_quality = self.get_parameter("compression_quality").get_parameter_value().integer_value
        self.max_pending_frames = self.get_parameter("max_pending_frames").get_parameter_value().integer_value
        self.request_ttl = self.get_parameter("request_ttl").get_parameter_value().double_value

        self.bridge = CvBridge()
        self.worker_pool = ThreadPoolExecutor(max_workers=2)
        self._seq = 0
        self._lock = threading.Lock()
        self._requests: Dict[str, RequestState] = {}

        self.output_waypoint_pub = self.create_publisher(PoseArray, self.output_waypoint_topic, 10)
        self.output_text_pub = self.create_publisher(String, self.output_text_topic, 10)
        self.qwen_result_pub = self.create_publisher(String, self.qwen_result_topic, 10)

        self.image_sub = self.create_subscription(
            Image,
            self.process_pic_topic,
            self.image_callback,
            10,
        )
        self.simple_waypoint_sub = self.create_subscription(
            PoseArray,
            self.simple_waypoint_topic,
            self._on_simple_waypoints,
            10,
        )
        self.complex_waypoint_sub = self.create_subscription(
            PoseArray,
            self.complex_waypoint_topic,
            self._on_complex_waypoints,
            10,
        )
        self.arbitration_timer = self.create_timer(0.05, self._arbitration_tick)

        self.get_logger().info(
            "vlm_complexity_router 启动: "
            f"process_pic={self.process_pic_topic}, "
            f"simple_wp={self.simple_waypoint_topic}, complex_wp={self.complex_waypoint_topic}, "
            f"output_wp={self.output_waypoint_topic}, qwen_result={self.qwen_result_topic}, "
            f"qwen_api={self.qwen_api_url}, timeout={self.qwen_timeout}s"
        )

    def image_callback(self, msg: Image):
        request_id = msg.header.frame_id.strip() if msg.header.frame_id else ""
        if not request_id:
            request_id = self._build_request_id(msg)

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            self.get_logger().error(f"图像转换失败 request_id={request_id}: {exc}")
            return

        encoded_image = self._encode_image_to_data_url(cv_image)
        if not encoded_image:
            self.get_logger().warn(f"图像编码失败，跳过 request_id={request_id}")
            return

        now = time.monotonic()
        state = RequestState(
            request_id=request_id,
            created_at=now,
            deadline=now + self.qwen_timeout,
        )
        with self._lock:
            self._requests[request_id] = state
            self._trim_pending_locked()

        self.worker_pool.submit(self._ask_qwen_complexity, request_id, encoded_image)

    def _build_request_id(self, msg: Image) -> str:
        self._seq += 1
        stamp = msg.header.stamp
        sec = int(stamp.sec) if stamp else 0
        nanosec = int(stamp.nanosec) if stamp else 0
        return f"req_{sec}_{nanosec}_{self._seq}"

    def _encode_image_to_data_url(self, cv_image) -> str:
        success, buffer = cv2.imencode(
            ".jpg",
            cv_image,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.compression_quality],
        )
        if not success:
            return ""
        jpg_as_text = base64.b64encode(buffer).decode("utf-8")
        return f"data:image/jpeg;base64,{jpg_as_text}"

    def _ask_qwen_complexity(self, request_id: str, image_data_url: str):
        payload = {
            "model": "/app/model",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "请按以下格式回答，不要输出额外内容。\n"
                                "第一行：仅输出 bool 值，true 表示场景复杂，false 表示场景不复杂。\n"
                                "第二行开始：描述场景中看到的物体、道路与环境情况。"
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": image_data_url},
                        },
                    ],
                }
            ],
            "max_tokens": 128,
            "temperature": 0.0,
            "top_p": 0.9,
        }

        start_time = time.time()
        try:
            response = requests.post(
                self.qwen_api_url,
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=self.qwen_timeout,
            )
            response.raise_for_status()
            data = response.json()
            content: Optional[str] = None
            if data.get("choices"):
                content = data["choices"][0].get("message", {}).get("content")

            parsed = self._parse_qwen_result(content or "")
            with self._lock:
                state = self._requests.get(request_id)
                if state is None or state.resolved:
                    return
                state.qwen_done = True
                state.qwen_complex = parsed.get("complex")
                state.qwen_confidence = parsed.get("confidence")
                state.qwen_scene_summary = parsed.get("scene_summary", "")
                state.qwen_raw_answer = content or ""
                state.qwen_objects = parsed.get("objects", [])
                if parsed.get("error"):
                    state.qwen_error = str(parsed.get("error"))

            elapsed = time.time() - start_time
            self.get_logger().info(
                f"Qwen完成 request_id={request_id}, cost={elapsed:.2f}s, complex={parsed.get('complex')}, "
                f"scene_summary={parsed.get('scene_summary')}"
            )
        except Exception as exc:
            elapsed = time.time() - start_time
            with self._lock:
                state = self._requests.get(request_id)
                if state is not None and not state.resolved:
                    state.qwen_done = True
                    state.qwen_complex = None
                    state.qwen_raw_answer = ""
                    state.qwen_error = str(exc)
            self.get_logger().warn(
                f"Qwen请求失败 request_id={request_id}, cost={elapsed:.2f}s, err={exc}"
            )

    def _parse_bool_line(self, first_line: str) -> Optional[bool]:
        token = first_line.strip().lower()
        token = token.strip("` ")

        if token in {"true", "t", "1", "yes", "y", "复杂", "是"}:
            return True
        if token in {"false", "f", "0", "no", "n", "不复杂", "否"}:
            return False

        # 容忍带前缀或标点，例如: "complex: true" / "复杂: false"
        match = re.search(r"(true|false)", token)
        if match:
            return match.group(1) == "true"

        return None

    def _parse_qwen_result(self, content: str) -> Dict[str, Any]:
        text = (content or "").strip()
        if not text:
            return {
                "complex": None,
                "objects": [],
                "scene_summary": "",
                "confidence": None,
                "error": "empty_response",
            }

        lines = text.splitlines()
        first_line = lines[0].strip()
        parsed_complex = self._parse_bool_line(first_line)
        scene_summary = "\n".join(line.rstrip() for line in lines[1:]).strip()

        return {
            "complex": parsed_complex,
            "objects": [],
            "scene_summary": scene_summary,
            "confidence": None,
            "error": "" if parsed_complex is not None else "invalid_bool_first_line",
        }

    def _on_simple_waypoints(self, msg: PoseArray):
        self._store_waypoints(msg, is_complex=False)

    def _on_complex_waypoints(self, msg: PoseArray):
        self._store_waypoints(msg, is_complex=True)

    def _store_waypoints(self, msg: PoseArray, is_complex: bool):
        request_id = msg.header.frame_id
        if not request_id:
            return
        with self._lock:
            state = self._requests.get(request_id)
            if state is None or state.resolved:
                return
            if is_complex:
                state.complex_waypoints = msg
            else:
                state.simple_waypoints = msg

    def _arbitration_tick(self):
        now = time.monotonic()
        to_resolve: List[RequestState] = []
        to_cleanup: List[str] = []

        with self._lock:
            for request_id, state in self._requests.items():
                if state.resolved:
                    continue
                chosen = self._choose_branch(state, now)
                if chosen is not None:
                    state.resolved = True
                    to_resolve.append(state)
                elif now - state.created_at > self.request_ttl:
                    to_cleanup.append(request_id)

            for request_id in to_cleanup:
                self._requests.pop(request_id, None)

        for state in to_resolve:
            self._publish_selected_result(state)

    def _choose_branch(self, state: RequestState, now: float) -> Optional[str]:
        simple_ready = state.simple_waypoints is not None
        complex_ready = state.complex_waypoints is not None
        if not simple_ready and not complex_ready:
            return None

        if state.qwen_done:
            if state.qwen_complex is True:
                if complex_ready:
                    return "complex"
                if simple_ready:
                    return "simple_fallback"
            else:
                if simple_ready:
                    return "simple"
                if complex_ready:
                    return "complex_fallback"
            return None

        if now < state.deadline:
            return None

        if simple_ready:
            return "simple_timeout"
        if complex_ready:
            return "complex_timeout_fallback"
        return None

    def _publish_selected_result(self, state: RequestState):
        reason = self._choose_reason(state)
        branch = self._choose_branch_for_publish(state)
        if branch is None:
            return

        qwen_result = {
            "request_id": state.request_id,
            "qwen_complex": state.qwen_complex,
            "qwen_confidence": state.qwen_confidence,
            "qwen_error": state.qwen_error,
            "qwen_answer": state.qwen_raw_answer,
            "qwen_scene_summary": state.qwen_scene_summary,
        }
        qwen_msg = String()
        qwen_msg.data = json.dumps(qwen_result, ensure_ascii=False)
        self.qwen_result_pub.publish(qwen_msg)

        if branch == "complex":
            waypoint_msg = state.complex_waypoints
        else:
            waypoint_msg = state.simple_waypoints

        if waypoint_msg is None:
            self.get_logger().warn(f"request_id={state.request_id} 缺少最终轨迹，丢弃")
            return

        waypoint_msg.header.frame_id = state.request_id
        waypoint_msg.header.stamp = self.get_clock().now().to_msg()
        self.output_waypoint_pub.publish(waypoint_msg)

        diag = {
            "request_id": state.request_id,
            "selected_branch": branch,
            "reason": reason,
            "qwen_complex": state.qwen_complex,
            "qwen_confidence": state.qwen_confidence,
            "objects": state.qwen_objects,
            "scene_summary": state.qwen_scene_summary,
            "qwen_error": state.qwen_error,
        }
        diag_msg = String()
        diag_msg.data = json.dumps(diag, ensure_ascii=False)
        self.output_text_pub.publish(diag_msg)

        self.get_logger().info(
            f"仲裁完成 request_id={state.request_id}, branch={branch}, reason={reason}, "
            f"complex={state.qwen_complex}, objects={state.qwen_objects}"
        )

        with self._lock:
            self._requests.pop(state.request_id, None)

    def _choose_branch_for_publish(self, state: RequestState) -> Optional[str]:
        if state.qwen_done:
            if state.qwen_complex is True:
                if state.complex_waypoints is not None:
                    return "complex"
                if state.simple_waypoints is not None:
                    return "simple"
            else:
                if state.simple_waypoints is not None:
                    return "simple"
                if state.complex_waypoints is not None:
                    return "complex"
            return None

        if state.simple_waypoints is not None:
            return "simple"
        if state.complex_waypoints is not None:
            return "complex"
        return None

    def _choose_reason(self, state: RequestState) -> str:
        if state.qwen_done:
            if state.qwen_complex is True:
                return "complex" if state.complex_waypoints is not None else "complex_fallback_simple"
            return "simple" if state.simple_waypoints is not None else "simple_fallback_complex"
        return "timeout_default_simple"

    def _trim_pending_locked(self):
        pending = [s for s in self._requests.values() if not s.resolved]
        if len(pending) <= max(1, self.max_pending_frames):
            return

        pending.sort(key=lambda s: s.created_at)
        overflow = len(pending) - max(1, self.max_pending_frames)
        for state in pending[:overflow]:
            self._requests.pop(state.request_id, None)
            self.get_logger().warn(f"丢弃过载帧 request_id={state.request_id}")

    def destroy_node(self):
        self.worker_pool.shutdown(wait=False)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = VLMComplexityRouterNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
