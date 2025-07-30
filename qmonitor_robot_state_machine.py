#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
이 스크립트는 robot_state_machine.py 노드의 상태를 모니터링하고, 제어 파라미터를
실시간으로 조절할 수 있는 PyQt5 기반의 종합적인 GUI 애플리케이션입니다.

주요 기능:
- Matplotlib 기반 실시간 Pose 시각화: 로봇의 현재 위치, 궤적, 목표 지점을 2D 격자 위에 그래픽으로 표시합니다.
- 클릭 및 드래그로 목표 발행: 맵 위에서 마우스를 이용해 새로운 target_pose를 발행합니다.
- 상태 흐름 시각화: 상태 머신의 현재 진행 상태를 다이어그램으로 표시합니다.
- 상태 및 오차 모니터링: 로봇의 현재 상태, 오차 등을 텍스트로 표시합니다.
- YAML 기반 동적 파라미터 제어: 'pid_config.yaml' 파일을 읽고 수정하여 모든 제어 파라미터를 동적으로 변경합니다.
- 다중 스레딩 및 QTimer: ROS 2 통신과 GUI 이벤트 루프가 충돌하지 않도록 하고, QTimer를 통해 주기적으로 GUI를 갱신하여 실시간 성을 보장합니다.
"""

import sys
import rclpy
import math
import yaml
import os
from rclpy.node import Node
from std_msgs.msg import String, Float64
from geometry_msgs.msg import PoseStamped
from rcl_interfaces.srv import SetParameters
from rcl_interfaces.msg import Parameter, ParameterValue, ParameterType
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QDoubleSpinBox,
                             QGroupBox, QGridLayout, QPushButton, QCheckBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer

# Matplotlib 임포트
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.patches import FancyBboxPatch

# ROS 2 노드 및 통신을 처리하는 스레드
class RosNodeThread(QThread):
    state_update = pyqtSignal(str)
    angle_error_update = pyqtSignal(float)
    distance_error_update = pyqtSignal(float)
    camera_pose_update = pyqtSignal(object)
    target_pose_update = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self.node = rclpy.create_node('qmonitor_robot_state_machine')
        
        self.param_client = self.node.create_client(SetParameters, '/robot_goal_controller/set_parameters')
        self.target_publisher = self.node.create_publisher(PoseStamped, 'target_pose', 10)

        self.node.create_subscription(String, 'state', self.state_callback, 10)
        self.node.create_subscription(Float64, 'angle_error', self.angle_error_callback, 10)
        self.node.create_subscription(Float64, 'distance_error', self.distance_error_callback, 10)
        self.node.create_subscription(PoseStamped, 'camera_pose', self.camera_pose_callback, 10)
        self.node.create_subscription(PoseStamped, 'target_pose', self.target_pose_callback, 10)
        
        self.current_state = "대기 중"

    def run(self):
        self.node.get_logger().info("ROS 2 모니터링 노드 스레드 시작")
        rclpy.spin(self.node)
        self.node.destroy_node()

    def state_callback(self, msg): self.state_update.emit(msg.data)
    def angle_error_callback(self, msg): self.angle_error_update.emit(msg.data)
    def distance_error_callback(self, msg): self.distance_error_update.emit(msg.data)
    def camera_pose_callback(self, msg): self.camera_pose_update.emit(msg)
    def target_pose_callback(self, msg): self.target_pose_update.emit(msg)

    def publish_target_pose(self, msg):
        self.target_publisher.publish(msg)

    def set_parameter(self, name, value):
        if not self.param_client.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().warn('파라미터 서비스에 연결할 수 없습니다.')
            return

        req = SetParameters.Request()
        param = Parameter(name=name, value=ParameterValue(type=ParameterType.PARAMETER_BOOL, bool_value=value) if isinstance(value, bool) else ParameterValue(type=ParameterType.PARAMETER_DOUBLE, double_value=value))
        req.parameters.append(param)
        self.param_client.call_async(req)

# 메인 GUI 윈도우
class MainWindow(QWidget):
    def __init__(self, ros_thread):
        super().__init__()
        self.ros_thread = ros_thread
        self.config_path = os.path.join(os.path.dirname(__file__), 'pid_config.yaml')
        
        self.camera_pose = None
        self.target_pose = None
        self.drag_start = None
        self.drag_current = None
        self.trajectory = []

        self.init_ui()
        self.connect_signals()
        self.load_config()
        
        self.state_list = ["RotateToGoal", "MoveToGoal", "RotateToFinal", "GoalReached"]
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_gui_elements)
        self.timer.start(100) # 100ms (10Hz) 업데이트

    def init_ui(self):
        main_layout = QHBoxLayout()
        
        map_group = QGroupBox("로봇 맵")
        map_layout = QVBoxLayout()
        self.figure_map = Figure()
        self.canvas_map = FigureCanvas(self.figure_map)
        self.ax_map = self.figure_map.add_subplot(111)
        map_layout.addWidget(self.canvas_map)
        map_group.setLayout(map_layout)
        main_layout.addWidget(map_group, 1)

        right_panel_layout = QVBoxLayout()
        
        monitor_group = QGroupBox("상태 모니터링")
        monitor_layout = QGridLayout()
        self.state_label = QLabel("현재 상태: 대기 중")
        self.angle_error_label = QLabel("각도 오차: 0.00 deg")
        self.distance_error_label = QLabel("거리 오차: 0.00 m")
        monitor_layout.addWidget(self.state_label, 0, 0, 1, 2)
        monitor_layout.addWidget(self.angle_error_label, 1, 0)
        monitor_layout.addWidget(self.distance_error_label, 1, 1)
        monitor_group.setLayout(monitor_layout)

        state_flow_group = QGroupBox("상태 흐름")
        state_flow_layout = QVBoxLayout()
        self.figure_state = Figure(figsize=(3, 4))
        self.canvas_state = FigureCanvas(self.figure_state)
        self.ax_state = self.figure_state.add_subplot(111)
        state_flow_layout.addWidget(self.canvas_state)
        state_flow_group.setLayout(state_flow_layout)

        control_group = QGroupBox("파라미터 제어")
        control_layout = QGridLayout()
        
        self.euclidean_checkbox = QCheckBox("유클리드 거리 사용")
        self.euclidean_checkbox.setChecked(True)
        control_layout.addWidget(self.euclidean_checkbox, 0, 0, 1, 2)

        self.spinboxes = {}
        params = [
            (('tolerances', 'angle'), "Angle Tol:"),
            (('tolerances', 'distance'), "Dist Tol:"),
            (('angular', 'P'), "Angular P:"),
            (('angular', 'I'), "Angular I:"),
            (('angular', 'D'), "Angular D:"),
            (('angular', 'max_state'), "Angular Max State:"),
            (('angular', 'min_state'), "Angular Min State:"),
            (('linear', 'P'), "Linear P:"),
            (('linear', 'I'), "Linear I:"),
            (('linear', 'D'), "Linear D:"),
            (('linear', 'max_state'), "Linear Max State:"),
            (('linear', 'min_state'), "Linear Min State:")
        ]

        for i, ((group_param, label_text)) in enumerate(params):
            label = QLabel(label_text)
            spinbox = QDoubleSpinBox()
            spinbox.setRange(-10.0, 10.0)
            spinbox.setDecimals(3)
            spinbox.setSingleStep(0.001)
            self.spinboxes[group_param] = spinbox
            control_layout.addWidget(label, i + 1, 0)
            control_layout.addWidget(spinbox, i + 1, 1)

        self.save_button = QPushButton("YAML에 설정 저장")
        control_layout.addWidget(self.save_button, len(params) + 1, 0, 1, 2)
        control_group.setLayout(control_layout)

        right_panel_layout.addWidget(monitor_group)
        right_panel_layout.addWidget(state_flow_group)
        right_panel_layout.addWidget(control_group)
        main_layout.addLayout(right_panel_layout, 0)

        self.setLayout(main_layout)
        self.setWindowTitle('로봇 상태 모니터 및 파라미터 제어기')
        self.setGeometry(100, 100, 1200, 800)

    def connect_signals(self):
        self.ros_thread.state_update.connect(self.update_state_label)
        self.ros_thread.angle_error_update.connect(lambda e: self.angle_error_label.setText(f"각도 오차: {math.degrees(e):.2f} deg"))
        self.ros_thread.distance_error_update.connect(lambda e: self.distance_error_label.setText(f"거리 오차: {e:.2f} m"))
        self.ros_thread.camera_pose_update.connect(self.handle_camera_pose_update)
        self.ros_thread.target_pose_update.connect(self.handle_target_pose_update)

        self.canvas_map.mpl_connect('button_press_event', self.on_mouse_press)
        self.canvas_map.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.canvas_map.mpl_connect('button_release_event', self.on_mouse_release)

        self.euclidean_checkbox.stateChanged.connect(self.update_distance_mode)
        self.save_button.clicked.connect(self.save_config)

    #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
이 스크립트는 robot_state_machine.py 노드의 상태를 모니터링하고, 제어 파라미터를
실시간으로 조절할 수 있는 PyQt5 기반의 종합적인 GUI 애플리케이션입니다.

주요 기능:
- Matplotlib 기반 실시간 Pose 시각화: 로봇의 현재 위치, 궤적, 목표 지점을 2D 격자 위에 그래픽으로 표시합니다.
- 클릭 및 드래그로 목표 발행: 맵 위에서 마우스를 이용해 새로운 target_pose를 발행합니다.
- 상태 흐름 시각화: 상태 머신의 현재 진행 상태를 다이어그램으로 표시합니다.
- 상태 및 오차 모니터링: 로봇의 현재 상태, 오차 등을 텍스트로 표시합니다.
- YAML 기반 동적 파라미터 제어: 'pid_config.yaml' 파일을 읽고 수정하여 모든 제어 파라미터를 동적으로 변경합니다.
- 다중 스레딩 및 QTimer: ROS 2 통신과 GUI 이벤트 루프가 충돌하지 않도록 하고, QTimer를 통해 주기적으로 GUI를 갱신하여 실시간 성을 보장합니다.
"""

import sys
import rclpy
import math
import yaml
import os
from rclpy.node import Node
from std_msgs.msg import String, Float64
from geometry_msgs.msg import PoseStamped
from rcl_interfaces.srv import SetParameters
from rcl_interfaces.msg import Parameter, ParameterValue, ParameterType
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QDoubleSpinBox,
                             QGroupBox, QGridLayout, QPushButton, QCheckBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer

# Matplotlib 임포트
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.patches import FancyBboxPatch

# ROS 2 노드 및 통신을 처리하는 스레드
class RosNodeThread(QThread):
    state_update = pyqtSignal(str)
    angle_error_update = pyqtSignal(float)
    distance_error_update = pyqtSignal(float)
    camera_pose_update = pyqtSignal(object)
    target_pose_update = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self.node = rclpy.create_node('qmonitor_robot_state_machine')
        
        self.param_client = self.node.create_client(SetParameters, '/robot_goal_controller/set_parameters')
        self.target_publisher = self.node.create_publisher(PoseStamped, 'target_pose', 10)

        self.node.create_subscription(String, 'state', self.state_callback, 10)
        self.node.create_subscription(Float64, 'angle_error', self.angle_error_callback, 10)
        self.node.create_subscription(Float64, 'distance_error', self.distance_error_callback, 10)
        self.node.create_subscription(PoseStamped, 'camera_pose', self.camera_pose_callback, 10)
        self.node.create_subscription(PoseStamped, 'target_pose', self.target_pose_callback, 10)
        
        self.current_state = "대기 중"

    def run(self):
        self.node.get_logger().info("ROS 2 모니터링 노드 스레드 시작")
        rclpy.spin(self.node)
        self.node.destroy_node()

    def state_callback(self, msg): self.state_update.emit(msg.data)
    def angle_error_callback(self, msg): self.angle_error_update.emit(msg.data)
    def distance_error_callback(self, msg): self.distance_error_update.emit(msg.data)
    def camera_pose_callback(self, msg): self.camera_pose_update.emit(msg)
    def target_pose_callback(self, msg): self.target_pose_update.emit(msg)

    def publish_target_pose(self, msg):
        self.target_publisher.publish(msg)

    def set_parameter(self, name, value):
        if not self.param_client.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().warn('파라미터 서비스에 연결할 수 없습니다.')
            return

        req = SetParameters.Request()
        param = Parameter(name=name, value=ParameterValue(type=ParameterType.PARAMETER_BOOL, bool_value=value) if isinstance(value, bool) else ParameterValue(type=ParameterType.PARAMETER_DOUBLE, double_value=value))
        req.parameters.append(param)
        self.param_client.call_async(req)

# 메인 GUI 윈도우
class MainWindow(QWidget):
    def __init__(self, ros_thread):
        super().__init__()
        self.ros_thread = ros_thread
        self.config_path = os.path.join(os.path.dirname(__file__), 'pid_config.yaml')
        
        self.camera_pose = None
        self.target_pose = None
        self.drag_start = None
        self.drag_current = None
        self.trajectory = []
        self.current_robot_state = "대기 중" # 로봇의 현재 상태를 저장할 변수

        self.init_ui()
        self.connect_signals()
        self.load_config()
        
        self.state_list = ["RotateToGoal", "MoveToGoal", "RotateToFinal", "GoalReached"]
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_gui_elements)
        self.timer.start(100) # 100ms (10Hz) 업데이트

    def init_ui(self):
        main_layout = QHBoxLayout()
        
        map_group = QGroupBox("로봇 맵")
        map_layout = QVBoxLayout()
        self.figure_map = Figure()
        self.canvas_map = FigureCanvas(self.figure_map)
        self.ax_map = self.figure_map.add_subplot(111)
        map_layout.addWidget(self.canvas_map)
        map_group.setLayout(map_layout)
        main_layout.addWidget(map_group, 1)

        right_panel_layout = QVBoxLayout()
        
        monitor_group = QGroupBox("상태 모니터링")
        monitor_layout = QGridLayout()
        self.state_label = QLabel("현재 상태: 대기 중")
        self.angle_error_label = QLabel("각도 오차: 0.00 deg")
        self.distance_error_label = QLabel("거리 오차: 0.00 m")
        monitor_layout.addWidget(self.state_label, 0, 0, 1, 2)
        monitor_layout.addWidget(self.angle_error_label, 1, 0)
        monitor_layout.addWidget(self.distance_error_label, 1, 1)
        monitor_group.setLayout(monitor_layout)

        state_flow_group = QGroupBox("상태 흐름")
        state_flow_layout = QVBoxLayout()
        self.figure_state = Figure(figsize=(3, 4))
        self.canvas_state = FigureCanvas(self.figure_state)
        self.ax_state = self.figure_state.add_subplot(111)
        state_flow_layout.addWidget(self.canvas_state)
        state_flow_group.setLayout(state_flow_layout)

        control_group = QGroupBox("파라미터 제어")
        control_layout = QGridLayout()
        
        self.euclidean_checkbox = QCheckBox("유클리드 거리 사용")
        self.euclidean_checkbox.setChecked(True)
        control_layout.addWidget(self.euclidean_checkbox, 0, 0, 1, 2)

        self.spinboxes = {}
        params = [
            (('tolerances', 'angle'), "Angle Tol:"),
            (('tolerances', 'distance'), "Dist Tol:"),
            (('angular', 'P'), "Angular P:"),
            (('angular', 'I'), "Angular I:"),
            (('angular', 'D'), "Angular D:"),
            (('angular', 'max_state'), "Angular Max State:"),
            (('angular', 'min_state'), "Angular Min State:"),
            (('linear', 'P'), "Linear P:"),
            (('linear', 'I'), "Linear I:"),
            (('linear', 'D'), "Linear D:"),
            (('linear', 'max_state'), "Linear Max State:"),
            (('linear', 'min_state'), "Linear Min State:")
        ]

        for i, ((group_param, label_text)) in enumerate(params):
            label = QLabel(label_text)
            spinbox = QDoubleSpinBox()
            spinbox.setRange(-10.0, 10.0)
            spinbox.setDecimals(3)
            spinbox.setSingleStep(0.001)
            self.spinboxes[group_param] = spinbox
            control_layout.addWidget(label, i + 1, 0)
            control_layout.addWidget(spinbox, i + 1, 1)

        self.save_button = QPushButton("YAML에 설정 저장")
        control_layout.addWidget(self.save_button, len(params) + 1, 0, 1, 2)
        control_group.setLayout(control_layout)

        right_panel_layout.addWidget(monitor_group)
        right_panel_layout.addWidget(state_flow_group)
        right_panel_layout.addWidget(control_group)
        main_layout.addLayout(right_panel_layout, 0)

        self.setLayout(main_layout)
        self.setWindowTitle('로봇 상태 모니터 및 파라미터 제어기')
        self.setGeometry(100, 100, 1200, 800)

    def connect_signals(self):
        self.ros_thread.state_update.connect(self.update_state_label)
        self.ros_thread.angle_error_update.connect(lambda e: self.angle_error_label.setText(f"각도 오차: {math.degrees(e):.2f} deg"))
        self.ros_thread.distance_error_update.connect(lambda e: self.distance_error_label.setText(f"거리 오차: {e:.2f} m"))
        self.ros_thread.camera_pose_update.connect(self.handle_camera_pose_update)
        self.ros_thread.target_pose_update.connect(self.handle_target_pose_update)

        self.canvas_map.mpl_connect('button_press_event', self.on_mouse_press)
        self.canvas_map.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.canvas_map.mpl_connect('button_release_event', self.on_mouse_release)

        self.euclidean_checkbox.stateChanged.connect(self.update_distance_mode)
        self.save_button.clicked.connect(self.save_config)

    def update_state_label(self, state):
        self.state_label.setText(f"현재 상태: {state}")
        self.current_robot_state = state # MainWindow에 현재 상태 저장

    def load_config(self):
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            for (group, param), spinbox in self.spinboxes.items():
                if group in config and param in config[group]:
                    spinbox.setValue(float(config[group][param]))
            self.ros_thread.node.get_logger().info("YAML 설정을 GUI에 로드했습니다.")
        except Exception as e:
            self.ros_thread.node.get_logger().error(f"YAML 설정 파일 로드 실패: {e}")

    def save_config(self):
        try:
            # 기존 설정을 로드하여 업데이트 (새로운 키 추가 및 기존 키 유지)
            current_config = {}
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    current_config = yaml.safe_load(f) or {}

            for (group, param), spinbox in self.spinboxes.items():
                if group not in current_config:
                    current_config[group] = {}
                current_config[group][param] = spinbox.value()
            
            with open(self.config_path, 'w') as f:
                yaml.dump(current_config, f, default_flow_style=False, sort_keys=False)
            self.ros_thread.node.get_logger().info(f"설정을 '{self.config_path}'에 저장했습니다.")
        except Exception as e:
            self.ros_thread.node.get_logger().error(f"YAML 설정 파일 저장 실패: {e}")

    def update_distance_mode(self, state):
        self.ros_thread.set_parameter('use_euclidean_distance', state == Qt.Checked)

    def handle_camera_pose_update(self, msg):
        self.camera_pose = msg
        if self.camera_pose:
            x, y = msg.pose.position.x, msg.pose.position.y
            self.trajectory.append((x, y))
            if len(self.trajectory) > 200: self.trajectory.pop(0)

    def handle_target_pose_update(self, msg):
        self.target_pose = msg
        self.trajectory.clear()

    def on_mouse_press(self, event):
        if event.button == 1 and event.inaxes == self.ax_map:
            self.drag_start = (event.xdata, event.ydata)
            self.drag_current = (event.xdata, event.ydata)

    def on_mouse_move(self, event):
        if self.drag_start and event.inaxes == self.ax_map:
            self.drag_current = (event.xdata, event.ydata)

    def on_mouse_release(self, event):
        if event.button == 1 and self.drag_start and self.drag_current:
            dx = self.drag_current[0] - self.drag_start[0]
            dy = self.drag_current[1] - self.drag_start[1]
            theta = math.atan2(dy, dx)

            target_msg = PoseStamped()
            target_msg.header.stamp = self.ros_thread.node.get_clock().now().to_msg()
            target_msg.header.frame_id = 'map'
            target_msg.pose.position.x = float(self.drag_start[0])
            target_msg.pose.position.y = float(self.drag_start[1])
            target_msg.pose.orientation.z = math.sin(theta / 2.0)
            target_msg.pose.orientation.w = math.cos(theta / 2.0)

            self.ros_thread.publish_target_pose(target_msg)
        self.drag_start, self.drag_current = None, None

    def update_gui_elements(self):
        self.update_map_display()
        self.update_state_diagram()

    def update_map_display(self):
        self.ax_map.clear()
        self.ax_map.set_xlim(0, 2.0)
        self.ax_map.set_ylim(0, 1.0)
        self.ax_map.set_aspect('equal')
        self.ax_map.set_title("로봇 맵")
        self.ax_map.set_xlabel("X (m)")
        self.ax_map.set_ylabel("Y (m)")
        self.ax_map.grid(True, linestyle='--', linewidth=0.5)

        if self.trajectory:
            traj_x, traj_y = zip(*self.trajectory)
            self.ax_map.plot(traj_x, traj_y, 'g-', linewidth=1, label='Trajectory')

        if self.camera_pose:
            x, y = self.camera_pose.pose.position.x, self.camera_pose.pose.position.y
            qz, qw = self.camera_pose.pose.orientation.z, self.camera_pose.pose.orientation.w
            yaw = 2 * math.atan2(qz, qw)
            self.ax_map.plot(x, y, 'bo', markersize=8, label='Robot')
            self.ax_map.arrow(x, y, 0.1 * math.cos(yaw), 0.1 * math.sin(yaw), head_width=0.05, fc='blue', ec='blue')

        if self.target_pose:
            x, y = self.target_pose.pose.position.x, self.target_pose.pose.position.y
            qz, qw = self.target_pose.pose.orientation.z, self.target_pose.pose.orientation.w
            yaw = 2 * math.atan2(qz, qw)
            self.ax_map.plot(x, y, 'rX', markersize=10, label='Target')
            self.ax_map.arrow(x, y, 0.1 * math.cos(yaw), 0.1 * math.sin(yaw), head_width=0.05, fc='red', ec='red')

        if self.drag_start and self.drag_current:
            sx, sy = self.drag_start
            cx, cy = self.drag_current
            self.ax_map.arrow(sx, sy, cx - sx, cy - sy, head_width=0.03, fc='green', ec='green', linestyle='--')
            
        self.ax_map.legend(loc='upper right')
        self.canvas_map.draw()

    def update_state_diagram(self):
        self.ax_state.clear()
        self.ax_state.set_xlim(0, 1.5)
        self.ax_state.set_ylim(0, 1)
        self.ax_state.axis('off')
        
        block_width, block_height, spacing = 1.2, 0.1, 0.1
        start_x = (1.5 - block_width) / 2
        n = len(self.state_list)
        total_height = n * block_height + (n - 1) * spacing
        group_bottom = 0.5 - total_height / 2
        
        for i, state in enumerate(self.state_list):
            y = group_bottom + (n - 1 - i) * (block_height + spacing)
            # self.ros_thread.current_state 대신 self.current_robot_state 사용
            face_color = '#A2D2FF' if self.current_robot_state == state else '#E9ECEF'
            rect = FancyBboxPatch((start_x, y), block_width, block_height, boxstyle="round,pad=0.02", fc=face_color, ec="black", lw=1.5)
            self.ax_state.add_patch(rect)
            self.ax_state.text(start_x + block_width/2, y + block_height/2, state, ha='center', va='center', fontsize=10)
        self.canvas_state.draw()

def main(args=None):
    rclpy.init(args=args)
    app = QApplication(sys.argv)
    ros_thread = RosNodeThread()
    main_win = MainWindow(ros_thread)
    main_win.show()
    ros_thread.start()
    
    exit_code = app.exec_()
    
    ros_thread.node.get_logger().info("GUI 종료. ROS 2 노드 종료 중...")
    rclpy.shutdown()
    sys.exit(exit_code)

if __name__ == '__main__':
    main()

    def load_config(self):
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            for (group, param), spinbox in self.spinboxes.items():
                if group in config and param in config[group]:
                    spinbox.setValue(float(config[group][param]))
            self.ros_thread.node.get_logger().info("YAML 설정을 GUI에 로드했습니다.")
        except Exception as e:
            self.ros_thread.node.get_logger().error(f"YAML 설정 파일 로드 실패: {e}")

    def save_config(self):
        try:
            # 기존 설정을 로드하여 업데이트 (새로운 키 추가 및 기존 키 유지)
            current_config = {}
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    current_config = yaml.safe_load(f) or {}

            for (group, param), spinbox in self.spinboxes.items():
                if group not in current_config:
                    current_config[group] = {}
                current_config[group][param] = spinbox.value()
            
            with open(self.config_path, 'w') as f:
                yaml.dump(current_config, f, default_flow_style=False, sort_keys=False)
            self.ros_thread.node.get_logger().info(f"설정을 '{self.config_path}'에 저장했습니다.")
        except Exception as e:
            self.ros_thread.node.get_logger().error(f"YAML 설정 파일 저장 실패: {e}")

    def update_distance_mode(self, state):
        self.ros_thread.set_parameter('use_euclidean_distance', state == Qt.Checked)

    def handle_camera_pose_update(self, msg):
        self.camera_pose = msg
        if self.camera_pose:
            x, y = msg.pose.position.x, msg.pose.position.y
            self.trajectory.append((x, y))
            if len(self.trajectory) > 200: self.trajectory.pop(0)

    def handle_target_pose_update(self, msg):
        self.target_pose = msg
        self.trajectory.clear()

    def on_mouse_press(self, event):
        if event.button == 1 and event.inaxes == self.ax_map:
            self.drag_start = (event.xdata, event.ydata)
            self.drag_current = (event.xdata, event.ydata)

    def on_mouse_move(self, event):
        if self.drag_start and event.inaxes == self.ax_map:
            self.drag_current = (event.xdata, event.ydata)

    def on_mouse_release(self, event):
        if event.button == 1 and self.drag_start and self.drag_current:
            dx = self.drag_current[0] - self.drag_start[0]
            dy = self.drag_current[1] - self.drag_start[1]
            theta = math.atan2(dy, dx)

            target_msg = PoseStamped()
            target_msg.header.stamp = self.ros_thread.node.get_clock().now().to_msg()
            target_msg.header.frame_id = 'map'
            target_msg.pose.position.x = float(self.drag_start[0])
            target_msg.pose.position.y = float(self.drag_start[1])
            target_msg.pose.orientation.z = math.sin(theta / 2.0)
            target_msg.pose.orientation.w = math.cos(theta / 2.0)

            self.ros_thread.publish_target_pose(target_msg)
        self.drag_start, self.drag_current = None, None

    def update_gui_elements(self):
        self.update_map_display()
        self.update_state_diagram()

    def update_map_display(self):
        self.ax_map.clear()
        self.ax_map.set_xlim(0, 2.0)
        self.ax_map.set_ylim(0, 1.0)
        self.ax_map.set_aspect('equal')
        self.ax_map.set_title("로봇 맵")
        self.ax_map.set_xlabel("X (m)")
        self.ax_map.set_ylabel("Y (m)")
        self.ax_map.grid(True, linestyle='--', linewidth=0.5)

        if self.trajectory:
            traj_x, traj_y = zip(*self.trajectory)
            self.ax_map.plot(traj_x, traj_y, 'g-', linewidth=1, label='Trajectory')

        if self.camera_pose:
            x, y = self.camera_pose.pose.position.x, self.camera_pose.pose.position.y
            qz, qw = self.camera_pose.pose.orientation.z, self.camera_pose.pose.orientation.w
            yaw = 2 * math.atan2(qz, qw)
            self.ax_map.plot(x, y, 'bo', markersize=8, label='Robot')
            self.ax_map.arrow(x, y, 0.1 * math.cos(yaw), 0.1 * math.sin(yaw), head_width=0.05, fc='blue', ec='blue')

        if self.target_pose:
            x, y = self.target_pose.pose.position.x, self.target_pose.pose.position.y
            qz, qw = self.target_pose.pose.orientation.z, self.target_pose.pose.orientation.w
            yaw = 2 * math.atan2(qz, qw)
            self.ax_map.plot(x, y, 'rX', markersize=10, label='Target')
            self.ax_map.arrow(x, y, 0.1 * math.cos(yaw), 0.1 * math.sin(yaw), head_width=0.05, fc='red', ec='red')

        if self.drag_start and self.drag_current:
            sx, sy = self.drag_start
            cx, cy = self.drag_current
            self.ax_map.arrow(sx, sy, cx - sx, cy - sy, head_width=0.03, fc='green', ec='green', linestyle='--')
            
        self.ax_map.legend(loc='upper right')
        self.canvas_map.draw()

    def update_state_diagram(self):
        self.ax_state.clear()
        self.ax_state.set_xlim(0, 1.5)
        self.ax_state.set_ylim(0, 1)
        self.ax_state.axis('off')
        
        block_width, block_height, spacing = 1.2, 0.1, 0.1
        start_x = (1.5 - block_width) / 2
        n = len(self.state_list)
        total_height = n * block_height + (n - 1) * spacing
        group_bottom = 0.5 - total_height / 2
        
        for i, state in enumerate(self.state_list):
            y = group_bottom + (n - 1 - i) * (block_height + spacing)
            face_color = '#A2D2FF' if self.ros_thread.current_state == state else '#E9ECEF'
            rect = FancyBboxPatch((start_x, y), block_width, block_height, boxstyle="round,pad=0.02", fc=face_color, ec="black", lw=1.5)
            self.ax_state.add_patch(rect)
            self.ax_state.text(start_x + block_width/2, y + block_height/2, state, ha='center', va='center', fontsize=10)
        self.canvas_state.draw()

def main(args=None):
    rclpy.init(args=args)
    app = QApplication(sys.argv)
    ros_thread = RosNodeThread()
    main_win = MainWindow(ros_thread)
    main_win.show()
    ros_thread.start()
    
    exit_code = app.exec_()
    
    ros_thread.node.get_logger().info("GUI 종료. ROS 2 노드 종료 중...")
    rclpy.shutdown()
    sys.exit(exit_code)

if __name__ == '__main__':
    main()
