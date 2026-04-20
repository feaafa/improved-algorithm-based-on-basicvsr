# video_quality_metric_gui.py

import sys
import os
import torch
import numpy as np
import cv2
from pathlib import Path
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QComboBox, QPushButton,
                             QLineEdit, QFileDialog, QProgressBar, QGridLayout,
                             QGroupBox, QMessageBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont
import matplotlib

matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

# 导入指标计算库
try:
    import lpips

    LPIPS_AVAILABLE = True
except:
    LPIPS_AVAILABLE = False
    print("Warning: lpips not installed. Install with: pip install lpips")

try:
    import pyiqa

    PYIQA_AVAILABLE = True
except:
    PYIQA_AVAILABLE = False
    print("Warning: pyiqa not installed. Install with: pip install pyiqa")

from skimage import color
from skimage.metrics import structural_similarity as ssim


class MetricCalculator:
    """指标计算器"""

    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device

        # 初始化LPIPS
        if LPIPS_AVAILABLE:
            try:
                self.lpips_fn = lpips.LPIPS(net='alex').to(device)
                self.lpips_fn.eval()
                print("✓ LPIPS loaded successfully")
            except Exception as e:
                print(f"✗ LPIPS initialization failed: {e}")
                self.lpips_fn = None
        else:
            self.lpips_fn = None

        # 初始化NIQE
        if PYIQA_AVAILABLE:
            try:
                self.niqe_fn = pyiqa.create_metric('niqe', device=device)
                print("✓ NIQE loaded successfully")
            except Exception as e:
                print(f"✗ NIQE initialization failed: {e}")
                self.niqe_fn = None
        else:
            self.niqe_fn = None

        self.prev_frames = {}
        self.prev_gt_frames = {}

    def reset_temporal(self):
        """重置时序帧缓存"""
        self.prev_frames = {}
        self.prev_gt_frames = {}

    def calculate_lpips(self, img_sr, img_gt):
        """计算LPIPS"""
        if self.lpips_fn is None:
            return np.nan

        try:
            # 转换为torch tensor [0,1] -> [-1,1]
            img_sr_t = self._to_tensor(img_sr) * 2 - 1
            img_gt_t = self._to_tensor(img_gt) * 2 - 1

            with torch.no_grad():
                lpips_val = self.lpips_fn(img_sr_t, img_gt_t)

            return lpips_val.item()
        except Exception as e:
            print(f"LPIPS calculation error: {e}")
            return np.nan

    def calculate_niqe(self, img_sr):
        """计算NIQE（无参考指标）"""
        if self.niqe_fn is None:
            # 使用OpenCV的简化NIQE实现
            try:
                if len(img_sr.shape) == 3:
                    img_gray = cv2.cvtColor(img_sr, cv2.COLOR_RGB2GRAY)
                else:
                    img_gray = img_sr

                # 简化的NIQE实现：使用图像质量特征
                # 这里使用拉普拉斯方差作为替代
                laplacian = cv2.Laplacian(img_gray, cv2.CV_64F)
                variance = laplacian.var()

                # 归一化到NIQE范围（0-100，越小越好）
                niqe_approx = 100 / (1 + variance)
                return niqe_approx
            except Exception as e:
                print(f"NIQE calculation error: {e}")
                return np.nan

        try:
            img_sr_t = self._to_tensor(img_sr)
            with torch.no_grad():
                niqe_val = self.niqe_fn(img_sr_t)
            return niqe_val.item()
        except Exception as e:
            print(f"NIQE calculation error: {e}")
            return np.nan

    def calculate_temporal_mse(self, img_sr, img_gt, sr_idx, frame_idx):
        """计算时序MSE"""
        try:
            if frame_idx == 0:
                # 第一帧，保存并返回0
                self.prev_frames[sr_idx] = img_sr.astype(np.float32) / 255.0
                self.prev_gt_frames[sr_idx] = img_gt.astype(np.float32) / 255.0
                return 0.0

            # 计算当前帧和前一帧的差异
            curr_sr = img_sr.astype(np.float32) / 255.0
            curr_gt = img_gt.astype(np.float32) / 255.0

            prev_sr = self.prev_frames.get(sr_idx, curr_sr)
            prev_gt = self.prev_gt_frames.get(sr_idx, curr_gt)

            # SR的帧差
            sr_diff = curr_sr - prev_sr
            # GT的帧差
            gt_diff = curr_gt - prev_gt

            # 计算帧差的MSE
            temporal_mse = np.mean((sr_diff - gt_diff) ** 2)

            # 更新缓存
            self.prev_frames[sr_idx] = curr_sr
            self.prev_gt_frames[sr_idx] = curr_gt

            return temporal_mse
        except Exception as e:
            print(f"Temporal MSE calculation error: {e}")
            return np.nan

    def calculate_sharpness_ratio(self, img_sr, img_gt):
        """计算清晰度比率"""
        try:
            sharpness_sr = self._compute_sharpness(img_sr)
            sharpness_gt = self._compute_sharpness(img_gt)

            ratio = sharpness_sr / (sharpness_gt + 1e-8)
            return ratio
        except Exception as e:
            print(f"Sharpness ratio calculation error: {e}")
            return np.nan

    def calculate_delta_e(self, img_sr, img_gt):
        """计算Delta E（Lab色彩空间）"""
        try:
            # 确保是RGB格式
            if len(img_sr.shape) == 2:
                img_sr = cv2.cvtColor(img_sr, cv2.COLOR_GRAY2RGB)
            if len(img_gt.shape) == 2:
                img_gt = cv2.cvtColor(img_gt, cv2.COLOR_GRAY2RGB)

            # 归一化到[0,1]
            img_sr_norm = img_sr.astype(np.float32) / 255.0
            img_gt_norm = img_gt.astype(np.float32) / 255.0

            # 转换到Lab空间
            img_sr_lab = color.rgb2lab(img_sr_norm)
            img_gt_lab = color.rgb2lab(img_gt_norm)

            # 计算欧氏距离
            delta_e = np.sqrt(np.sum((img_sr_lab - img_gt_lab) ** 2, axis=2))

            return np.mean(delta_e)
        except Exception as e:
            print(f"Delta E calculation error: {e}")
            return np.nan

    def _compute_sharpness(self, img):
        """计算图像清晰度（Laplacian方差）"""
        if len(img.shape) == 3:
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            img_gray = img

        laplacian = cv2.Laplacian(img_gray, cv2.CV_64F)
        sharpness = laplacian.var()

        return sharpness

    def _to_tensor(self, img):
        """将numpy图像转换为torch tensor"""
        # img: (H, W, C) numpy array, [0, 255]
        img = img.astype(np.float32) / 255.0

        # 转换为 (C, H, W)
        if len(img.shape) == 2:
            img = img[np.newaxis, :, :]
        else:
            img = img.transpose(2, 0, 1)

        # 转换为tensor并添加batch维度
        img_t = torch.from_numpy(img).unsqueeze(0).to(self.device)

        return img_t


class EvaluationThread(QThread):
    """评估线程"""

    progress_update = pyqtSignal(int)
    result_update = pyqtSignal(int, str)  # SR index, result string
    plot_update = pyqtSignal(dict)  # 绘图数据
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, files, num_sr, metric_type, is_video):
        super().__init__()
        self.files = files
        self.num_sr = num_sr
        self.metric_type = metric_type
        self.is_video = is_video
        self.calculator = MetricCalculator()

    def run(self):
        try:
            if self.is_video:
                self._evaluate_videos()
            else:
                self._evaluate_images()

            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))

    def _evaluate_videos(self):
        """评估视频"""
        num_sr = self.num_sr

        # 打开所有视频
        caps_sr = []
        for i in range(num_sr):
            cap = cv2.VideoCapture(self.files[i])
            if not cap.isOpened():
                raise ValueError(f"Cannot open SR video {i + 1}")
            caps_sr.append(cap)

        cap_gt = cv2.VideoCapture(self.files[-1])
        if not cap_gt.isOpened():
            raise ValueError("Cannot open GT video")

        # 获取总帧数
        total_frames = int(cap_gt.get(cv2.CAP_PROP_FRAME_COUNT))

        # 重置时序缓存
        self.calculator.reset_temporal()

        # 存储所有帧的指标值
        metric_values = {i: [] for i in range(num_sr)}

        frame_idx = 0

        while True:
            # 读取GT帧
            ret_gt, frame_gt = cap_gt.read()
            if not ret_gt:
                break

            # 读取所有SR帧
            frames_sr = []
            all_valid = True

            for cap_sr in caps_sr:
                ret_sr, frame_sr = cap_sr.read()
                if not ret_sr:
                    all_valid = False
                    break
                frames_sr.append(frame_sr)

            if not all_valid:
                break

            # 转换颜色空间 (BGR -> RGB)
            frame_gt = cv2.cvtColor(frame_gt, cv2.COLOR_BGR2RGB)
            frames_sr = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_sr]

            # 计算每个SR的指标
            for sr_idx, frame_sr in enumerate(frames_sr):
                value = self._compute_metric(frame_sr, frame_gt, sr_idx, frame_idx)
                metric_values[sr_idx].append(value)

            # 更新进度
            progress = int((frame_idx + 1) / total_frames * 90)
            self.progress_update.emit(progress)

            frame_idx += 1

        # 释放资源
        for cap in caps_sr:
            cap.release()
        cap_gt.release()

        # 计算平均值并更新结果
        for sr_idx in range(num_sr):
            values = metric_values[sr_idx]
            if len(values) > 0:
                avg = np.mean([v for v in values if not np.isnan(v)])
                self.result_update.emit(sr_idx, f"SR{sr_idx + 1}: {avg:.4f}")

        # 发送绘图数据
        self.plot_update.emit({
            'type': 'curve',
            'data': metric_values,
            'num_sr': num_sr
        })

        self.progress_update.emit(100)

    def _evaluate_images(self):
        """评估图像"""
        num_sr = self.num_sr

        # 读取GT图像
        img_gt = cv2.imread(self.files[-1])
        if img_gt is None:
            raise ValueError("Cannot read GT image")
        img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB)

        metric_values = {}

        # 计算每个SR的指标
        for sr_idx in range(num_sr):
            img_sr = cv2.imread(self.files[sr_idx])
            if img_sr is None:
                raise ValueError(f"Cannot read SR image {sr_idx + 1}")
            img_sr = cv2.cvtColor(img_sr, cv2.COLOR_BGR2RGB)

            value = self._compute_metric(img_sr, img_gt, sr_idx, 0)
            metric_values[sr_idx] = value

            # 更新结果
            self.result_update.emit(sr_idx, f"SR{sr_idx + 1}: {value:.4f}")

            # 更新进度
            progress = int((sr_idx + 1) / num_sr * 90)
            self.progress_update.emit(progress)

        # 发送绘图数据
        self.plot_update.emit({
            'type': 'bar',
            'data': metric_values,
            'num_sr': num_sr
        })

        self.progress_update.emit(100)

    def _compute_metric(self, img_sr, img_gt, sr_idx, frame_idx):
        """计算指标"""
        if self.metric_type == 'LPIPS_Alex':
            return self.calculator.calculate_lpips(img_sr, img_gt)
        elif self.metric_type == 'NIQE':
            return self.calculator.calculate_niqe(img_sr)
        elif self.metric_type == 'Temporal_MSE':
            return self.calculator.calculate_temporal_mse(img_sr, img_gt, sr_idx, frame_idx)
        elif self.metric_type == 'Sharpness_Ratio':
            return self.calculator.calculate_sharpness_ratio(img_sr, img_gt)
        elif self.metric_type == 'Delta_E':
            return self.calculator.calculate_delta_e(img_sr, img_gt)
        else:
            return np.nan


class MatplotlibCanvas(FigureCanvas):
    """Matplotlib画布"""

    def __init__(self, parent=None, width=8, height=3, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)

    def plot_curves(self, data, num_sr, metric_type):
        """绘制曲线"""
        self.axes.clear()

        colors = ['r', 'g', 'b', 'orange', 'purple']

        for sr_idx in range(num_sr):
            values = data[sr_idx]
            if len(values) > 0:
                self.axes.plot(values, color=colors[sr_idx],
                               linewidth=2, label=f'SR Method {sr_idx + 1}')

        self.axes.set_xlabel('Frame Index', fontsize=10)
        self.axes.set_ylabel(self._get_ylabel(metric_type), fontsize=10)
        self.axes.set_title(f'{metric_type} Curve', fontsize=12, fontweight='bold')
        self.axes.legend(loc='best')
        self.axes.grid(True, alpha=0.3)

        self.fig.tight_layout()
        self.draw()

    def plot_bars(self, data, num_sr, metric_type):
        """绘制柱状图"""
        self.axes.clear()

        colors = ['r', 'g', 'b', 'orange', 'purple']

        x = np.arange(num_sr)
        values = [data[i] for i in range(num_sr)]

        bars = self.axes.bar(x, values, color=colors[:num_sr], width=0.6)

        # 在柱状图上显示数值
        for i, (bar, value) in enumerate(zip(bars, values)):
            height = bar.get_height()
            self.axes.text(bar.get_x() + bar.get_width() / 2., height,
                           f'{value:.4f}',
                           ha='center', va='bottom', fontsize=9)

        self.axes.set_xlabel('SR Methods', fontsize=10)
        self.axes.set_ylabel(self._get_ylabel(metric_type), fontsize=10)
        self.axes.set_title(f'{metric_type} Comparison', fontsize=12, fontweight='bold')
        self.axes.set_xticks(x)
        self.axes.set_xticklabels([f'SR{i + 1}' for i in range(num_sr)])
        self.axes.grid(True, alpha=0.3, axis='y')

        self.fig.tight_layout()
        self.draw()

    def _get_ylabel(self, metric_type):
        """获取Y轴标签"""
        labels = {
            'LPIPS_Alex': 'LPIPS (Lower is Better)',
            'NIQE': 'NIQE (Lower is Better)',
            'Temporal_MSE': 'Temporal MSE (Lower is Better)',
            'Sharpness_Ratio': 'Sharpness Ratio (Closer to 1)',
            'Delta_E': 'Delta E (Lower is Better)'
        }
        return labels.get(metric_type, metric_type)


class MainWindow(QMainWindow):
    """主窗口"""

    def __init__(self):
        super().__init__()

        self.num_sr = 3
        self.metric_type = 'LPIPS_Alex'
        self.files = []
        self.file_edits = []

        self.init_ui()
        self.check_environment()

    def init_ui(self):
        """初始化UI"""
        self.setWindowTitle('Advanced Video Quality Evaluation Tool')
        self.setGeometry(100, 100, 1100, 800)

        # 主widget
        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        # 主布局
        main_layout = QVBoxLayout()
        main_widget.setLayout(main_layout)

        # ===== 顶部控制区 =====
        control_layout = QHBoxLayout()

        # 指标选择
        metric_label = QLabel('Metric Type:')
        metric_label.setFont(QFont('Arial', 10))
        self.metric_combo = QComboBox()
        self.metric_combo.addItems(['LPIPS_Alex', 'NIQE', 'Temporal_MSE',
                                    'Sharpness_Ratio', 'Delta_E'])
        self.metric_combo.currentTextChanged.connect(self.on_metric_changed)

        # SR数量选择
        num_sr_label = QLabel('Number of SR Results:')
        num_sr_label.setFont(QFont('Arial', 10))
        self.num_sr_combo = QComboBox()
        self.num_sr_combo.addItems(['3', '4', '5'])
        self.num_sr_combo.currentTextChanged.connect(self.on_num_sr_changed)

        # 环境状态
        self.status_label = QLabel('Status: Checking...')
        self.status_label.setFont(QFont('Arial', 9))

        control_layout.addWidget(metric_label)
        control_layout.addWidget(self.metric_combo)
        control_layout.addSpacing(20)
        control_layout.addWidget(num_sr_label)
        control_layout.addWidget(self.num_sr_combo)
        control_layout.addSpacing(20)
        control_layout.addWidget(self.status_label)
        control_layout.addStretch()

        main_layout.addLayout(control_layout)

        # ===== 文件选择区 =====
        self.file_group = QGroupBox('File Selection')
        self.file_group.setFont(QFont('Arial', 10, QFont.Bold))
        self.file_layout = QGridLayout()
        self.file_group.setLayout(self.file_layout)

        main_layout.addWidget(self.file_group)

        # ===== 开始按钮 =====
        self.start_btn = QPushButton('Start Evaluation')
        self.start_btn.setFont(QFont('Arial', 12, QFont.Bold))
        self.start_btn.setMinimumHeight(50)
        self.start_btn.clicked.connect(self.start_evaluation)
        self.start_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)

        main_layout.addWidget(self.start_btn)

        # ===== 结果显示区 =====
        result_group = QGroupBox('Results')
        result_group.setFont(QFont('Arial', 10, QFont.Bold))
        result_layout = QHBoxLayout()
        result_group.setLayout(result_layout)

        self.result_labels = []
        colors = ['red', 'green', 'blue', 'orange', 'purple']
        for i in range(5):  # 最多5个SR
            label = QLabel(f'SR{i + 1}: --')
            label.setFont(QFont('Arial', 11, QFont.Bold))
            label.setStyleSheet(f'color: {colors[i]};')
            label.setAlignment(Qt.AlignLeft)
            result_layout.addWidget(label)
            self.result_labels.append(label)

        main_layout.addWidget(result_group)

        # ===== 进度条 =====
        progress_group = QGroupBox('Progress')
        progress_layout = QVBoxLayout()
        progress_group.setLayout(progress_layout)

        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimumHeight(25)
        progress_layout.addWidget(self.progress_bar)

        main_layout.addWidget(progress_group)

        # ===== 绘图区 =====
        plot_group = QGroupBox('Visualization')
        plot_group.setFont(QFont('Arial', 10, QFont.Bold))
        plot_layout = QVBoxLayout()
        plot_group.setLayout(plot_layout)

        self.canvas = MatplotlibCanvas(self, width=10, height=4, dpi=100)
        plot_layout.addWidget(self.canvas)

        main_layout.addWidget(plot_group)

        # 初始化文件选择器
        self.update_file_selectors()

    def check_environment(self):
        """检查环境"""
        issues = []

        if not LPIPS_AVAILABLE:
            issues.append("LPIPS")
        if not PYIQA_AVAILABLE:
            issues.append("PyIQA")

        if not torch.cuda.is_available():
            device_info = "CPU mode (slower)"
        else:
            device_info = f"GPU: {torch.cuda.get_device_name(0)}"

        if len(issues) == 0:
            self.status_label.setText(f'Status: ✓ Ready ({device_info})')
            self.status_label.setStyleSheet('color: green;')
        else:
            self.status_label.setText(f'Status: ⚠ Missing: {", ".join(issues)} ({device_info})')
            self.status_label.setStyleSheet('color: orange;')

    def on_metric_changed(self, metric):
        """指标类型改变"""
        self.metric_type = metric

        # 清空结果
        for label in self.result_labels:
            label.setText(label.text().split(':')[0] + ': --')

        # 清空图表
        self.canvas.axes.clear()
        self.canvas.draw()

    def on_num_sr_changed(self, num):
        """SR数量改变"""
        self.num_sr = int(num)
        self.update_file_selectors()

    def update_file_selectors(self):
        """更新文件选择器"""
        # 清空旧的
        for i in reversed(range(self.file_layout.count())):
            self.file_layout.itemAt(i).widget().setParent(None)

        self.file_edits = []
        self.files = [None] * (self.num_sr + 1)

        # 创建新的
        labels = [f'SR Result {i + 1}' for i in range(self.num_sr)] + ['Ground Truth']

        for i, label_text in enumerate(labels):
            # 标签
            label = QLabel(label_text + ':')
            label.setFont(QFont('Arial', 9))
            self.file_layout.addWidget(label, i, 0)

            # 文本框
            edit = QLineEdit()
            edit.setReadOnly(True)
            edit.setPlaceholderText('No file selected')
            self.file_layout.addWidget(edit, i, 1)
            self.file_edits.append(edit)

            # 浏览按钮
            btn = QPushButton('Browse')
            btn.setMaximumWidth(100)
            btn.clicked.connect(lambda checked, idx=i: self.select_file(idx))
            self.file_layout.addWidget(btn, i, 2)

        # 更新结果标签显示
        for i, label in enumerate(self.result_labels):
            if i < self.num_sr:
                label.setVisible(True)
                label.setText(f'SR{i + 1}: --')
            else:
                label.setVisible(False)

    def select_file(self, idx):
        """选择文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            'Select File',
            '',
            'Video/Image Files (*.mp4 *.avi *.png *.jpg *.jpeg *.bmp);;All Files (*)'
        )

        if file_path:
            self.files[idx] = file_path
            self.file_edits[idx].setText(file_path)

    def start_evaluation(self):
        """开始评估"""
        # 检查文件是否都已选择
        if None in self.files:
            QMessageBox.warning(self, 'Warning', 'Please select all files!')
            return

        # 检查文件格式
        extensions = [Path(f).suffix.lower() for f in self.files]
        if len(set(extensions)) > 1:
            QMessageBox.warning(self, 'Warning', 'All files must have the same format!')
            return

        # 判断是视频还是图像
        is_video = extensions[0] in ['.mp4', '.avi', '.mov', '.mkv']

        # 检查NIQE指标的特殊性（不需要GT）
        if self.metric_type == 'NIQE':
            # NIQE只需要SR结果
            pass

        # 禁用开始按钮
        self.start_btn.setEnabled(False)
        self.progress_bar.setValue(0)

        # 清空旧结果
        for label in self.result_labels[:self.num_sr]:
            label.setText(label.text().split(':')[0] + ': Computing...')

        # 创建并启动评估线程
        self.eval_thread = EvaluationThread(
            self.files, self.num_sr, self.metric_type, is_video
        )

        self.eval_thread.progress_update.connect(self.update_progress)
        self.eval_thread.result_update.connect(self.update_result)
        self.eval_thread.plot_update.connect(self.update_plot)
        self.eval_thread.finished.connect(self.evaluation_finished)
        self.eval_thread.error.connect(self.evaluation_error)

        self.eval_thread.start()

    def update_progress(self, value):
        """更新进度"""
        self.progress_bar.setValue(value)

    def update_result(self, sr_idx, result_text):
        """更新结果"""
        self.result_labels[sr_idx].setText(result_text)

    def update_plot(self, plot_data):
        """更新图表"""
        if plot_data['type'] == 'curve':
            self.canvas.plot_curves(
                plot_data['data'],
                plot_data['num_sr'],
                self.metric_type
            )
        elif plot_data['type'] == 'bar':
            self.canvas.plot_bars(
                plot_data['data'],
                plot_data['num_sr'],
                self.metric_type
            )

    def evaluation_finished(self):
        """评估完成"""
        self.start_btn.setEnabled(True)
        self.progress_bar.setValue(100)
        QMessageBox.information(self, 'Success', 'Evaluation completed!')

    def evaluation_error(self, error_msg):
        """评估错误"""
        self.start_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        QMessageBox.critical(self, 'Error', f'Evaluation failed:\n{error_msg}')


def main():
    app = QApplication(sys.argv)

    # 设置样式
    app.setStyle('Fusion')

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
