import sys
import time
import os
import csv
import numpy as np
from PySide6.QtCore import Qt, QThread, Signal, Slot, QTimer
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QProgressBar, QTableWidget,
    QTableWidgetItem, QHeaderView, QFrame, QFileDialog, QSplitter,
    QGroupBox, QSizePolicy, QDialog, QScrollArea, QDialogButtonBox
)
from PySide6.QtGui import QFont, QColor, QPalette, QIcon, QPainter, QBrush, QPen

# Import custom audio engine & chord detector
from audio_engine import AudioEngine
from chord_detector import NOTE_NAMES_SHARP, ALTO_PREFERRED_NAMES, get_chord_note_info
from key_estimator import estimate_keys, analyze_chord_progression



class ClickableCard(QFrame):
    """
    Custom clickable frame used for mode selection cards.
    """
    clicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setCursor(Qt.PointingHandCursor)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)


class ChromaVisualizer(QWidget):
    """
    Custom widget to display real-time 12-bin Chromagram energy bars.
    Supports switching between Concert Pitch mode and Alto Sax Eb transposed pitch mode.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.chroma = np.zeros(12)
        self.pitch_mode = "concert"  # "concert" or "alto"
        self.setMinimumHeight(140)

    def set_chroma(self, chroma: np.ndarray):
        self.chroma = chroma
        self.update()

    def set_pitch_mode(self, mode: str):
        self.pitch_mode = mode
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        width = self.width()
        height = self.height()
        padding = 10
        bottom_margin = 36
        top_margin = 25

        usable_w = width - 2 * padding
        usable_h = height - top_margin - bottom_margin

        n_bars = 12
        bar_gap = 6
        bar_width = (usable_w - (n_bars - 1) * bar_gap) / n_bars

        max_val = np.max(self.chroma) if len(self.chroma) > 0 and np.max(self.chroma) > 0 else 1.0

        for i in range(n_bars):
            val = float(self.chroma[i]) if i < len(self.chroma) else 0.0
            norm_val = min(1.0, max(0.0, val / max_val)) if max_val > 0 else 0.0

            bar_h = norm_val * usable_h
            x = padding + i * (bar_width + bar_gap)
            y = top_margin + (usable_h - bar_h)

            # Gradient color based on intensity (Blue to Violet/Red)
            hue = int(240 - norm_val * 180)
            color = QColor.fromHsv(hue, 200, 240)

            # Draw bar
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(color))
            painter.drawRoundedRect(int(x), int(y), int(bar_width), int(bar_h), 4, 4)

            # Pitch names based on active mode (Concert vs Alto Sax Eb)
            concert_note = NOTE_NAMES_SHARP[i]
            alto_note = ALTO_PREFERRED_NAMES[(i + 9) % 12]

            if self.pitch_mode == "alto":
                main_note = alto_note
                sub_note = f"({concert_note})"
                main_color = QColor("#A6E3A1")
            else:
                main_note = concert_note
                sub_note = f"({alto_note})"
                main_color = QColor("#89B4FA")

            # Draw primary note text
            painter.setPen(QPen(main_color))
            font = QFont("Segoe UI", 9.5, QFont.Bold)
            painter.setFont(font)
            painter.drawText(
                int(x - 4), int(height - 30), int(bar_width + 8), 16,
                Qt.AlignCenter, main_note
            )

            # Draw secondary (sub) note text
            painter.setPen(QPen(QColor("#7F849C")))
            font_sub = QFont("Segoe UI", 7.5)
            painter.setFont(font_sub)
            painter.drawText(
                int(x - 4), int(height - 14), int(bar_width + 8), 14,
                Qt.AlignCenter, sub_note
            )

            # Draw percentage above bar
            if norm_val > 0.15:
                painter.setPen(QPen(QColor("#A6ADC8")))
                font_val = QFont("Segoe UI", 8)
                painter.setFont(font_val)
                painter.drawText(
                    int(x), int(max(0, y - 4)), int(bar_width), 15,
                    Qt.AlignCenter, f"{int(norm_val * 100)}%"
                )


class SelectedChordVisualizer(QWidget):
    """
    Widget to display detailed note distribution (Chromagram snapshot) of a selected chord
    from history, highlighting its constituent notes (Root, 3rd, 5th, 7th) and pitch names.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.chord_name = None
        self.alto_chord = None
        self.confidence = None
        self.timestamp = None
        self.chroma = np.zeros(12)
        self.note_info = None
        self.pitch_mode = "concert"  # "concert" or "alto"
        self.setMinimumHeight(160)

    def set_selected_chord(self, chord_name: str, alto_chord: str, confidence: str, timestamp: str, chroma: np.ndarray):
        self.chord_name = chord_name
        self.alto_chord = alto_chord
        self.confidence = confidence
        self.timestamp = timestamp
        self.chroma = chroma if chroma is not None else np.zeros(12)
        self.note_info = get_chord_note_info(chord_name) if chord_name else None
        self.update()

    def set_pitch_mode(self, mode: str):
        self.pitch_mode = mode
        self.update()

    def clear_selected_chord(self):
        self.chord_name = None
        self.alto_chord = None
        self.confidence = None
        self.timestamp = None
        self.chroma = np.zeros(12)
        self.note_info = None
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        width = self.width()
        height = self.height()

        if not self.chord_name or self.chord_name == "N.C.":
            painter.setPen(QPen(QColor("#7F849C")))
            font = QFont("Segoe UI", 11)
            painter.setFont(font)
            painter.drawText(
                0, 0, width, height,
                Qt.AlignCenter,
                "לחץ על שורה בהיסטוריה להצגת התפלגות הצלילים המרכיבים"
            )
            return

        # 1. Top summary header line
        info_str = f"זמן: {self.timestamp}  |  ודאות: {self.confidence}"
        painter.setPen(QPen(QColor("#BAC2DE")))
        painter.setFont(QFont("Segoe UI", 9.5))
        painter.drawText(10, 10, width - 20, 18, Qt.AlignRight | Qt.AlignVCenter, info_str)

        if self.pitch_mode == "alto":
            main_chord_title = f"אקורד אלט (Eb): {self.alto_chord}   [קונצרט: {self.chord_name}]"
            title_color = QColor("#A6E3A1")
        else:
            main_chord_title = f"אקורד: {self.chord_name}   (אלט Eb: {self.alto_chord})"
            title_color = QColor("#89B4FA")

        painter.setPen(QPen(title_color))
        painter.setFont(QFont("Segoe UI", 12, QFont.Bold))
        painter.drawText(10, 10, width - 20, 22, Qt.AlignLeft | Qt.AlignVCenter, main_chord_title)

        # Notes breakdown line below title
        notes_list = self.note_info.get('chord_notes', []) if self.note_info else []
        if notes_list:
            if self.pitch_mode == "alto":
                notes_str = "הרכב צלילי אלט: " + "  ·  ".join([f"{n['alto_note']} ({n['interval']})" for n in notes_list])
            else:
                notes_str = "הרכב הצלילים: " + "  ·  ".join([f"{n['concert_note']} ({n['interval']})" for n in notes_list])
        else:
            notes_str = ""
        painter.setFont(QFont("Segoe UI", 9.5, QFont.Bold))
        painter.setPen(QPen(QColor("#A6E3A1" if self.pitch_mode == "alto" else "#89B4FA")))
        painter.drawText(10, 36, width - 20, 18, Qt.AlignLeft | Qt.AlignVCenter, notes_str)

        # 2. Draw 12 Chromagram bars
        padding = 10
        top_margin = 80
        bottom_margin = 40

        usable_w = width - 2 * padding
        usable_h = height - top_margin - bottom_margin

        n_bars = 12
        bar_gap = 5
        bar_width = max(1.0, (usable_w - (n_bars - 1) * bar_gap) / n_bars)

        max_val = np.max(self.chroma) if len(self.chroma) > 0 and np.max(self.chroma) > 0 else 1.0
        active_indices = self.note_info.get('active_pitch_indices', set()) if self.note_info else set()

        interval_map = {}
        if self.note_info:
            for n in self.note_info.get('chord_notes', []):
                interval_map[n['pitch_idx']] = n['interval']

        for i in range(n_bars):
            val = float(self.chroma[i]) if i < len(self.chroma) else 0.0
            norm_val = min(1.0, max(0.0, val / max_val)) if max_val > 0 else 0.0

            bar_h = norm_val * usable_h
            x = padding + i * (bar_width + bar_gap)
            y = top_margin + (usable_h - bar_h)

            is_active = i in active_indices

            if is_active:
                color = QColor("#F9E2AF") if interval_map.get(i) == "Root" else QColor("#A6E3A1")
                pen = QPen(QColor("#F5E0DC"), 2)
            else:
                color = QColor(49, 50, 68, 140)
                pen = Qt.NoPen

            # Draw bar
            painter.setPen(pen)
            painter.setBrush(QBrush(color))
            painter.drawRoundedRect(int(x), int(y), int(bar_width), int(bar_h), 4, 4)

            # Draw interval badge above bar if active
            if is_active and i in interval_map:
                interval_txt = interval_map[i]
                painter.setPen(QPen(QColor("#F9E2AF" if interval_txt == "Root" else "#A6E3A1")))
                font_badge = QFont("Segoe UI", 8.5, QFont.Bold)
                painter.setFont(font_badge)
                painter.drawText(
                    int(x - 4), int(max(top_margin - 20, y - 18)), int(bar_width + 8), 16,
                    Qt.AlignCenter, interval_txt
                )

            concert_note = NOTE_NAMES_SHARP[i]
            alto_note = ALTO_PREFERRED_NAMES[(i + 9) % 12]

            if self.pitch_mode == "alto":
                top_note = alto_note
                sub_note = f"({concert_note})"
                top_color = QColor("#F5E0DC") if is_active else QColor("#A6E3A1")
            else:
                top_note = concert_note
                sub_note = f"({alto_note})"
                top_color = QColor("#F5E0DC") if is_active else QColor("#89B4FA")

            # Draw Primary Note Name below bar
            painter.setPen(QPen(top_color))
            font_note = QFont("Segoe UI", 9, QFont.Bold if is_active else QFont.Normal)
            painter.setFont(font_note)
            painter.drawText(
                int(x - 4), int(height - 36), int(bar_width + 8), 16,
                Qt.AlignCenter, top_note
            )

            # Draw Secondary Transposed Note Name below primary note
            painter.setPen(QPen(QColor("#A6E3A1" if (is_active and self.pitch_mode != "alto") else "#585B70")))
            font_alto = QFont("Segoe UI", 8)
            painter.setFont(font_alto)
            painter.drawText(
                int(x - 4), int(height - 18), int(bar_width + 8), 16,
                Qt.AlignCenter, sub_note
            )


class KeyAnalysisDialog(QDialog):
    """
    Modal dialog showing chord frequency statistics and musical key estimation
    based on the current chord history session.
    """
    def __init__(self, analysis: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("ניתוח סולם וסיכום אקורדים (Key & Chord Analysis)")
        self.resize(780, 620)
        self.setMinimumSize(680, 500)
        self._build_ui(analysis)
        self._apply_style()

    def _build_ui(self, analysis: dict):
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(14)
        main_layout.setContentsMargins(20, 20, 20, 20)

        total = analysis.get('total_chords', 0)
        unique = analysis.get('unique_chords_count', 0)
        chord_stats = analysis.get('chord_stats', [])
        key_candidates = analysis.get('key_candidates', [])

        # --- Summary header ---
        hdr = QLabel(f"סה\"כ אקורדים שנלכדו: {total}   |   סוגים שונים: {unique}")
        hdr.setFont(QFont("Segoe UI", 11, QFont.Bold))
        hdr.setStyleSheet("color: #CDD6F4;")
        main_layout.addWidget(hdr)

        splitter = QSplitter(Qt.Horizontal)

        # ==== LEFT: Chord frequency table ====
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 8, 0)

        chords_lbl = QLabel("שכיחות אקורדים (Chord Frequency)")
        chords_lbl.setFont(QFont("Segoe UI", 10, QFont.Bold))
        chords_lbl.setStyleSheet("color: #89B4FA;")
        left_layout.addWidget(chords_lbl)

        tbl = QTableWidget(len(chord_stats), 4)
        tbl.setHorizontalHeaderLabels(["קונצרט", "אלט Eb", "פעמים", "%"])
        tbl.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        tbl.setAlternatingRowColors(True)
        tbl.setEditTriggers(QTableWidget.NoEditTriggers)
        tbl.setSelectionMode(QTableWidget.NoSelection)
        tbl.verticalHeader().setVisible(False)

        for row, cs in enumerate(chord_stats):
            tbl.setItem(row, 0, QTableWidgetItem(cs['concert_chord']))
            tbl.setItem(row, 1, QTableWidgetItem(cs['alto_chord']))
            tbl.setItem(row, 2, QTableWidgetItem(str(cs['count'])))
            tbl.setItem(row, 3, QTableWidgetItem(cs['percentage_str']))

        left_layout.addWidget(tbl)
        splitter.addWidget(left_widget)

        # ==== RIGHT: Key estimation results ====
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(8, 0, 0, 0)

        keys_lbl = QLabel("סבירות סולמות (Key Probability Ranking)")
        keys_lbl.setFont(QFont("Segoe UI", 10, QFont.Bold))
        keys_lbl.setStyleSheet("color: #A6E3A1;")
        right_layout.addWidget(keys_lbl)

        if not key_candidates:
            no_data = QLabel("אין מספיק נתונים לאמידת סולם\n(Play some chords first)")
            no_data.setAlignment(Qt.AlignCenter)
            no_data.setStyleSheet("color: #7F849C; font-size: 12px;")
            right_layout.addWidget(no_data)
        else:
            # Top key highlighted
            top = key_candidates[0]
            top_frame = QFrame()
            top_frame.setObjectName("topKeyFrame")
            top_layout = QVBoxLayout(top_frame)
            top_layout.setContentsMargins(16, 12, 16, 12)

            top_concert_lbl = QLabel(f"🎵  {top['concert_key']}")
            top_concert_lbl.setFont(QFont("Segoe UI", 18, QFont.Bold))
            top_concert_lbl.setStyleSheet("color: #F9E2AF;")
            top_concert_lbl.setAlignment(Qt.AlignCenter)

            top_alto_lbl = QLabel(f"סקסופון אלט (Eb): {top['alto_key']}")
            top_alto_lbl.setFont(QFont("Segoe UI", 12, QFont.Bold))
            top_alto_lbl.setStyleSheet("color: #A6E3A1;")
            top_alto_lbl.setAlignment(Qt.AlignCenter)

            scale_c = "  ·  ".join(top.get('concert_scale_notes', []))
            scale_a = "  ·  ".join(top.get('alto_scale_notes', []))
            scale_c_lbl = QLabel(f"צלילי קונצרט: {scale_c}")
            scale_c_lbl.setStyleSheet("color: #89B4FA; font-size: 10.5px;")
            scale_c_lbl.setAlignment(Qt.AlignCenter)
            scale_a_lbl = QLabel(f"צלילי אלט: {scale_a}")
            scale_a_lbl.setStyleSheet("color: #A6E3A1; font-size: 10.5px;")
            scale_a_lbl.setAlignment(Qt.AlignCenter)

            conf_bar = QProgressBar()
            conf_bar.setRange(0, 100)
            conf_bar.setValue(top.get('confidence_pct', 0))
            conf_bar.setFormat(f"סבירות: {top['confidence_str']}")
            conf_bar.setTextVisible(True)
            conf_bar.setFixedHeight(18)

            top_layout.addWidget(top_concert_lbl)
            top_layout.addWidget(top_alto_lbl)
            top_layout.addSpacing(6)
            top_layout.addWidget(scale_c_lbl)
            top_layout.addWidget(scale_a_lbl)
            top_layout.addSpacing(6)
            top_layout.addWidget(conf_bar)
            right_layout.addWidget(top_frame)

            # Remaining candidates
            if len(key_candidates) > 1:
                other_lbl = QLabel("מועמדים נוספים:")
                other_lbl.setFont(QFont("Segoe UI", 9, QFont.Bold))
                other_lbl.setStyleSheet("color: #BAC2DE; margin-top: 8px;")
                right_layout.addWidget(other_lbl)

                for cand in key_candidates[1:]:
                    row_widget = QWidget()
                    row_h = QHBoxLayout(row_widget)
                    row_h.setContentsMargins(4, 2, 4, 2)

                    key_name_lbl = QLabel(f"{cand['concert_key']}  /  Alto: {cand['alto_key']}")
                    key_name_lbl.setFont(QFont("Segoe UI", 9.5))
                    key_name_lbl.setStyleSheet("color: #CDD6F4;")

                    cand_bar = QProgressBar()
                    cand_bar.setRange(0, 100)
                    cand_bar.setValue(cand.get('confidence_pct', 0))
                    cand_bar.setFormat(cand['confidence_str'])
                    cand_bar.setTextVisible(True)
                    cand_bar.setFixedHeight(14)
                    cand_bar.setMaximumWidth(130)

                    row_h.addWidget(key_name_lbl, stretch=2)
                    row_h.addWidget(cand_bar, stretch=1)
                    right_layout.addWidget(row_widget)

        right_layout.addStretch()
        splitter.addWidget(right_widget)
        splitter.setSizes([340, 400])
        main_layout.addWidget(splitter, stretch=1)

        # Close button
        btn_box = QDialogButtonBox(QDialogButtonBox.Close)
        btn_box.rejected.connect(self.reject)
        main_layout.addWidget(btn_box)

    def _apply_style(self):
        self.setStyleSheet("""
            QDialog {
                background-color: #1E1E2E;
                color: #CDD6F4;
            }
            QLabel { color: #CDD6F4; }
            QGroupBox {
                color: #CDD6F4;
                border: 1px solid #45475A;
                border-radius: 8px;
                margin-top: 8px;
                padding-top: 12px;
            }
            QFrame#topKeyFrame {
                background-color: #181825;
                border: 2px solid #F9E2AF;
                border-radius: 10px;
            }
            QTableWidget {
                background-color: #181825;
                color: #CDD6F4;
                gridline-color: #313244;
                border: 1px solid #313244;
                border-radius: 6px;
            }
            QTableWidget::item:alternate { background-color: #1E1E2E; }
            QHeaderView::section {
                background-color: #313244;
                color: #CDD6F4;
                padding: 4px;
                border: none;
                font-weight: bold;
            }
            QProgressBar {
                background-color: #313244;
                border-radius: 4px;
                color: #CDD6F4;
                text-align: center;
            }
            QProgressBar::chunk { background-color: #F9E2AF; border-radius: 4px; }
            QPushButton {
                background-color: #313244;
                color: #CDD6F4;
                border: 1px solid #45475A;
                border-radius: 6px;
                padding: 6px 14px;
            }
            QPushButton:hover { background-color: #45475A; color: #F5E0DC; }
            QSplitter::handle { background-color: #313244; }
        """)


class AudioBridge(QThread):

    """
    Bridge connecting AudioEngine background thread safely to Qt GUI thread.
    """
    chord_signal = Signal(str, str, float, object, float)

    def __init__(self, audio_engine: AudioEngine):
        super().__init__()
        self.engine = audio_engine
        self.engine.on_chord_detected = self._handle_detection

    def _handle_detection(self, concert_chord, alto_chord, confidence, chroma, db_level):
        self.chord_signal.emit(concert_chord, alto_chord, confidence, chroma, db_level)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("זיהוי אקורדים בזמן אמת מהרמקולים / מיקרופון - סקסופון אלט | Real-time Chord Detector")
        self.resize(1180, 780)
        self.setMinimumSize(980, 680)

        # Core audio engine & bridge
        self.audio_engine = AudioEngine()
        self.bridge = AudioBridge(self.audio_engine)
        self.bridge.chord_signal.connect(self.on_chord_update)

        self.history_data = []
        self.last_added_chord = ""
        self.last_added_time = 0
        self.current_pitch_mode = "concert"  # "concert" or "alto"
        self.live_chroma_accum = np.zeros(12)  # accumulated chroma for live key estimation
        self.live_chroma_count = 0


        self.init_ui()
        self.apply_theme()
        self.populate_audio_devices()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        main_layout.setSpacing(14)
        main_layout.setContentsMargins(18, 18, 18, 18)

        # --- Top Header & Controls Panel ---
        header_box = QGroupBox("הגדרות מקור השמע (Audio Source: System Speakers / Microphone)")
        header_layout = QHBoxLayout(header_box)
        header_layout.setSpacing(14)

        # Device selection dropdown
        device_label = QLabel("מקור שמע:")
        device_label.setFont(QFont("Segoe UI", 10, QFont.Bold))
        self.combo_devices = QComboBox()
        self.combo_devices.setMinimumWidth(340)
        self.combo_devices.currentIndexChanged.connect(self.on_device_changed)

        # Start / Stop Listen Button
        self.btn_toggle = QPushButton("התחל זיהוי (Start)")
        self.btn_toggle.setFont(QFont("Segoe UI", 10, QFont.Bold))
        self.btn_toggle.setMinimumHeight(38)
        self.btn_toggle.setMinimumWidth(160)
        self.btn_toggle.setCursor(Qt.PointingHandCursor)
        self.btn_toggle.clicked.connect(self.toggle_audio_stream)

        # Level meter
        level_label = QLabel("ווליום:")
        level_label.setFont(QFont("Segoe UI", 9))
        self.progress_level = QProgressBar()
        self.progress_level.setRange(0, 100)
        self.progress_level.setValue(0)
        self.progress_level.setTextVisible(False)
        self.progress_level.setMaximumWidth(120)
        self.progress_level.setFixedHeight(14)

        header_layout.addWidget(device_label)
        header_layout.addWidget(self.combo_devices)
        header_layout.addWidget(self.btn_toggle)
        header_layout.addSpacing(15)
        header_layout.addWidget(level_label)
        header_layout.addWidget(self.progress_level)
        header_layout.addStretch()

        main_layout.addWidget(header_box)

        # --- Middle Main Section: Interactive Pitch Mode Cards ---
        cards_layout = QHBoxLayout()
        cards_layout.setSpacing(20)

        # 1. Concert Pitch Chord Card (Clickable)
        self.card_concert = ClickableCard()
        self.card_concert.setFrameShape(QFrame.StyledPanel)
        self.card_concert.setObjectName("cardConcertActive")
        layout_c = QVBoxLayout(self.card_concert)
        layout_c.setContentsMargins(18, 14, 18, 14)

        lbl_c_title = QLabel("אקורד מקור (Concert Pitch)")
        lbl_c_title.setFont(QFont("Segoe UI", 12, QFont.Bold))
        lbl_c_title.setStyleSheet("color: #89B4FA;")
        
        self.lbl_concert_status = QLabel("● נבחר לתצוגת תדרים (Click to select)")
        self.lbl_concert_status.setStyleSheet("color: #89B4FA; font-weight: bold; font-size: 11px;")

        self.lbl_concert_chord = QLabel("N.C.")
        self.lbl_concert_chord.setFont(QFont("Segoe UI", 46, QFont.Bold))
        self.lbl_concert_chord.setAlignment(Qt.AlignCenter)
        self.lbl_concert_chord.setStyleSheet("color: #89B4FA;")

        self.lbl_concert_conf = QLabel("רמת ודאות: 0%")
        self.lbl_concert_conf.setAlignment(Qt.AlignCenter)
        self.lbl_concert_conf.setStyleSheet("color: #BAC2DE; font-size: 11px;")

        layout_c.addWidget(lbl_c_title)
        layout_c.addWidget(self.lbl_concert_status)
        layout_c.addStretch()
        layout_c.addWidget(self.lbl_concert_chord)
        layout_c.addStretch()
        layout_c.addWidget(self.lbl_concert_conf)

        self.card_concert.clicked.connect(lambda: self.set_pitch_mode("concert"))

        # 2. Alto Saxophone Chord Card (Clickable Transposed Card)
        self.card_alto = ClickableCard()
        self.card_alto.setFrameShape(QFrame.StyledPanel)
        self.card_alto.setObjectName("cardAltoInactive")
        layout_a = QVBoxLayout(self.card_alto)
        layout_a.setContentsMargins(18, 14, 18, 14)

        lbl_a_title = QLabel("טרנספוזיציה לסקסופון אלט (Alto Sax Eb)")
        lbl_a_title.setFont(QFont("Segoe UI", 12, QFont.Bold))
        lbl_a_title.setStyleSheet("color: #A6E3A1;")

        self.lbl_alto_status = QLabel("לחץ לבחירה לתצוגת תדרים (Click to select)")
        self.lbl_alto_status.setStyleSheet("color: #6C7086; font-size: 11px;")

        self.lbl_alto_chord = QLabel("N.C.")
        self.lbl_alto_chord.setFont(QFont("Segoe UI", 46, QFont.Bold))
        self.lbl_alto_chord.setAlignment(Qt.AlignCenter)
        self.lbl_alto_chord.setStyleSheet("color: #A6E3A1;")

        lbl_a_info = QLabel("כלי Eb transposing (+9 סמיטונים)")
        lbl_a_info.setAlignment(Qt.AlignCenter)
        lbl_a_info.setStyleSheet("color: #BAC2DE; font-size: 11px;")

        layout_a.addWidget(lbl_a_title)
        layout_a.addWidget(self.lbl_alto_status)
        layout_a.addStretch()
        layout_a.addWidget(self.lbl_alto_chord)
        layout_a.addStretch()
        layout_a.addWidget(lbl_a_info)

        self.card_alto.clicked.connect(lambda: self.set_pitch_mode("alto"))

        # 3. Live Key Estimation Card
        self.card_key = QFrame()
        self.card_key.setFrameShape(QFrame.StyledPanel)
        self.card_key.setObjectName("cardKey")
        layout_k = QVBoxLayout(self.card_key)
        layout_k.setContentsMargins(18, 14, 18, 14)

        lbl_k_title = QLabel("סולם השיר (Estimated Key)")
        lbl_k_title.setFont(QFont("Segoe UI", 12, QFont.Bold))
        lbl_k_title.setStyleSheet("color: #F9E2AF;")

        lbl_k_sub = QLabel("אמידה בזמן אמת לפי תדרי השמע")
        lbl_k_sub.setStyleSheet("color: #A6ADC8; font-size: 11px;")

        self.lbl_live_key = QLabel("---")
        self.lbl_live_key.setFont(QFont("Segoe UI", 28, QFont.Bold))
        self.lbl_live_key.setAlignment(Qt.AlignCenter)
        self.lbl_live_key.setStyleSheet("color: #F9E2AF;")

        self.lbl_live_key_alto = QLabel("אלט Eb: ---")
        self.lbl_live_key_alto.setFont(QFont("Segoe UI", 14, QFont.Bold))
        self.lbl_live_key_alto.setAlignment(Qt.AlignCenter)
        self.lbl_live_key_alto.setStyleSheet("color: #A6E3A1;")

        self.lbl_live_key_conf = QLabel("סבירות: ---%")
        self.lbl_live_key_conf.setAlignment(Qt.AlignCenter)
        self.lbl_live_key_conf.setStyleSheet("color: #BAC2DE; font-size: 11px;")

        self.live_key_bar = QProgressBar()
        self.live_key_bar.setRange(0, 100)
        self.live_key_bar.setValue(0)
        self.live_key_bar.setTextVisible(False)
        self.live_key_bar.setFixedHeight(8)

        layout_k.addWidget(lbl_k_title)
        layout_k.addWidget(lbl_k_sub)
        layout_k.addStretch()
        layout_k.addWidget(self.lbl_live_key)
        layout_k.addWidget(self.lbl_live_key_alto)
        layout_k.addStretch()
        layout_k.addWidget(self.lbl_live_key_conf)
        layout_k.addWidget(self.live_key_bar)

        cards_layout.addWidget(self.card_concert)
        cards_layout.addWidget(self.card_alto)
        cards_layout.addWidget(self.card_key)


        main_layout.addLayout(cards_layout, stretch=2)

        # --- Lower Middle Section: Chromagram Visualizer ---
        self.chroma_box = QGroupBox("התפלגות תדרים בזמן אמת (Real-time Chromagram - Concert Pitch)")
        chroma_layout = QVBoxLayout(self.chroma_box)
        self.chroma_visualizer = ChromaVisualizer()
        chroma_layout.addWidget(self.chroma_visualizer)
        main_layout.addWidget(self.chroma_box, stretch=2)

        # --- Bottom Section: Chord History Table + Selected Chord Inspector ---
        bottom_splitter = QSplitter(Qt.Horizontal)

        # Left Group Box: History Log Table
        history_box = QGroupBox("היסטוריית אקורדים שנלכדו (Chord History Log)")
        history_layout = QVBoxLayout(history_box)

        btn_bar = QHBoxLayout()
        btn_clear = QPushButton("ניקוי היסטוריה")
        btn_clear.clicked.connect(self.clear_history)
        btn_export = QPushButton("ייצוא לקובץ CSV")
        btn_export.clicked.connect(self.export_history)
        btn_bar.addWidget(btn_clear)
        btn_bar.addWidget(btn_export)
        btn_bar.addStretch()

        self.table_history = QTableWidget(0, 4)
        self.table_history.setHorizontalHeaderLabels(["זמן (Time)", "קונצרט (Concert)", "סקסופון אלט (Alto Sax Eb)", "רמת ודאות"])
        self.table_history.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table_history.setAlternatingRowColors(True)
        self.table_history.setSelectionBehavior(QTableWidget.SelectRows)
        self.table_history.setSelectionMode(QTableWidget.SingleSelection)
        self.table_history.itemSelectionChanged.connect(self.on_history_selection_changed)

        history_layout.addLayout(btn_bar)
        history_layout.addWidget(self.table_history)

        # Right Group Box: Selected Chord Diagram Visualizer
        self.selected_box = QGroupBox("התפלגות צלילי אקורד מסומן (Selected Chord - Concert Pitch)")
        selected_layout = QVBoxLayout(self.selected_box)

        self.selected_chord_visualizer = SelectedChordVisualizer()
        selected_layout.addWidget(self.selected_chord_visualizer)

        bottom_splitter.addWidget(history_box)
        bottom_splitter.addWidget(self.selected_box)
        bottom_splitter.setSizes([540, 540])

        main_layout.addWidget(bottom_splitter, stretch=3)

    def set_pitch_mode(self, mode: str):
        self.current_pitch_mode = mode

        if mode == "concert":
            self.card_concert.setObjectName("cardConcertActive")
            self.card_alto.setObjectName("cardAltoInactive")
            self.lbl_concert_status.setText("● נבחר לתצוגת תדרים (Concert Pitch)")
            self.lbl_concert_status.setStyleSheet("color: #89B4FA; font-weight: bold; font-size: 11px;")
            self.lbl_alto_status.setText("לחץ להעברת התדרים לטרנספוזיציית אלט")
            self.lbl_alto_status.setStyleSheet("color: #6C7086; font-size: 11px;")
            self.chroma_box.setTitle("התפלגות תדרים בזמן אמת (Real-time Chromagram - Concert Pitch)")
            self.selected_box.setTitle("התפלגות צלילי אקורד מסומן (Selected Chord - Concert Pitch)")
        else:
            self.card_concert.setObjectName("cardConcertInactive")
            self.card_alto.setObjectName("cardAltoActive")
            self.lbl_concert_status.setText("לחץ להעברת התדרים לפיץ' קונצרט")
            self.lbl_concert_status.setStyleSheet("color: #6C7086; font-size: 11px;")
            self.lbl_alto_status.setText("● נבחר לתצוגת תדרים (Alto Sax Eb Pitch)")
            self.lbl_alto_status.setStyleSheet("color: #A6E3A1; font-weight: bold; font-size: 11px;")
            self.chroma_box.setTitle("התפלגות תדרים בזמן אמת (Real-time Chromagram - Alto Sax Eb)")
            self.selected_box.setTitle("התפלגות צלילי אקורד מסומן (Selected Chord - Alto Sax Eb)")

        # Re-evaluate style for object names
        self.card_concert.setStyle(self.card_concert.style())
        self.card_alto.setStyle(self.card_alto.style())

        self.apply_theme()
        self.chroma_visualizer.set_pitch_mode(mode)
        self.selected_chord_visualizer.set_pitch_mode(mode)

    def apply_theme(self):
        """Applies Catppuccin Mocha Dark theme style."""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1E1E2E;
            }
            QGroupBox {
                color: #CDD6F4;
                font-weight: bold;
                border: 1px solid #45475A;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 14px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 14px;
                padding: 0 5px;
            }
            QFrame#cardConcertActive {
                background-color: #1E1E2E;
                border: 3px solid #89B4FA;
                border-radius: 12px;
            }
            QFrame#cardConcertInactive {
                background-color: #181825;
                border: 1px solid #313244;
                border-radius: 12px;
            }
            QFrame#cardAltoActive {
                background-color: #1E1E2E;
                border: 3px solid #A6E3A1;
                border-radius: 12px;
            }
            QFrame#cardAltoInactive {
                background-color: #181825;
                border: 1px solid #313244;
                border-radius: 12px;
            }
            QLabel {
                color: #CDD6F4;
            }
            QPushButton {
                background-color: #313244;
                color: #CDD6F4;
                border: 1px solid #45475A;
                border-radius: 6px;
                padding: 6px 14px;
            }
            QPushButton:hover {
                background-color: #45475A;
                color: #F5E0DC;
            }
            QPushButton:pressed {
                background-color: #585B70;
            }
            QComboBox {
                background-color: #313244;
                color: #CDD6F4;
                border: 1px solid #45475A;
                border-radius: 6px;
                padding: 4px 8px;
            }
            QComboBox QAbstractItemView {
                background-color: #181825;
                color: #CDD6F4;
                selection-background-color: #45475A;
            }
            QProgressBar {
                background-color: #313244;
                border-radius: 4px;
            }
            QProgressBar::chunk {
                background-color: #A6E3A1;
                border-radius: 4px;
            }
            QTableWidget {
                background-color: #181825;
                color: #CDD6F4;
                gridline-color: #313244;
                border: 1px solid #313244;
                border-radius: 6px;
            }
            QTableWidget::item:selected {
                background-color: #45475A;
                color: #F9E2AF;
                font-weight: bold;
            }
            QHeaderView::section {
                background-color: #313244;
                color: #CDD6F4;
                padding: 4px;
                border: none;
                font-weight: bold;
            }
            QSplitter::handle {
                background-color: #313244;
                width: 4px;
            }
        """)

    def populate_audio_devices(self):
        self.combo_devices.clear()
        devices = AudioEngine.get_input_devices()
        if not devices:
            self.combo_devices.addItem("לא נמצאו התקני שמע (No Audio Devices)", None)
            return

        for dev in devices:
            self.combo_devices.addItem(dev['display_name'], dev)

        # Select default system speaker loopback automatically
        self.combo_devices.setCurrentIndex(0)

    def on_device_changed(self, index):
        dev_data = self.combo_devices.currentData()
        if dev_data and isinstance(dev_data, dict):
            self.audio_engine.set_device(dev_data['id'], dev_data['is_loopback'])

    def toggle_audio_stream(self):
        if self.audio_engine.is_running:
            self.audio_engine.stop()
            self.btn_toggle.setText("התחל זיהוי (Start)")
            self.btn_toggle.setStyleSheet("background-color: #313244;")
            self.lbl_concert_chord.setText("N.C.")
            self.lbl_alto_chord.setText("N.C.")
            self.progress_level.setValue(0)
            self.chroma_visualizer.set_chroma(np.zeros(12))
        else:
            try:
                dev_data = self.combo_devices.currentData()
                if dev_data and isinstance(dev_data, dict):
                    self.audio_engine.set_device(dev_data['id'], dev_data['is_loopback'])
                self.audio_engine.start()
                self.btn_toggle.setText("עצור זיהוי (Stop)")
                self.btn_toggle.setStyleSheet("background-color: #F38BA8; color: #11111B;")
            except Exception as e:
                print(f"Error starting audio stream: {e}")

    @Slot(str, str, float, object, float)
    def on_chord_update(self, concert_chord, alto_chord, confidence, chroma, db_level):
        # Update audio level meter (-60 dB to 0 dB mapped to 0..100)
        norm_db = int(max(0, min(100, (db_level + 60) * (100 / 60))))
        self.progress_level.setValue(norm_db)

        # Update real-time visualizer
        self.chroma_visualizer.set_chroma(chroma)

        # Update chord display
        self.lbl_concert_chord.setText(concert_chord)
        self.lbl_alto_chord.setText(alto_chord)
        self.lbl_concert_conf.setText(f"רמת ודאות: {int(confidence * 100)}%")

        # Log chord into history if it's sustained
        now = time.time()
        if concert_chord != "N.C." and confidence > 0.40:
            if concert_chord != self.last_added_chord or (now - self.last_added_time) > 2.0:
                self.add_history_entry(concert_chord, alto_chord, confidence, chroma)
                self.last_added_chord = concert_chord
                self.last_added_time = now

    def add_history_entry(self, concert_chord, alto_chord, confidence, chroma):
        timestamp = time.strftime("%H:%M:%S")
        chroma_copy = np.copy(chroma) if chroma is not None else np.zeros(12)
        conf_str = f"{int(confidence * 100)}%"

        self.history_data.append({
            'timestamp': timestamp,
            'concert_chord': concert_chord,
            'alto_chord': alto_chord,
            'confidence': conf_str,
            'chroma': chroma_copy
        })

        row = self.table_history.rowCount()
        self.table_history.insertRow(row)
        self.table_history.setItem(row, 0, QTableWidgetItem(timestamp))
        self.table_history.setItem(row, 1, QTableWidgetItem(concert_chord))
        self.table_history.setItem(row, 2, QTableWidgetItem(alto_chord))
        self.table_history.setItem(row, 3, QTableWidgetItem(conf_str))

        # Scroll to bottom
        self.table_history.scrollToBottom()

    def on_history_selection_changed(self):
        selected_rows = self.table_history.selectionModel().selectedRows()
        if not selected_rows:
            self.selected_chord_visualizer.clear_selected_chord()
            return
        row = selected_rows[0].row()
        if 0 <= row < len(self.history_data):
            entry = self.history_data[row]
            self.selected_chord_visualizer.set_selected_chord(
                chord_name=entry['concert_chord'],
                alto_chord=entry['alto_chord'],
                confidence=entry['confidence'],
                timestamp=entry['timestamp'],
                chroma=entry['chroma']
            )

    def clear_history(self):
        self.history_data.clear()
        self.table_history.setRowCount(0)
        self.selected_chord_visualizer.clear_selected_chord()

    def export_history(self):
        if not self.history_data:
            return
        filename, _ = QFileDialog.getSaveFileName(self, "ייצוא היסטוריה", "chords_history.csv", "CSV Files (*.csv)")
        if filename:
            try:
                with open(filename, 'w', newline='', encoding='utf-8-sig') as f:
                    writer = csv.writer(f)
                    writer.writerow(["Time", "Concert Pitch Chord", "Alto Sax (Eb) Transposed Chord", "Confidence"])
                    for item in self.history_data:
                        writer.writerow([item['timestamp'], item['concert_chord'], item['alto_chord'], item['confidence']])
            except Exception as e:
                print(f"Error saving history: {e}")

    def closeEvent(self, event):
        self.audio_engine.stop()
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()

