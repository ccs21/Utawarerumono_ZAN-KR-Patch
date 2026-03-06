import csv
import io
import os
import sys
from dataclasses import dataclass
from typing import List, Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QGuiApplication, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QSplitter,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
    QAbstractItemView,
    QHeaderView,
)


APP_TITLE = "Utawarerumono Translation CSV Helper"
DEFAULT_ENCODING = "utf-8-sig"


@dataclass
class SelectionState:
    row_indices: List[int]
    copied_text: str


class CsvTranslationHelper(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(1600, 900)

        self.file_path: Optional[str] = None
        self.headers: List[str] = []
        self.rows: List[List[str]] = []
        self.selection_state: Optional[SelectionState] = None
        self.last_open_dir: str = os.getcwd()
        self.is_dirty: bool = False

        self.jp_col: Optional[int] = None
        self.en_col: Optional[int] = None
        self.kr_col: Optional[int] = None

        self._build_ui()
        self._build_menu()
        self._update_ui_state()

    def _build_ui(self) -> None:
        central = QWidget(self)
        self.setCentralWidget(central)

        root = QHBoxLayout(central)

        splitter = QSplitter(Qt.Horizontal, self)
        root.addWidget(splitter)

        left = QWidget(self)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(6, 6, 6, 6)

        self.table = QTableWidget(self)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.setSelectionBehavior(QAbstractItemView.SelectItems)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(True)
        self.table.horizontalHeader().setStretchLastSection(False)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.table.itemSelectionChanged.connect(self._on_selection_changed)
        left_layout.addWidget(self.table)

        splitter.addWidget(left)

        right = QWidget(self)
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(8, 8, 8, 8)
        right_layout.setSpacing(8)

        self.file_label = QLabel("파일: (없음)", self)
        self.file_label.setWordWrap(True)
        right_layout.addWidget(self.file_label)

        self.next_blank_label = QLabel("다음 미번역 행: -", self)
        right_layout.addWidget(self.next_blank_label)

        self.selection_label = QLabel("현재 선택: -", self)
        self.selection_label.setWordWrap(True)
        right_layout.addWidget(self.selection_label)

        help_label = QLabel(
            "표에서 원하는 셀을 직접 선택한 뒤 Ctrl+C로 복사하세요.\n"
            "복사 시 모든 셀을 항상 큰따옴표로 감싸서 멀티라인 셀을 쉽게 확인할 수 있게 합니다.\n"
            "번역 결과를 아래 칸에 붙여넣고 '검증' 또는 'kr 열에 반영'을 누르세요.",
            self,
        )
        help_label.setWordWrap(True)
        right_layout.addWidget(help_label)

        self.translation_edit = QPlainTextEdit(self)
        self.translation_edit.setPlaceholderText("여기에 ChatGPT가 리턴한 번역 결과를 그대로 붙여넣으세요.")
        right_layout.addWidget(self.translation_edit, stretch=1)

        self.validation_label = QLabel("검증 결과: -", self)
        self.validation_label.setWordWrap(True)
        right_layout.addWidget(self.validation_label)

        self.validate_button = QPushButton("붙여넣은 번역문 검증", self)
        self.validate_button.clicked.connect(self.validate_translation_input)
        right_layout.addWidget(self.validate_button)

        self.apply_button = QPushButton("kr 열에 반영", self)
        self.apply_button.clicked.connect(self.apply_translation_to_kr)
        right_layout.addWidget(self.apply_button)

        self.save_button = QPushButton("CSV 저장", self)
        self.save_button.clicked.connect(self.save_csv)
        right_layout.addWidget(self.save_button)

        self.reload_button = QPushButton("파일 다시 읽기", self)
        self.reload_button.clicked.connect(self.reload_csv)
        right_layout.addWidget(self.reload_button)

        self.log_edit = QPlainTextEdit(self)
        self.log_edit.setReadOnly(True)
        self.log_edit.setPlaceholderText("작업 로그")
        self.log_edit.setMaximumBlockCount(300)
        self.log_edit.setFixedHeight(150)
        right_layout.addWidget(self.log_edit)

        right_layout.addStretch(1)
        splitter.addWidget(right)
        splitter.setSizes([1100, 500])

        self.status_bar = QStatusBar(self)
        self.setStatusBar(self.status_bar)

    def _build_menu(self) -> None:
        menu = self.menuBar()

        file_menu = menu.addMenu("파일")

        open_action = QAction("열기", self)
        open_action.setShortcut(QKeySequence.Open)
        open_action.triggered.connect(self.open_csv)
        file_menu.addAction(open_action)

        save_action = QAction("저장", self)
        save_action.setShortcut(QKeySequence.Save)
        save_action.triggered.connect(self.save_csv)
        file_menu.addAction(save_action)

        reload_action = QAction("다시 읽기", self)
        reload_action.setShortcut(QKeySequence.Refresh)
        reload_action.triggered.connect(self.reload_csv)
        file_menu.addAction(reload_action)

        file_menu.addSeparator()

        exit_action = QAction("종료", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        edit_menu = menu.addMenu("작업")

        copy_action = QAction("선택 셀 복사", self)
        copy_action.setShortcut(QKeySequence.Copy)
        copy_action.triggered.connect(self.copy_selected_cells)
        edit_menu.addAction(copy_action)

        validate_action = QAction("검증", self)
        validate_action.triggered.connect(self.validate_translation_input)
        edit_menu.addAction(validate_action)

        apply_action = QAction("kr 열에 반영", self)
        apply_action.triggered.connect(self.apply_translation_to_kr)
        edit_menu.addAction(apply_action)

    def _log(self, message: str) -> None:
        self.log_edit.appendPlainText(message)
        self.status_bar.showMessage(message, 4000)

    def _update_ui_state(self) -> None:
        has_file = self.file_path is not None
        has_selection = self.selection_state is not None and len(self.selection_state.row_indices) > 0

        self.validate_button.setEnabled(has_file and has_selection)
        self.apply_button.setEnabled(has_file and has_selection)
        self.save_button.setEnabled(has_file)
        self.reload_button.setEnabled(has_file)

        title = APP_TITLE
        if self.file_path:
            title += f" - {os.path.basename(self.file_path)}"
        if self.is_dirty:
            title += " *"
        self.setWindowTitle(title)

    def _show_error(self, message: str) -> None:
        QMessageBox.critical(self, "오류", message)
        self._log(f"[오류] {message}")

    def _confirm(self, message: str) -> bool:
        return QMessageBox.question(self, "확인", message) == QMessageBox.Yes

    def open_csv(self) -> None:
        if self.is_dirty and not self._confirm("저장되지 않은 변경사항이 있습니다. 계속 진행할까요?"):
            return

        path, _ = QFileDialog.getOpenFileName(
            self,
            "CSV 파일 열기",
            self.last_open_dir,
            "CSV Files (*.csv);;All Files (*)",
        )
        if not path:
            return

        self.last_open_dir = os.path.dirname(path)
        self.load_csv(path)

    def reload_csv(self) -> None:
        if not self.file_path:
            return
        if self.is_dirty and not self._confirm("저장되지 않은 변경사항이 있습니다. 파일을 다시 읽으면 사라집니다. 계속할까요?"):
            return
        self.load_csv(self.file_path)

    def load_csv(self, path: str) -> None:
        try:
            with open(path, "r", encoding=DEFAULT_ENCODING, newline="") as f:
                reader = csv.reader(f)
                rows = list(reader)
        except UnicodeDecodeError:
            try:
                with open(path, "r", encoding="cp949", newline="") as f:
                    reader = csv.reader(f)
                    rows = list(reader)
            except Exception as e:
                self._show_error(f"CSV 읽기 실패:\n{e}")
                return
        except Exception as e:
            self._show_error(f"CSV 읽기 실패:\n{e}")
            return

        if not rows:
            self._show_error("비어 있는 CSV 파일입니다.")
            return

        self.headers = rows[0]
        self.rows = rows[1:]
        self.file_path = path
        self.is_dirty = False
        self.selection_state = None

        try:
            self._detect_columns()
        except ValueError as e:
            self._show_error(str(e))
            return

        self._populate_table()
        self._jump_to_first_blank_kr()
        self.file_label.setText(f"파일: {path}")
        self.translation_edit.clear()
        self.validation_label.setText("검증 결과: -")
        self._update_selection_label([])
        self._update_ui_state()
        self._log("CSV 파일을 불러왔습니다.")

    def _detect_columns(self) -> None:
        lower_headers = [h.strip().lower() for h in self.headers]
        self.jp_col = lower_headers.index("jp") if "jp" in lower_headers else None
        self.en_col = lower_headers.index("en") if "en" in lower_headers else None
        self.kr_col = lower_headers.index("kr") if "kr" in lower_headers else None

        if self.jp_col is None or self.kr_col is None:
            raise ValueError("필수 열(jp, kr)을 찾을 수 없습니다.")

    def _populate_table(self) -> None:
        self.table.clear()
        self.table.setRowCount(len(self.rows))
        self.table.setColumnCount(len(self.headers))
        self.table.setHorizontalHeaderLabels(self.headers)

        for row_idx, row in enumerate(self.rows):
            if len(row) < len(self.headers):
                row = row + [""] * (len(self.headers) - len(row))
                self.rows[row_idx] = row
            for col_idx, value in enumerate(row):
                item = QTableWidgetItem(value)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                if col_idx == self.kr_col and value.strip() == "":
                    item.setBackground(Qt.yellow)
                self.table.setItem(row_idx, col_idx, item)
            self.table.setVerticalHeaderItem(row_idx, QTableWidgetItem(str(row_idx + 2)))

        self.table.resizeColumnsToContents()
        if self.jp_col is not None:
            self.table.setColumnWidth(self.jp_col, 420)
        if self.en_col is not None:
            self.table.setColumnWidth(self.en_col, 420)
        if self.kr_col is not None:
            self.table.setColumnWidth(self.kr_col, 420)

    def _update_kr_cell_style(self, row_idx: int) -> None:
        if self.kr_col is None:
            return
        item = self.table.item(row_idx, self.kr_col)
        if item is None:
            return
        if item.text().strip() == "":
            item.setBackground(Qt.yellow)
        else:
            item.setBackground(Qt.white)

    def _find_first_blank_kr_row(self) -> Optional[int]:
        if self.kr_col is None:
            return None
        for row_idx, row in enumerate(self.rows):
            if self.kr_col >= len(row) or row[self.kr_col].strip() == "":
                return row_idx
        return None

    def _jump_to_first_blank_kr(self) -> None:
        row_idx = self._find_first_blank_kr_row()
        if row_idx is None:
            self.next_blank_label.setText("다음 미번역 행: 없음")
            return

        visible_row_number = row_idx + 2
        self.next_blank_label.setText(f"다음 미번역 행: {visible_row_number}행")

        target_item = self.table.item(row_idx, self.kr_col if self.kr_col is not None else 0)
        if target_item is not None:
            self.table.scrollToItem(target_item, QAbstractItemView.PositionAtCenter)
        self.table.clearSelection()
        self.table.selectRow(row_idx)

    def _on_selection_changed(self) -> None:
        row_indices = self._get_selected_row_indices()
        self._update_selection_label(row_indices)
        self._update_ui_state()

    def _update_selection_label(self, row_indices: List[int]) -> None:
        if not row_indices:
            self.selection_label.setText("현재 선택: -")
            return

        first = row_indices[0] + 2
        last = row_indices[-1] + 2
        count = len(row_indices)
        self.selection_label.setText(f"현재 선택: {first}행 ~ {last}행 / 총 {count}개")

    def _get_selected_row_indices(self) -> List[int]:
        indexes = self.table.selectedIndexes()
        if not indexes:
            return []
        rows = sorted({index.row() for index in indexes})
        return rows

    def _csv_escape_cell_always_quoted(self, value: str) -> str:
        output = io.StringIO()
        writer = csv.writer(output, lineterminator="", quoting=csv.QUOTE_ALL)
        writer.writerow([value])
        line = output.getvalue()
        if line.endswith("\r\n"):
            line = line[:-2]
        elif line.endswith("\n"):
            line = line[:-1]
        return line

    def build_copy_text_from_selection(self, selected_indexes) -> str:
        if not selected_indexes:
            return ""

        row_to_cols = {}
        for index in selected_indexes:
            row_to_cols.setdefault(index.row(), set()).add(index.column())

        lines: List[str] = []
        for row_idx in sorted(row_to_cols.keys()):
            cols = sorted(row_to_cols[row_idx])
            row_values: List[str] = []
            for col_idx in cols:
                row = self.rows[row_idx]
                value = row[col_idx] if col_idx < len(row) else ""
                row_values.append(self._csv_escape_cell_always_quoted(value))
            lines.append(",".join(row_values))

        return "\n".join(lines)

    def copy_selected_cells(self) -> None:
        if self.file_path is None:
            self._show_error("먼저 CSV 파일을 열어주세요.")
            return

        selected_indexes = self.table.selectedIndexes()
        if not selected_indexes:
            self._show_error("먼저 복사할 셀/행을 선택해주세요.")
            return

        row_indices = self._get_selected_row_indices()
        copied_text = self.build_copy_text_from_selection(selected_indexes)
        QGuiApplication.clipboard().setText(copied_text)
        self.selection_state = SelectionState(
            row_indices=row_indices,
            copied_text=copied_text,
        )
        self._update_ui_state()

        first = row_indices[0] + 2
        last = row_indices[-1] + 2
        self._log(f"복사됨: {first}행 ~ {last}행 / {len(row_indices)}개")

    def _parse_translation_cells(self, text: str) -> List[str]:
        if not text.strip():
            return []

        try:
            reader = csv.reader(io.StringIO(text), skipinitialspace=False)
            parsed_rows = list(reader)
        except Exception as e:
            raise ValueError(f"CSV 형식 파싱 실패: {e}") from e

        cells: List[str] = []
        for row in parsed_rows:
            for cell in row:
                cells.append(cell)
        return cells

    def validate_translation_input(self) -> bool:
        if self.selection_state is None:
            self._show_error("먼저 원문 셀을 복사해서 작업 대상을 지정해주세요.")
            return False

        text = self.translation_edit.toPlainText()
        try:
            parsed_cells = self._parse_translation_cells(text)
        except ValueError as e:
            self.validation_label.setText(f"검증 결과: 실패 - {e}")
            self._show_error(str(e))
            return False

        expected = len(self.selection_state.row_indices)
        actual = len(parsed_cells)
        if actual != expected:
            self.validation_label.setText(
                f"검증 결과: 실패 - 기대 셀 수 {expected}개 / 입력된 셀 수 {actual}개"
            )
            self._show_error(
                f"셀 수가 맞지 않습니다.\n기대: {expected}개\n입력: {actual}개\n\n"
                "멀티라인 셀의 큰따옴표가 유지되었는지 확인해주세요."
            )
            return False

        self.validation_label.setText(
            f"검증 결과: 성공 - 기대 셀 수 {expected}개 / 입력된 셀 수 {actual}개"
        )
        self._log(f"검증 성공: 기대 {expected}개 / 입력 {actual}개")
        return True

    def apply_translation_to_kr(self) -> None:
        if self.file_path is None:
            self._show_error("먼저 CSV 파일을 열어주세요.")
            return
        if self.selection_state is None:
            self._show_error("먼저 원문 셀을 복사해서 작업 대상을 지정해주세요.")
            return
        if self.kr_col is None:
            self._show_error("kr 열을 찾을 수 없습니다.")
            return
        if not self.validate_translation_input():
            return

        parsed_cells = self._parse_translation_cells(self.translation_edit.toPlainText())
        target_rows = self.selection_state.row_indices

        filled_rows = []
        for row_idx in target_rows:
            current = self.rows[row_idx][self.kr_col] if self.kr_col < len(self.rows[row_idx]) else ""
            if current.strip() != "":
                filled_rows.append(row_idx + 2)

        if filled_rows:
            sample = ", ".join(map(str, filled_rows[:10]))
            if len(filled_rows) > 10:
                sample += ", ..."
            proceed = self._confirm(
                f"이미 kr 값이 들어있는 행이 있습니다.\n행 번호: {sample}\n\n덮어쓸까요?"
            )
            if not proceed:
                return

        for row_idx, translated in zip(target_rows, parsed_cells):
            self.rows[row_idx][self.kr_col] = translated
            item = self.table.item(row_idx, self.kr_col)
            if item is None:
                item = QTableWidgetItem(translated)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.table.setItem(row_idx, self.kr_col, item)
            else:
                item.setText(translated)
            self._update_kr_cell_style(row_idx)

        self.is_dirty = True
        self._update_ui_state()
        self._refresh_next_blank_label_only()
        first = target_rows[0] + 2
        last = target_rows[-1] + 2
        self._log(f"kr 반영 완료: {first}행 ~ {last}행 / {len(target_rows)}개")

    def _refresh_next_blank_label_only(self) -> None:
        row_idx = self._find_first_blank_kr_row()
        if row_idx is None:
            self.next_blank_label.setText("다음 미번역 행: 없음")
        else:
            self.next_blank_label.setText(f"다음 미번역 행: {row_idx + 2}행")

    def save_csv(self) -> None:
        if self.file_path is None:
            self._show_error("저장할 CSV 파일이 없습니다.")
            return

        backup_path = self.file_path + ".bak"
        try:
            if os.path.exists(self.file_path):
                try:
                    with open(self.file_path, "r", encoding=DEFAULT_ENCODING, newline="") as src:
                        original_text = src.read()
                except UnicodeDecodeError:
                    with open(self.file_path, "r", encoding="cp949", newline="") as src:
                        original_text = src.read()
                with open(backup_path, "w", encoding=DEFAULT_ENCODING, newline="") as bak:
                    bak.write(original_text)

            with open(self.file_path, "w", encoding=DEFAULT_ENCODING, newline="") as f:
                writer = csv.writer(f)
                writer.writerow(self.headers)
                writer.writerows(self.rows)
        except Exception as e:
            self._show_error(f"CSV 저장 실패:\n{e}")
            return

        self.is_dirty = False
        self._update_ui_state()
        self._log(f"저장 완료: {self.file_path}")

    def closeEvent(self, event) -> None:  # type: ignore[override]
        if self.is_dirty:
            reply = QMessageBox.question(
                self,
                "저장되지 않은 변경사항",
                "저장되지 않은 변경사항이 있습니다. 종료하시겠습니까?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                event.ignore()
                return
        event.accept()


def main() -> None:
    app = QApplication(sys.argv)
    win = CsvTranslationHelper()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
