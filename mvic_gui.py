#!/usr/bin/env python3
"""
mvic_gui.py

Minimal, safe GUI to run mvic_cli.py with only the public flags:
  --nu, --fthscale, --fthtime, --save-csv, --animate

Dependencies: PyQt5
Install: pip install PyQt5
"""

import sys
import os
from PyQt5 import QtCore, QtGui, QtWidgets

class ProcessRunner(QtCore.QObject):
    outputReady = QtCore.pyqtSignal(str)
    finished = QtCore.pyqtSignal(int, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._proc = QtCore.QProcess(self)
        self._proc.readyReadStandardOutput.connect(self._on_stdout)
        self._proc.readyReadStandardError.connect(self._on_stderr)
        self._proc.finished.connect(self._on_finished)

    def start(self, python_exe, script_path, args):
        if self._proc.state() != QtCore.QProcess.NotRunning:
            self.outputReady.emit("Process already running\n")
            return
        program = python_exe
        arguments = [script_path] + args
        self.outputReady.emit(f"Starting: {program} {' '.join(arguments)}\n\n")
        self._proc.start(program, arguments)

    def terminate(self):
        if self._proc.state() != QtCore.QProcess.NotRunning:
            self._proc.terminate()
            QtCore.QTimer.singleShot(1500, self._force_kill)

    def _force_kill(self):
        if self._proc.state() != QtCore.QProcess.NotRunning:
            self._proc.kill()

    def _on_stdout(self):
        data = bytes(self._proc.readAllStandardOutput()).decode('utf-8', errors='replace')
        self.outputReady.emit(data)

    def _on_stderr(self):
        data = bytes(self._proc.readAllStandardError()).decode('utf-8', errors='replace')
        self.outputReady.emit(data)

    def _on_finished(self, exitCode, exitStatus):
        status_text = "NormalExit" if exitStatus == QtCore.QProcess.NormalExit else "Crash/Terminated"
        self.finished.emit(exitCode, status_text)

class MVICGui(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MVIC CLI Runner")
        self.setMinimumSize(820, 600)
        self._runner = ProcessRunner(self)
        self._runner.outputReady.connect(self.append_console)
        self._runner.finished.connect(self.on_finished)
        self._build_ui()

    def _build_ui(self):
        main = QtWidgets.QVBoxLayout(self)

        # Parameters group (only public flags)
        params = QtWidgets.QGroupBox("Simulation Parameters (public flags)")
        grid = QtWidgets.QGridLayout()
        params.setLayout(grid)

        # --nu
        grid.addWidget(QtWidgets.QLabel("Number of motor units (--nu):"), 0, 0)
        self.spin_nu = QtWidgets.QSpinBox()
        self.spin_nu.setRange(1, 10000)
        self.spin_nu.setValue(200)
        self.spin_nu.setSingleStep(1)
        grid.addWidget(self.spin_nu, 0, 1)

        # --fthscale
        # --fthscale as percentage
        grid.addWidget(QtWidgets.QLabel("Target % MVIC (--fthscale):"), 1, 0)
        self.spin_fth_percent = QtWidgets.QSpinBox()
        self.spin_fth_percent.setRange(1, 100)
        self.spin_fth_percent.setValue(80)   # default 80%
        grid.addWidget(self.spin_fth_percent, 1, 1)


        # --fthtime
        grid.addWidget(QtWidgets.QLabel("Task duration seconds (--fthtime):"), 2, 0)
        self.spin_time = QtWidgets.QDoubleSpinBox()
        self.spin_time.setRange(0.1, 86400.0)
        self.spin_time.setSingleStep(1.0)
        self.spin_time.setValue(20.0)
        grid.addWidget(self.spin_time, 2, 1)

        # Booleans
        self.chk_save = QtWidgets.QCheckBox("Save CSV (--save-csv)")
        self.chk_animate = QtWidgets.QCheckBox("Enable animation (--animate)")
        bool_layout = QtWidgets.QHBoxLayout()
        bool_layout.addWidget(self.chk_save)
        bool_layout.addWidget(self.chk_animate)
        grid.addLayout(bool_layout, 3, 0, 1, 2)

        # Script selection (default mvic_cli.py)
        script_layout = QtWidgets.QHBoxLayout()
        self.script_path_edit = QtWidgets.QLineEdit(os.path.join(os.getcwd(), "mvic_cli.py"))
        self.script_path_edit.setToolTip("Path to mvic_cli.py (default: current directory)")
        btn_browse = QtWidgets.QPushButton("Browse")
        btn_browse.clicked.connect(self._browse_script)
        script_layout.addWidget(QtWidgets.QLabel("Script:"))
        script_layout.addWidget(self.script_path_edit)
        script_layout.addWidget(btn_browse)

        # Command preview
        self.cmd_preview = QtWidgets.QLineEdit()
        self.cmd_preview.setReadOnly(True)
        self._update_cmd_preview()

        # Buttons
        btn_row = QtWidgets.QHBoxLayout()
        self.btn_run = QtWidgets.QPushButton("Run Simulation")
        self.btn_cancel = QtWidgets.QPushButton("Cancel")
        self.btn_cancel.setEnabled(False)
        btn_row.addWidget(self.btn_run)
        btn_row.addWidget(self.btn_cancel)

        # Status label
        self.status_label = QtWidgets.QLabel("Ready")
        self.status_label.setStyleSheet("color: green; font-weight: bold;")

        # Console
        console_group = QtWidgets.QGroupBox("Console Output")
        console_layout = QtWidgets.QVBoxLayout()
        console_group.setLayout(console_layout)
        self.console = QtWidgets.QPlainTextEdit()
        self.console.setReadOnly(True)
        self.console.setMaximumBlockCount(20000)
        console_layout.addWidget(self.console)

        # Assemble
        main.addWidget(params)
        main.addLayout(script_layout)
        main.addWidget(QtWidgets.QLabel("Command preview"))
        main.addWidget(self.cmd_preview)
        main.addLayout(btn_row)
        main.addWidget(self.status_label)
        main.addWidget(console_group, stretch=1)

        # Signals
        self.btn_run.clicked.connect(self.on_run)
        self.btn_cancel.clicked.connect(self.on_cancel)
        self.spin_nu.valueChanged.connect(self._update_cmd_preview)
        self.spin_fth_percent.valueChanged.connect(self._update_cmd_preview)
        self.spin_time.valueChanged.connect(self._update_cmd_preview)
        self.chk_save.stateChanged.connect(self._update_cmd_preview)
        self.chk_animate.stateChanged.connect(self._update_cmd_preview)
        self.script_path_edit.textChanged.connect(self._update_cmd_preview)

    def _browse_script(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select mvic_cli.py", os.getcwd(), "Python files (*.py)")
        if path:
            self.script_path_edit.setText(path)

    def _build_args(self):
        nu = int(self.spin_nu.value())
        fth_percent = int(self.spin_fth_percent.value())
        fth_fraction = fth_percent / 100.0

        fthtime = float(self.spin_time.value())

        args = [
            "--nu", str(nu),
            "--fthscale", str(fth_fraction),
            "--fthtime", str(fthtime),
        ]

        if self.chk_save.isChecked():
            args.append("--save-csv")
        if self.chk_animate.isChecked():
            args.append("--animate")

        return args


    def _update_cmd_preview(self):
        script = self.script_path_edit.text().strip() or "mvic_cli.py"
        args = self._build_args()
        preview = f"{sys.executable} {script} " + " ".join(args)
        self.cmd_preview.setText(preview)

    def append_console(self, text):
        self.console.moveCursor(QtGui.QTextCursor.End)
        self.console.insertPlainText(text)
        self.console.moveCursor(QtGui.QTextCursor.End)

    def on_run(self):
        script = self.script_path_edit.text().strip() or os.path.join(os.getcwd(), "mvic_cli.py")
        if not os.path.isfile(script):
            QtWidgets.QMessageBox.warning(self, "Script not found", "mvic_cli.py not found. Please select the script.")
            return

        args = self._build_args()
        python_exe = sys.executable

        # UI state
        self.console.clear()
        self.status_label.setText("Running...")
        self.status_label.setStyleSheet("color: orange; font-weight: bold;")
        self.btn_run.setEnabled(False)
        self.btn_cancel.setEnabled(True)

        self._runner.start(python_exe, script, args)

    def on_cancel(self):
        self.append_console("\nTerminating process...\n")
        self._runner.terminate()
        self.btn_cancel.setEnabled(False)

    def on_finished(self, exitCode, status_text):
        self.append_console(f"\nProcess finished: exitCode={exitCode}, status={status_text}\n")
        self.status_label.setText("Ready")
        self.status_label.setStyleSheet("color: green; font-weight: bold;")
        self.btn_run.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self._update_cmd_preview()

def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    w = MVICGui()
    w.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
