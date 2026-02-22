#!/usr/bin/env python3
"""
Import Validator Script
Checks for unused and missing imports in Python files
"""

import ast
import os
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

class ImportChecker:
    """Check for missing and unused imports in Python files"""
    
    # Common PySide6 widgets and their modules
    PYSIDE6_WIDGETS = {
        'QtWidgets': [
            'QWidget', 'QMainWindow', 'QApplication', 'QLabel', 'QPushButton',
            'QVBoxLayout', 'QHBoxLayout', 'QGridLayout', 'QFormLayout',
            'QFrame', 'QScrollArea', 'QSplitter', 'QTabWidget',
            'QComboBox', 'QCheckBox', 'QRadioButton', 'QLineEdit', 'QTextEdit',
            'QSpinBox', 'QDoubleSpinBox', 'QSlider', 'QProgressBar',
            'QProgressDialog', 'QFileDialog', 'QMessageBox', 'QInputDialog',
            'QGroupBox', 'QMenuBar', 'QMenu', 'QAction', 'QToolBar',
            'QStatusBar', 'QDockWidget', 'QTableWidget', 'QTreeWidget',
            'QListWidget', 'QGraphicsView', 'QGraphicsScene',
        ],
        'QtCore': [
            'Qt', 'QTimer', 'Signal', 'Slot', 'QThread', 'QObject',
            'QSize', 'QPoint', 'QRect', 'QUrl', 'QEvent',
        ],
        'QtGui': [
            'QImage', 'QPixmap', 'QIcon', 'QFont', 'QColor', 'QBrush',
            'QPainter', 'QPen', 'QKeySequence', 'QAction',
        ],
    }
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.tree = None
        self.imports: Dict[str, Set[str]] = {}
        self.used_names: Set[str] = set()
        
    def parse_file(self) -> bool:
        """Parse the Python file into AST"""
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            self.tree = ast.parse(content, filename=self.file_path)
            return True
        except SyntaxError as e:
            print(f"❌ Syntax error in {self.file_path}: {e}")
            return False
        except Exception as e:
            print(f"❌ Error reading {self.file_path}: {e}")
            return False
    
    def extract_imports(self):
        """Extract all imports from the file"""
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module_name = alias.asname or alias.name
                    if module_name not in self.imports:
                        self.imports[module_name] = set()
            
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                for alias in node.names:
                    name = alias.asname or alias.name
                    self.imports[name] = {module}
    
    def extract_used_names(self):
        """Extract all used names in the file"""
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Name):
                self.used_names.add(node.id)
            elif isinstance(node, ast.Attribute):
                # Handle things like Qt.AlignCenter
                if isinstance(node.value, ast.Name):
                    self.used_names.add(node.value.id)
    
    def find_unused_imports(self) -> List[str]:
        """Find imported names that are never used"""
        unused = []
        for name in self.imports:
            if name not in self.used_names:
                # Skip __all__, __version__, etc.
                if not name.startswith('__'):
                    unused.append(name)
        return sorted(unused)
    
    def check(self) -> Tuple[List[str], bool]:
        """
        Run the check
        Returns: (list of issues, has_critical_errors)
        """
        if not self.parse_file():
            return [f"Failed to parse {self.file_path}"], True
        
        self.extract_imports()
        self.extract_used_names()
        
        issues = []
        has_critical = False
        
        # Check for unused imports
        unused = self.find_unused_imports()
        if unused:
            issues.append(f"⚠️  Unused imports: {', '.join(unused)}")
        
        # Check for common missing imports (basic check)
        # This is not exhaustive but catches obvious issues
        for name in self.used_names:
            if name not in self.imports and name.startswith('Q'):
                # Likely a Qt class
                issues.append(f"❌ Potentially missing import: {name}")
                has_critical = True
        
        return issues, has_critical


def check_directory(directory: str) -> Dict[str, List[str]]:
    """Check all Python files in a directory"""
    results = {}
    path = Path(directory)
    
    python_files = list(path.glob('**/*.py'))
    print(f"\n🔍 Checking {len(python_files)} Python files in {directory}\n")
    
    for py_file in python_files:
        if '__pycache__' in str(py_file) or 'venv' in str(py_file):
            continue
        
        checker = ImportChecker(str(py_file))
        issues, critical = checker.check()
        
        if issues:
            results[str(py_file)] = issues
            status = "❌" if critical else "⚠️ "
            print(f"{status} {py_file.name}")
            for issue in issues:
                print(f"   {issue}")
        else:
            print(f"✅ {py_file.name}")
    
    return results


def main():
    """Main entry point"""
    if len(sys.argv) > 1:
        target = sys.argv[1]
    else:
        # Default to desktop_app directory
        target = os.path.join(os.path.dirname(__file__), '..', 'desktop_app')
    
    if not os.path.exists(target):
        print(f"❌ Path does not exist: {target}")
        sys.exit(1)
    
    if os.path.isfile(target):
        checker = ImportChecker(target)
        issues, critical = checker.check()
        if issues:
            for issue in issues:
                print(issue)
            sys.exit(1 if critical else 0)
        else:
            print(f"✅ No issues found in {target}")
            sys.exit(0)
    else:
        results = check_directory(target)
        
        print("\n" + "="*60)
        if results:
            print(f"⚠️  Found issues in {len(results)} file(s)")
            sys.exit(1)
        else:
            print("✅ All files passed import validation!")
            sys.exit(0)


if __name__ == '__main__':
    main()
