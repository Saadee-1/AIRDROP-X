# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for AIRDROP-X Qt desktop app.
# Build: pyinstaller --noconfirm --clean airdrop_x_qt.spec
# Output: dist/AIRDROP-X/AIRDROP-X.exe (one-folder bundle)

block_cipher = None

# Hidden imports for PyQt6 and Matplotlib when frozen
hidden_imports = [
    'PyQt6.QtCore',
    'PyQt6.QtGui',
    'PyQt6.QtWidgets',
    'matplotlib',
    'matplotlib.backends.backend_qtagg',
    'matplotlib.figure',
    'numpy',
]

a = Analysis(
    ['qt_app.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='AIRDROP-X',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # windowed (no console on Windows)
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='AIRDROP-X',
)
