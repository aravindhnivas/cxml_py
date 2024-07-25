# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['/Users/aravindhnivas/Documents/GitHub/umdapy/src/main.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=['umdalib'],
    hookspath=['/Users/aravindhnivas/Documents/GitHub/umdapy/src/hooks'],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=True,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='umdapy',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['/Users/aravindhnivas/Documents/GitHub/umdapy/src/icons/icon.ico'],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='umdapy',
)
