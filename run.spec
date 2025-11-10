# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['run.py'],
    pathex=[],
    binaries=[],
    datas=[ 
        ('gy.py', '.'),
        ('C:/Users/28646/anaconda3/envs/StudentScoreAnalyzer/Lib/site-packages/streamlit/static', './streamlit/static'),
        ('C:/Users/28646/anaconda3/envs/StudentScoreAnalyzer/Lib/site-packages/streamlit/runtime', './streamlit/runtime'),
        ('C:/Users/28646/anaconda3/envs/StudentScoreAnalyzer/Lib/site-packages/streamlit_sortables', './streamlit_sortables'),
    ],
    hiddenimports=[],
    hookspath=['./hooks'],
    hooksconfig={},
    runtime_hooks=['hooks/close_on_console_exit.py'],
    excludes=[],
    noarchive=False,
    optimize=0,
    icons=,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='run',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
