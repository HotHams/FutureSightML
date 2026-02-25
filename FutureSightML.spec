# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for FutureSightML backend server.

Bundles scripts/run_server.py + all showdown package dependencies
into a one-dir distribution at dist/run_server/.

Usage:
    pyinstaller FutureSightML.spec
"""

import sys
import os
from pathlib import Path

block_cipher = None
project_root = Path(SPECPATH)

# Find xgboost native files (DLL + VERSION)
def find_xgboost_files():
    binaries = []
    datas = []
    try:
        import xgboost
        xgb_dir = Path(xgboost.__file__).parent
        # Native DLL
        dll = xgb_dir / 'lib' / 'xgboost.dll'
        if dll.exists():
            binaries.append((str(dll), 'xgboost/lib'))
        # VERSION file (required at runtime)
        version_file = xgb_dir / 'VERSION'
        if version_file.exists():
            datas.append((str(version_file), 'xgboost'))
    except ImportError:
        pass
    return binaries, datas

_xgb_binaries, _xgb_datas = find_xgboost_files()

a = Analysis(
    [str(project_root / 'scripts' / 'run_server.py')],
    pathex=[str(project_root)],
    binaries=_xgb_binaries,
    datas=[
        # Config
        (str(project_root / 'config.yaml'), '.'),
        # GUI static files (served by FastAPI)
        (str(project_root / 'gui' / 'static'), 'gui/static'),
        # Pre-exported pool data
        (str(project_root / 'data' / 'pools'), 'data/pools'),
        # Model checkpoints
        (str(project_root / 'data' / 'checkpoints'), 'data/checkpoints'),
    ] + _xgb_datas,
    hiddenimports=[
        # Uvicorn internals
        'uvicorn',
        'uvicorn.logging',
        'uvicorn.loops',
        'uvicorn.loops.auto',
        'uvicorn.loops.asyncio',
        'uvicorn.protocols',
        'uvicorn.protocols.http',
        'uvicorn.protocols.http.auto',
        'uvicorn.protocols.http.h11_impl',
        'uvicorn.protocols.http.httptools_impl',
        'uvicorn.protocols.websockets',
        'uvicorn.protocols.websockets.auto',
        'uvicorn.protocols.websockets.wsproto_impl',
        'uvicorn.lifespan',
        'uvicorn.lifespan.on',
        'uvicorn.lifespan.off',
        # FastAPI / Starlette
        'fastapi',
        'starlette',
        'starlette.routing',
        'starlette.responses',
        'starlette.middleware',
        'starlette.middleware.cors',
        'starlette.staticfiles',
        'anyio._backends._asyncio',
        # Showdown package
        'showdown',
        'showdown.api',
        'showdown.api.server',
        'showdown.api.state',
        'showdown.config',
        'showdown.data',
        'showdown.data.database',
        'showdown.data.features',
        'showdown.data.pokemon_data',
        'showdown.data.preprocessor',
        'showdown.data.schema',
        'showdown.data.damage_calc',
        'showdown.data.mechanics',
        'showdown.models',
        'showdown.models.embeddings',
        'showdown.models.win_predictor',
        'showdown.models.xgb_predictor',
        'showdown.models.ensemble',
        'showdown.models.trainer',
        'showdown.scraper',
        'showdown.scraper.replay_parser',
        'showdown.scraper.replay_scraper',
        'showdown.scraper.stats_scraper',
        'showdown.teambuilder',
        'showdown.teambuilder.constraints',
        'showdown.teambuilder.evaluator',
        'showdown.teambuilder.genetic',
        'showdown.teambuilder.meta_analysis',
        'showdown.teambuilder.analysis',
        'showdown.teambuilder.spread_inference',
        'showdown.simulator',
        'showdown.simulator.battle_sim',
        'showdown.utils',
        'showdown.utils.constants',
        'showdown.utils.logging_config',
        # ML libraries
        'torch',
        'torch.nn',
        'torch.nn.functional',
        'xgboost',
        'xgboost.core',
        'xgboost.tracker',
        'xgboost.training',
        'xgboost.compat',
        'sklearn',
        'sklearn.utils',
        'sklearn.utils._typedefs',
        'sklearn.utils._heap',
        'sklearn.utils._sorting',
        'sklearn.utils._vector_sentinel',
        'sklearn.neighbors._partition_nodes',
        # Async DB
        'aiosqlite',
        # Data libs
        'numpy',
        'yaml',
        'pydantic',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude heavy unused packages
        'matplotlib',
        'tkinter',
        'PIL',
        'IPython',
        'jupyter',
        'notebook',
        'sphinx',
        'pytest',
    ],
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
    name='run_server',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,  # Keep console for server stdout
    disable_windowed_traceback=False,
    argv_emulation=False,
    icon=str(project_root / 'gui' / 'static' / 'icon.ico'),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='run_server',
)
