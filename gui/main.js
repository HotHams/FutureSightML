const { app, BrowserWindow, shell } = require('electron');
const { spawn } = require('child_process');
const path = require('path');
const http = require('http');

const PORT = 8765;
const API_URL = `http://127.0.0.1:${PORT}`;

let mainWindow;
let serverProcess;

function getIconPath() {
    if (process.platform === 'win32') {
        return path.join(__dirname, 'static', 'icon.ico');
    }
    // macOS and Linux use PNG (electron-builder handles .icns conversion for mac .app bundle)
    return path.join(__dirname, 'static', 'icon.png');
}

function createWindow() {
    mainWindow = new BrowserWindow({
        width: 1400,
        height: 900,
        minWidth: 1000,
        minHeight: 700,
        title: 'FutureSightML',
        backgroundColor: '#0c0c1e',
        icon: getIconPath(),
        webPreferences: {
            nodeIntegration: false,
            contextIsolation: true,
        },
        autoHideMenuBar: true,
    });

    // Handle external links
    mainWindow.webContents.setWindowOpenHandler(({ url }) => {
        shell.openExternal(url);
        return { action: 'deny' };
    });

    mainWindow.on('closed', () => { mainWindow = null; });
}

function startServer() {
    const isDev = !app.isPackaged;
    let projectRoot;
    let cmd;
    let args;

    if (isDev) {
        projectRoot = path.join(__dirname, '..');
        cmd = 'python';
        args = [
            path.join(projectRoot, 'scripts', 'run_server.py'),
            '--port', String(PORT),
            '--no-browser',
        ];
    } else {
        // Packaged mode: launch PyInstaller-bundled exe
        projectRoot = path.join(process.resourcesPath, 'backend');
        const exeName = process.platform === 'win32' ? 'run_server.exe' : 'run_server';
        cmd = path.join(projectRoot, 'dist', 'run_server', exeName);
        args = ['--port', String(PORT), '--no-browser'];
    }

    console.log(`Starting server: ${cmd} ${args.join(' ')}`);

    serverProcess = spawn(cmd, args, {
        cwd: projectRoot,
        env: { ...process.env, PYTHONPATH: projectRoot },
    });

    serverProcess.stdout.on('data', (data) => console.log(`[Server] ${data}`));
    serverProcess.stderr.on('data', (data) => console.error(`[Server] ${data}`));
    serverProcess.on('close', (code) => console.log(`Server exited with code ${code}`));
}

function waitForServer(retries = 60) {
    return new Promise((resolve, reject) => {
        function check() {
            http.get(API_URL + '/api/formats', (res) => {
                resolve();
            }).on('error', () => {
                if (retries <= 0) {
                    reject(new Error('Server did not start'));
                    return;
                }
                retries--;
                setTimeout(check, 1000);
            });
        }
        check();
    });
}

app.whenReady().then(async () => {
    createWindow();

    // Retro pixel-art loading screen
    mainWindow.loadURL(`data:text/html,
        <html style="background:#0c0c1e;color:white;font-family:'Courier New',monospace;display:flex;align-items:center;justify-content:center;height:100vh;margin:0;image-rendering:pixelated">
            <div style="text-align:center">
                <pre style="font-size:10px;line-height:1.1;color:#9b59b6;margin-bottom:24px">
    @@@@@@@@@@@@
  @@%%%%%%%%%%%%@@
 @%%%%%%%%%%%%%%%@
 @%% M  M M M %%@
@################@
@## [========] ##@
@################@
 @...............@
 @...............@
  @@...........@@
    @@@@@@@@@@@@
                </pre>
                <h1 style="font-size:28px;letter-spacing:2px;margin:0">
                    FutureSight<span style="color:#e63946">ML</span>
                </h1>
                <p style="color:#888;margin-top:16px;font-size:14px">Loading models<span id="dots"></span></p>
                <p style="color:#e63946;font-size:18px;margin-top:12px;animation:blink 1s step-end infinite" id="cursor">&#9654; INITIALIZING...</p>
            </div>
            <style>
                @keyframes blink { 50% { opacity: 0; } }
            </style>
            <script>
                let d = 0;
                setInterval(() => { d = (d + 1) % 4; document.getElementById('dots').textContent = '.'.repeat(d); }, 400);
            </script>
        </html>
    `);

    startServer();

    try {
        await waitForServer();
        mainWindow.loadURL(API_URL);
    } catch (e) {
        mainWindow.loadURL(`data:text/html,
            <html style="background:#0c0c1e;color:white;font-family:'Courier New',monospace;display:flex;align-items:center;justify-content:center;height:100vh;margin:0">
                <div style="text-align:center">
                    <h1 style="color:#e63946;font-size:24px">SERVER ERROR</h1>
                    <p style="color:#888;margin-top:12px">Could not start the backend server.</p>
                    <p style="color:#666;font-size:12px;margin-top:8px">Check that all dependencies are installed.</p>
                    <p style="color:#e63946;font-size:16px;margin-top:16px;animation:blink 1s step-end infinite">&#9654; PRESS ALT+F4 TO EXIT</p>
                </div>
                <style>@keyframes blink { 50% { opacity: 0; } }</style>
            </html>
        `);
    }
});

app.on('window-all-closed', () => {
    if (serverProcess) {
        serverProcess.kill();
    }
    app.quit();
});

app.on('before-quit', () => {
    if (serverProcess) {
        serverProcess.kill();
    }
});
