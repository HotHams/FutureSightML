const { app, BrowserWindow, shell } = require('electron');
const { spawn } = require('child_process');
const path = require('path');
const http = require('http');

const PORT = 8765;
const API_URL = `http://127.0.0.1:${PORT}`;

let mainWindow;
let serverProcess;

function createWindow() {
    mainWindow = new BrowserWindow({
        width: 1400,
        height: 900,
        minWidth: 1000,
        minHeight: 700,
        title: 'FutureSightML',
        backgroundColor: '#0f0f23',
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
    // Determine paths
    const isDev = !app.isPackaged;
    let projectRoot;
    let pythonCmd;

    if (isDev) {
        projectRoot = path.join(__dirname, '..');
        pythonCmd = 'python';
    } else {
        projectRoot = path.join(process.resourcesPath, 'backend');
        pythonCmd = 'python';
    }

    const scriptPath = path.join(projectRoot, 'scripts', 'run_server.py');

    console.log(`Starting server: ${pythonCmd} ${scriptPath} --port ${PORT} --no-browser`);

    serverProcess = spawn(pythonCmd, [scriptPath, '--port', String(PORT), '--no-browser'], {
        cwd: projectRoot,
        env: { ...process.env, PYTHONPATH: projectRoot },
    });

    serverProcess.stdout.on('data', (data) => console.log(`[Server] ${data}`));
    serverProcess.stderr.on('data', (data) => console.error(`[Server] ${data}`));
    serverProcess.on('close', (code) => console.log(`Server exited with code ${code}`));
}

function waitForServer(retries = 30) {
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

    // Show loading screen
    mainWindow.loadURL(`data:text/html,
        <html style="background:#0f0f23;color:white;font-family:sans-serif;display:flex;align-items:center;justify-content:center;height:100vh;margin:0">
            <div style="text-align:center">
                <h1 style="font-family:monospace">FutureSight<span style="color:#e63946">ML</span></h1>
                <p style="color:#888;margin-top:20px">Loading models...</p>
                <div style="margin-top:20px;width:40px;height:40px;border:3px solid #e63946;border-top-color:white;border-radius:50%;animation:spin 1s linear infinite;margin:20px auto"></div>
            </div>
            <style>@keyframes spin{to{transform:rotate(360deg)}}</style>
        </html>
    `);

    startServer();

    try {
        await waitForServer();
        mainWindow.loadURL(API_URL);
    } catch (e) {
        mainWindow.loadURL(`data:text/html,
            <html style="background:#0f0f23;color:white;font-family:sans-serif;display:flex;align-items:center;justify-content:center;height:100vh;margin:0">
                <div style="text-align:center">
                    <h1 style="color:#e63946">Server Error</h1>
                    <p style="color:#888">Could not start the backend server.</p>
                    <p style="color:#666;font-size:12px">Make sure Python and dependencies are installed.</p>
                </div>
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
