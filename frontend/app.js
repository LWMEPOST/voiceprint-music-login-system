let mediaRecorder;
let audioChunks = [];
let recordedBlob = null;
let isRecording = false;
let recordingStartTs = 0;

const RECORD_SECONDS = 4;
const MIN_VALID_SECONDS = 2.2;

const TOKEN_KEY = "token";
const LOGIN_USER_KEY = "login_username";
const LOGIN_METHOD_KEY = "login_method";

function switchTab(tab) {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.getElementById(`tab-${tab}`).classList.add('active');

    if (tab === 'login') {
        document.getElementById('form-login').style.display = 'block';
        document.getElementById('form-register').style.display = 'none';
    } else {
        document.getElementById('form-login').style.display = 'none';
        document.getElementById('form-register').style.display = 'block';
    }

    hideMessage();
    resetRecordingUI();
}

function showMessage(msg, type) {
    const box = document.getElementById('msg-box');
    box.className = `message ${type}`;
    box.innerText = msg;
}

function hideMessage() {
    document.getElementById('msg-box').className = 'message';
}

function resetRecordingUI() {
    recordedBlob = null;
    document.getElementById('btn-submit-reg').disabled = true;
    document.getElementById('btn-reg-record').classList.remove('recording');
    document.getElementById('btn-login-record').classList.remove('recording');
    document.getElementById('btn-reg-record').innerText = `🎤 点击录制声纹 (${RECORD_SECONDS}秒)`;
    document.getElementById('btn-login-record').innerText = "🎤 声纹安全登录";
}

function showAuthPage(defaultTab = 'login') {
    document.getElementById('welcome-page').style.display = 'none';
    document.getElementById('auth-page').style.display = 'block';
    switchTab(defaultTab);
}

function showWelcomePage(username, method, extra = '') {
    document.getElementById('auth-page').style.display = 'none';
    document.getElementById('welcome-page').style.display = 'block';
    document.getElementById('welcome-text').innerText = `欢迎，${username}`;
    document.getElementById('welcome-subtext').innerText =
        `你已通过${method}成功登录。${extra}`.trim();
}

function onLoginSuccess(username, token, method, extra = '') {
    localStorage.setItem(TOKEN_KEY, token);
    localStorage.setItem(LOGIN_USER_KEY, username);
    localStorage.setItem(LOGIN_METHOD_KEY, method);
    showWelcomePage(username, method, extra);
}

function goBackToAuth() {
    hideMessage();
    showAuthPage('login');
}

function logout() {
    localStorage.removeItem(TOKEN_KEY);
    localStorage.removeItem(LOGIN_USER_KEY);
    localStorage.removeItem(LOGIN_METHOD_KEY);
    showAuthPage('login');
    showMessage("已注销账号，返回登录/注册页，可继续测试。", "success");
}

async function startRecording(type) {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            audio: {
                channelCount: 1,
                sampleRate: 16000,
                noiseSuppression: true,
                echoCancellation: true,
                autoGainControl: true
            }
        });
        mediaRecorder = new MediaRecorder(stream);
        audioChunks = [];
        recordingStartTs = Date.now();

        mediaRecorder.ondataavailable = event => {
            if (event.data.size > 0) audioChunks.push(event.data);
        };

        mediaRecorder.onstop = () => {
            let mimeType = 'audio/webm;codecs=opus';
            if (!MediaRecorder.isTypeSupported(mimeType)) {
                mimeType = 'audio/mp4';
                if (!MediaRecorder.isTypeSupported(mimeType)) {
                    mimeType = 'audio/webm';
                }
            }

            recordedBlob = new Blob(audioChunks, { type: mimeType });
            stream.getTracks().forEach(track => track.stop());
            const recordedSeconds = (Date.now() - recordingStartTs) / 1000;

            if (recordedSeconds < MIN_VALID_SECONDS) {
                recordedBlob = null;
                showMessage(`录音时长不足 ${MIN_VALID_SECONDS.toFixed(1)} 秒，请重录`, 'error');
                document.getElementById('btn-reg-record').innerText = `🎤 点击录制声纹 (${RECORD_SECONDS}秒)`;
                document.getElementById('btn-login-record').innerText = "🎤 声纹安全登录";
                return;
            }

            if (type === 'register') {
                document.getElementById('btn-reg-record').innerText = "✅ 录音完成 (点击重录)";
                document.getElementById('btn-reg-record').classList.remove('recording');
                document.getElementById('btn-submit-reg').disabled = false;
            } else {
                document.getElementById('btn-login-record').innerText = "✅ 录音完成 (正在验证...)";
                document.getElementById('btn-login-record').classList.remove('recording');
                submitVoiceLogin();
            }
        };

        mediaRecorder.start();
        isRecording = true;

        setTimeout(() => {
            if (isRecording) {
                stopRecording();
            }
        }, RECORD_SECONDS * 1000);

        const btn = type === 'register' ? 'btn-reg-record' : 'btn-login-record';
        document.getElementById(btn).classList.add('recording');
        document.getElementById(btn).innerText = `🔴 录音中... (${RECORD_SECONDS}秒后自动停止)`;

    } catch (err) {
        showMessage(`获取麦克风失败: ${err.message}`, 'error');
    }
}

function stopRecording() {
    if (mediaRecorder && mediaRecorder.state === "recording") {
        mediaRecorder.stop();
        isRecording = false;
    }
}

function toggleRecording(type) {
    if (isRecording) {
        stopRecording();
    } else {
        startRecording(type);
    }
}

async function submitRegister() {
    const username = document.getElementById('reg-username').value.trim();
    const password = document.getElementById('reg-password').value.trim();

    if (!username || !password) return showMessage("用户名和密码不能为空", "error");
    if (!recordedBlob) return showMessage("请先录制声纹", "error");

    const formData = new FormData();
    formData.append("username", username);
    formData.append("password", password);
    formData.append("audio", recordedBlob, "reg.webm");

    showMessage("注册中，请稍候...", "info");
    document.getElementById('btn-submit-reg').disabled = true;

    try {
        const res = await fetch("/api/register", {
            method: "POST",
            body: formData
        });
        const data = await res.json();

        if (res.ok) {
            showMessage("✅ 注册成功，请切换到登录页面", "success");
            setTimeout(() => switchTab('login'), 1200);
        } else {
            showMessage(`❌ 注册失败: ${data.detail}`, "error");
            document.getElementById('btn-submit-reg').disabled = false;
        }
    } catch (e) {
        showMessage(`❌ 网络错误: ${e.message}`, "error");
        document.getElementById('btn-submit-reg').disabled = false;
    }
}

async function submitVoiceLogin() {
    const username = document.getElementById('login-username').value.trim();
    if (!username) return showMessage("请输入用户名", "error");
    if (!recordedBlob) return showMessage("未检测到录音", "error");

    const formData = new FormData();
    formData.append("username", username);
    formData.append("audio", recordedBlob, "login.webm");

    showMessage("正在比对声纹...", "info");

    try {
        const res = await fetch("/api/login/voice", {
            method: "POST",
            body: formData
        });
        const data = await res.json();

        if (res.ok) {
            const similarityText = `相似度：${(data.similarity * 100).toFixed(1)}%`;
            onLoginSuccess(username, data.access_token, "声纹", similarityText);
        } else {
            showMessage(`⚠️ 登录失败: ${data.detail}`, "error");
            document.getElementById('btn-login-record').innerText = "🎤 声纹安全登录";
        }
    } catch (e) {
        showMessage(`❌ 网络错误: ${e.message}`, "error");
        document.getElementById('btn-login-record').innerText = "🎤 声纹安全登录";
    }
}

async function submitLoginPassword() {
    const username = document.getElementById('login-username').value.trim();
    const password = document.getElementById('login-password').value.trim();

    if (!username || !password) return showMessage("用户名和密码不能为空", "error");

    const formData = new FormData();
    formData.append("username", username);
    formData.append("password", password);

    showMessage("密码验证中...", "info");

    try {
        const res = await fetch("/api/login/password", {
            method: "POST",
            body: formData
        });
        const data = await res.json();

        if (res.ok) {
            onLoginSuccess(username, data.access_token, "密码");
        } else {
            showMessage(`❌ 登录失败: ${data.detail}`, "error");
        }
    } catch (e) {
        showMessage(`❌ 网络错误: ${e.message}`, "error");
    }
}

document.addEventListener("DOMContentLoaded", () => {
    const token = localStorage.getItem(TOKEN_KEY);
    const username = localStorage.getItem(LOGIN_USER_KEY);
    const method = localStorage.getItem(LOGIN_METHOD_KEY);

    if (token && username && method) {
        showWelcomePage(username, method);
    } else {
        showAuthPage('login');
    }
});
