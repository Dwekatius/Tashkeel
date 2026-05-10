// =====================================================
// Sukoon Tashkeel - container-backed frontend
// =====================================================

(function () {
    'use strict';

    const MODEL_ID = 'eo';
    const MODEL_LABEL = 'Sukoon';
    const FAST_REVEAL_THRESHOLD_MS = 900;

    let chats = [];
    let activeChatId = null;
    let isDark = true;
    let sidebarCollapsed = false;
    let runtime = {
        ok: false,
        maxChars: 0,
        devices: [],
        demos: [],
    };
    let selectedDevice = localStorage.getItem('sukoon-device') || 'cpu';

    const $ = (sel) => document.querySelector(sel);
    const messagesContainer = $('#messages-container');
    const messagesColumn = $('#messages-column');
    const welcomeScreen = $('#welcome-screen');
    const messageInput = $('#message-input');
    const sendBtn = $('#send-btn');
    const newChatBtn = $('#new-chat-btn');
    const chatHistoryEl = $('#chat-history');
    const demoListEl = $('#demo-list');
    const deviceOptionsEl = $('#device-options');
    const runtimeStatusEl = $('#runtime-status');
    const runtimeDetailsEl = $('#runtime-details');
    const inputHintEl = $('#input-hint');
    const sidebarWrapper = $('#sidebar-wrapper');
    const sidebarToggleBtn = $('#sidebar-toggle-btn');
    const settingsPanel = $('#settings-panel');
    const settingsBtn = $('#settings-btn');
    const closeSettingsBtn = $('#close-settings');
    const themeToggleBtn = $('#theme-toggle');
    const heroStartBtn = $('#hero-start-btn');
    const heroExampleBtn = $('#hero-example-btn');

    function uid() {
        return Date.now().toString(36) + Math.random().toString(36).slice(2, 7);
    }

    function createChat(title = 'New session') {
        const chat = { id: uid(), title, messages: [] };
        chats.unshift(chat);
        activeChatId = chat.id;
        renderChatHistory();
        showWelcomeScreen();
        return chat;
    }

    function getActiveChat() {
        return chats.find((chat) => chat.id === activeChatId);
    }

    function deleteChat(id, event) {
        if (event) event.stopPropagation();
        chats = chats.filter((chat) => chat.id !== id);
        if (activeChatId === id) {
            activeChatId = chats.length > 0 ? chats[0].id : null;
        }
        renderChatHistory();
        if (activeChatId) {
            renderMessages(getActiveChat());
        } else {
            showWelcomeScreen();
        }
    }

    function setActiveChat(chat) {
        activeChatId = chat.id;
        renderMessages(chat);
        renderChatHistory();
    }

    function renderChatHistory() {
        chatHistoryEl.innerHTML = '';
        chats.forEach((chat) => {
            const div = document.createElement('div');
            const isActive = chat.id === activeChatId;
            div.className = `chat-history-item ${isActive ? 'active' : ''}`;
            div.innerHTML = `
                <span class="truncate flex-1">${escapeHtml(chat.title)}</span>
                <button class="delete-chat-btn" aria-label="Delete chat" title="Delete chat">
                    <svg xmlns="http://www.w3.org/2000/svg" class="icon" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6M1 7h22"/></svg>
                </button>`;

            div.addEventListener('click', () => setActiveChat(chat));
            div.querySelector('.delete-chat-btn').addEventListener('click', (event) => deleteChat(chat.id, event));
            chatHistoryEl.appendChild(div);
        });
    }

    function showWelcomeScreen() {
        messagesColumn.innerHTML = '';
        messagesColumn.appendChild(welcomeScreen);
        welcomeScreen.style.display = '';
    }

    function renderMessages(chat) {
        if (!chat || chat.messages.length === 0) {
            showWelcomeScreen();
            return;
        }
        messagesColumn.innerHTML = '';
        chat.messages.forEach((msg) => appendMessageBubble(msg.role, msg.content, false, msg.meta || ''));
        scrollToBottom();
    }

    function appendMessageBubble(role, content, animate = true, meta = '') {
        const wrapper = document.createElement('div');
        wrapper.className = `message-wrap ${role === 'user' ? 'user' : 'model'} ${animate ? 'message-enter' : ''}`;

        const label = document.createElement('span');
        label.className = 'message-label';
        label.textContent = role === 'user' ? 'You' : (meta ? `${MODEL_LABEL} / ${meta}` : `${MODEL_LABEL} AI`);

        const bubble = document.createElement('div');
        bubble.className = role === 'user' ? 'user-message' : 'model-message';
        bubble.dir = 'auto';
        bubble.textContent = content;

        wrapper.appendChild(label);
        wrapper.appendChild(bubble);
        messagesColumn.appendChild(wrapper);
        return bubble;
    }

    function setBubbleMeta(bubble, meta) {
        const label = bubble?.previousElementSibling;
        if (label) label.textContent = meta ? `${MODEL_LABEL} / ${meta}` : `${MODEL_LABEL} AI`;
    }

    function showLoading() {
        const el = document.createElement('div');
        el.id = 'loading-indicator';
        el.className = 'message-wrap model message-enter';

        const label = document.createElement('span');
        label.className = 'message-label';
        label.textContent = `${MODEL_LABEL} is diacritizing...`;

        const dots = document.createElement('div');
        dots.className = 'model-message';
        dots.innerHTML = '<div class="flex items-center gap-2"><div class="loading-dot"></div><div class="loading-dot"></div><div class="loading-dot"></div></div>';

        el.appendChild(label);
        el.appendChild(dots);
        messagesColumn.appendChild(el);
        scrollToBottom();
    }

    function hideLoading() {
        const el = $('#loading-indicator');
        if (el) el.remove();
    }

    function updateRuntimeUI() {
        runtimeStatusEl.textContent = runtime.ok ? 'Ready' : 'Offline';
        runtimeStatusEl.classList.toggle('ready', runtime.ok);
        runtimeStatusEl.classList.toggle('offline', !runtime.ok);
        runtimeDetailsEl.textContent = runtime.ok
            ? `${MODEL_LABEL} / ${formatLimit(runtime.maxChars)}`
            : `${MODEL_LABEL} / waiting for API`;
        inputHintEl.textContent = `Local ${MODEL_LABEL} runtime. Input limit: ${formatLimit(runtime.maxChars)}.`;
    }

    async function refreshStatus() {
        try {
            const response = await fetch('/api/status', { cache: 'no-store' });
            const payload = await response.json();
            if (!response.ok || payload.error) throw new Error(payload.error || `HTTP ${response.status}`);
            runtime = {
                ok: Boolean(payload.ok),
                maxChars: Number(payload.max_chars || 0),
                devices: Array.isArray(payload.devices) ? payload.devices : [],
                demos: Array.isArray(payload.demos) ? payload.demos : [],
            };
            const defaultDevice = normalizeDevice(payload.default_device || selectedDevice);
            if (!runtime.devices.some((item) => item.id === selectedDevice && item.available)) {
                selectedDevice = runtime.devices.some((item) => item.id === defaultDevice && item.available)
                    ? defaultDevice
                    : 'cpu';
                localStorage.setItem('sukoon-device', selectedDevice);
            }
            renderDeviceOptions(runtime.devices);
            renderDemos(runtime.demos);
        } catch {
            runtime.ok = false;
        } finally {
            updateRuntimeUI();
        }
    }

    function normalizeDevice(device) {
        return device === 'cuda' ? 'gpu' : String(device || 'cpu').toLowerCase();
    }

    function formatLimit(value) {
        return Number(value || 0) > 0 ? `max ${value} chars` : 'unlimited input';
    }

    function renderDeviceOptions(devices) {
        const visibleDevices = devices.length ? devices : [{ id: 'cpu', label: 'CPU', available: true }];
        deviceOptionsEl.innerHTML = '';
        visibleDevices.forEach((device) => {
            const button = document.createElement('button');
            button.type = 'button';
            button.className = `device-option ${device.id === selectedDevice ? 'active' : ''}`;
            button.disabled = !device.available;
            button.title = device.description || device.label || device.id;
            button.textContent = device.label || device.id.toUpperCase();
            button.addEventListener('click', () => {
                selectedDevice = device.id;
                localStorage.setItem('sukoon-device', selectedDevice);
                renderDeviceOptions(runtime.devices);
            });
            deviceOptionsEl.appendChild(button);
        });
    }

    function renderDemos(demos) {
        demoListEl.innerHTML = '';
        demos.forEach((demo) => {
            const button = document.createElement('button');
            button.type = 'button';
            button.className = 'demo-chip';
            button.innerHTML = `
                <strong>${escapeHtml(demo.title || 'Example')}</strong>
                <span>${escapeHtml(demo.tag || 'demo')} / ${Number(demo.letters || 0)} letters</span>`;
            button.addEventListener('click', () => loadDemo(demo));
            demoListEl.appendChild(button);
        });
    }

    async function loadDemo(demo) {
        try {
            const response = await fetch('/api/demos', { cache: 'no-store' });
            const payload = await response.json();
            const fullDemo = payload.demos?.find((item) => item.id === demo.id) || demo;
            const chat = createChat(fullDemo.title || 'Demo');
            messageInput.value = fullDemo.plain || '';
            autoResizeInput();
            setActiveChat(chat);
            messageInput.focus();
        } catch {
            messageInput.value = demo.plain || '';
            autoResizeInput();
            messageInput.focus();
        }
    }

    function loadFirstExample() {
        const first = runtime.demos?.[0];
        if (first) {
            loadDemo(first);
            return;
        }
        messageInput.value = 'كتب الطالب الدرس ثم ذهب إلى المكتبة.';
        autoResizeInput();
        messageInput.focus();
    }

    async function sendMessage() {
        const text = messageInput.value.trim();
        if (!text) return;
        if (runtime.maxChars > 0 && text.length > runtime.maxChars) {
            appendSystemMessage(`Input is longer than ${runtime.maxChars} characters.`);
            return;
        }

        let chat = getActiveChat();
        if (!chat) {
            chat = createChat(text.slice(0, 30) + (text.length > 30 ? '...' : ''));
        } else if (chat.title === 'New session' && chat.messages.length === 0) {
            chat.title = text.slice(0, 30) + (text.length > 30 ? '...' : '');
            renderChatHistory();
        }

        const ws = $('#welcome-screen');
        if (ws) ws.remove();

        chat.messages.push({ role: 'user', content: text });
        appendMessageBubble('user', text);
        messageInput.value = '';
        autoResizeInput();
        disableInput(true);
        showLoading();

        try {
            const result = await streamDiacritizedText(text, chat);
            chat.messages.push({ role: 'model', content: result.output, meta: result.meta });
            refreshStatus();
        } catch (error) {
            hideLoading();
            const message = error.message || 'Inference failed.';
            chat.messages.push({ role: 'model', content: message, meta: 'error' });
            appendMessageBubble('model', message, true, 'error');
        } finally {
            disableInput(false);
            scrollToBottom();
        }
    }

    async function streamDiacritizedText(text) {
        const requestStarted = performance.now();
        const response = await fetch('/api/diacritize_stream', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text, model: MODEL_ID, device: selectedDevice }),
        });

        if (!response.ok || !response.body) {
            return fallbackDiacritize(text);
        }

        hideLoading();
        const bubble = appendMessageBubble('model', '', true, 'working');
        bubble.classList.add('streaming');

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let pendingLine = '';
        let displayed = '';
        let buffered = '';
        let finalPayload = null;
        let flushedLive = false;

        const appendNow = (chunk) => {
            if (!chunk) return;
            displayed += chunk;
            bubble.textContent = displayed;
            scrollToBottom();
        };

        const flushBuffer = () => {
            if (!flushedLive) {
                flushedLive = true;
                appendNow(buffered);
                buffered = '';
            }
        };

        while (true) {
            const { value, done } = await reader.read();
            if (value) {
                pendingLine += decoder.decode(value, { stream: !done });
                const lines = pendingLine.split('\n');
                pendingLine = lines.pop() || '';
                for (const line of lines) {
                    if (!line.trim()) continue;
                    const event = JSON.parse(line);
                    if (event.type === 'error') throw new Error(event.error || 'Inference failed.');
                    if (event.type === 'start') {
                        const meta = formatMeta(event);
                        if (meta) setBubbleMeta(bubble, meta);
                    }
                    if (event.type === 'chunk') {
                        const age = performance.now() - requestStarted;
                        if (age < FAST_REVEAL_THRESHOLD_MS && !flushedLive) {
                            buffered += event.text || '';
                        } else {
                            flushBuffer();
                            appendNow(event.text || '');
                        }
                    }
                    if (event.type === 'done') {
                        finalPayload = event;
                    }
                }
            }
            if (done) break;
        }

        if (pendingLine.trim()) {
            const event = JSON.parse(pendingLine);
            if (event.type === 'done') finalPayload = event;
            if (event.type === 'error') throw new Error(event.error || 'Inference failed.');
        }

        const elapsedMs = performance.now() - requestStarted;
        const finalOutput = finalPayload?.output || displayed + buffered;
        const meta = formatMeta(finalPayload || {});
        setBubbleMeta(bubble, meta);

        if (!flushedLive && elapsedMs < FAST_REVEAL_THRESHOLD_MS) {
            await fastRevealText(bubble, finalOutput);
        } else {
            flushBuffer();
            bubble.textContent = finalOutput;
        }

        bubble.classList.remove('streaming');
        scrollToBottom();
        return { output: finalOutput, meta };
    }

    async function fallbackDiacritize(text) {
        const response = await fetch('/api/diacritize', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text, model: MODEL_ID, device: selectedDevice }),
        });
        const payload = await response.json();
        if (!response.ok || payload.error) throw new Error(payload.error || `HTTP ${response.status}`);

        hideLoading();
        const output = payload.output || '';
        const meta = formatMeta(payload);
        const bubble = appendMessageBubble('model', '', true, meta);
        if (Number(payload.elapsed || 0) < 1) {
            await fastRevealText(bubble, output);
        } else {
            bubble.textContent = output;
        }
        return { output, meta };
    }

    function formatMeta(payload) {
        return [
            payload.device || '',
            payload.elapsed ? `${Number(payload.elapsed).toFixed(2)}s` : '',
            payload.segments ? `${payload.segments} segments` : '',
        ].filter(Boolean).join(' / ');
    }

    function appendSystemMessage(message) {
        let chat = getActiveChat();
        if (!chat) chat = createChat('Notice');
        const ws = $('#welcome-screen');
        if (ws) ws.remove();
        chat.messages.push({ role: 'model', content: message, meta: 'notice' });
        appendMessageBubble('model', message, true, 'notice');
    }

    function disableInput(disabled) {
        messageInput.disabled = disabled;
        sendBtn.disabled = disabled;
        if (!disabled) messageInput.focus();
    }

    function scrollToBottom() {
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }

    function escapeHtml(str) {
        const div = document.createElement('div');
        div.textContent = String(str);
        return div.innerHTML;
    }

    function splitRevealUnits(text) {
        const units = text.match(/\S+\s*|\s+/g);
        return units && units.length ? units : Array.from(text);
    }

    function fastRevealText(element, text) {
        return new Promise((resolve) => {
            const units = splitRevealUnits(text);
            const duration = Math.min(650, Math.max(180, units.length * 7));
            const started = performance.now();
            element.textContent = '';
            element.classList.add('streaming');

            function frame(now) {
                const progress = Math.min(1, (now - started) / duration);
                const count = Math.max(1, Math.floor(progress * units.length));
                element.textContent = units.slice(0, count).join('');
                scrollToBottom();
                if (progress < 1) {
                    requestAnimationFrame(frame);
                } else {
                    element.textContent = text;
                    element.classList.remove('streaming');
                    resolve();
                }
            }

            requestAnimationFrame(frame);
        });
    }

    const arrowOpen = '<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2.5" d="M15 19l-7-7 7-7"/>';
    const arrowClose = '<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2.5" d="M9 5l7 7-7 7"/>';

    function toggleSidebar(collapse) {
        sidebarCollapsed = collapse;
        if (sidebarCollapsed) {
            sidebarWrapper.style.width = '0px';
            sidebarToggleBtn.querySelector('svg').innerHTML = arrowClose;
        } else {
            sidebarWrapper.style.width = '';
            sidebarToggleBtn.querySelector('svg').innerHTML = arrowOpen;
        }
    }

    function setTheme(dark) {
        isDark = dark;
        localStorage.setItem('sukoon-theme', dark ? 'dark' : 'light');
        applyTheme();
    }

    function applyTheme() {
        document.documentElement.classList.toggle('dark', isDark);
        const moonPath = '<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z"/>';
        const sunPath = '<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z"/>';
        themeToggleBtn.querySelector('svg').innerHTML = isDark ? moonPath : sunPath;
    }

    function autoResizeInput() {
        messageInput.style.height = 'auto';
        messageInput.style.height = Math.min(messageInput.scrollHeight, 170) + 'px';
    }

    function syncResponsiveSidebar() {
        if (window.matchMedia('(max-width: 980px)').matches) {
            toggleSidebar(true);
        } else if (sidebarCollapsed) {
            toggleSidebar(false);
        }
    }

    messageInput.addEventListener('input', autoResizeInput);

    sendBtn.addEventListener('click', sendMessage);
    messageInput.addEventListener('keydown', (event) => {
        if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault();
            sendMessage();
        }
    });

    newChatBtn.addEventListener('click', () => {
        createChat();
        messageInput.value = '';
        autoResizeInput();
        messageInput.focus();
    });

    sidebarToggleBtn.addEventListener('click', () => toggleSidebar(!sidebarCollapsed));
    settingsBtn.addEventListener('click', () => settingsPanel.classList.remove('hidden'));
    closeSettingsBtn.addEventListener('click', () => settingsPanel.classList.add('hidden'));
    $('#settings-backdrop').addEventListener('click', () => settingsPanel.classList.add('hidden'));
    themeToggleBtn.addEventListener('click', () => setTheme(!isDark));
    if (heroStartBtn) heroStartBtn.addEventListener('click', () => messageInput.focus());
    if (heroExampleBtn) heroExampleBtn.addEventListener('click', loadFirstExample);

    isDark = localStorage.getItem('sukoon-theme') !== 'light';
    applyTheme();
    syncResponsiveSidebar();
    updateRuntimeUI();
    renderDeviceOptions(runtime.devices);
    refreshStatus();
    window.addEventListener('resize', syncResponsiveSidebar);
    window.setInterval(refreshStatus, 5000);
})();
