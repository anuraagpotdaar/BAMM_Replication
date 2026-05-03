/**
 * Main application logic
 * Handles UI interactions, API calls, and 3D rendering loop
 */

class MotionApp {
    constructor() {
        this.isRunning = false;
        this.targetFps = 20; // Model generates data at 20fps
        this.frameInterval = 1000 / this.targetFps; // 50ms
        this.nextFetchTime = 0;  // Scheduled time for next fetch
        this.frameCount = 0;

        // Motion FPS tracking (frame consumption rate)
        this.motionFpsCounter = 0;
        this.motionFpsUpdateTime = 0;
        
        // Request throttling
        this.isFetchingFrame = false;  // Prevent concurrent requests
        this.consecutiveWaiting = 0;   // Count consecutive 'waiting' responses

        // Local frame queue for batch fetching (reduces HTTP round-trip overhead)
        this.localFrameQueue = [];
        this.batchSize = 2;  // Small batch = lower input-to-screen latency
        this.broadcastLastId = 0;  // For spectator mode (broadcast buffer cursor)
        
        // Session management
        this.sessionId = this.generateSessionId();
        
        // Camera follow settings
        this.lastUserInteraction = 0;
        this.autoFollowDelay = 2000; // Auto-follow after 2 seconds of inactivity (reduced from 3s)
        this.currentRootPos = new THREE.Vector3(0, 1, 0);
        
        this.initThreeJS();
        this.initUI();
        this.updateStatus();
        this.setupBeforeUnload();
        
        console.log('Session ID:', this.sessionId);
    }
    
    generateSessionId() {
        // Generate a simple unique session ID
        return 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    }
    
    setupBeforeUnload() {
        // Handle page close/refresh - send reset request
        window.addEventListener('beforeunload', () => {
            // Send synchronous reset if we're generating
            if (!this.isIdle) {
                // Use Blob to set correct Content-Type for JSON
                const blob = new Blob(
                    [JSON.stringify({session_id: this.sessionId})],
                    {type: 'application/json'}
                );
                navigator.sendBeacon('/api/reset', blob);
                console.log('Sent reset beacon on page unload');
            }
        });
        
        // Also handle visibility change (tab hidden, mobile app switch)
        document.addEventListener('visibilitychange', () => {
            if (document.hidden && !this.isIdle && this.isRunning) {
                // User switched away while generating - they might not come back
                // Note: Don't reset immediately, let the frame consumption monitor handle it
                console.log('Tab hidden while generating - consumption monitor will auto-reset if needed');
            }
        });
    }
    
    initThreeJS() {
        // Get canvas
        const canvas = document.getElementById('renderCanvas');
        const container = document.getElementById('canvas-container');
        
        // Create scene
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0xffffff);  // White background
        
        // Create camera
        this.camera = new THREE.PerspectiveCamera(
            60,
            container.clientWidth / container.clientHeight,
            0.1,
            1000
        );
        this.camera.position.set(3, 1.5, 3);
        this.camera.lookAt(0, 1, 0);
        
        // Create renderer
        this.renderer = new THREE.WebGLRenderer({
            canvas: canvas,
            antialias: true
        });
        this.renderer.setSize(container.clientWidth, container.clientHeight);
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
        this.renderer.toneMappingExposure = 1.0;
        
        // Add lights - bright and soft
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.7);
        this.scene.add(ambientLight);
        
        const keyLight = new THREE.DirectionalLight(0xffffff, 0.8);
        keyLight.position.set(5, 8, 3);
        keyLight.castShadow = true;
        keyLight.shadow.mapSize.width = 2048;
        keyLight.shadow.mapSize.height = 2048;
        keyLight.shadow.camera.near = 0.5;
        keyLight.shadow.camera.far = 50;
        keyLight.shadow.camera.left = -5;
        keyLight.shadow.camera.right = 5;
        keyLight.shadow.camera.top = 5;
        keyLight.shadow.camera.bottom = -5;
        keyLight.shadow.bias = -0.0001;
        this.scene.add(keyLight);
        
        // Fill light
        const fillLight = new THREE.DirectionalLight(0xffffff, 0.4);
        fillLight.position.set(-3, 5, -3);
        this.scene.add(fillLight);
        
        // Add ground plane - light gray, very large
        const groundGeometry = new THREE.PlaneGeometry(1000, 1000);
        const groundMaterial = new THREE.ShadowMaterial({
            opacity: 0.15
        });
        const ground = new THREE.Mesh(groundGeometry, groundMaterial);
        ground.rotation.x = -Math.PI / 2;
        ground.position.y = 0;
        ground.receiveShadow = true;
        this.scene.add(ground);
        
        // Add infinite-looking grid - very large grid
        const gridHelper = new THREE.GridHelper(1000, 1000, 0xdddddd, 0xeeeeee);
        gridHelper.position.y = 0.01;
        this.scene.add(gridHelper);
        
        // Add orbit controls
        this.controls = new THREE.OrbitControls(this.camera, canvas);
        this.controls.target.set(0, 1, 0);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;
        this.controls.update();
        
        // Listen for user interaction - record time
        const updateInteractionTime = () => {
            this.lastUserInteraction = Date.now();
        };
        canvas.addEventListener('mousedown', updateInteractionTime);
        canvas.addEventListener('wheel', updateInteractionTime);
        canvas.addEventListener('touchstart', updateInteractionTime);
        
        // Default to the stick-figure skeleton — it draws every bone of the
        // 22-joint HumanML3D chain directly from the predicted joint positions,
        // so the model output is faithfully visualised. The toggle cycles
        // Skeleton → Soldier (rigged glTF) → SMPL (mannequin) → Skeleton.
        this.skeleton = new Skeleton3D(this.scene);
        
        // Handle window resize
        window.addEventListener('resize', () => this.onWindowResize());
        
        // Start render loop
        this.animate();
    }
    
    initUI() {
        // Get UI elements
        this.motionText = document.getElementById('motionText');
        this.currentSmoothing = document.getElementById('currentSmoothing');
        this.currentHistory = document.getElementById('currentHistory');
        this.startResetBtn = document.getElementById('startResetBtn');
        this.updateBtn = document.getElementById('updateBtn');
        this.pauseResumeBtn = document.getElementById('pauseResumeBtn');
        this.configBtn = document.getElementById('configBtn');
        this.characterToggleBtn = document.getElementById('characterToggleBtn');
        this.characterCurrentEl = document.getElementById('characterCurrent');
        this.statusEl = document.getElementById('status');
        this.bufferSizeEl = document.getElementById('bufferSize');
        this.fpsEl = document.getElementById('fps');
        this.frameCountEl = document.getElementById('frameCount');
        this.conflictWarning = document.getElementById('conflictWarning');
        this.forceTakeoverBtn = document.getElementById('forceTakeoverBtn');
        this.cancelTakeoverBtn = document.getElementById('cancelTakeoverBtn');

        // Stored runtime parameter values (updated by Config modal)
        this.historyLengthValue = null;
        this.smoothingAlphaValue = 0.5;  // Default

        // Track state
        this.isPaused = false;
        this.isIdle = true;
        this.isWatching = false;  // Spectator mode
        this.isProcessing = false;  // Prevent concurrent API calls
        this.pendingStartRequest = null;  // Store pending start request data

        // Character toggle state
        this.currentCharacter = 'skeleton';   // matches the constructor in initThreeJS
        this.lastJointFrame = null;          // most recent pose, replayed after a swap

        // Attach event listeners
        this.startResetBtn.addEventListener('click', () => this.toggleStartReset());
        // Auto-update text while running:
        //  - immediately after a space (word completion) or Enter (full submit)
        //  - or after the user pauses typing for 600 ms
        // Button is hidden but kept in DOM as a fallback.
        this.updateBtn.addEventListener('click', () => this.updateText());
        this._autoUpdateTimer = null;
        this._lastAutoUpdatedText = '';
        const flushAuto = (delayMs) => {
            clearTimeout(this._autoUpdateTimer);
            this._autoUpdateTimer = setTimeout(() => this._autoUpdateText(), delayMs);
        };
        this.motionText.addEventListener('keydown', (e) => {
            if (e.key === ' ' || e.code === 'Space') {
                console.log('[auto] space pressed, scheduling update');
                flushAuto(60);
            } else if (e.key === 'Enter') {
                e.preventDefault();
                console.log('[auto] enter pressed, scheduling update');
                flushAuto(0);
            }
        });
        this.motionText.addEventListener('input', () => flushAuto(600));

        // Motion-preset chips: click → set text + fire update immediately.
        // Each click fires regardless of dedupe (so re-clicking a chip replays
        // the action), and the chip auto-deselects after a short visual flash
        // so the same chip can be clicked again.
        this.motionChips = Array.from(document.querySelectorAll('.motion-chip'));
        this.motionChips.forEach(chip => {
            chip.addEventListener('click', () => {
                const text = chip.dataset.motion;
                if (!text) return;
                this.motionText.value = text;
                this._setActiveChip(chip);
                this._telem('client_chip_click', { text });
                clearTimeout(this._autoUpdateTimer);
                this._autoUpdateText(true);  // force-fire (bypass dedupe)
                setTimeout(() => chip.classList.remove('active'), 350);
            });
        });

        this.pauseResumeBtn.addEventListener('click', () => this.togglePauseResume());
        this.configBtn.addEventListener('click', () => this.openConfigEditor());
        if (this.characterToggleBtn) {
            this.characterToggleBtn.addEventListener('click', () => this.switchCharacter());
        }
        this.forceTakeoverBtn.addEventListener('click', () => this.handleForceTakeover());
        this.cancelTakeoverBtn.addEventListener('click', () => this.handleCancelTakeover());

        // Modal event listeners
        document.getElementById('configDiscardBtn').addEventListener('click', () => this.closeConfigEditor());
        document.getElementById('configSaveBtn').addEventListener('click', () => this.saveConfigAndReset());
        document.getElementById('modalSmoothingAlpha').addEventListener('input', (e) => {
            document.getElementById('modalSmoothingValue').textContent = parseFloat(e.target.value).toFixed(2);
        });

        // Fetch config from server on page load
        fetch('/api/config')
            .then(r => {
                if (!r.ok) throw new Error(`HTTP ${r.status}`);
                return r.json();
            })
            .then(data => {
                if (data.status === 'error') throw new Error(data.message);
                this.historyLengthValue = data.history_length;
                this.smoothingAlphaValue = data.smoothing_alpha;
            })
            .catch(e => {
                this.statusEl.textContent = 'Error: failed to load config';
                this.startResetBtn.disabled = true;
                console.error('Failed to fetch config:', e);
            });
    }
    
    async toggleStartReset() {
        if (this.isProcessing) return;  // Prevent concurrent operations

        if (this.isIdle || this.isWatching) {
            // Idle or spectator watching, so start (will force takeover if needed)
            await this.startGeneration(this.isWatching);
        } else {
            // Currently running/paused, so reset
            await this.reset();
        }
    }
    
    async startGeneration(force = false) {
        if (this.isProcessing) return;  // Prevent concurrent operations

        const text = this.motionText.value.trim();
        if (!text) {
            alert('Please enter a motion description');
            return;
        }

        const historyLength = this.historyLengthValue || 30;
        const smoothingAlpha = this.smoothingAlphaValue;

        this.isProcessing = true;
        this.statusEl.textContent = 'Initializing...';

        try {
            const response = await fetch('/api/start', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    session_id: this.sessionId,
                    text: text,
                    history_length: historyLength,
                    smoothing_alpha: smoothingAlpha,
                    force: force
                })
            });
            
            const data = await response.json();
            
            if (data.status === 'success') {
                this.isRunning = true;
                this.isPaused = false;
                this.isIdle = false;
                this.frameCount = 0;
                this.motionFpsCounter = 0;
                this.motionFpsUpdateTime = performance.now();
                this.isFetchingFrame = false;
                this.consecutiveWaiting = 0;
                this.startResetBtn.textContent = 'Reset';
                this.startResetBtn.classList.remove('btn-primary');
                this.startResetBtn.classList.add('btn-danger');
                this.updateBtn.disabled = false;
                this.pauseResumeBtn.disabled = false;
                this.pauseResumeBtn.textContent = 'Pause';
                this.statusEl.textContent = 'Running';
                this.startFrameLoop();
            } else if (response.status === 409 && data.conflict) {
                // Another session is running, show warning UI
                this.statusEl.textContent = 'Conflict - Another user is generating';
                this.conflictWarning.style.display = 'block';
                
                // Store request data for later
                this.pendingStartRequest = {
                    text: text,
                    history_length: historyLength
                };
                
                return;
            } else {
                // Other errors
                alert('Error: ' + data.message);
                this.statusEl.textContent = 'Idle';
                this.isIdle = true;
                this.isRunning = false;
                this.isPaused = false;
            }
        } catch (error) {
            console.error('Error starting generation:', error);
            alert('Failed to start generation: ' + error.message);
            this.statusEl.textContent = 'Idle';
            // Keep idle state on error
            this.isIdle = true;
            this.isRunning = false;
            this.isPaused = false;
        } finally {
            this.isProcessing = false;
        }
    }
    
    async updateText() {
        if (this.isProcessing) return;  // Prevent concurrent operations
        
        const text = this.motionText.value.trim();
        if (!text) {
            alert('Please enter a motion description');
            return;
        }
        
        this.isProcessing = true;
        try {
            const response = await fetch('/api/update_text', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    session_id: this.sessionId,
                    text: text
                })
            });
            
            const data = await response.json();
            
            if (data.status === 'success') {
                console.log('Text updated:', text);
                // Drop already-buffered "old text" frames on the client so the new
                // motion appears as soon as the next batch arrives from the server.
                this.localFrameQueue = [];
            } else {
                alert('Error: ' + data.message);
            }
        } catch (error) {
            console.error('Error updating text:', error);
        } finally {
            this.isProcessing = false;
        }
    }

    _setActiveChip(activeChip) {
        if (!this.motionChips) return;
        for (const c of this.motionChips) c.classList.toggle('active', c === activeChip);
    }

    _telem(src, payload) {
        try {
            fetch('/api/telemetry', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ src, ...(payload || {}) }),
                keepalive: true,
            }).catch(() => {});
        } catch (_) {}
    }

    // Fired automatically after a space, Enter, or typing pause. Pass force=true
    // to bypass the unchanged-text dedupe (chips do this so re-clicking the same
    // chip replays the action).
    async _autoUpdateText(force = false) {
        const reason = !this.isRunning ? 'not running'
                     : this.isPaused ? 'paused'
                     : this.isIdle ? 'idle'
                     : null;
        if (reason) {
            console.log('[auto] skip:', reason);
            return;
        }
        if (this.isProcessing) {
            console.log('[auto] skip: already processing — will retry on next event');
            return;
        }
        const text = this.motionText.value.trim();
        if (!text) {
            console.log('[auto] skip: empty text');
            return;
        }
        if (text === this._lastAutoUpdatedText && !force) {
            console.log('[auto] skip: unchanged text');
            return;
        }
        this._lastAutoUpdatedText = text;
        this.isProcessing = true;
        const fetchStart = performance.now();
        this._telem('client_fetch_start', { text });
        try {
            const response = await fetch('/api/update_text', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ session_id: this.sessionId, text }),
            });
            const data = await response.json();
            const fetchMs = performance.now() - fetchStart;
            this._telem('client_fetch_done', { text, ms: fetchMs, ok: data.status === 'success' });
            if (data.status === 'success') {
                console.log('[auto] sent:', text);
                // No queue manipulation — already-decoded frames represent specific
                // moments in time, so dropping them mid-stream makes the character
                // teleport. The small batch (2) drains in ~100 ms naturally.
            } else {
                console.warn('[auto] server rejected:', data.message);
                // Allow retry if it failed (e.g. session not yet active)
                this._lastAutoUpdatedText = '';
            }
        } catch (e) {
            console.warn('[auto] fetch failed:', e);
            this._lastAutoUpdatedText = '';
        } finally {
            this.isProcessing = false;
        }
    }

    async togglePauseResume() {
        if (this.isProcessing) return;  // Prevent concurrent operations
        if (this.isPaused) {
            // Currently paused, so resume
            await this.resumeGeneration();
        } else {
            // Currently running, so pause
            await this.pauseGeneration();
        }
    }
    
    async pauseGeneration() {
        this.isProcessing = true;
        try {
            const response = await fetch('/api/pause', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({session_id: this.sessionId})
            });
            
            const data = await response.json();
            
            if (data.status === 'success') {
                this.isRunning = false;
                this.isPaused = true;
                this.pauseResumeBtn.textContent = 'Resume';
                this.pauseResumeBtn.classList.remove('btn-warning');
                this.pauseResumeBtn.classList.add('btn-success');
                this.updateBtn.disabled = true;
                this.statusEl.textContent = 'Paused';
                console.log('Generation paused (state preserved)');
            }
        } catch (error) {
            console.error('Error pausing generation:', error);
        } finally {
            this.isProcessing = false;
        }
    }
    
    async resumeGeneration() {
        this.isProcessing = true;
        try {
            const response = await fetch('/api/resume', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({session_id: this.sessionId})
            });
            
            const data = await response.json();
            
            if (data.status === 'success') {
                this.isRunning = true;
                this.isPaused = false;
                this.pauseResumeBtn.textContent = 'Pause';
                this.pauseResumeBtn.classList.remove('btn-success');
                this.pauseResumeBtn.classList.add('btn-warning');
                this.updateBtn.disabled = false;
                this.statusEl.textContent = 'Running';
                this.startFrameLoop();
                console.log('Generation resumed');
            }
        } catch (error) {
            console.error('Error resuming generation:', error);
        } finally {
            this.isProcessing = false;
        }
    }
    
    async reset() {
        if (this.isProcessing) return;  // Prevent concurrent operations

        const historyLength = this.historyLengthValue || 30;
        const smoothingAlpha = this.smoothingAlphaValue;

        this.isProcessing = true;
        try {
            const response = await fetch('/api/reset', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    session_id: this.sessionId,
                    history_length: historyLength,
                    smoothing_alpha: smoothingAlpha,
                })
            });
            
            const data = await response.json();
            
            if (data.status === 'success') {
                this._resetUIToIdle();
                console.log('Reset complete - all state cleared');
            }
        } catch (error) {
            console.error('Error resetting:', error);
        } finally {
            this.isProcessing = false;
        }
    }
    
    async handleForceTakeover() {
        // Hide warning
        this.conflictWarning.style.display = 'none';
        
        if (!this.pendingStartRequest) return;
        
        // Retry with force=true
        this.isProcessing = false;
        await this.startGeneration(true);
        
        this.pendingStartRequest = null;
    }
    
    handleCancelTakeover() {
        // Hide warning
        this.conflictWarning.style.display = 'none';
        this.statusEl.textContent = 'Idle';
        this.isProcessing = false;
        this.pendingStartRequest = null;
    }
    
    startFrameLoop() {
        const now = performance.now();
        this.nextFetchTime = now + this.frameInterval;
        this.fetchFrame();
    }
    
    fetchFrame() {
        if (!this.isRunning) return;

        const now = performance.now();

        // Play back from local queue at target FPS
        if (now >= this.nextFetchTime && this.localFrameQueue.length > 0) {
            this.nextFetchTime += this.frameInterval;
            if (this.nextFetchTime < now) {
                this.nextFetchTime = now + this.frameInterval;
            }

            const joints = this.localFrameQueue.shift();
            this.lastJointFrame = joints;
            this.skeleton.updatePose(joints);
            this.frameCount++;
            this.frameCountEl.textContent = this.frameCount;
            this.motionFpsCounter++;

            this.currentRootPos.set(joints[0][0], joints[0][1], joints[0][2]);
            this.updateAutoFollow();

            // Sample frame display telemetry (every 4th frame ≈ 5 Hz)
            if ((this.frameCount & 3) === 0) {
                this._telem('client_frame_displayed', {
                    n: this.frameCount,
                    root: [joints[0][0], joints[0][1], joints[0][2]],
                });
            }
        }

        // Fetch a batch from server when local queue is running low
        if (this.localFrameQueue.length < this.batchSize && !this.isFetchingFrame) {
            this.isFetchingFrame = true;

            let url = `/api/get_frame?session_id=${this.sessionId}&count=${this.batchSize}`;
            if (this.broadcastLastId > 0) {
                url += `&after_id=${this.broadcastLastId}`;
            }
            fetch(url)
                .then(response => response.json())
                .then(data => {
                    if (data.status === 'success') {
                        for (const frame of data.frames) {
                            this.localFrameQueue.push(frame);
                        }
                        if (data.last_id !== undefined) {
                            this.broadcastLastId = data.last_id;
                        }
                        this.consecutiveWaiting = 0;
                    } else if (data.status === 'waiting') {
                        this.consecutiveWaiting++;
                    }
                })
                .catch(error => {
                    console.error('Error fetching frames:', error);
                })
                .finally(() => {
                    this.isFetchingFrame = false;
                });
        }

        // Use requestAnimationFrame for continuous checking
        requestAnimationFrame(() => this.fetchFrame());
    }
    
    updateAutoFollow() {
        const timeSinceInteraction = Date.now() - this.lastUserInteraction;
        
        // Auto-follow if user hasn't interacted for more than 3 seconds
        if (timeSinceInteraction > this.autoFollowDelay) {
            // Calculate camera offset relative to current target
            const currentOffset = new THREE.Vector3().subVectors(
                this.camera.position, 
                this.controls.target
            );
            
            // New target position (character position, waist height)
            const newTarget = this.currentRootPos.clone();
            newTarget.y = 1.0;
            
            // Calculate new camera position (maintain relative offset)
            const newCameraPos = newTarget.clone().add(currentOffset);
            
            // Smooth interpolation follow (increased lerp factor for more obvious following)
            // 0.2 = more aggressive following, 0.05 = gentle following
            this.controls.target.lerp(newTarget, 0.2);
            this.camera.position.lerp(newCameraPos, 0.2);
            
            // Debug log (comment out in production)
            // console.log('Auto-follow active, tracking:', newTarget);
        }
    }
    
    _resetUIToIdle() {
        this.isRunning = false;
        this.isPaused = false;
        this.isIdle = true;
        this.isWatching = false;
        this.frameCount = 0;
        this.motionFpsCounter = 0;
        this.isFetchingFrame = false;
        this.consecutiveWaiting = 0;
        this.localFrameQueue = [];
        this.broadcastLastId = 0;
        this.startResetBtn.textContent = 'Start';
        this.startResetBtn.classList.remove('btn-danger');
        this.startResetBtn.classList.add('btn-primary');
        this.updateBtn.disabled = true;
        this.pauseResumeBtn.disabled = true;
        this.pauseResumeBtn.textContent = 'Pause';
        this.pauseResumeBtn.classList.remove('btn-success');
        this.pauseResumeBtn.classList.add('btn-warning');
        this.statusEl.textContent = 'Idle';
        this.bufferSizeEl.textContent = '0 / 4';
        this.frameCountEl.textContent = '0';
        this.fpsEl.textContent = '0';
        if (this.skeleton) this.skeleton.clearTrail();
    }

    // --- Config Editor ---

    async openConfigEditor() {
        try {
            const response = await fetch('/api/config');
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            const data = await response.json();
            if (data.status === 'error') throw new Error(data.message);

            // Render schedule_config fields
            this.renderConfigSection('schedule_config', data.schedule_config,
                document.getElementById('scheduleConfigFields'));

            // Render cfg_config fields
            this.renderConfigSection('cfg_config', data.cfg_config,
                document.getElementById('cfgConfigFields'));

            // Populate runtime params
            document.getElementById('modalHistoryLength').value = data.history_length;
            const slider = document.getElementById('modalSmoothingAlpha');
            slider.value = data.smoothing_alpha;
            document.getElementById('modalSmoothingValue').textContent =
                parseFloat(data.smoothing_alpha).toFixed(2);

            // Show modal
            document.getElementById('configModal').style.display = 'flex';
        } catch (error) {
            console.error('Error opening config editor:', error);
            alert('Failed to load config: ' + error.message);
        }
    }

    renderConfigSection(sectionName, obj, container) {
        container.innerHTML = '';
        for (const [key, value] of Object.entries(obj)) {
            const field = document.createElement('div');
            field.className = 'config-field';

            const label = document.createElement('label');
            label.textContent = key;
            field.appendChild(label);

            let input;
            // Length keys get a slider (paper-faithful exposure of the model's
            // length input — `--motion_length` for BAMM, `--length` for MMM).
            if (key === 'motion_length' || key === 'length') {
                const wrap = document.createElement('div');
                wrap.className = 'slider-container';
                input = document.createElement('input');
                input.type = 'range';
                input.min = (key === 'motion_length') ? 0 : 40;
                input.max = 196;
                input.step = 4;
                input.value = value;
                const valSpan = document.createElement('span');
                valSpan.className = 'slider-value';
                valSpan.textContent = value;
                input.addEventListener('input', () => { valSpan.textContent = input.value; });
                wrap.appendChild(input);
                wrap.appendChild(valSpan);
                input.dataset.section = sectionName;
                input.dataset.key = key;
                input.dataset.type = 'number';
                input.className = 'config-input';
                field.appendChild(wrap);
                if (key === 'motion_length') {
                    const hint = document.createElement('span');
                    hint.className = 'param-hint';
                    hint.textContent = '0 = use length predictor; otherwise frames (multiple of 4)';
                    field.appendChild(hint);
                }
                container.appendChild(field);
                continue;
            }
            if (typeof value === 'boolean') {
                input = document.createElement('select');
                input.innerHTML =
                    `<option value="true" ${value ? 'selected' : ''}>true</option>` +
                    `<option value="false" ${!value ? 'selected' : ''}>false</option>`;
            } else {
                input = document.createElement('input');
                input.type = typeof value === 'number' ? 'number' : 'text';
                if (typeof value === 'number' && !Number.isInteger(value)) {
                    input.step = 'any';
                }
                input.value = value;
            }
            input.dataset.section = sectionName;
            input.dataset.key = key;
            input.dataset.type = typeof value;
            input.className = 'config-input';
            field.appendChild(input);

            container.appendChild(field);
        }
    }

    async saveConfigAndReset() {
        try {
            // Collect config values from dynamically rendered fields
            const scheduleConfig = {};
            const cfgConfig = {};

            document.querySelectorAll('.config-input').forEach(input => {
                const section = input.dataset.section;
                const key = input.dataset.key;
                const type = input.dataset.type;

                let value;
                if (type === 'boolean') {
                    value = input.value === 'true';
                } else if (type === 'number') {
                    value = Number(input.value);
                } else {
                    value = input.value;
                }

                if (section === 'schedule_config') {
                    scheduleConfig[key] = value;
                } else if (section === 'cfg_config') {
                    cfgConfig[key] = value;
                }
            });

            const historyLength = parseInt(document.getElementById('modalHistoryLength').value);
            const smoothingAlpha = parseFloat(document.getElementById('modalSmoothingAlpha').value);

            const response = await fetch('/api/config', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    schedule_config: scheduleConfig,
                    cfg_config: cfgConfig,
                    history_length: historyLength,
                    smoothing_alpha: smoothingAlpha,
                })
            });

            const data = await response.json();

            if (data.status === 'success') {
                this.historyLengthValue = historyLength;
                this.smoothingAlphaValue = smoothingAlpha;
                this._resetUIToIdle();
                this.closeConfigEditor();
                console.log('Config updated and reset complete');
            } else {
                alert('Error: ' + data.message);
            }
        } catch (error) {
            console.error('Error saving config:', error);
            alert('Failed to save config: ' + error.message);
        }
    }

    closeConfigEditor() {
        document.getElementById('configModal').style.display = 'none';
    }

    // Swap between the rigged Soldier and the procedural SMPL mannequin.
    // Disposes the current renderer and instantiates the other; replays the
    // last drawn pose so the swap is visually continuous when a generation
    // is running. (For Soldier, the glTF loads async — its constructor snaps
    // to STANDING_POSE once ready, and the next streamed frame overwrites
    // within ~50 ms at 20 FPS.)
    switchCharacter() {
        // Cycle: skeleton → soldier → smpl → skeleton
        const order = ['skeleton', 'soldier', 'smpl'];
        const ctors = { skeleton: Skeleton3D, soldier: MeshBody, smpl: SmplBody };
        const labels = { skeleton: 'Skeleton', soldier: 'Soldier', smpl: 'SMPL' };
        const idx = order.indexOf(this.currentCharacter);
        const next = order[(idx + 1) % order.length];
        const after = order[(idx + 2) % order.length];

        this.skeleton.dispose();
        this.skeleton = new ctors[next](this.scene);
        this.currentCharacter = next;
        if (this.characterCurrentEl) {
            this.characterCurrentEl.textContent = labels[next];
        }
        if (this.characterToggleBtn) {
            this.characterToggleBtn.textContent = `Switch to ${labels[after]}`;
        }
        if (this.lastJointFrame) {
            this.skeleton.updatePose(this.lastJointFrame);
        }
    }

    async updateStatus() {
        try {
            const response = await fetch(`/api/status?session_id=${this.sessionId}`);
            const data = await response.json();
            
            if (data.initialized) {
                this.bufferSizeEl.textContent = `${data.buffer_size} / ${data.target_size}`;

                // Update current smoothing display
                if (data.smoothing_alpha !== undefined) {
                    this.currentSmoothing.textContent = data.smoothing_alpha.toFixed(2);
                }

                // Update current history length display
                if (data.history_length !== undefined) {
                    this.currentHistory.textContent = data.history_length;
                }

                // Auto-start spectator mode if someone else is generating
                if (data.is_generating && !data.is_active_session && this.isIdle && !this.isWatching) {
                    this.isWatching = true;
                    this.isRunning = true;
                    this.statusEl.textContent = 'Watching';
                    this.startResetBtn.textContent = 'Take Over';
                    this.startResetBtn.classList.remove('btn-danger');
                    this.startResetBtn.classList.add('btn-primary');
                    this.startFrameLoop();
                }
                // Stop spectator mode when generation stops
                if (!data.is_generating && !data.is_active_session && this.isWatching) {
                    this.isWatching = false;
                    this.isRunning = false;
                    this.isIdle = true;
                    this.statusEl.textContent = 'Idle';
                    this.startResetBtn.textContent = 'Start';
                    this.localFrameQueue = [];
                    this.broadcastLastId = 0;
                }
            }
            
            // Update motion FPS (frame consumption rate)
            const now = performance.now();
            if (now - this.motionFpsUpdateTime > 1000) {
                this.fpsEl.textContent = this.motionFpsCounter;
                this.motionFpsCounter = 0;
                this.motionFpsUpdateTime = now;
            }
        } catch (error) {
            // Silently fail for status updates
        }
        
        // Update status every 500ms
        setTimeout(() => this.updateStatus(), 500);
    }
    
    animate() {
        requestAnimationFrame(() => this.animate());
        
        // Update controls
        this.controls.update();
        
        // Render scene
        this.renderer.render(this.scene, this.camera);
    }
    
    onWindowResize() {
        const container = document.getElementById('canvas-container');
        this.camera.aspect = container.clientWidth / container.clientHeight;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(container.clientWidth, container.clientHeight);
    }
}

// Initialize app when page loads
window.addEventListener('DOMContentLoaded', () => {
    window.app = new MotionApp();
});

