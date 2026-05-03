/**
 * SmplBody: drives a rigged glTF (smpl_neutral.glb) from a 22-joint
 * HumanML3D stream — the same approach as MeshBody but with SMPL bone
 * names. Falls back to a procedural mannequin if the asset is missing.
 */

class SmplBody {
    constructor(scene) {
        this.scene = scene;
        this.modelLoaded = false;
        this.usingFallback = false;
        this.root = null;
        this.bones = {};
        this.bindWorldQuat = {};
        this.bindHipsOffset = new THREE.Vector3();
        this.updateSpecs = [];
        this.yawSpecs = [];

        // HumanML3D 22 → SMPL bone names. The SMPL Blender add-on exports
        // bones as Pelvis / L_Hip / R_Hip / Spine1 / etc. The matcher
        // normalises (lowercase + strip separators) so 'L_Hip' and 'lhip'
        // and 'left_hip' all collapse to the same key — meaning this map
        // works for both the Blender add-on output and SMPL canonical
        // conventions provided one of the aliases below appears.
        this.boneNameMap = {
            0:  ['Pelvis', 'pelvis'],
            1:  ['L_Hip', 'left_hip', 'lefthip'],
            2:  ['R_Hip', 'right_hip', 'righthip'],
            3:  ['Spine1', 'spine1'],
            4:  ['L_Knee', 'left_knee', 'leftknee'],
            5:  ['R_Knee', 'right_knee', 'rightknee'],
            6:  ['Spine2', 'spine2'],
            7:  ['L_Ankle', 'left_ankle', 'leftankle'],
            8:  ['R_Ankle', 'right_ankle', 'rightankle'],
            9:  ['Spine3', 'spine3'],
            10: ['L_Foot', 'left_foot', 'leftfoot'],
            11: ['R_Foot', 'right_foot', 'rightfoot'],
            12: ['Neck', 'neck'],
            13: ['L_Collar', 'left_collar', 'leftcollar'],
            14: ['R_Collar', 'right_collar', 'rightcollar'],
            15: ['Head', 'head'],
            16: ['L_Shoulder', 'left_shoulder', 'leftshoulder'],
            17: ['R_Shoulder', 'right_shoulder', 'rightshoulder'],
            18: ['L_Elbow', 'left_elbow', 'leftelbow'],
            19: ['R_Elbow', 'right_elbow', 'rightelbow'],
            20: ['L_Wrist', 'left_wrist', 'leftwrist'],
            21: ['R_Wrist', 'right_wrist', 'rightwrist'],
        };

        // Same parent→child chains as MeshBody.updateOrder
        this.updateOrder = [
            [3, 6], [6, 9], [9, 12], [12, 15],
            [13, 16], [16, 18], [18, 20],
            [14, 17], [17, 19], [19, 21],
            [1, 4], [4, 7], [7, 10],
            [2, 5], [5, 8], [8, 11],
        ];

        this.yawDefs = [{ own: 0, joints: [1, 2] }];

        this.trailPoints = [];
        this.maxTrailPoints = 200;
        this.trailLine = null;
        this.trailGeometry = null;
        this.trailMaterial = null;

        this.initTrail();
        this.loadModel();
    }

    loadModel() {
        const url = '/static/models/smpl_neutral.glb';
        const loader = new THREE.GLTFLoader();
        loader.load(
            url,
            (gltf) => this.onGltfLoaded(gltf),
            undefined,
            (err) => {
                console.warn(
                    `SmplBody: failed to load ${url} — falling back to procedural ` +
                    `mannequin. Drop a SMPL-rigged .glb at FloodDiffusion/static/models/` +
                    `smpl_neutral.glb to use a real anatomical mesh. Error:`, err
                );
                this.buildFallbackMannequin();
            }
        );
    }

    onGltfLoaded(gltf) {
        this.root = gltf.scene;
        this.root.traverse((obj) => {
            if (obj.isMesh) {
                obj.castShadow = true;
                obj.receiveShadow = true;
            }
            if (obj.isSkinnedMesh) {
                obj.frustumCulled = false;
            }
        });
        this.scene.add(this.root);

        const normalize = (s) =>
            s.toLowerCase().replace(/[:._\/\\\[\]\s-]/g, '');
        // Some SMPL exports prefix bones with things like 'm_avg_' or 'Bone_';
        // strip a small set of common prefixes before comparing.
        const stripPrefix = (s) => {
            const prefixes = ['mavg', 'mixamorig', 'bone', 'smpl', 'b'];
            for (const p of prefixes) {
                if (s.startsWith(p) && s.length > p.length) return s.slice(p.length);
            }
            return s;
        };

        const allBones = [];
        this.root.traverse((obj) => {
            if (!obj.isBone) return;
            allBones.push(obj.name);
            const n = stripPrefix(normalize(obj.name));
            for (const [jIdx, aliases] of Object.entries(this.boneNameMap)) {
                const aliasList = Array.isArray(aliases) ? aliases : [aliases];
                for (const a of aliasList) {
                    if (n === stripPrefix(normalize(a))) {
                        this.bones[+jIdx] = obj;
                        break;
                    }
                }
            }
        });
        const matched = Object.keys(this.bones).length;
        console.log(`SmplBody: matched ${matched}/22 bones`);
        if (matched < 22) {
            const missing = Object.entries(this.boneNameMap)
                .filter(([j]) => !this.bones[+j])
                .map(([j, aliases]) => `${j}:${Array.isArray(aliases) ? aliases[0] : aliases}`);
            console.warn('SmplBody: missing bones:', missing.join(', '));
            console.warn('SmplBody: bones in rig:\n' + allBones.join('\n'));
        }
        if (matched < 18) {
            console.warn(
                'SmplBody: too few bones matched — falling back to procedural mannequin. ' +
                'Adjust the boneNameMap in static/js/smpl_body.js to match your rig.'
            );
            this.scene.remove(this.root);
            this.buildFallbackMannequin();
            return;
        }

        // Capture bind-pose world quaternions
        this.root.updateMatrixWorld(true);
        for (const [jIdx, bone] of Object.entries(this.bones)) {
            this.bindWorldQuat[+jIdx] = bone.getWorldQuaternion(new THREE.Quaternion());
        }

        // Aim-at-child specs in parent-local space (same math as MeshBody)
        for (const [own, target] of this.updateOrder) {
            const ownBone = this.bones[own];
            const targetBone = this.bones[target];
            if (!ownBone || !targetBone || !ownBone.parent) continue;
            const ownPos = ownBone.getWorldPosition(new THREE.Vector3());
            const targetPos = targetBone.getWorldPosition(new THREE.Vector3());
            const dirWorld = new THREE.Vector3().subVectors(targetPos, ownPos);
            if (dirWorld.lengthSq() < 1e-10) continue;
            const parentBindWorldInv = ownBone.parent
                .getWorldQuaternion(new THREE.Quaternion()).invert();
            const bindDirInParent = dirWorld.clone()
                .applyQuaternion(parentBindWorldInv).normalize();
            const bindLocalQuat = ownBone.quaternion.clone();
            this.updateSpecs.push({
                own, target, bone: ownBone, bindDirInParent, bindLocalQuat,
            });
        }

        if (this.bones[0]) {
            this.bindHipsOffset.copy(this.bones[0].getWorldPosition(new THREE.Vector3()));
            this.bindHipsOffset.sub(this.root.position);
        }

        for (const def of this.yawDefs) {
            const bone = this.bones[def.own];
            const bL = this.bones[def.joints[0]];
            const bR = this.bones[def.joints[1]];
            if (!bone || !bL || !bR) continue;
            const lPos = bL.getWorldPosition(new THREE.Vector3());
            const rPos = bR.getWorldPosition(new THREE.Vector3());
            const dir = new THREE.Vector3(rPos.x - lPos.x, 0, rPos.z - lPos.z);
            if (dir.lengthSq() < 1e-10) continue;
            this.yawSpecs.push({
                own: def.own, joints: def.joints, bone,
                bindRightXZ: dir.normalize(),
            });
        }

        this.modelLoaded = true;
        console.log(`SmplBody loaded with ${this.updateSpecs.length} bone updates`);
        this.updatePose(window.MeshBody.STANDING_POSE);
    }

    updatePose(jointPositions) {
        if (this.usingFallback) return this._fallbackUpdatePose(jointPositions);
        if (!this.modelLoaded) return;
        if (!jointPositions || jointPositions.length !== 22) return;

        const j0 = jointPositions[0];
        this.root.position.set(
            j0[0] - this.bindHipsOffset.x,
            j0[1] - this.bindHipsOffset.y,
            j0[2] - this.bindHipsOffset.z,
        );
        this.root.updateMatrixWorld(true);

        const tmpDir = new THREE.Vector3();
        const tmpDelta = new THREE.Quaternion();
        const tmpParentWorld = new THREE.Quaternion();
        const tmpBoneWorld = new THREE.Quaternion();
        const tmpYawDelta = new THREE.Quaternion();
        const tmpLiveRight = new THREE.Vector3();

        for (const spec of this.yawSpecs) {
            const jL = jointPositions[spec.joints[0]];
            const jR = jointPositions[spec.joints[1]];
            tmpLiveRight.set(jR[0] - jL[0], 0, jR[2] - jL[2]);
            const lenSq = tmpLiveRight.lengthSq();
            if (lenSq < 1e-8) continue;
            tmpLiveRight.multiplyScalar(1.0 / Math.sqrt(lenSq));
            tmpYawDelta.setFromUnitVectors(spec.bindRightXZ, tmpLiveRight);
            tmpBoneWorld.copy(tmpYawDelta).multiply(this.bindWorldQuat[spec.own]);
            spec.bone.parent.getWorldQuaternion(tmpParentWorld);
            tmpParentWorld.invert();
            spec.bone.quaternion.copy(tmpParentWorld.multiply(tmpBoneWorld));
            spec.bone.updateMatrixWorld(true);
        }

        for (const spec of this.updateSpecs) {
            const { own, target, bone, bindDirInParent, bindLocalQuat } = spec;
            const jOwn = jointPositions[own];
            const jTgt = jointPositions[target];
            tmpDir.set(jTgt[0] - jOwn[0], jTgt[1] - jOwn[1], jTgt[2] - jOwn[2]);
            const lenSq = tmpDir.lengthSq();
            if (lenSq < 1e-8) continue;
            tmpDir.multiplyScalar(1.0 / Math.sqrt(lenSq));
            bone.parent.getWorldQuaternion(tmpParentWorld);
            tmpParentWorld.invert();
            tmpDir.applyQuaternion(tmpParentWorld);
            tmpDelta.setFromUnitVectors(bindDirInParent, tmpDir);
            bone.quaternion.copy(tmpDelta).multiply(bindLocalQuat);
            bone.updateMatrixWorld(true);
        }

        this.updateTrail(jointPositions[0]);
    }

    // ---- Procedural fallback (used when smpl_neutral.glb is missing) ----

    buildFallbackMannequin() {
        this.usingFallback = true;
        this.modelLoaded = true;
        this.root = new THREE.Group();
        this.scene.add(this.root);

        this.fallback = { joints: new Array(22), bones: [], head: null };

        const jointGeom = new THREE.SphereGeometry(0.035, 12, 10);
        const mat = new THREE.MeshStandardMaterial({
            color: 0xc9ccd1, roughness: 0.55, metalness: 0.0,
        });
        for (let i = 0; i < 22; i++) {
            const m = new THREE.Mesh(jointGeom, mat);
            m.castShadow = true; m.receiveShadow = true;
            this.root.add(m);
            this.fallback.joints[i] = m;
        }

        this.fallback.boneSegments = [
            [0, 3], [3, 6], [6, 9], [9, 12], [12, 15],
            [12, 13], [13, 16], [16, 18], [18, 20],
            [12, 14], [14, 17], [17, 19], [19, 21],
            [0, 1], [1, 4], [4, 7], [7, 10],
            [0, 2], [2, 5], [5, 8], [8, 11],
        ];
        const trunk = new Set([0, 3, 6, 9, 12]);
        const cylGeom = new THREE.CylinderGeometry(1, 1, 1, 14, 1, false);
        this.fallback.bones = this.fallback.boneSegments.map(([p, c]) => {
            const r = (trunk.has(p) && trunk.has(c)) ? 0.085 : 0.045;
            const m = new THREE.Mesh(cylGeom, mat);
            m.castShadow = true; m.receiveShadow = true;
            m.userData.radius = r;
            this.root.add(m);
            return m;
        });

        const headGeom = new THREE.SphereGeometry(0.11, 18, 14);
        this.fallback.head = new THREE.Mesh(headGeom, mat);
        this.fallback.head.castShadow = true;
        this.fallback.head.receiveShadow = true;
        this.root.add(this.fallback.head);

        this._fallbackUpdatePose(window.MeshBody.STANDING_POSE);
    }

    _fallbackUpdatePose(jointPositions) {
        if (!jointPositions || jointPositions.length !== 22) return;
        const fb = this.fallback;
        for (let i = 0; i < 22; i++) {
            const j = jointPositions[i];
            fb.joints[i].position.set(j[0], j[1], j[2]);
        }
        const yAxis = SmplBody._Y_AXIS;
        const tmpDir = new THREE.Vector3();
        const tmpQuat = new THREE.Quaternion();
        for (let i = 0; i < fb.boneSegments.length; i++) {
            const [p, c] = fb.boneSegments[i];
            const jp = jointPositions[p], jc = jointPositions[c];
            tmpDir.set(jc[0] - jp[0], jc[1] - jp[1], jc[2] - jp[2]);
            const length = tmpDir.length();
            const mesh = fb.bones[i];
            if (length < 1e-6) { mesh.visible = false; continue; }
            mesh.visible = true;
            tmpDir.multiplyScalar(1.0 / length);
            mesh.position.set(
                0.5 * (jp[0] + jc[0]),
                0.5 * (jp[1] + jc[1]),
                0.5 * (jp[2] + jc[2]),
            );
            tmpQuat.setFromUnitVectors(yAxis, tmpDir);
            mesh.quaternion.copy(tmpQuat);
            const r = mesh.userData.radius;
            mesh.scale.set(r, length, r);
        }
        const head = jointPositions[15];
        fb.head.position.set(head[0], head[1] + 0.06, head[2]);
        this.updateTrail(jointPositions[0]);
    }

    // ---- trail / visibility / dispose (shared) ----

    initTrail() {
        this.trailGeometry = new THREE.BufferGeometry();
        const positions = new Float32Array(this.maxTrailPoints * 3);
        const colors = new Float32Array(this.maxTrailPoints * 4);
        this.trailGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        this.trailGeometry.setAttribute('color', new THREE.BufferAttribute(colors, 4));
        this.trailMaterial = new THREE.LineBasicMaterial({
            vertexColors: true, transparent: true, opacity: 1.0, linewidth: 2,
        });
        this.trailLine = new THREE.Line(this.trailGeometry, this.trailMaterial);
        this.trailLine.frustumCulled = false;
        this.scene.add(this.trailLine);
    }

    updateTrail(rootPos) {
        const trailPoint = { x: rootPos[0], y: 0.01, z: rootPos[2] };
        if (this.trailPoints.length === 0) {
            this.trailPoints.push(trailPoint);
        } else {
            const last = this.trailPoints[this.trailPoints.length - 1];
            const dx = trailPoint.x - last.x;
            const dz = trailPoint.z - last.z;
            if ((dx * dx + dz * dz) > 0.0004) this.trailPoints.push(trailPoint);
        }
        if (this.trailPoints.length > this.maxTrailPoints) this.trailPoints.shift();

        const positions = this.trailGeometry.attributes.position.array;
        const colors = this.trailGeometry.attributes.color.array;
        const numPoints = this.trailPoints.length;
        for (let i = 0; i < this.maxTrailPoints; i++) {
            if (i < numPoints) {
                const p = this.trailPoints[i];
                positions[i * 3] = p.x;
                positions[i * 3 + 1] = p.y;
                positions[i * 3 + 2] = p.z;
                const alpha = numPoints > 1 ? i / (numPoints - 1) : 1;
                const opacity = Math.pow(alpha, 1.5) * 0.8;
                colors[i * 4] = 0.0;
                colors[i * 4 + 1] = 0.67;
                colors[i * 4 + 2] = 0.85;
                colors[i * 4 + 3] = opacity;
            } else {
                positions[i * 3] = 0;
                positions[i * 3 + 1] = 0;
                positions[i * 3 + 2] = 0;
                colors[i * 4 + 3] = 0;
            }
        }
        this.trailGeometry.attributes.position.needsUpdate = true;
        this.trailGeometry.attributes.color.needsUpdate = true;
        this.trailGeometry.setDrawRange(0, numPoints);
    }

    clearTrail() {
        this.trailPoints = [];
        this.trailGeometry.setDrawRange(0, 0);
    }

    setVisible(visible) {
        if (this.root) this.root.visible = visible;
        if (this.trailLine) this.trailLine.visible = visible;
    }

    dispose() {
        if (this.root) {
            this.scene.remove(this.root);
            this.root.traverse((obj) => {
                if (obj.geometry) obj.geometry.dispose();
                if (obj.material) {
                    if (Array.isArray(obj.material)) obj.material.forEach(m => m.dispose());
                    else obj.material.dispose();
                }
            });
        }
        if (this.trailLine) {
            this.scene.remove(this.trailLine);
            this.trailGeometry.dispose();
            this.trailMaterial.dispose();
        }
    }
}

SmplBody._Y_AXIS = new THREE.Vector3(0, 1, 0);

window.SmplBody = SmplBody;
