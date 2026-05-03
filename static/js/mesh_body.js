/**
 * MeshBody: drives a rigged glTF (Soldier.glb) from a 22-joint HumanML3D stream.
 * Same public surface as Skeleton3D so main.js can swap by changing the constructor.
 */

class MeshBody {
    constructor(scene) {
        this.scene = scene;
        this.modelLoaded = false;
        this.root = null;
        this.bones = {};            // jointIdx -> THREE.Bone
        this.bindWorldQuat = {};    // jointIdx -> THREE.Quaternion
        this.bindDirs = {};         // jointIdx -> THREE.Vector3 (unit world dir to target child)
        this.bindHipsOffset = new THREE.Vector3();  // hips world pos when root.position == 0
        this.updateSpecs = [];      // [{own, target, bone}, ...] in topological order

        // HumanML3D 22-joint -> Soldier (Mixamo) bone name
        this.boneNameMap = {
            0:  'mixamorig:Hips',
            1:  'mixamorig:LeftUpLeg',
            2:  'mixamorig:RightUpLeg',
            3:  'mixamorig:Spine',
            4:  'mixamorig:LeftLeg',
            5:  'mixamorig:RightLeg',
            6:  'mixamorig:Spine1',
            7:  'mixamorig:LeftFoot',
            8:  'mixamorig:RightFoot',
            9:  'mixamorig:Spine2',
            10: 'mixamorig:LeftToeBase',
            11: 'mixamorig:RightToeBase',
            12: 'mixamorig:Neck',
            13: 'mixamorig:LeftShoulder',
            14: 'mixamorig:RightShoulder',
            15: 'mixamorig:Head',
            16: 'mixamorig:LeftArm',
            17: 'mixamorig:RightArm',
            18: 'mixamorig:LeftForeArm',
            19: 'mixamorig:RightForeArm',
            20: 'mixamorig:LeftHand',
            21: 'mixamorig:RightHand',
        };

        // Aim-at-child specs: bone for ownJoint is rotated to point at targetJoint.
        // Two-axis only — loses twist around the bone direction. Fine for limbs and the
        // upper spine. Hips is handled separately by a 3-axis frame (see frameDefs).
        this.updateOrder = [
            [3, 6],            // Spine -> Spine2
            [6, 9],            // Spine1 -> Spine3
            [9, 12],           // Spine2 -> Neck
            [12, 15],          // Neck -> Head
            [13, 16], [16, 18], [18, 20],  // L: Collar -> Shoulder -> Elbow -> Wrist
            [14, 17], [17, 19], [19, 21],  // R: Collar -> Shoulder -> Elbow -> Wrist
            [1, 4], [4, 7], [7, 10],       // L leg: Hip -> Knee -> Ankle -> Toe
            [2, 5], [5, 8], [8, 11],       // R leg
        ];

        // Yaw specs: bone gets a Y-axis rotation derived from a horizontal "right" vector.
        // Pitch/roll is left to aim-at-child on descendants. Used to recover the yaw
        // that aim-at-child loses when a bone is roughly vertical (Hips → Spine).
        // joints = [left, right] — the across-body horizontal direction.
        this.yawDefs = [
            { own: 0, joints: [1, 2] },   // Hips: L_hip → R_hip
        ];
        this.yawSpecs = [];

        // Trail (root path on the ground)
        this.trailPoints = [];
        this.maxTrailPoints = 200;
        this.trailLine = null;
        this.trailGeometry = null;
        this.trailMaterial = null;

        this.initTrail();
        this.loadModel();
    }

    loadModel() {
        const loader = new THREE.GLTFLoader();
        loader.load('/static/models/Soldier.glb', (gltf) => {
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

            // Find the bones we care about. Three.js r128 sanitizes ":" -> "_" in node
            // names, so match against both forms.
            const normalize = (s) => s.replace(/[:._\/\\\[\]]/g, '');
            const allBones = [];
            this.root.traverse((obj) => {
                if (obj.isBone) {
                    allBones.push(obj.name);
                    const n = normalize(obj.name);
                    for (const [jIdx, name] of Object.entries(this.boneNameMap)) {
                        if (n === normalize(name)) this.bones[+jIdx] = obj;
                    }
                }
            });
            const matched = Object.keys(this.bones).length;
            console.log(`MeshBody: matched ${matched}/22 bones`);
            if (matched < 22) {
                const missing = Object.entries(this.boneNameMap)
                    .filter(([j, _]) => !this.bones[+j])
                    .map(([j, n]) => `${j}:${n}`);
                console.warn('MeshBody: missing bones', missing.join(', '));
                console.warn('MeshBody: bones in rig:\n' + allBones.join('\n'));
            }

            // Capture bind-pose world transforms (T-pose)
            this.root.updateMatrixWorld(true);
            for (const [jIdx, bone] of Object.entries(this.bones)) {
                this.bindWorldQuat[+jIdx] = bone.getWorldQuaternion(new THREE.Quaternion());
            }

            // For each (own -> target), capture in PARENT-LOCAL space so children inherit
            // their parent's runtime rotation (yaw, etc.) rather than fighting it.
            for (const [own, target] of this.updateOrder) {
                const ownBone = this.bones[own];
                const targetBone = this.bones[target];
                if (!ownBone || !targetBone || !ownBone.parent) {
                    console.warn(`MeshBody: missing bone for ${own}->${target}, skipping`);
                    continue;
                }
                const ownPos = ownBone.getWorldPosition(new THREE.Vector3());
                const targetPos = targetBone.getWorldPosition(new THREE.Vector3());
                const dirWorld = new THREE.Vector3().subVectors(targetPos, ownPos);
                if (dirWorld.lengthSq() < 1e-10) {
                    console.warn(`MeshBody: zero bind dir for ${own}->${target}, skipping`);
                    continue;
                }
                // Rotate bind world direction into parent's bind frame
                const parentBindWorldInv = ownBone.parent
                    .getWorldQuaternion(new THREE.Quaternion())
                    .invert();
                const bindDirInParent = dirWorld.clone()
                    .applyQuaternion(parentBindWorldInv)
                    .normalize();
                // Bone's bind LOCAL quat is just whatever's in bone.quaternion at load
                const bindLocalQuat = ownBone.quaternion.clone();
                this.updateSpecs.push({
                    own, target, bone: ownBone,
                    bindDirInParent, bindLocalQuat,
                });
            }

            // Hips world offset when root sits at origin (used to place root each frame
            // so the rendered Hips coincides with joints[0])
            if (this.bones[0]) {
                this.bindHipsOffset.copy(this.bones[0].getWorldPosition(new THREE.Vector3()));
                this.bindHipsOffset.sub(this.root.position);
            }

            // Yaw specs: capture the bind-pose horizontal "right" vector for each.
            for (const def of this.yawDefs) {
                const ji = def.joints;
                const bone = this.bones[def.own];
                const bL = this.bones[ji[0]];
                const bR = this.bones[ji[1]];
                if (!bone || !bL || !bR) {
                    console.warn(`MeshBody: missing bone(s) for yaw spec own=${def.own}`);
                    continue;
                }
                const lPos = bL.getWorldPosition(new THREE.Vector3());
                const rPos = bR.getWorldPosition(new THREE.Vector3());
                const dir = new THREE.Vector3(rPos.x - lPos.x, 0, rPos.z - lPos.z);
                if (dir.lengthSq() < 1e-10) {
                    console.warn(`MeshBody: degenerate bind yaw vector for own=${def.own}`);
                    continue;
                }
                this.yawSpecs.push({
                    own: def.own,
                    joints: ji,
                    bone,
                    bindRightXZ: dir.normalize(),
                });
            }

            this.modelLoaded = true;
            console.log(`MeshBody loaded with ${this.updateSpecs.length} bone updates`);

            // Snap to a neutral standing pose so the soldier never appears in
            // T-pose between page-load and the first generated frame.
            this.updatePose(MeshBody.STANDING_POSE);
        }, undefined, (err) => {
            console.error('MeshBody: failed to load Soldier.glb', err);
        });
    }

    updatePose(jointPositions) {
        if (!this.modelLoaded) return;
        if (!jointPositions || jointPositions.length !== 22) {
            console.error('Invalid joint positions:', jointPositions);
            return;
        }

        const j0 = jointPositions[0];

        // Place model so Hips is at joints[0] (Hips' world pos = root.pos + bindHipsOffset
        // since Hips is at the top of the bone hierarchy and only its rotation changes)
        this.root.position.set(
            j0[0] - this.bindHipsOffset.x,
            j0[1] - this.bindHipsOffset.y,
            j0[2] - this.bindHipsOffset.z
        );
        this.root.updateMatrixWorld(true);

        const tmpDir = new THREE.Vector3();
        const tmpDelta = new THREE.Quaternion();
        const tmpParentWorld = new THREE.Quaternion();
        const tmpBoneWorld = new THREE.Quaternion();

        // Yaw bones first: pure Y-axis rotation from horizontal "right" vector.
        const tmpYawDelta = new THREE.Quaternion();
        const tmpLiveRight = new THREE.Vector3();
        for (const spec of this.yawSpecs) {
            const jL = jointPositions[spec.joints[0]];
            const jR = jointPositions[spec.joints[1]];
            tmpLiveRight.set(jR[0] - jL[0], 0, jR[2] - jL[2]);
            const lenSq = tmpLiveRight.lengthSq();
            if (lenSq < 1e-8) continue;
            tmpLiveRight.multiplyScalar(1.0 / Math.sqrt(lenSq));

            // Both vectors are in the XZ plane, so this rotation is purely around Y.
            tmpYawDelta.setFromUnitVectors(spec.bindRightXZ, tmpLiveRight);
            // boneWorld = yawDelta * bindWorldQuat
            tmpBoneWorld.copy(tmpYawDelta).multiply(this.bindWorldQuat[spec.own]);
            // bone.quat = parentWorld^-1 * boneWorld
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

            // Rotate live world direction into parent's CURRENT frame (so any rotation
            // we've already applied to the parent — e.g., Hips yaw — flows through).
            bone.parent.getWorldQuaternion(tmpParentWorld);
            tmpParentWorld.invert();
            tmpDir.applyQuaternion(tmpParentWorld);

            // Delta in parent-local frame: bind direction → live direction
            tmpDelta.setFromUnitVectors(bindDirInParent, tmpDir);

            // bone.local = delta * bindLocalQuat (preserves bind twist)
            bone.quaternion.copy(tmpDelta).multiply(bindLocalQuat);
            bone.updateMatrixWorld(true);
        }

        this.updateTrail(jointPositions[0]);
    }

    _asVec(arr) {
        return new THREE.Vector3(arr[0], arr[1], arr[2]);
    }

    // Right-handed orthonormal frame from 4 points: center, left, right, up.
    // x = (right - left)            -- across the body
    // y = re-orthogonalised "up"
    // z = right × up                 -- forward (where the body faces)
    _frameQuat(pCenter, pLeft, pRight, pUp) {
        const xAxis = new THREE.Vector3().subVectors(pRight, pLeft);
        if (xAxis.lengthSq() < 1e-10) return new THREE.Quaternion();
        xAxis.normalize();
        const upApprox = new THREE.Vector3().subVectors(pUp, pCenter);
        if (upApprox.lengthSq() < 1e-10) return new THREE.Quaternion();
        upApprox.normalize();
        const zAxis = new THREE.Vector3().crossVectors(xAxis, upApprox);
        if (zAxis.lengthSq() < 1e-10) return new THREE.Quaternion();
        zAxis.normalize();
        const yAxis = new THREE.Vector3().crossVectors(zAxis, xAxis).normalize();
        const m = new THREE.Matrix4().makeBasis(xAxis, yAxis, zAxis);
        return new THREE.Quaternion().setFromRotationMatrix(m);
    }

    // ---- trail (copied from Skeleton3D) ----

    initTrail() {
        this.trailGeometry = new THREE.BufferGeometry();
        const positions = new Float32Array(this.maxTrailPoints * 3);
        const colors = new Float32Array(this.maxTrailPoints * 4);
        this.trailGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        this.trailGeometry.setAttribute('color', new THREE.BufferAttribute(colors, 4));
        this.trailMaterial = new THREE.LineBasicMaterial({
            vertexColors: true,
            transparent: true,
            opacity: 1.0,
            linewidth: 2,
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
            if ((dx * dx + dz * dz) > 0.0004) {  // 0.02^2
                this.trailPoints.push(trailPoint);
            }
        }
        if (this.trailPoints.length > this.maxTrailPoints) this.trailPoints.shift();

        const positions = this.trailGeometry.attributes.position.array;
        const colors = this.trailGeometry.attributes.color.array;
        const numPoints = this.trailPoints.length;
        for (let i = 0; i < this.maxTrailPoints; i++) {
            if (i < numPoints) {
                const p = this.trailPoints[i];
                positions[i * 3]     = p.x;
                positions[i * 3 + 1] = p.y;
                positions[i * 3 + 2] = p.z;
                const alpha = numPoints > 1 ? i / (numPoints - 1) : 1;
                const opacity = Math.pow(alpha, 1.5) * 0.8;
                colors[i * 4]     = 0.0;
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

// Hardcoded neutral standing pose in HumanML3D 22-joint world coords. Used to
// pre-pose the rig at load time and as a fallback "rest" pose. Approximate
// dimensions of a 1.6 m human standing at the origin facing +Z.
MeshBody.STANDING_POSE = [
    [ 0.00, 0.95, 0.00],   // 0  Pelvis
    [ 0.10, 0.85, 0.00],   // 1  L_Hip
    [-0.10, 0.85, 0.00],   // 2  R_Hip
    [ 0.00, 1.05, 0.00],   // 3  Spine1
    [ 0.10, 0.50, 0.00],   // 4  L_Knee
    [-0.10, 0.50, 0.00],   // 5  R_Knee
    [ 0.00, 1.20, 0.00],   // 6  Spine2
    [ 0.10, 0.10, 0.00],   // 7  L_Ankle
    [-0.10, 0.10, 0.00],   // 8  R_Ankle
    [ 0.00, 1.35, 0.00],   // 9  Spine3
    [ 0.10, 0.00, 0.10],   // 10 L_Foot
    [-0.10, 0.00, 0.10],   // 11 R_Foot
    [ 0.00, 1.45, 0.00],   // 12 Neck
    [ 0.08, 1.42, 0.00],   // 13 L_Collar
    [-0.08, 1.42, 0.00],   // 14 R_Collar
    [ 0.00, 1.60, 0.00],   // 15 Head
    [ 0.18, 1.40, 0.00],   // 16 L_Shoulder
    [-0.18, 1.40, 0.00],   // 17 R_Shoulder
    [ 0.20, 1.10, 0.00],   // 18 L_Elbow
    [-0.20, 1.10, 0.00],   // 19 R_Elbow
    [ 0.20, 0.85, 0.00],   // 20 L_Wrist
    [-0.20, 0.85, 0.00],   // 21 R_Wrist
];

window.MeshBody = MeshBody;
