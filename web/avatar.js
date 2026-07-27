/**
 * SMPL のモーションを VRM アバターへリターゲットする（issue #5）。
 *
 * リターゲットに使うのは関節「位置」ではなく「回転」。位置は体格差で破綻するが、
 * 回転なら体型の違うキャラにも移せる。
 *
 * ## レストポーズ差の吸収（この issue の本番）
 *
 * SMPL と VRM は基準姿勢が違うため、回転をそのまま入れると腕がねじれる。
 * ボーンごとの補正 A を挟んで解決する。
 *
 *   G_smpl[i]        SMPL のグローバル回転（FKで算出）
 *   d_rest_smpl[i]   SMPL の基準姿勢での骨の向き
 *   d_rest_vrm[i]    VRM  の基準姿勢での骨の向き
 *   A[i]             d_rest_vrm → d_rest_smpl に回す補正
 *
 *   G_vrm[i] = G_smpl[i] · A[i]
 *     ⇒ G_vrm[i] · d_rest_vrm = G_smpl[i] · d_rest_smpl = 現在の骨の向き ✓
 *
 * d_rest_smpl は外部から与えなくてよい。G[i]⁻¹ · (現在の向き) が全フレームで
 * 一定になる性質を使い、手元のデータから逆算する。
 */
import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { VRMLoaderPlugin } from "@pixiv/three-vrm";

/** SMPL 24関節の親（-1 = 根） */
const PARENT = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9,
                12, 13, 14, 16, 17, 18, 19, 20, 21];

/** SMPL joint → VRM humanoid bone */
const TO_VRM = {
  0: "hips", 1: "leftUpperLeg", 2: "rightUpperLeg", 3: "spine",
  4: "leftLowerLeg", 5: "rightLowerLeg", 6: "chest",
  7: "leftFoot", 8: "rightFoot", 9: "upperChest",
  10: "leftToes", 11: "rightToes", 12: "neck",
  13: "leftShoulder", 14: "rightShoulder", 15: "head",
  16: "leftUpperArm", 17: "rightUpperArm",
  18: "leftLowerArm", 19: "rightLowerArm",
  20: "leftHand", 21: "rightHand",
};

/** 骨の「向き」を決める子関節（joint → この子 への方向を骨の向きとする） */
const DIR_CHILD = {
  0: 3, 1: 4, 2: 5, 3: 6, 4: 7, 5: 8, 6: 9, 7: 10, 8: 11, 9: 12,
  12: 15, 13: 16, 14: 17, 16: 18, 17: 19, 18: 20, 19: 21, 20: 22, 21: 23,
};

const IDS = Object.keys(TO_VRM).map(Number);

function aaToQuat(x, y, z, out = new THREE.Quaternion()) {
  const a = Math.hypot(x, y, z);
  if (a < 1e-8) return out.identity();
  return out.setFromAxisAngle(new THREE.Vector3(x / a, y / a, z / a), a);
}

/** SMPL パラメータから各関節のグローバル回転を求める（FK） */
function globalRotations(pose, f) {
  const go = pose.global_orient[f], bp = pose.body_pose[f];
  const G = new Array(24);
  G[0] = aaToQuat(go[0], go[1], go[2]);
  for (let i = 1; i < 24; i++) {
    // body_pose は pelvis を除く21関節。body_pose[i-1] が joint i に対応。
    const L = i <= 21
      ? aaToQuat(bp[(i - 1) * 3], bp[(i - 1) * 3 + 1], bp[(i - 1) * 3 + 2])
      : new THREE.Quaternion();
    G[i] = G[PARENT[i]].clone().multiply(L);
  }
  return G;
}

/**
 * SMPL の基準姿勢での骨の向きをデータから逆算する。
 * 現在の向き = G[i] · d_rest[i] なので d_rest[i] = G[i]⁻¹ · 現在の向き。
 * 理論上どのフレームでも同じ値になるため、複数フレームで平均して安定させる。
 */
function smplRestDirs(joints, pose) {
  const F = joints.length;
  const samples = [0, F >> 2, F >> 1, (F * 3) >> 2, F - 1].filter(
    (v, i, a) => v >= 0 && v < F && a.indexOf(v) === i);
  const acc = {}, cur = new THREE.Vector3(), inv = new THREE.Quaternion();

  for (const f of samples) {
    const G = globalRotations(pose, f);
    for (const i of IDS) {
      const c = DIR_CHILD[i];
      if (c === undefined) continue;
      const a = joints[f][i], b = joints[f][c];
      cur.set(b[0] - a[0], b[1] - a[1], b[2] - a[2]);
      if (cur.lengthSq() < 1e-12) continue;
      cur.normalize().applyQuaternion(inv.copy(G[i]).invert());
      (acc[i] ||= new THREE.Vector3()).add(cur);
    }
  }
  for (const k in acc) acc[k].normalize();
  return acc;
}

/**
 * 骨の向きの集合から、体の基準フレーム（右・上・前）を作る。
 *
 * VRM と SMPL は基準姿勢の「体の向き」自体が違う（およそ180°ヨー）。これを先に
 * 揃えておかないと、ボーンごとの補正が約180°の最短回転になってしまい、
 * その軸が不定になる ＝ ねじれが暴れる。全体を揃えてから個別補正を出す。
 */
function bodyFrame(dirs) {
  const up = new THREE.Vector3();
  for (const i of [3, 6, 9]) if (dirs[i]) up.add(dirs[i]);   // 背骨は上を向く
  if (up.lengthSq() < 1e-9) up.set(0, 1, 0);
  up.normalize();

  // 左右の上腕（無ければ左右の腿）の差 ＝ 体の「右」方向。上下成分は打ち消される。
  const right = new THREE.Vector3();
  if (dirs[17] && dirs[16]) right.copy(dirs[17]).sub(dirs[16]);
  else if (dirs[2] && dirs[1]) right.copy(dirs[2]).sub(dirs[1]);
  else right.set(1, 0, 0);
  right.sub(up.clone().multiplyScalar(right.dot(up)));        // up と直交化
  if (right.lengthSq() < 1e-9) right.set(1, 0, 0);
  right.normalize();

  const fwd = new THREE.Vector3().crossVectors(right, up).normalize();
  return new THREE.Matrix4().makeBasis(right, up, fwd);
}

/** VRM の基準姿勢（T-pose）での骨の向き。正規化リグから読む。 */
function vrmRestDirs(vrm) {
  const h = vrm.humanoid;
  vrm.scene.updateWorldMatrix(true, true);
  const node = (i) => h.getNormalizedBoneNode(TO_VRM[i]);
  const dirs = {}, pa = new THREE.Vector3(), pb = new THREE.Vector3();

  for (const i of IDS) {
    const c = DIR_CHILD[i];
    if (c === undefined || TO_VRM[c] === undefined) continue;
    const na = node(i), nb = node(c);
    if (!na || !nb) continue;
    na.getWorldPosition(pa);
    nb.getWorldPosition(pb);
    const d = pb.clone().sub(pa);
    if (d.lengthSq() < 1e-12) continue;
    dirs[i] = d.normalize();
  }
  return dirs;
}

/**
 * VRM を読み込み、SMPL のモーションを流し込めるようにする。
 * @returns {Promise<{root:THREE.Object3D, applyFrame:Function, setVisible:Function, report:Object}>}
 */
export async function createAvatar(url, joints, pose) {
  const loader = new GLTFLoader();
  loader.register((parser) => new VRMLoaderPlugin(parser));
  const gltf = await loader.loadAsync(url);
  const vrm = gltf.userData.vrm;
  if (!vrm) throw new Error("VRM として読めませんでした");

  // 影・カリングの都合で frustumCulled を切る（腕を上げると消えることがある）
  vrm.scene.traverse((o) => { o.frustumCulled = false; });

  const dSmpl = smplRestDirs(joints, pose);
  const dVrm = vrmRestDirs(vrm);

  // ① 体の向きを丸ごと揃える W（VRM基準フレーム → SMPL基準フレーム）
  const Mv = bodyFrame(dVrm), Ms = bodyFrame(dSmpl);
  const W = new THREE.Quaternion().setFromRotationMatrix(
    Ms.clone().multiply(Mv.clone().transpose()));  // 回転行列なので逆行列＝転置

  // ② その上でボーンごとの残差を補正する A[i]。①のおかげで小さく安定した回転になる。
  //    最終的な補正 C[i] = A[i]・W（先に W、次に A[i]）
  const A = {};
  const report = { globalYawDeg: 0, corrected: [], fallback: [] };
  report.globalYawDeg = +THREE.MathUtils.radToDeg(2 * Math.acos(
    Math.min(1, Math.abs(W.w)))).toFixed(1);

  for (const i of IDS) {
    if (!dSmpl[i] || !dVrm[i]) continue;
    const aligned = dVrm[i].clone().applyQuaternion(W);
    const resid = new THREE.Quaternion().setFromUnitVectors(aligned, dSmpl[i]);
    A[i] = resid.clone().multiply(W);
    report.corrected.push({
      bone: TO_VRM[i],
      deg: +THREE.MathUtils.radToDeg(aligned.angleTo(dSmpl[i])).toFixed(1),
    });
  }
  // 末端（手・つま先・頭）は子が無く向きを定義できないため、親の補正を借りる
  for (const i of IDS) {
    if (A[i]) continue;
    let p = PARENT[i];
    while (p >= 0 && !A[p]) p = PARENT[p];
    A[i] = (p >= 0 && A[p]) ? A[p].clone() : new THREE.Quaternion();
    report.fallback.push(TO_VRM[i]);
  }

  const Gv = new Array(24);
  const tmp = new THREE.Quaternion();

  function applyFrame(f, hipsPos) {
    const G = globalRotations(pose, f);
    for (const i of IDS) Gv[i] = G[i].clone().multiply(A[i]);

    for (const i of IDS) {
      const node = vrm.humanoid.getNormalizedBoneNode(TO_VRM[i]);
      if (!node) continue;
      if (i === 0) {
        node.quaternion.copy(Gv[0]);
        if (hipsPos) node.position.copy(hipsPos);
      } else {
        // ローカル回転 = 親のグローバル⁻¹ · 自分のグローバル
        node.quaternion.copy(tmp.copy(Gv[PARENT[i]]).invert().multiply(Gv[i]));
      }
    }
    vrm.humanoid.update();
    // 髪などの揺れ物は「連続した時間」を前提にしている。ここはスライダーで
    // 任意のフレームへ飛ぶ使い方なので、毎回レストに戻して暴れるのを防ぐ。
    vrm.springBoneManager?.reset();
  }

  /**
   * ラケットを利き手に持たせる（見た目だけ）。
   * ラケット自体は追跡していないので、前腕の延長方向に固定するだけの飾り。
   */
  function attachRacket(side = "right") {
    const h = vrm.humanoid;
    const hand = (h.getRawBoneNode ? h.getRawBoneNode(`${side}Hand`) : null)
      || h.getNormalizedBoneNode(`${side}Hand`);
    const lower = (h.getRawBoneNode ? h.getRawBoneNode(`${side}LowerArm`) : null)
      || h.getNormalizedBoneNode(`${side}LowerArm`);
    if (!hand) return null;

    const g = new THREE.Group();
    const mat = new THREE.MeshStandardMaterial({ color: 0x8a94a6, roughness: 0.5 });
    const grip = new THREE.Mesh(new THREE.CylinderGeometry(0.014, 0.016, 0.20, 12), mat);
    grip.position.y = 0.10;
    const head = new THREE.Mesh(new THREE.TorusGeometry(0.115, 0.008, 8, 28), mat);
    head.position.y = 0.315; head.scale.set(0.85, 1, 1); head.rotation.x = Math.PI / 2;
    const face = new THREE.Mesh(
      new THREE.CircleGeometry(0.108, 24),
      new THREE.MeshBasicMaterial({ color: 0xdbe4f0, transparent: true, opacity: 0.22,
                                    side: THREE.DoubleSide }));
    face.position.y = 0.315; face.rotation.x = Math.PI / 2; face.scale.set(0.85, 1, 1);
    g.add(grip, head, face);

    // 前腕の延長方向（手のローカル座標系）にラケットの軸(+Y)を向ける
    if (lower) {
      vrm.scene.updateWorldMatrix(true, true);
      const a = new THREE.Vector3(), b = new THREE.Vector3(), q = new THREE.Quaternion();
      lower.getWorldPosition(a); hand.getWorldPosition(b);
      const dir = b.sub(a).normalize();
      hand.getWorldQuaternion(q);
      const local = dir.applyQuaternion(q.invert()).normalize();
      g.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), local);
    }
    hand.add(g);
    return g;
  }

  return {
    root: vrm.scene,
    vrm,
    applyFrame,
    attachRacket,
    /** 毎フレーム呼ぶ（MToonマテリアルや揺れ物の更新に必要） */
    update: (dt) => vrm.update(dt),
    setVisible: (v) => { vrm.scene.visible = v; },
    report,
  };
}
