/**
 * 3D骨格の表示。compare.html（2本の動画を比べる）と models.html
 * （1本を複数の手法で比べる）が共有する。
 *
 * ここに置いてあるのは「復元結果を見られる形に直す」処理で、
 * ページ固有のUI（再生・同期・アバター・描き込み）は各ページに置く。
 */
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

/** SMPL 24関節の骨（子, 親） */
export const BONES = [[1,0],[2,0],[3,0],[4,1],[5,2],[6,3],[7,4],[8,5],[9,6],[10,7],
  [11,8],[12,9],[13,9],[14,9],[15,12],[16,13],[17,14],[18,16],[19,17],
  [20,18],[21,19],[22,20],[23,21]];

/** 左右の足首・つま先。接地の判定に使う */
export const FOOT_IDS = [7, 8, 10, 11];
const L_HIP = 1, R_HIP = 2;

/** 局面の色。オレンジ＝沈み込み、赤＝打点 */
export const LOADING_COLOR = 0xf59e0b;
export const CONTACT_COLOR = 0xef4444;

/**
 * 復元結果を表示座標へ直す。3つ揃える:
 *   1. up軸を +Y に
 *   2. 体の向きを揃える（腰の左→右を +X に回す）
 *   3. 足元を床(0)に
 *
 * 2つ目が無いと、2本の動画を並べたとき片方が背中向きになる（実測で187°違った）。
 *
 * 3つ目の基準は最低値ではなく**中央値**。理由が2つある。最低値だとクリップ中に
 * 一度でも沈んだ瞬間が床になり、復元がドリフトしていると開始時点で体が浮く。
 * さらに、世界座標の原点の置き方は手法ごとに違う（GVHMR は床を y≈0 に置くが
 * TRAM は置かない。同じ動画で頭が +1.4m と −0.2m）。足から測れば吸収できる。
 */
export function toDisplay(clip) {
  const [ax, sg] = clip.up_axis;
  const hz = [0, 1, 2].filter(k => k !== ax);
  const d = clip.joints.map(fr => fr.map(p => [p[hz[0]], p[ax] * sg, p[hz[1]]]));

  let rx = 0, rz = 0;
  for (const fr of d) {
    rx += fr[R_HIP][0] - fr[L_HIP][0];
    rz += fr[R_HIP][2] - fr[L_HIP][2];
  }
  const ang = Math.atan2(rz, rx);            // 現在の「右」方向
  const ca = Math.cos(-ang), sa = Math.sin(-ang);
  for (const fr of d) for (const p of fr) {
    const x = p[0], z = p[2];
    p[0] = x * ca - z * sa;
    p[2] = x * sa + z * ca;
  }

  const lows = d.map(fr => Math.min(...FOOT_IDS.map(i => fr[i][1]))).sort((a, b) => a - b);
  const ground = lows[Math.floor(lows.length / 2)];
  let sx = 0, sz = 0;
  for (const fr of d) { sx += fr[0][0]; sz += fr[0][2]; }
  const cx = sx / d.length, cz = sz / d.length;
  for (const fr of d) for (const p of fr) { p[0] -= cx; p[1] -= ground; p[2] -= cz; }
  return d;
}

/**
 * 1画面ぶんの骨格ビューを作る。
 *
 * 戻り値の `scene` にアバター等を足せるので、compare.html はそれで拡張している。
 *
 * @param host  描画先の要素。大きさはCSSで決める
 * @param clip  make_compare.py / make_models.py が出す1クリップぶんのJSON
 * @param color 関節の色（局面の色が優先される）
 */
export function createPane(host, clip, color) {
  const size = () => [host.clientWidth || 380, host.clientHeight || 400];
  let [W, H] = size();

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(42, W / H, 0.01, 100);
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  // 第3引数 false が要点。true だと three がキャンバスに px 幅をインラインで
  // 書き込み、それがグリッド列を押し広げ、次のフレームでさらに広がる、という
  // 増幅ループになる。表示サイズはCSS（100%）に任せ、解像度だけ倍率を掛ける。
  renderer.setSize(W, H, false);
  renderer.setPixelRatio(devicePixelRatio);
  host.appendChild(renderer.domElement);

  const controls = new OrbitControls(camera, renderer.domElement);
  camera.position.set(2.6, 1.2, 2.6);
  controls.target.set(0, 0.9, 0);

  scene.add(new THREE.HemisphereLight(0xffffff, 0x444444, 1.2));
  const key = new THREE.DirectionalLight(0xffffff, 1.0);
  key.position.set(2, 3, 2);
  scene.add(key);                                   // アバターの陰影用
  scene.add(new THREE.GridHelper(4, 8, 0x333a44, 0x222831));

  const disp = toDisplay(clip);
  const J = disp[0].length;

  const skel = new THREE.Group();
  scene.add(skel);
  const boneGeom = new THREE.BufferGeometry();
  const bonePos = new Float32Array(BONES.length * 2 * 3);
  boneGeom.setAttribute("position", new THREE.BufferAttribute(bonePos, 3));
  skel.add(new THREE.LineSegments(boneGeom,
    new THREE.LineBasicMaterial({ color: 0x9db2cc })));
  const spheres = [];
  for (let j = 0; j < J; j++) {
    const s = new THREE.Mesh(new THREE.SphereGeometry(0.023, 10, 10),
      new THREE.MeshBasicMaterial({ color }));
    skel.add(s);
    spheres.push(s);
  }

  function setFrame(f) {
    const fr = disp[Math.max(0, Math.min(disp.length - 1, f))];
    for (let j = 0; j < J; j++) spheres[j].position.set(...fr[j]);
    let k = 0;
    for (const [a, b] of BONES) {
      bonePos[k++] = fr[a][0]; bonePos[k++] = fr[a][1]; bonePos[k++] = fr[a][2];
      bonePos[k++] = fr[b][0]; bonePos[k++] = fr[b][1]; bonePos[k++] = fr[b][2];
    }
    boneGeom.attributes.position.needsUpdate = true;

    // 局面の色。1フレームだけだと再生中に見逃すので前後1フレームまで含める。
    const ph = clip.metrics.phases;
    const near = (a, b) => Math.abs(a - b) <= 1;
    const col = near(f, ph.contact) ? CONTACT_COLOR
              : near(f, ph.loading) ? LOADING_COLOR : color;
    for (const s of spheres) s.material.color.setHex(col);
    return fr;
  }

  function syncSize() {
    const [w, h] = size();
    if (w === W && h === H) return;
    [W, H] = [w, h];
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h, false);
  }

  return {
    clip, scene, camera, controls, skel, frames: disp.length, disp,
    setFrame, syncSize,
    render: () => { syncSize(); controls.update(); renderer.render(scene, camera); },
    setOrbit: (on) => { controls.enabled = on; },
  };
}

/**
 * 複数の画面でカメラを連動させる。3つ以上でも同じ。
 * 揃っていないと形の違いが分からないので、既定は同期。
 */
export function linkCameras(panes, isOn) {
  let syncing = false;
  for (const src of panes) {
    src.controls.addEventListener("change", () => {
      if (!isOn() || syncing) return;
      syncing = true;
      for (const dst of panes) {
        if (dst === src) continue;
        dst.camera.position.copy(src.camera.position);
        dst.controls.target.copy(src.controls.target);
      }
      syncing = false;
    });
  }
}
