import { useEffect, useRef, useState, useCallback } from "react";
import * as THREE from "three";

/* ─── Google Fonts ─────────────────────────────────────────── */
const fontLink = document.createElement("link");
fontLink.rel = "stylesheet";
fontLink.href =
  "https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@300;400;500&display=swap";
document.head.appendChild(fontLink);

/* ─── Design tokens ─────────────────────────────────────────── */
const C = {
  bg: "#040a15",
  panel: "rgba(4,10,21,0.94)",
  border: "rgba(50, 100, 200, 0.12)",
  accent: "#3578e5",
  accentDim: "rgba(53, 120, 229, 0.16)",
  text: "#b8c8e0",
  textDim: "#35476a",
  textMid: "#5a80ac",
  mono: "'JetBrains Mono', monospace",
  sans: "'Syne', sans-serif",
};

/* ─── Types ─────────────────────────────────────────────────── */
interface Mineral {
  id: string;
  name: string;
  formula: string;
  composition: number;
  color: string;
  glowColor: string;
  lat: number;
  lon: number;
  depth: string;
  confidence: number;
}

interface Pin {
  mesh: THREE.Group;
  mineral: Mineral;
  screenPos: { x: number; y: number };
  visible: boolean;
  offset: number;
}

/* ─── Sample data ───────────────────────────────────────────── */
// Generate minerals from mineral stats with colors and positions
const generateMineralsFromStats = (stats: MineralStat[]): Mineral[] => {
  const colors = [
    { color: "#22d97a", glow: "#6efbb0" },
    { color: "#f97316", glow: "#fdba74" },
    { color: "#38bdf8", glow: "#7dd3fc" },
    { color: "#a78bfa", glow: "#c4b5fd" },
    { color: "#fb7185", glow: "#fda4af" },
    { color: "#2dd4bf", glow: "#5eead4" },
    { color: "#facc15", glow: "#fde68a" },
    { color: "#ec4899", glow: "#f9a8d4" },
    { color: "#14b8a6", glow: "#5eead4" },
    { color: "#8b5cf6", glow: "#c4b5fd" },
    { color: "#10b981", glow: "#6ee7b7" },
    { color: "#f59e0b", glow: "#fcd34d" },
  ];

  // Filter out background and very small percentages, then take top minerals
  return stats
    .filter((s) => s.mineral_class !== 0 && s.percentage > 0.1)
    .sort((a, b) => b.percentage - a.percentage)
    .slice(0, 15)
    .map((stat, i) => {
      const colorSet = colors[i % colors.length];
      // Generate deterministic but varied positions based on mineral class
      const lat = ((stat.mineral_class * 37) % 180) - 90;
      const lon = ((stat.mineral_class * 73) % 360) - 180;
      const confidence = Math.min(95, 65 + stat.percentage * 2);

      return {
        id: stat.mineral_class.toString(),
        name: stat.mineral_name,
        formula: "", // Formula not included in current data
        composition: stat.percentage,
        color: colorSet.color,
        glowColor: colorSet.glow,
        lat,
        lon,
        depth: "Surface scan",
        confidence: Math.round(confidence),
      };
    });
};

/* ─── Breakdown input (from inference response) ─────────────────────────── */
interface MineralStat {
  mineral_class: number;
  mineral_name: string;
  pixel_count: number;
  percentage: number;
}

/* ─── Helpers ───────────────────────────────────────────────── */
function latLonToVec3(lat: number, lon: number, r: number): THREE.Vector3 {
  const phi = (90 - lat) * (Math.PI / 180);
  const theta = (lon + 180) * (Math.PI / 180);
  return new THREE.Vector3(
    -r * Math.sin(phi) * Math.cos(theta),
    r * Math.cos(phi),
    r * Math.sin(phi) * Math.sin(theta),
  );
}
const rng = (s: number) => {
  const x = Math.sin(s) * 43758.5453;
  return x - Math.floor(x);
};

/* ═══════════════════════════════════════════════════════════ */
interface PlanetMineralGlobeProps {
  results?: MineralStat[] | null;
  onResults?: (data: any) => void;
  onUploadState?: (b: boolean) => void;
}

// Only destructure `results` here; other props exist for future use but are not
// currently referenced which would cause TS6133 (unused) errors during build.
export default function PlanetMineralGlobe({
  results,
}: PlanetMineralGlobeProps) {
  const mountRef = useRef<HTMLDivElement>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const rendRef = useRef<THREE.WebGLRenderer | null>(null);
  const camRef = useRef<THREE.PerspectiveCamera | null>(null);
  const pivotRef = useRef<THREE.Group | null>(null);
  const pinsRef = useRef<Pin[]>([]);
  const frameRef = useRef<number>(0);
  const isDragging = useRef(false);
  const prevMouse = useRef({ x: 0, y: 0 });
  const rotVel = useRef({ x: 0, y: 0 });

  const [hovered, setHovered] = useState<Mineral | null>(null);
  const [selected, setSelected] = useState<Mineral | null>(null);
  const [tipPos, setTipPos] = useState({ x: 0, y: 0 });
  const [loaded, setLoaded] = useState(false);
  const [dragging, setDragging] = useState(false);
  // local state for stats coming from parent
  const [stats, setStats] = useState<MineralStat[] | null>(results ?? null);
  const [minerals, setMinerals] = useState<Mineral[]>([]);

  // Update stats when results prop changes
  useEffect(() => {
    if (!results) {
      setStats(null);
      setMinerals([]);
      return;
    }
    console.log("[PlanetMineralGlobe] received results prop:", results);
    // If it's already an array of MineralStat, use it directly
    if (Array.isArray(results)) {
      setStats(results);
      setMinerals(generateMineralsFromStats(results));
      return;
    }
    // If results is an object, try to normalize
    try {
      const vals = Object.values(results) as any[];
      if (
        vals.length &&
        (vals[0].mineral_name || vals[0].percentage !== undefined)
      ) {
        setStats(vals as MineralStat[]);
        setMinerals(generateMineralsFromStats(vals as MineralStat[]));
        return;
      }
    } catch (e) {
      console.warn("[PlanetMineralGlobe] Unrecognized results shape", results);
    }
  }, [results]);

  useEffect(() => {
    if (!mountRef.current || minerals.length === 0) return;
    const W = mountRef.current.clientWidth;
    const H = mountRef.current.clientHeight;

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(W, H);
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.1;
    // Ensure we don't accidentally append multiple canvases (HMR/dev hot reload can re-run)
    if (mountRef.current) {
      const existing = mountRef.current.querySelectorAll("canvas");
      existing.forEach((c) => mountRef.current!.removeChild(c));
      mountRef.current.appendChild(renderer.domElement);
    }
    rendRef.current = renderer;

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(42, W / H, 0.1, 100);
    camera.position.set(0, 0, 3.6);
    camRef.current = camera;

    const pivot = new THREE.Group();
    scene.add(pivot);
    pivotRef.current = pivot;

    /* lights */
    scene.add(new THREE.AmbientLight(0x0a1530, 3));
    const sun = new THREE.DirectionalLight(0xb8d4ff, 5);
    sun.position.set(4, 2, 4);
    scene.add(sun);
    const fill = new THREE.DirectionalLight(0x1a4fff, 1.8);
    fill.position.set(-4, -1, -3);
    scene.add(fill);

    /* ── planet texture ── */
    const tc = document.createElement("canvas");
    tc.width = 1024;
    tc.height = 512;
    const ctx = tc.getContext("2d")!;

    const bg = ctx.createLinearGradient(0, 0, 0, 512);
    bg.addColorStop(0, "#040e28");
    bg.addColorStop(0.25, "#071830");
    bg.addColorStop(0.5, "#0a2040");
    bg.addColorStop(0.75, "#071830");
    bg.addColorStop(1, "#040e28");
    ctx.fillStyle = bg;
    ctx.fillRect(0, 0, 1024, 512);

    const landColors = [
      "#0d2a4a",
      "#10324f",
      "#0e3d5c",
      "#163350",
      "#1a3f60",
      "#0a2238",
      "#16486a",
    ];
    for (let i = 0; i < 500; i++) {
      ctx.fillStyle = landColors[i % landColors.length];
      ctx.globalAlpha = 0.3 + rng(i) * 0.45;
      ctx.beginPath();
      ctx.ellipse(
        rng(i * 3.7) * 1024,
        rng(i * 8.1) * 512,
        rng(i * 2.3) * 80 + 20,
        rng(i * 5.9) * 40 + 10,
        rng(i) * Math.PI,
        0,
        Math.PI * 2,
      );
      ctx.fill();
    }
    const iceCap = (y0: number, y1: number) => {
      for (let i = 0; i < 120; i++) {
        ctx.fillStyle = "#a8c8e8";
        ctx.globalAlpha = 0.08 + rng(i) * 0.18;
        ctx.beginPath();
        ctx.ellipse(
          rng(i * 11.3) * 1024,
          y0 + rng(i * 4.7) * (y1 - y0),
          rng(i) * 50 + 10,
          rng(i * 2) * 20 + 4,
          0,
          0,
          Math.PI * 2,
        );
        ctx.fill();
      }
    };
    iceCap(0, 60);
    iceCap(450, 512);
    for (let i = 0; i < 200; i++) {
      ctx.fillStyle = "#ffffff";
      ctx.globalAlpha = 0.02 + rng(i) * 0.04;
      ctx.beginPath();
      ctx.ellipse(
        rng(i * 6.1) * 1024,
        rng(i * 9.5) * 512,
        rng(i) * 90 + 20,
        rng(i * 3) * 8 + 2,
        rng(i) * 0.4,
        0,
        Math.PI * 2,
      );
      ctx.fill();
    }
    ctx.globalAlpha = 1;
    const planetTex = new THREE.CanvasTexture(tc);

    const nc = document.createElement("canvas");
    nc.width = 512;
    nc.height = 256;
    const nctx = nc.getContext("2d")!;
    nctx.fillStyle = "#8080ff";
    nctx.fillRect(0, 0, 512, 256);
    for (let i = 0; i < 200; i++) {
      const cx = rng(i * 5.1) * 512,
        cy = rng(i * 9.3) * 256,
        rad = rng(i * 3.7) * 25 + 5;
      const g2 = nctx.createRadialGradient(cx, cy, 0, cx, cy, rad);
      g2.addColorStop(0, "#9898ff");
      g2.addColorStop(1, "#8080ff");
      nctx.fillStyle = g2;
      nctx.globalAlpha = 0.45;
      nctx.beginPath();
      nctx.arc(cx, cy, rad, 0, Math.PI * 2);
      nctx.fill();
    }
    nctx.globalAlpha = 1;
    const normTex = new THREE.CanvasTexture(nc);

    const RADIUS = 1;
    pivot.add(
      new THREE.Mesh(
        new THREE.SphereGeometry(RADIUS, 72, 72),
        new THREE.MeshStandardMaterial({
          map: planetTex,
          normalMap: normTex,
          normalScale: new THREE.Vector2(1.2, 1.2),
          roughness: 0.75,
          metalness: 0.15,
        }),
      ),
    );
    pivot.add(
      new THREE.Mesh(
        new THREE.SphereGeometry(RADIUS * 1.06, 32, 32),
        new THREE.MeshBasicMaterial({
          color: 0x1a6aff,
          transparent: true,
          opacity: 0.07,
          side: THREE.BackSide,
        }),
      ),
    );
    pivot.add(
      new THREE.Mesh(
        new THREE.SphereGeometry(RADIUS * 1.02, 32, 32),
        new THREE.MeshBasicMaterial({
          color: 0x4488ff,
          transparent: true,
          opacity: 0.04,
          side: THREE.BackSide,
        }),
      ),
    );

    /* stars */
    const sv: number[] = [];
    for (let i = 0; i < 3500; i++)
      sv.push(
        (Math.random() - 0.5) * 80,
        (Math.random() - 0.5) * 80,
        (Math.random() - 0.5) * 80,
      );
    const sg = new THREE.BufferGeometry();
    sg.setAttribute("position", new THREE.Float32BufferAttribute(sv, 3));
    scene.add(
      new THREE.Points(
        sg,
        new THREE.PointsMaterial({
          color: 0xffffff,
          size: 0.04,
          sizeAttenuation: true,
        }),
      ),
    );

    /* ── Professional pins ── */
    const pins: Pin[] = [];
    minerals.forEach((mineral, i) => {
      const pos = latLonToVec3(mineral.lat, mineral.lon, RADIUS);
      const group = new THREE.Group();
      group.position.copy(pos);
      group.lookAt(new THREE.Vector3(0, 0, 0));
      group.rotateX(Math.PI);

      const col = new THREE.Color(mineral.color);
      const glow = new THREE.Color(mineral.glowColor);

      // Layout (local Y = radially outward from surface):
      //   y = 0        → globe surface
      //   y = 0..0.10  → stem
      //   y = 0.10     → diamond centre
      //   y = 0.10     → precision ring & ticks
      //   y = 0.10     → pulse disc

      const STEM_H = 0.1;
      const HEAD_Y = STEM_H; // diamond sits on top of stem
      const RING_Y = HEAD_Y;

      /* stem — bottom at y=0, top at y=STEM_H */
      const stem = new THREE.Mesh(
        new THREE.CylinderGeometry(0.004, 0.004, STEM_H, 8),
        new THREE.MeshStandardMaterial({
          color: col,
          emissive: col,
          emissiveIntensity: 0.5,
          roughness: 0.2,
          metalness: 0.9,
        }),
      );
      stem.position.set(0, STEM_H / 2, 0); // cylinder origin is at its centre
      group.add(stem);

      /* diamond (two cones meeting at HEAD_Y) */
      const headMat = new THREE.MeshStandardMaterial({
        color: col,
        emissive: glow,
        emissiveIntensity: 0.9,
        roughness: 0.1,
        metalness: 1.0,
      });
      const CONE_H = 0.03;
      const cu = new THREE.Mesh(
        new THREE.ConeGeometry(0.018, CONE_H, 6),
        headMat,
      );
      cu.position.set(0, HEAD_Y + CONE_H / 2, 0); // upper cone
      group.add(cu);
      const cd = new THREE.Mesh(
        new THREE.ConeGeometry(0.018, CONE_H, 6),
        headMat,
      );
      cd.position.set(0, HEAD_Y - CONE_H / 2, 0); // lower cone (flipped)
      cd.rotation.x = Math.PI;
      group.add(cd);

      /* precision ring — flat around the diamond waist */
      const ring = new THREE.Mesh(
        new THREE.TorusGeometry(0.036, 0.003, 8, 40),
        new THREE.MeshBasicMaterial({
          color: glow,
          transparent: true,
          opacity: 0.9,
        }),
      );
      ring.rotation.x = Math.PI / 2;
      ring.position.set(0, RING_Y, 0);
      group.add(ring);

      /* crosshair ticks — around the ring */
      const tickMat = new THREE.MeshBasicMaterial({
        color: col,
        transparent: true,
        opacity: 0.8,
      });
      [0, 90, 180, 270].forEach((deg) => {
        const tick = new THREE.Mesh(
          new THREE.BoxGeometry(0.002, 0.012, 0.002),
          tickMat,
        );
        const a = (deg * Math.PI) / 180;
        tick.position.set(Math.sin(a) * 0.036, RING_Y, Math.cos(a) * 0.036);
        group.add(tick);
      });

      /* pulse disc — same plane as ring, animates outward */
      const pulse = new THREE.Mesh(
        new THREE.RingGeometry(0.042, 0.048, 40),
        new THREE.MeshBasicMaterial({
          color: glow,
          transparent: true,
          opacity: 0.5,
          side: THREE.DoubleSide,
        }),
      );
      pulse.rotation.x = Math.PI / 2;
      pulse.position.set(0, RING_Y, 0);
      group.add(pulse);

      pivot.add(group);
      pins.push({
        mesh: group,
        mineral,
        screenPos: { x: 0, y: 0 },
        visible: true,
        offset: i * 0.9,
      });
    });
    pinsRef.current = pins;

    /* ── Animation ── */
    let t = 0;
    const animate = () => {
      frameRef.current = requestAnimationFrame(animate);
      t += 0.016;

      if (!isDragging.current) {
        rotVel.current.y *= 0.93;
        rotVel.current.x *= 0.93;
        pivot.rotation.y += 0.0012 + rotVel.current.y;
        pivot.rotation.x += rotVel.current.x;
        pivot.rotation.x = Math.max(-0.6, Math.min(0.6, pivot.rotation.x));
      }

      const W2 = renderer.domElement.clientWidth;
      const H2 = renderer.domElement.clientHeight;

      pins.forEach((pin) => {
        const pulse = pin.mesh.children[
          pin.mesh.children.length - 1
        ] as THREE.Mesh;
        const s = 1 + 0.45 * Math.abs(Math.sin(t * 1.8 + pin.offset));
        pulse.scale.set(s, s, s);
        (pulse.material as THREE.MeshBasicMaterial).opacity =
          0.55 * (1 - Math.abs(Math.sin(t * 1.8 + pin.offset)));

        const wp = new THREE.Vector3();
        pin.mesh.getWorldPosition(wp);
        const proj = wp.clone().project(camera);
        pin.screenPos = {
          x: (proj.x * 0.5 + 0.5) * W2,
          y: (-proj.y * 0.5 + 0.5) * H2,
        };
        const dot = camera.position
          .clone()
          .normalize()
          .dot(wp.clone().normalize());
        pin.visible = dot > 0.08;
        pin.mesh.visible = pin.visible;
      });

      renderer.render(scene, camera);
    };
    animate();

    const onResize = () => {
      if (!mountRef.current) return;
      const w = mountRef.current.clientWidth,
        h = mountRef.current.clientHeight;
      renderer.setSize(w, h);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    };
    window.addEventListener("resize", onResize);
    setTimeout(() => setLoaded(true), 500);

    return () => {
      cancelAnimationFrame(frameRef.current);
      window.removeEventListener("resize", onResize);
      renderer.dispose();
      // remove canvas if still present
      try {
        mountRef.current
          ?.querySelectorAll("canvas")
          .forEach((c) => mountRef.current?.removeChild(c));
      } catch (e) {}
      rendRef.current = null;
    };
  }, [minerals]);

  /* ── Mouse ── */
  const onMouseDown = useCallback((e: React.MouseEvent) => {
    isDragging.current = true;
    setDragging(true);
    prevMouse.current = { x: e.clientX, y: e.clientY };
  }, []);
  const onMouseMove = useCallback((e: React.MouseEvent) => {
    if (!isDragging.current || !pivotRef.current) return;
    const dx = e.clientX - prevMouse.current.x,
      dy = e.clientY - prevMouse.current.y;
    rotVel.current = { x: dy * 0.005, y: dx * 0.005 };
    pivotRef.current.rotation.y += dx * 0.005;
    pivotRef.current.rotation.x = Math.max(
      -0.6,
      Math.min(0.6, pivotRef.current.rotation.x + dy * 0.005),
    );
    prevMouse.current = { x: e.clientX, y: e.clientY };
  }, []);
  const onMouseUp = useCallback(() => {
    isDragging.current = false;
    setDragging(false);
  }, []);

  const onOverlayMove = useCallback((e: React.MouseEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const mx = e.clientX - rect.left,
      my = e.clientY - rect.top;
    let found: Mineral | null = null,
      best = 28;
    pinsRef.current.forEach((p) => {
      if (!p.visible) return;
      const d = Math.hypot(p.screenPos.x - mx, p.screenPos.y - my);
      if (d < best) {
        best = d;
        found = p.mineral;
        setTipPos({ x: p.screenPos.x, y: p.screenPos.y });
      }
    });
    setHovered(found);
  }, []);
  const onOverlayClick = useCallback((e: React.MouseEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const mx = e.clientX - rect.left,
      my = e.clientY - rect.top;
    let found: Mineral | null = null,
      best = 28;
    pinsRef.current.forEach((p) => {
      if (!p.visible) return;
      const d = Math.hypot(p.screenPos.x - mx, p.screenPos.y - my);
      if (d < best) {
        best = d;
        found = p.mineral;
      }
    });
    setSelected(found);
  }, []);

  // `total` was used in the old composition UI; it's not needed with the
  // current `stats`-driven breakdown, so omit it to avoid unused variable errors.

  /* ── JSX ── */
  // Don't render globe if no results
  if (!results || !stats || minerals.length === 0) {
    return (
      <div
        style={{
          width: "100%",
          height: "100vh",
          background: C.bg,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          fontFamily: C.sans,
        }}
      >
        <div style={{ textAlign: "center" }}>
          <div
            style={{
              color: C.textDim,
              fontSize: 11,
              letterSpacing: 2,
              fontFamily: C.mono,
              textTransform: "uppercase",
            }}
          >
            Waiting for mineral analysis results...
          </div>
        </div>
      </div>
    );
  }

  return (
    <div
      style={{
        width: "100%",
        height: "100vh",
        background: C.bg,
        display: "flex",
        flexDirection: "column",
        overflowX: "hidden",
        overflowY: "hidden",
        fontFamily: C.sans,
      }}
    >
      {/* HEADER */}
      <header
        style={{
          padding: "13px 28px",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          zIndex: 10,
          borderBottom: `1px solid ${C.border}`,
          background: "rgba(3,7,15,0.88)",
          backdropFilter: "blur(14px)",
        }}
      >
        {/* <div style={{ display:"flex", gap:16, alignItems:"center" }}>
          {file && (
            <div style={{ fontSize:9.5, color:C.textMid, fontFamily:C.mono,
              background:C.accentDim, padding:"4px 10px", borderRadius:4,
              border:`1px solid ${C.border}` }}>
              ◉ {file}
            </div>
          )}
          <label style={{
            padding:"6px 16px", border:`1px solid rgba(61,139,255,0.3)`,
            color:C.accent, fontSize:10, letterSpacing:2, cursor:"pointer",
            background:"rgba(61,139,255,0.06)", borderRadius:4,
            textTransform:"uppercase", fontFamily:C.mono,
          }}>
            ↑ Upload Image
            <input type="file" accept="image/*" style={{ display:"none" }}
              onChange={(e) => setFile(e.target.files?.[0]?.name ?? null)} />
          </label>
          <div style={{ width:1, height:22, background:C.border }} />
          <div style={{ textAlign:"right" }}>
            <div style={{ fontSize:10.5, color:C.textMid, fontFamily:C.mono }}>
              {MINERALS.length} minerals
            </div>
            <div style={{ fontSize:8, color:C.textDim, fontFamily:C.mono, marginTop:1 }}>
              detected
            </div>
          </div>
        </div> */}
      </header>

      {/* BODY */}
      <div style={{ flex: 1, display: "flex", overflow: "hidden" }}>
        {/* Globe */}
        <div
          ref={containerRef}
          style={{ flex: "0 0 55%", position: "relative" }}
        >
          <div
            ref={mountRef}
            style={{ width: "100%", height: "100%", pointerEvents: "none" }}
          />

          {/* Interaction overlay */}
          <div
            style={{
              position: "absolute",
              inset: 0,
              pointerEvents: "all",
              cursor: dragging ? "grabbing" : "grab",
            }}
            onMouseDown={onMouseDown}
            onMouseMove={(e) => {
              onMouseMove(e);
              onOverlayMove(e);
            }}
            onMouseUp={onMouseUp}
            onMouseLeave={onMouseUp}
            onClick={onOverlayClick}
          />

          {/* Pin labels (clamped to globe container) */}
          {pinsRef.current.map((pin) =>
            pin.visible ? (
              <div
                key={pin.mineral.id}
                style={{
                  position: "absolute",
                  left: Math.min(
                    Math.max(pin.screenPos.x + 20, 6),
                    (containerRef.current?.clientWidth ?? window.innerWidth) -
                      140,
                  ),
                  top: Math.min(
                    Math.max(pin.screenPos.y - 14, 6),
                    (containerRef.current?.clientHeight ?? window.innerHeight) -
                      32,
                  ),
                  pointerEvents: "none",
                  opacity: hovered?.id === pin.mineral.id ? 1 : 0.42,
                  transition: "opacity 0.18s",
                }}
              >
                <div style={{ display: "flex", alignItems: "center", gap: 5 }}>
                  <div
                    style={{
                      width: 3,
                      height: 3,
                      background: pin.mineral.color,
                      transform: "rotate(45deg)",
                      boxShadow: `0 0 5px ${pin.mineral.glowColor}`,
                    }}
                  />
                  <span
                    style={{
                      fontSize: 9,
                      fontFamily: C.mono,
                      letterSpacing: 1.8,
                      color: pin.mineral.color,
                      fontWeight: 500,
                      textTransform: "uppercase",
                      textShadow: `0 0 10px ${pin.mineral.glowColor}77`,
                    }}
                  >
                    {pin.mineral.name}
                  </span>
                </div>
                <div
                  style={{
                    fontSize: 7.5,
                    fontFamily: C.mono,
                    color: C.textDim,
                    marginLeft: 8,
                    marginTop: 1,
                  }}
                >
                  {pin.mineral.composition.toFixed(1)}%
                </div>
              </div>
            ) : null,
          )}

          {/* Hover tooltip (clamped to globe container) */}
          {hovered && (
            <div
              style={{
                position: "absolute",
                left: Math.min(
                  tipPos.x + 24,
                  Math.max(
                    8,
                    (containerRef.current?.clientWidth ?? window.innerWidth) -
                      250,
                  ),
                ),
                top: Math.max(
                  Math.min(
                    tipPos.y - 88,
                    (containerRef.current?.clientHeight ?? window.innerHeight) -
                      120,
                  ),
                  8,
                ),
                background: "rgba(3,8,20,0.97)",
                border: `1px solid ${hovered.color}33`,
                borderLeft: `2px solid ${hovered.color}`,
                padding: "14px 18px",
                borderRadius: 6,
                pointerEvents: "none",
                zIndex: 20,
                boxShadow: `0 12px 48px rgba(0,0,0,0.7), 0 0 24px ${hovered.glowColor}18`,
                minWidth: 210,
                backdropFilter: "blur(10px)",
              }}
            >
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 8,
                  marginBottom: 6,
                }}
              >
                <div
                  style={{
                    width: 7,
                    height: 7,
                    background: hovered.color,
                    transform: "rotate(45deg)",
                    boxShadow: `0 0 8px ${hovered.glowColor}`,
                  }}
                />
                <span
                  style={{
                    fontSize: 12,
                    fontWeight: 700,
                    color: hovered.color,
                    letterSpacing: 2,
                    textTransform: "uppercase",
                  }}
                >
                  {hovered.name}
                </span>
              </div>
              <div
                style={{
                  fontSize: 8.5,
                  fontFamily: C.mono,
                  color: C.textDim,
                  marginBottom: 10,
                  letterSpacing: 0.3,
                }}
              >
                {hovered.formula}
              </div>
              {[
                ["Composition", `${hovered.composition}%`],
                ["Depth", hovered.depth],
                ["Confidence", `${hovered.confidence}%`],
              ].map(([k, v]) => (
                <div
                  key={k}
                  style={{
                    display: "flex",
                    justifyContent: "space-between",
                    marginBottom: 5,
                    gap: 16,
                  }}
                >
                  <span
                    style={{
                      fontSize: 9,
                      fontFamily: C.mono,
                      color: C.textDim,
                    }}
                  >
                    {k}
                  </span>
                  <span
                    style={{
                      fontSize: 9,
                      fontFamily: C.mono,
                      color: C.text,
                      fontWeight: 500,
                    }}
                  >
                    {v}
                  </span>
                </div>
              ))}
              <div
                style={{
                  marginTop: 10,
                  paddingTop: 8,
                  borderTop: `1px solid rgba(255,255,255,0.05)`,
                  fontSize: 8,
                  fontFamily: C.mono,
                  color: C.textDim,
                  letterSpacing: 1.5,
                }}
              >
                CLICK TO INSPECT ›
              </div>
            </div>
          )}

          {/* Drag hint */}
          {loaded && (
            <div
              style={{
                position: "absolute",
                bottom: 20,
                left: "50%",
                transform: "translateX(-50%)",
                fontSize: 8,
                letterSpacing: 2.5,
                color: C.textDim,
                textTransform: "uppercase",
                pointerEvents: "none",
                fontFamily: C.mono,
                display: "flex",
                alignItems: "center",
                gap: 7,
              }}
            >
              <div
                style={{
                  width: 13,
                  height: 13,
                  borderRadius: "50%",
                  border: `1px solid ${C.textDim}33`,
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  fontSize: 8,
                  opacity: 0.6,
                }}
              >
                ↻
              </div>
              Drag to rotate
            </div>
          )}

          {/* Loading veil */}
          <div
            style={{
              position: "absolute",
              inset: 0,
              background: C.bg,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              opacity: loaded ? 0 : 1,
              pointerEvents: loaded ? "none" : "all",
              transition: "opacity 0.9s ease",
              zIndex: 30,
            }}
          >
            <div style={{ textAlign: "center" }}>
              <div
                style={{
                  width: 44,
                  height: 44,
                  borderRadius: "50%",
                  border: `1px solid ${C.accentDim}`,
                  borderTop: `1px solid ${C.accent}`,
                  animation: "spin 1.2s linear infinite",
                  margin: "0 auto 14px",
                }}
              />
              <div
                style={{
                  color: C.textDim,
                  fontSize: 8.5,
                  letterSpacing: 3,
                  fontFamily: C.mono,
                }}
              >
                INITIALISING SURFACE SCAN…
              </div>
            </div>
          </div>
        </div>

        {/* RIGHT PANEL */}
        <aside
          style={{
            flex: "1",
            background: C.panel,
            borderLeft: `1px solid ${C.border}`,
            display: "flex",
            flexDirection: "column",
            overflow: "hidden",
          }}
        >
          {/* Inspector */}
          <div
            style={{
              padding: "18px 20px 16px",
              borderBottom: `1px solid ${C.border}`,
            }}
          >
            {selected ? (
              <>
                <div
                  style={{
                    fontSize: 7.5,
                    fontFamily: C.mono,
                    color: C.textDim,
                    letterSpacing: 3,
                    marginBottom: 10,
                    textTransform: "uppercase",
                  }}
                >
                  Selected Mineral
                </div>
                <div
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: 8,
                    marginBottom: 3,
                  }}
                >
                  <div
                    style={{
                      width: 7,
                      height: 7,
                      background: selected.color,
                      transform: "rotate(45deg)",
                      boxShadow: `0 0 10px ${selected.glowColor}`,
                    }}
                  />
                  <span
                    style={{
                      fontSize: 18,
                      fontWeight: 700,
                      color: "#deeeff",
                      letterSpacing: 0.5,
                    }}
                  >
                    {selected.name}
                  </span>
                </div>
                <div
                  style={{
                    fontSize: 8.5,
                    fontFamily: C.mono,
                    color: C.textDim,
                    marginBottom: 14,
                  }}
                >
                  {selected.formula}
                </div>

                <div style={{ marginBottom: 14 }}>
                  <div
                    style={{
                      display: "flex",
                      justifyContent: "space-between",
                      marginBottom: 5,
                    }}
                  >
                    <span
                      style={{
                        fontSize: 7.5,
                        fontFamily: C.mono,
                        color: C.textDim,
                        letterSpacing: 2,
                      }}
                    >
                      COMPOSITION
                    </span>
                    <span
                      style={{
                        fontSize: 7.5,
                        fontFamily: C.mono,
                        color: selected.color,
                      }}
                    >
                      {selected.composition}%
                    </span>
                  </div>
                  <div
                    style={{
                      height: 2,
                      background: "rgba(255,255,255,0.05)",
                      borderRadius: 1,
                      overflow: "hidden",
                    }}
                  >
                    <div
                      style={{
                        height: "100%",
                        width: `${selected.composition}%`,
                        background: `linear-gradient(90deg, ${selected.color}, ${selected.glowColor})`,
                        borderRadius: 1,
                        transition: "width 0.7s ease",
                        boxShadow: `0 0 8px ${selected.glowColor}77`,
                      }}
                    />
                  </div>
                  <div
                    style={{
                      fontSize: 26,
                      fontWeight: 800,
                      color: "#deeeff",
                      marginTop: 5,
                      letterSpacing: -1,
                    }}
                  >
                    {selected.composition}
                    <span
                      style={{
                        fontSize: 13,
                        color: C.textMid,
                        fontWeight: 400,
                        marginLeft: 2,
                      }}
                    >
                      %
                    </span>
                  </div>
                </div>

                <div
                  style={{
                    display: "grid",
                    gridTemplateColumns: "1fr 1fr",
                    gap: 6,
                    marginBottom: 12,
                  }}
                >
                  {[
                    ["Depth", selected.depth],
                    ["Confidence", `${selected.confidence}%`],
                    ["Latitude", `${selected.lat}°`],
                    ["Longitude", `${selected.lon}°`],
                  ].map(([k, v]) => (
                    <div
                      key={k}
                      style={{
                        background: "rgba(61,139,255,0.04)",
                        border: `1px solid ${C.border}`,
                        borderRadius: 5,
                        padding: "7px 10px",
                      }}
                    >
                      <div
                        style={{
                          fontSize: 7,
                          fontFamily: C.mono,
                          color: C.textDim,
                          letterSpacing: 1.5,
                          marginBottom: 3,
                          textTransform: "uppercase",
                        }}
                      >
                        {k}
                      </div>
                      <div
                        style={{
                          fontSize: 10.5,
                          fontFamily: C.mono,
                          color: C.text,
                        }}
                      >
                        {v}
                      </div>
                    </div>
                  ))}
                </div>

                <button
                  onClick={() => setSelected(null)}
                  style={{
                    width: "100%",
                    padding: "6px 0",
                    background: "transparent",
                    border: `1px solid ${C.border}`,
                    color: C.textDim,
                    fontSize: 8,
                    letterSpacing: 2.5,
                    cursor: "pointer",
                    borderRadius: 4,
                    fontFamily: C.mono,
                    textTransform: "uppercase",
                  }}
                >
                  ✕ Deselect
                </button>
              </>
            ) : (
              <>
                <div
                  style={{
                    fontSize: 7.5,
                    fontFamily: C.mono,
                    color: C.textDim,
                    letterSpacing: 3,
                    marginBottom: 8,
                    textTransform: "uppercase",
                  }}
                >
                  Mineral Inspector
                </div>
                <div
                  style={{
                    fontSize: 10.5,
                    color: C.textDim,
                    lineHeight: 1.75,
                    fontFamily: C.mono,
                  }}
                >
                  Select a pin on the globe to view detailed spectroscopic
                  analysis.
                </div>
              </>
            )}
          </div>

          {/* Breakdown list */}
          <div style={{ flex: 1, overflowY: "auto", padding: "14px 20px" }}>
            <div
              style={{
                fontSize: 7.5,
                fontFamily: C.mono,
                color: C.textDim,
                letterSpacing: 3,
                marginBottom: 14,
                textTransform: "uppercase",
              }}
            >
              Composition Breakdown
            </div>
            {stats!
              .slice()
              .sort((a, b) => b.percentage - a.percentage)
              .map((m) => (
                <div key={m.mineral_class} style={{ marginBottom: 13 }}>
                  <div
                    style={{
                      display: "flex",
                      justifyContent: "space-between",
                      alignItems: "center",
                      marginBottom: 6,
                    }}
                  >
                    <div
                      style={{ display: "flex", alignItems: "center", gap: 8 }}
                    >
                      <div
                        style={{
                          width: 6,
                          height: 6,
                          background: C.accent,
                          transform: "rotate(45deg)",
                          flexShrink: 0,
                          boxShadow: `0 0 6px ${C.accent}55`,
                        }}
                      />
                      <div style={{ display: "flex", flexDirection: "column" }}>
                        <span
                          style={{
                            fontSize: 11,
                            fontWeight: 600,
                            color: "#c8d8f0",
                            letterSpacing: 0.2,
                          }}
                        >
                          {m.mineral_name}
                        </span>
                        <span
                          style={{
                            fontSize: 9,
                            fontFamily: C.mono,
                            color: C.textDim,
                          }}
                        >
                          {m.pixel_count.toLocaleString()} pixels
                        </span>
                      </div>
                    </div>
                    <span
                      style={{
                        fontSize: 9,
                        fontFamily: C.mono,
                        color: C.textMid,
                      }}
                    >
                      {m.percentage.toFixed(6)}%
                    </span>
                  </div>

                  <div
                    style={{
                      height: 6,
                      background: "rgba(255,255,255,0.03)",
                      borderRadius: 4,
                      overflow: "hidden",
                    }}
                  >
                    <div
                      style={{
                        height: "100%",
                        width: `${m.percentage}%`,
                        background: `linear-gradient(90deg, ${C.accent}, ${C.accentDim})`,
                        borderRadius: 4,
                        boxShadow: "none",
                        transition: "width 0.5s ease",
                      }}
                    />
                  </div>
                </div>
              ))}
          </div>

          {/* Footer */}
          <div
            style={{
              padding: "11px 20px",
              borderTop: `1px solid ${C.border}`,
              display: "grid",
              gridTemplateColumns: "1fr 1fr",
              gap: 10,
            }}
          >
            {[
              ["SCAN RES", "4K · 12-band"],
              ["SPECTRA", "VIS + NIR"],
              ["COVERAGE", "94.3%"],
              ["DATE", new Date().toLocaleDateString()],
            ].map(([k, v]) => (
              <div key={k}>
                <div
                  style={{
                    fontSize: 7,
                    fontFamily: C.mono,
                    color: C.textDim,
                    letterSpacing: 1.5,
                    textTransform: "uppercase",
                  }}
                >
                  {k}
                </div>
                <div
                  style={{
                    fontSize: 9.5,
                    fontFamily: C.mono,
                    color: C.textMid,
                    marginTop: 2,
                  }}
                >
                  {v}
                </div>
              </div>
            ))}
          </div>
        </aside>
      </div>

      <style>{`
        @keyframes spin { to { transform: rotate(360deg); } }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        ::-webkit-scrollbar { width: 3px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: rgba(61,139,255,0.18); border-radius:2px; }
      `}</style>
    </div>
  );
}
