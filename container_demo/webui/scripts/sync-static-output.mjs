import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const webRoot = path.resolve(__dirname, "..");
const outDir = path.join(webRoot, "out");
const distDir = path.join(webRoot, "dist");

const assertInside = (target, parent) => {
  const relative = path.relative(parent, target);
  if (relative.startsWith("..") || path.isAbsolute(relative)) {
    throw new Error(`Refusing to touch path outside ${parent}: ${target}`);
  }
};

const copyRecursive = (from, to) => {
  const stat = fs.statSync(from);
  if (stat.isDirectory()) {
    fs.mkdirSync(to, { recursive: true });
    for (const entry of fs.readdirSync(from)) {
      copyRecursive(path.join(from, entry), path.join(to, entry));
    }
    return;
  }
  fs.copyFileSync(from, to);
};

if (!fs.existsSync(outDir)) {
  throw new Error(`Next export output was not found: ${outDir}`);
}

fs.mkdirSync(distDir, { recursive: true });

const nextStatic = path.join(distDir, "_next");
assertInside(nextStatic, distDir);
fs.rmSync(nextStatic, { recursive: true, force: true });

for (const entry of fs.readdirSync(outDir)) {
  if (entry === "app.html") {
    continue;
  }
  copyRecursive(path.join(outDir, entry), path.join(distDir, entry));
}

console.log(`Synced Next static landing output into ${distDir}`);
