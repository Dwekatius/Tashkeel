import { execFileSync } from 'node:child_process';
import { cpSync, mkdirSync, rmSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const dist = resolve(root, 'dist');
const tailwindCli = resolve(root, 'node_modules', 'tailwindcss', 'lib', 'cli.js');

rmSync(dist, { recursive: true, force: true });
mkdirSync(resolve(dist, 'css'), { recursive: true });

cpSync(resolve(root, 'index.html'), resolve(dist, 'index.html'));
cpSync(resolve(root, 'js'), resolve(dist, 'js'), { recursive: true });
cpSync(resolve(root, 'css', 'style.css'), resolve(dist, 'css', 'style.css'));

execFileSync(
  process.execPath,
  [
    tailwindCli,
    '-c',
    resolve(root, 'tailwind.config.js'),
    '-i',
    resolve(root, 'src', 'tailwind.css'),
    '-o',
    resolve(dist, 'css', 'tailwind.css'),
    '--minify',
  ],
  { cwd: root, stdio: 'inherit' },
);
