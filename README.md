# FreeStyle Project Page

Static site for the FreeStyle project. Plain HTML / CSS / JS, no build step.

## 本地预览

页面通过 `fetch('./data/gallery.json')` 加载数据，双击 `index.html` 用 `file://`
打开会被浏览器 CORS 拦下，必须走一个本地静态服务器。

### 方式一：Python（无需额外依赖）

```bash
python3 -m http.server 8765 --bind 127.0.0.1
```

然后浏览器打开 <http://127.0.0.1:8765/>。

`Ctrl+C` 停止服务器。

### 方式二：Node

```bash
npx serve .
```

终端会打印实际端口（通常是 3000）。

### 自检建议

打开浏览器 DevTools → Network → 勾选 Img，刷新页面：

- 列表里应是 `assets/thumbs/.../*.webp`，每张几十 KB（首屏加载控制在几 MB 以内）。
- 滚动到画廊后段，新缩略图才陆续出现 —— `IntersectionObserver` 懒挂 `src` 生效。
- 点任意一张图打开 lightbox，这时才会请求 `assets/.../*.png|jpg` 原图（全尺寸）。

修改代码后刷新浏览器即可生效（无热重载）。如果发现旧资源被缓存，按
`Cmd+Shift+R`（macOS）/ `Ctrl+Shift+R`（Win/Linux）跳过缓存。

## 缩略图构建

画廊缩略图位于 `assets/thumbs/`，由 `scripts/build_thumbs.py` 从 `assets/`
下的原图生成（800px 长边、WebP、quality=80）。脚本按 mtime 增量构建。

新增或替换原图后，重新跑一遍：

```bash
python3 scripts/build_thumbs.py
```

依赖 `Pillow`：

```bash
python3 -m pip install Pillow
```

## 目录结构

```
index.html               页面骨架
styles.css               样式
app.js                   渲染画廊 + lightbox + 懒加载
data/gallery.json        画廊数据（样本 id、图片路径、prompt）
assets/                  原图（PNG / JPG，全尺寸）
assets/thumbs/           生成的 WebP 缩略图，画廊使用
scripts/build_thumbs.py  缩略图生成脚本
```

## 部署

仓库根目录有 `.nojekyll`，可直接作为 GitHub Pages 静态站点发布；推送 `main`
分支后等 Pages 构建完成即可访问。`assets/thumbs/` 必须一并提交，否则线上画廊会
404。
