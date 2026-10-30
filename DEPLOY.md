# 请柬视频工作台 · 服务器部署手册

> 包名：请柬视频工作台_部署包_2026-10-30.zip / 内部根目录 qingjian_studio_online/

## 一、包内容
- `server.py`          服务入口
- `core/`              渲染引擎（本地 OpenCV / Pillow / ffmpeg + 火山方舟文生视频·图，走标准库 urllib，无需额外 SDK）
- `static/`            前端页面 + 模板封面/样片 + 字体（9 个 ttf，已含中文）
- `bgm/`               背景音乐（各模板固定选曲用，约 28 个文件）
- `config.example.json` 运行配置模板（不含密钥）
- `bgm_meta.json`      BGM 元数据
- `requirements.txt`   Python 依赖
- `data/`              运行期目录（用户工程 / 上传图 / 成片），首次运行自动创建

## 二、环境要求
- Python 3.11 或 3.12（numpy<3，3.13 亦可运行）
- 无需单独安装 ffmpeg：依赖 imageio-ffmpeg 会自带 ffmpeg 二进制
- 服务器需能出网访问火山方舟 API（文生视频 / 文生图功能需要；仅本地 CV 渲染不需要）

## 三、安装依赖
```bash
cd qingjian_studio_online
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 四、配置 config.json（上线前必看）
首次部署时，从模板创建本机配置：
```bash
cp config.example.json config.json
```
Windows PowerShell：
```powershell
Copy-Item config.example.json config.json
```

- `port`：服务端口（默认 8790）
- `video_api.api_key` / `storyboard_api.api_key`：分别配置火山方舟 ARK Key；也可在工作台设置中填写。
- `access_password`：工作台 Web 访问口令；也可通过 `STUDIO_PASSWORD` 环境变量配置。
- `config.json` 是本机配置，已加入 Git 忽略规则。不要提交 API Key、访问口令等凭据。

## 五、启动（关键！绑定地址取决于是否设置 PORT）
本服务绑定逻辑：
- **不设 `PORT`** → 仅绑定 `127.0.0.1`（只有本机能访问，外网/局域网不可达）
- **设置 `PORT`** → 自动绑定 `0.0.0.0`（对外提供服务）

因此部署到服务器务必设置 `PORT`（或显式 `HOST=0.0.0.0`）：
```bash
# Linux / macOS
PORT=8790 python server.py

# Windows (PowerShell)
$env:PORT=8790; python server.py
# 或 cmd:
set PORT=8790 && python server.py
```
服务读取 `PORT` 环境变量，绑定 `0.0.0.0`，监听端口 = PORT 值（与 `config.json.port` 保持一致即可）。

## 六、反向代理（推荐生产用法）
生产环境建议用 nginx 把 80/443 反代到本服务端口，并处理 HTTPS、静态资源与大文件上传：
```nginx
server {
    listen 80;
    server_name your.domain.com;
    client_max_body_size 200m;        # 允许大视频/图片上传
    location / {
        proxy_pass http://127.0.0.1:8790;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_read_timeout 600s;       # 视频生成耗时较长
        proxy_send_timeout 600s;
    }
}
```
说明：本服务为单进程 `ThreadingHTTPServer`。并发量高时可在 nginx 后跑多个实例，或用 gunicorn/uwsgi 多 worker 托管 `server.py` 的 WSGI/Handler。

## 七、防火墙 / 端口
- 若由 nginx 反代：对外只需开放 80/443。
- 若直连服务：开放 `PORT`（如 8790）入站。

## 八、数据目录 data/
存放用户工程、上传图片、成片视频，会随使用持续增长，请确保磁盘空间充足并定期清理。首次运行自动创建，无需手动建。

## 九、常见问题排查
- 视频生成返回 401：ARK Key 失效或额度不足 → 更新工作台设置或本机 `config.json` 的 `api_key`。
- 视频生成返回 404：模型名不存在 → 核对 `storyboard_api.model` / `video_api.model`。
- 中文界面 / 字幕乱码：服务器需有中文字体，`static/fonts/` 已内置 9 个 ttf。
- 上传或成片失败：检查 nginx `client_max_body_size` 与 `proxy_read_timeout` 是否够大。
- 服务起不来：确认 `PORT` 已设置（否则只绑 127.0.0.1，外部连不上），端口未被占用。
