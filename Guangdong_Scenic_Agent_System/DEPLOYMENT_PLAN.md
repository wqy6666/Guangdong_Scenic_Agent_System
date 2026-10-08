# 广东景区智能问答系统免费部署规划

核查日期：2026-10-08。目标是本科毕业设计的小规模在线展示。本文是部署规划；尚未创建远程仓库、云资源或修改运行代码。

## 1. 建议选择

优先采用 **GitHub 保存代码 + Vercel Hobby 托管 frontend + Oracle Cloud Always Free ARM 虚拟机运行 Python 后端**。该路线能保留现有 BGE 检索、重排序、反思和文件存储，代码迁移量相对较小。

前提是能完成 Oracle 账户验证并取得免费实例。当前官方文档对 Always Free 账户给出的 A1 额度相当于 **2 OCPU / 12 GB 内存**，另有总计 200 GB 的免费块存储；不是旧教程常写的 4 OCPU / 24 GB。免费容量可能不足，闲置实例可能被回收，所以需要备份和答辩时的本地备用服务。[Oracle 免费资源](https://docs.oracle.com/en-us/iaas/Content/FreeTier/freetier_topic-Always_Free_Resources.htm)

Oracle 注册要求有效信用卡或符合要求的借记卡做身份验证。账户审核、可分配容量和部署后的访问速度仍需实际验证。[注册条件](https://www.oracle.com/cloud/free/faq/)

若无法取得免费虚拟机，采用 **Vercel + Render Free + Supabase Free** 的精简版本。此版本必须调整检索实现与存储，不能直接运行现有重模型后端。

“服务器免费”不包含外部大模型调用费用。目前系统使用用户配置的模型 API；保留自带密钥方式时费用由相应用户承担，改成共享演示密钥时需要设置每日调用额度。

## 2. 当前目录与资源判断

| 文件或目录 | 本机大小约值 | 部署处理 |
|---|---:|---|
| `frontend/`，含 README | 0.10 MiB | Vercel 静态站点；运行资产为 HTML、CSS、JS |
| `src/` Python 源代码 | — | 后端服务器，保留实际运行模块及测试 |
| `model/bge-small-zh-v1.5/` | 183.34 MiB | 后端下载；目录包含重复格式权重 |
| `model/bge-reranker-base/` | 3203.51 MiB | 后端只准备所需格式；单个 safetensors 权重约 1060.70 MiB |
| `model/Qwen3-Reranker-0.6B/` | 1151.55 MiB | 当前 config 未选用；不纳入当前运行部署 |
| `src/vector_array.npy` | 78.15 MiB | 后端检索资产，与模型和元数据配套版本管理 |
| `src/chunk_meta.pkl` | 11.62 MiB | 后端检索资产，仅加载项目可信来源文件 |
| `shujuku/travel_final2.db` | 9.27 MiB | 当前实际使用的景区 SQLite 数据库 |
| `users/` | — | 用户账号、配置、历史、反馈和统计；私有可写数据 |
| `session_short2_store/` | — | 对话溢出记忆；可写数据 |
| `logs/` | — | 后端日志，轮转并限制保留时间 |

模型目录大小包括多种权重格式，不等于运行时内存占用。实际内存和延迟需要在 Linux ARM 上测量。不要上传模型目录内的 `.venv`，也不要搬运 Windows 的 Python 环境。

`requirements-web.txt` 目前只列 FastAPI、Uvicorn、HTTPX，不能独立安装完整问答后端。需要补齐 CPU 版 torch、numpy、transformers、sentence-transformers、openai、jieba、tiktoken 等实际运行依赖，在 Linux 环境验证后锁定版本。数据重建工具需要的依赖另列。

## 3. 为什么不直接全部放 Vercel

Vercel 支持 Python，但现有后端依赖常驻模型、进程内登录表、SESSION_STORE、全局锁以及本地文件写入。函数实例不能承担这些共享状态和持久存储职责。

当前 Vercel Fluid Compute 下 Hobby 函数上限为 2 GB 内存和 300 秒。Python 标准包体上限为 500 MB，符合条件的 Large Functions Beta 可到 5 GB；因此不能用“Vercel 永远只能装 250 MB”作为判断依据。即使包体放得下，本项目的内存、冷启动、状态和调用耗时仍然不适合原样迁入函数。[函数限制](https://vercel.com/docs/functions/limitations)

Vercel Hobby 适用于个人、非商业用途；本规划面向个人毕设展示。[Hobby 计划](https://vercel.com/docs/plans/hobby)

## 4. GitHub 仓库应怎样整理

当前检查时项目根目录不是 Git 仓库。首次上传前整理提交范围，再初始化 Git；先建立个人账户下的私有仓库，确认提交内容后再决定是否公开。

建议保留结构：

```text
frontend/
  index.html
  styles.css
  app.js
  vercel.json             # 后续实现：静态站点与 API 转发规则
src/
  web_api.py
  web_service.py
  main_agent.py
  config.py
  ...                     # 其余实际运行模块和测试
deploy/
  Dockerfile              # 后续实现：Linux CPU 后端环境
  compose.yaml            # 后续实现：持久目录挂载、单实例
  assets-manifest.json    # 后续实现：模型/索引版本和校验值
requirements.txt          # 后续实现：完整后端运行依赖
requirements-web.txt
.env.example              # 后续实现：变量名称和空值，无密钥
.gitignore
README.md
DEPLOYMENT_PLAN.md
```

上面的新增配置只是规划，不代表文件已经生成。

现有 `.gitignore` 范围太小，建议增加：

```gitignore
.venv/
**/.venv/
.web-deps/
.web-venv/
.runtime-smoke/
.idea/
__pycache__/
*.pyc
.env
.env.*
!.env.example
users/
src/user_data/
session_short2_store/
logs/
model/
long_term_memory.json
src/long_term_memory.json
src/vector_array.npy
src/chunk_meta.pkl
*.7z
*.zip
*.xlsx
```

数据库选择一个渠道发布：在确认内容适合发布后提交唯一运行库，或者与向量索引一起作为独立版本资产分发。原始 HTML、旧库、CSV 导出和与本项目无关的文档不放入部署包。已有真实用户目录与密钥配置都不提交，服务器从空用户目录初始化。

GitHub 普通仓库限制单个对象大小约 100 MB，现有模型权重远超该值；不要直接 `git add .` 后推送整份工作目录。[GitHub 仓库限制](https://docs.github.com/en/repositories/creating-and-managing-repositories/repository-limits)

模型在服务器构建/准备阶段从可信模型仓库下载，固定版本，仅下载所需权重格式及配置、tokenizer 文件。向量文件、元数据和数据库可以作为一组独立资产发布，记录 SHA-256。已有索引要求查询 embedding 与生成索引时的模型、维度、归一化和指令一致；更换 embedding 模型时重建索引并一起发布。

## 5. 首选路线：Vercel + Oracle 免费 VM

浏览器访问 Vercel 站点；静态页面由 Vercel 提供，`/api/*` 经 Vercel rewrite 转发到 VM 上的 HTTPS 后端。VM 常驻运行 FastAPI、BGE embedding、BGE reranker，调用外部模型服务，并在持久磁盘保存账户和会话。

### Vercel 设置

| 项目设置 | 建议值 |
|---|---|
| Git 仓库 | 个人账户下的项目仓库 |
| Root Directory | `frontend` |
| Framework Preset | `Other` |
| Build Command | 关闭自动构建覆盖，不需要 npm 构建 |
| Output Directory | `.`，发布现有静态文件 |
| 域名 | 使用平台提供的 `.vercel.app` 地址 |

`frontend/vercel.json` 的计划示例，必须把占位地址换成真实 HTTPS 后端：

```json
{
  "rewrites": [
    {
      "source": "/api/:path*",
      "destination": "https://BACKEND_HOST.example/api/:path*"
    }
  ]
}
```

现有 `app.js` 已使用相对地址 `/api` 和同源 cookie，可保留这种调用模式。rewrite 要保留 `/api` 前缀，不增加会把 API 错误变成 HTML 的全路径兜底。API 响应继续设置 `Cache-Control: no-store`。[外部 rewrite](https://vercel.com/docs/routing/rewrites)

### VM 设置

选择免费额度内的一台 Ubuntu ARM VM，建议使用可取得的 2 OCPU / 12 GB 配置，磁盘先分配约 50–60 GB。该配置是待压测的部署目标，不代表已验证性能。

在 VM 的持久磁盘准备代码、所需模型和固定版本检索资产，用 Docker 或独立 Linux Python 3.11/3.12 环境运行。先验证 CPU/ARM 依赖能安装以及模型可加载。

计划启动方式：

```bash
python -m uvicorn web_api:app --app-dir src --host 0.0.0.0 --port 8000 --workers 1
```

容器需要监听 `0.0.0.0`。使用 Docker 发布端口时，可只绑定 VM 的 `127.0.0.1:8000`，由 HTTPS 反向代理对外提供入口；直接运行 Uvicorn 且采用本机反向代理时则监听 `127.0.0.1`。现有 `start_web.ps1` 留作本地 Windows 启动入口。

首次上线保留 **一个进程、一个 worker**。当前 auth 字典、会话缓存和推理锁都是进程内状态，直接加多个 worker 会导致登录和会话表现不一致。

容器至少挂载 `users/`、`session_short2_store/`、`logs/` 到持久目录。`SCENIC_USER_ROOT` 已有实现，可指向持久用户目录；它不会自动迁移另两个目录。模型和检索资产使用只读目录。用重启策略或 systemd 管理服务，不依赖手动终端。

HTTPS 后端需要可用的证书和访问地址。Vercel 前端地址免费，VM 若使用自购域名会另有域名费用；坚持零域名费用时需验证所选免费 DNS 名称及证书方案。证书、网络访问、目标用户所在网络的可达性都是联调项。

### 必须修正的代理与请求行为

1. `src/web_api.py` 当前要求 Origin 等于后端 `request.url` 的源。rewrite 后浏览器 Origin 是 Vercel 前端地址，可能被拒绝。新增明确的前端 Origin 白名单，保留 `X-Requested-With` 检查；不要简单移除来源校验。
2. 生产登录 cookie 强制 Secure，保留 HttpOnly 和 SameSite=Strict。只信任实际反向代理的转发头，测试 cookie 经 rewrite 下发与回传。开发和生产配置分开，预览域名只显式允许需要联调的部署。
3. `/api/*` 同源代理路线不需要泛化 CORS；若改为浏览器跨域直连，就需要重新设计凭证、CORS 和 cookie 策略，不能只改 fetch URL。
4. 首页由 Vercel 提供以后，`web_api.py` 的 `/` 不再接收浏览器首页请求，当前打开次数统计会失效。需要增加页面访问事件接口或采用前端访问统计，并重新说明计数口径。

### 解决长回答：异步任务与短请求轮询

Vercel 外部 rewrite 的代理请求上限是 **120 秒**。当前前端设置的 `900000` 毫秒不能突破它；这与 Vercel 函数的 300 秒上限是两个不同限制。[代理超时](https://vercel.com/docs/limits#proxied-request-timeout)

计划为完整反思路径增加：

```text
POST /api/chat/jobs
  输入 session_id、message、request_id
  返回 202 和 job_id，尽快结束 HTTP 请求

GET /api/chat/jobs/{job_id}
  返回 queued / running / succeeded / failed
  成功时携带现有协议的完整回答
```

VM 在常驻进程中使用有界任务队列执行问答；前端每隔数秒查询，失败或超时后采用退避，页面隐藏时降低轮询频率。快查询可继续使用原接口，但必须验证在代理时限内完成。

任务与 request_id 持久化，按用户和会话校验归属；重复提交返回同一任务，成功答案只写一次历史。首次部署用现有全局推理锁控制串行处理，显示排队状态并限制队列长度。重启时把未完成任务标记为中断或恢复执行，不能永久显示 running。任务完成后保留结果一段时间，允许刷新页面找回。

不在 Vercel 函数返回之后依靠后台线程继续运行长问答；后台执行属于 VM 的常驻服务。

## 6. 备选路线：Vercel + Render + Supabase

Render Free 当前只有 512 MB RAM、0.1 CPU，无法承载本项目当前约 1 GiB 权重的 reranker 加 torch 运行环境。模型应精简为 SQL + 分词检索/BM25；如需 embedding，可验证小模型 ONNX 量化版本的实际峰值内存，或使用外部 embedding 服务并单独核算费用。[Render 计算规格](https://render.com/docs/compute-plans)

精简版保留景区资料、结构化筛选、生成回答、账号和会话。现有本地 BGE 重排序不承诺保留；混合检索若改为其他模型，要重建索引。仅删除 requirements 中的 torch 不够，还需解除 `config.py` 和 `rag_hybrid_query.py` 的顶层 torch/模型加载依赖。

Render 服务配置：仓库根目录部署，构建安装精简依赖；启动 `uvicorn web_api:app --app-dir src --host 0.0.0.0 --port $PORT --workers 1`。这条命令只有在精简模式和存储适配完成后才可用。

Render 免费服务在 15 分钟无流量后休眠，重启、重新部署和休眠会丢失本地写入；不能把生产账号和会话继续放在 `users/*.json` 中。其免费 Postgres 30 天后过期，不适合跨学期毕设。[Render 免费服务限制](https://render.com/docs/free)

建议将用户、密码哈希、模型配置、会话、消息、反馈、统计、任务和幂等记录迁入 Supabase Postgres。至少设计 users、sessions、messages、feedback、site_stats、auth_sessions、chat_jobs 表。它不是给 `SCENIC_USER_ROOT` 换一个路径，需要改造 UserAccount、SessionMemoryManager、WebService 的存储接口，以及 auth 字典。

景区 SQLite 可作为只读构建资产保留。后续后台需要编辑景区时，再迁移相应景区表。Supabase Free 当前包含 500 MB 数据库和 1 GB 文件存储，低活跃项目约一周可能暂停；上线前恢复项目，定期导出备份，数据库管理密钥仅放后端。[Supabase 额度](https://supabase.com/pricing)、[项目暂停规则](https://supabase.com/docs/guides/platform/free-project-pausing)

此方案的部署操作更托管化，但代码改动量比 VM 路线更大，并且平台休眠和小内存会影响体验。

## 7. 不作为零费用主方案的平台

Hugging Face 当前官方说明：新建运行计算资源的 Docker/Gradio Space 需要付费计划；免费个人账户的 ZeroGPU Gradio 例外不等同于常驻 FastAPI Docker 后端。不能根据旧文档中的“CPU Basic 每小时免费”推导新账号可免费创建 Docker Space。[Spaces 当前规则](https://huggingface.co/docs/hub/spaces-overview)

GitHub 用于代码托管和可选 CI；GitHub Pages 只服务静态内容，不运行当前 Python Agent。外部模型的免费试用额度也不作为长期免费保证。

## 8. 上线前需要完成的代码与验证

| 范围 | 首选 VM 路线要做的事 |
|---|---|
| `.gitignore`、根 README | 明确源码、模型资产和真实用户数据的边界 |
| 依赖、Dockerfile、部署变量 | CPU/ARM 安装验证、锁版本、配置持久目录和自动重启 |
| `frontend/vercel.json` | API rewrite，替换真实后端地址 |
| `src/web_api.py` | 代理来源校验、生产 cookie、任务提交/查询端点 |
| `src/web_service.py` | 有界任务队列、持久任务状态、幂等结果、访问统计适配 |
| `frontend/app.js` | 任务轮询、排队状态、刷新恢复、兼容代理/休眠时返回的非 JSON 错误 |
| 管理员初始化 | 通过部署初始化步骤创建管理员；注册接口禁止注册保留管理员名 |
| 模型配置 | 公网模式限制可访问的模型服务地址，避免现有任意 URL 配置触达服务器内网 |
| 公网调用额度 | 登录、注册、提问限流；若共享演示密钥，设置账户/每日调用上限 |

管理员角色目前依据用户名 `admin` 判断，管理员创建代码却在控制台 `__main__` 入口。用 Uvicorn 启动不会执行该入口，需要独立初始化方案，不能依赖真实用户目录随仓库发布。

验证顺序：

1. 全新 Linux 环境从源码和指定资产启动，`/api/health` 显示模型就绪；不能只确认网页可打开。
2. 云端测模型加载时间、峰值内存、典型问答耗时和队列等待时间，再确定候选数量与反思次数。
3. Vercel 域名测试注册、登录、Secure cookie、配置保存、历史、删除会话和管理后台权限。
4. 超过 120 秒的任务仍可通过轮询取得答案；刷新、重试同一 request_id 不重复写对话。
5. 两个用户的会话和任务互不混淆；跨账户读取被拒绝。
6. 重启后端及重新部署后，会话、配置、反馈和已完成任务保留；登录若按现有内存令牌实现则允许要求重新登录。
7. 验证 VM/数据库备份可恢复，并保留本地 Windows 演示入口。
8. 从答辩所用网络和设备访问前端、API、模型供应商；实际验证可达性与延迟。

建议实施顺序：整理仓库与 Linux 依赖 → 判断能否取得 Oracle 免费实例 → 后端模型和持久目录验证 → 异步任务与代理适配 → Vercel 联调 → 回归、备份及答辩演练。若免费实例不可取得，再实施精简版，而不是先把重模型原样部署到 Render。
