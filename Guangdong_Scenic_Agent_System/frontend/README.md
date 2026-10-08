# 粤游智答 Web 版

前端已通过同源 `/api` 接口连接现有 Python Agent。网页与 API 由一个进程提供，不能再通过双击 HTML 或 `http.server` 使用真实问答。

## 启动

在项目根目录运行：

```powershell
.\start_web.ps1
```

首次部署或管理员密码丢失时，在项目根目录运行 `python -B src/admin_init.py`，按提示创建或重置 `admin` 账户。公开部署必须使用 HTTPS 反向代理，并以精确来源配置 `SCENIC_ALLOWED_ORIGINS`；模型服务默认只允许 `api.deepseek.com` 和 `api.openai.com`。

公开部署还必须设置 `SCENIC_CONFIG_ENCRYPTION_KEY`，可用 `python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"` 生成。该密钥只放在服务端环境变量或密钥管理器中，不要提交到项目文件；已有明文模型配置在下次保存时会加密。

脚本优先选择项目内或项目父目录的 `.venv`。本机现有模型环境位于父目录，也可以直接运行：

```powershell
& '..\.venv\Scripts\python.exe' -B src/web_api.py
```

等待终端出现 `Application startup complete`，打开 <http://127.0.0.1:8000/>。首次启动需要加载本地向量库、BGE embedding 和 reranker。停止服务使用 Ctrl+C。

可指定环境和端口：

```powershell
.\start_web.ps1 -Python 'D:\path\to\python.exe' -Port 8000
```

### 手机通过同一局域网访问

启动脚本默认监听 `0.0.0.0`，终端会显示 `Phone (same Wi-Fi)` 的访问地址。电脑和手机连接同一个 Wi-Fi，等待启动完成后，在手机浏览器打开该地址（例如 `http://192.168.0.102:8000`）。手机上不要填写 `127.0.0.1` 或 `0.0.0.0`；更换网络后以启动脚本显示的新地址为准。电脑需保持开机且服务运行，手机需单独登录。

如果电脑能打开、手机打不开，检查 Windows 防火墙是否允许 TCP 8000 的入站访问，以及路由器是否开启访客网络/设备隔离。保持服务运行，在项目目录另开管理员 PowerShell，执行：

```powershell
.\enable_lan_access.ps1
```

自定义端口时使用 `-Port 你的端口`。脚本先确认端口属于当前项目，再创建或更新允许规则，范围仅为正在运行的 Python 程序、指定 TCP 端口、Wi-Fi 和同一子网。如果此前曾在 Windows 提示中拒绝 Python 联网，脚本还会禁用同一个 Python 程序由该提示生成的 TCP 阻止规则，并打印恢复命令；其他阻止规则保持原样。显式阻止规则优先于允许规则，因此只添加端口允许规则可能仍无法访问。此脚本需要 Windows 管理员权限，不会关闭防火墙。

前端请求保持同源，局域网 HTTP 下也能生成提问请求编号并复用编号重试。浏览器可能限制 HTTP 页面的“复制回答”按钮，可长按回答文字复制。

只允许电脑本机访问时运行：

```powershell
.\start_web.ps1 -BindAddress 127.0.0.1
```

直接运行 `src/web_api.py` 时也默认支持局域网；可通过 `SCENIC_HOST` 和 `SCENIC_PORT` 环境变量修改监听地址和端口。

使用已安装原项目推理依赖的 Python 环境。新增 Web 依赖已安装到当前项目 `.web-deps`，Web 入口会自动加载此目录；迁移到另一台机器时执行：

```powershell
python -m pip install --target .web-deps -r requirements-web.txt
```

原有推理依赖包括 torch、numpy、transformers、sentence-transformers、openai、jieba、tiktoken；本地数据库、模型和向量文件继续使用 `src/config.py` 的路径配置。

## 使用

1. 点击账户菜单登录已有账号，或注册新账号（密码至少 8 位）。
2. 配置服务地址、模型名称和 API 密钥。已有完整配置的账号可直接提问。编辑配置时，密钥留空表示保留原值。
3. 新建会话，或点击左侧历史会话继续。刷新页面会恢复当前登录和最近选择的会话。
4. 输入问题，等待完整回答；可展开相关景区资料、打开存在的来源链接、复制回答。
5. 点击“加载更早的对话”读取历史。历史消息目前保存为文本，旧消息不重建卡片和来源链接。

修改密码有两种方式：已登录时从账户菜单选择“修改密码”，输入当前密码后设置新密码；忘记当前密码时，从登录窗口选择“忘记密码”，回答此前在账户设置保存的姓名、出生日期和身份验证语句。三项答案以加盐校验值保存在账号元信息中且不会回传；未设置或全部清空时，问答找回不可用，不影响其他功能。管理员从账户菜单进入独立的 `/admin` 页面。

问答、门票、推荐、统计、名单、多轮追问均使用同一个聊天接口。侧栏景区内容是固定探索入口，不是实时热榜。附件、反馈、评分和临时密钥模式未纳入首版。

## 接口

| 方法 | 路径 | 功能 |
|---|---|---|
| GET | `/api/health` | 本地问答引擎状态，不代表外部模型密钥有效 |
| POST | `/api/auth/register` | 注册并登录 |
| POST | `/api/auth/login` | 登录 |
| POST | `/api/auth/logout` | 退出 |
| GET | `/api/auth/me` | 当前用户及不含密钥的配置状态 |
| GET / PUT | `/api/settings/llm` | 读取 / 保存模型配置 |
| GET / PUT | `/api/settings/security-questions` | 查询是否已设置 / 保存或清除密码找回答案 |
| POST | `/api/auth/reset-password-by-answers` | 使用三项验证信息重设密码 |
| GET / POST | `/api/sessions` | 列出 / 新建会话 |
| GET | `/api/sessions/{id}/messages?before=段落编号` | 最近六轮 / 更早六轮 |
| POST | `/api/chat` | `{session_id, message, request_id}`，同步兼容接口 |
| POST | `/api/chat/jobs` | 提交后台问答任务，立即返回任务状态 |
| GET | `/api/chat/jobs/{session_id}/{request_id}` | 查询后台问答任务状态和结果 |

修改类请求需要 `X-Requested-With: scenic-web`。浏览器使用 HttpOnly、SameSite=Strict 的登录 cookie，有效期 8 小时。服务重启后需要重新登录，会话和配置保留。

仅启动一个 worker：登录状态、会话上下文和锁均在当前进程中。不同账户使用独立记忆键，同一用户的存储操作串行，本地模型推理也串行，网页问答进入有界后台队列。重启时排队或运行中的任务标记为中断，不会自动重复调用外部模型。公开模式还启用登录、注册、找回密码和聊天限流，以及模型地址白名单。

管理员监控仅通过管理员鉴权接口提供。系统页展示 CPU、内存、磁盘、队列、引擎状态及近 7 天请求成功/失败和平均耗时；错误页展示近 7/30/90 天问答失败摘要；Token 页按北京时间统计每位用户每日或本月的输入/输出 Token。仅统计模型服务在流式响应中返回的 usage；未返回 usage 的调用单独标记，不估算。运营数据保存在 `users/operations/`，错误事件保留 90 天、用量与请求汇总保留 400 天。错误面板不保存用户提问或回答正文。

公开模式示例（反向代理终止 TLS）：

```powershell
.\start_web.ps1 -BindAddress 127.0.0.1 -Port 8000 -PublicMode `
  -AllowedOrigins 'https://travel.example.com' `
  -AllowedModelHosts 'api.deepseek.com','api.openai.com'
```

配置沿用 `users/<用户名>/llm_config.json` 文件保存方式，完整密钥不会由接口返回；配置了 `SCENIC_CONFIG_ENCRYPTION_KEY` 后文件保存为 Fernet 密文。新密码采用 PBKDF2，已有 SHA-256 密码在成功登录时自动升级。

`SCENIC_USER_ROOT` 可覆盖用户数据根目录，用于隔离联调环境。模型加载失败时网页仍可打开，状态显示“服务未就绪”；查看启动日志解决模型或依赖问题后重启。

若网页可以打开，但提问提示后端被禁止联网（日志中为 `WinError 10013`），请停止原来的后端进程，在自己的 PowerShell 或 PyCharm 终端运行 `start_web.ps1`。本地网页可访问不代表后端进程能访问外部模型，受限环境中启动的进程可能只能响应本地请求。不要在旧进程占用端口时重复启动。

模型错误现在区分网络权限、连接、响应超时、证书、密钥权限、余额、模型名称和服务端异常；错误下方可点击“检查模型配置”或重试。Web 模型连接诊断日志仅记录异常类别、HTTP 状态和系统错误编号，不记录密钥或服务商错误正文。

回答核验使用 DeepSeek 官方接口时关闭思考模式，首次输出上限为 1500 tokens；空回答或输出截断时最多重试一次，上限提高到 3000 tokens。核验失败时保留已生成的回答，附上“未完成自动核验”提示，协议类型为 `agent_unverified_answer`。空 JSON 或缺少判决字段不算核验通过。日志只记录结束原因、正文/思考字符数和服务商返回的 token 用量，不记录思考原文。

最终回答整理（`rewrite`）使用 DeepSeek 官方接口时也关闭思考模式，避免 2000 tokens 全部用于思考而没有正文。整理返回空正文或达到长度上限时最多重试一次，输出上限提高到 4000 tokens。仍失败时保留已有草稿，附上已发现的缺项/无依据表述说明和“回答整理未完成”提示，不再额外调用模型，协议类型为 `agent_unverified_answer`。其他厂商不发送 DeepSeek 专用参数，普通回答生成保留原有行为。

## 验证

```powershell
& '..\.venv\Scripts\python.exe' -B src/test_suite/test_web_api.py
& '..\.venv\Scripts\python.exe' -B src/test_suite/test_web_llm_transport.py
& '..\.venv\Scripts\python.exe' -B src/test_suite/test_answer_verification.py
node src/test_suite/test_frontend_request_id.cjs
```

这些测试使用真实文件存储、临时账户和可控的问答引擎，覆盖登录、配置、历史分页、会话隔离、重试幂等和错误状态；模型传输测试使用真实 SDK 与本地模拟响应，验证流式拼接、密钥错误、限流、超时和空回答，不调用外部付费模型。旧 `test_suite` 中部分测试仍对应旧接口，不作为本次 Web 接入的验收入口。

回答核验测试加载本地检索模型，但模拟外部模型响应，覆盖空回答/截断重试、无效判决、首次及后置核验失败降级，以及正常核验通过的路径。

前端请求编号测试使用 Node.js 自带模块，覆盖安全上下文、局域网 HTTP 无 `randomUUID`、失败重试复用编号，以及编号生成异常后恢复发送按钮，不访问外部模型。
