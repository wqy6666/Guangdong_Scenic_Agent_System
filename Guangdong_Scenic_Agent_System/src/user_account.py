# -*- coding: utf-8 -*-
# user_account.py
import os
import json
import hashlib
import hmac
import re
import secrets
import uuid
from typing import Optional, Tuple, List
from pathlib import Path
from config import ROOT_PATH, DEFAULT_LLM_TEMPLATE
from security_controls import PUBLIC_MODE
from safe_guard import logger
from file_store import write_json

# 用户根目录：项目根目录下 users 文件夹
USER_ROOT_DIR = os.environ.get("SCENIC_USER_ROOT", str(ROOT_PATH / "users"))
os.makedirs(USER_ROOT_DIR, exist_ok=True)
CONFIG_KEY_ENV = "SCENIC_CONFIG_ENCRYPTION_KEY"

try:
    from cryptography.fernet import Fernet, InvalidToken
except ImportError:  # Local legacy mode can still read existing plaintext configs.
    Fernet = None
    InvalidToken = ValueError


def _config_fernet():
    key = os.environ.get(CONFIG_KEY_ENV, "").strip()
    if not key:
        return None
    if Fernet is None:
        raise RuntimeError("配置了加密密钥，但运行环境缺少 cryptography 依赖。")
    try:
        return Fernet(key.encode("ascii"))
    except (ValueError, TypeError):
        raise RuntimeError(f"{CONFIG_KEY_ENV} 格式无效，请使用 Fernet.generate_key() 生成。") from None


def get_pwd_hash(plain_pwd: str) -> str:
    salt = secrets.token_hex(16)
    rounds = 600000
    digest = hashlib.pbkdf2_hmac("sha256", plain_pwd.encode("utf-8"), salt.encode(), rounds).hex()
    return f"pbkdf2_sha256${rounds}${salt}${digest}"


def validate_user_id(user_id: str) -> str:
    uid = user_id.strip()
    if not re.fullmatch(r"[A-Za-z0-9_\-\u4e00-\u9fff]{1,64}", uid):
        raise ValueError("用户名限 1–64 位中文、字母、数字、下划线或短横线。")
    if re.fullmatch(r"CON|PRN|AUX|NUL|COM[0-9]|LPT[0-9]", uid, re.IGNORECASE):
        raise ValueError("请换一个用户名。")
    return uid


# 注册/改密共用的密码规则：仅字母和数字，禁止空格、短横线等特殊字符
PASSWORD_CHARSET_PATTERN = re.compile(r"[A-Za-z0-9]+")


def validate_password(plain_pwd: str) -> str:
    """校验密码规则：8–16位、仅字母和数字、且不能以数字开头；不合法抛 ValueError"""
    if not 8 <= len(plain_pwd) <= 16:
        raise ValueError("密码长度需要 8 到 16 位。")
    if not PASSWORD_CHARSET_PATTERN.fullmatch(plain_pwd):
        raise ValueError("密码只能由字母和数字组成，不能包含空格或其他特殊字符。")
    if plain_pwd[0].isdigit():
        raise ValueError("密码不能以数字开头。")
    return plain_pwd


class UserAccount:
    """
    通用用户账号管理类
    管理员账户只能通过 admin_init.py 初始化，不从普通注册入口创建。
    """
    def __init__(self, user_id: str):
        self.user_id = validate_user_id(user_id)
        self.user_folder = os.path.join(USER_ROOT_DIR, self.user_id)
        self.meta_file = os.path.join(self.user_folder, "user_meta.json")
        self.llm_cfg_file = os.path.join(self.user_folder, "llm_config.json")
        os.makedirs(self.user_folder, exist_ok=True)
        self._load_or_init_meta()

    def _load_or_init_meta(self):
        """加载/初始化用户元信息文件 user_meta.json"""
        if os.path.exists(self.meta_file):
            with open(self.meta_file, "r", encoding="utf-8") as f:
                self.meta = json.load(f)
        else:
            import time
            self.meta = {
                "user_id": self.user_id,
                "create_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "password_hash": "",
                "session_list": []
            }
            self._save_meta()

    def _save_meta(self):
        """保存用户元信息"""
        write_json(self.meta_file, self.meta)

    def set_password(self, plain_pwd: str):
        """设置用户密码，保存哈希值"""
        self.meta["password_hash"] = get_pwd_hash(plain_pwd)
        self._save_meta()

    def verify_password(self, plain_pwd: str) -> bool:
        """校验登录密码"""
        stored = self.meta.get("password_hash", "")
        if stored.startswith("pbkdf2_sha256$"):
            try:
                _, rounds, salt, expected = stored.split("$")
                actual = hashlib.pbkdf2_hmac("sha256", plain_pwd.encode("utf-8"), salt.encode(), int(rounds)).hex()
                return hmac.compare_digest(expected, actual)
            except (ValueError, TypeError):
                return False
        # Existing console accounts remain usable and upgrade on successful login.
        valid = hmac.compare_digest(stored, hashlib.sha256(plain_pwd.encode("utf-8")).hexdigest())
        if valid:
            self.set_password(plain_pwd)
        return valid

    def is_disabled(self) -> bool:
        """账户是否已被管理员关停"""
        return bool(self.meta.get("disabled", False))

    def set_disabled(self, disabled: bool):
        """管理员关停/激活账户；关停后的账户无法登录"""
        self.meta["disabled"] = bool(disabled)
        self._save_meta()

    def set_identity_answers(self, full_name: str, birth_date: str, verification_phrase: str):
        """保存带独立盐值的答案校验值到账户元信息文件。"""
        self.meta["identity_answers"] = {
            "full_name": get_pwd_hash(full_name),
            "birth_date": get_pwd_hash(birth_date),
            "verification_phrase": get_pwd_hash(verification_phrase),
        } if full_name and birth_date and verification_phrase else {}
        self._save_meta()

    def has_identity_answers(self) -> bool:
        answers = self.meta.get("identity_answers")
        if not isinstance(answers, dict):
            return False
        return all(isinstance(answers.get(key), str) and answers[key]
                   for key in ("full_name", "birth_date", "verification_phrase"))

    def verify_identity_answers(self, full_name: str, birth_date: str, verification_phrase: str) -> bool:
        answers = self.meta.get("identity_answers")
        if not self.has_identity_answers() or not isinstance(answers, dict):
            return False
        supplied = (full_name, birth_date, verification_phrase)
        keys = ("full_name", "birth_date", "verification_phrase")
        matched = []
        for key, value in zip(keys, supplied):
            stored = answers[key]
            if stored.startswith("pbkdf2_sha256$"):
                try:
                    _, rounds, salt, expected = stored.split("$")
                    actual = hashlib.pbkdf2_hmac("sha256", value.encode("utf-8"), salt.encode(), int(rounds)).hex()
                    matched.append(hmac.compare_digest(expected, actual))
                except (ValueError, TypeError):
                    matched.append(False)
            else:
                matched.append(hmac.compare_digest(stored, value))
        return all(matched)

    def save_user_llm_config(self, llm_cfg: dict) -> Tuple[bool, str]:
        """
        保存该用户的llm配置到 llm_config.json
        :param llm_cfg: {"api_key":"xxx","base_url":"xxx","model_name":"xxx"}
        :return: (成功bool, 提示消息)
        """
        try:
            fernet = _config_fernet()
            if PUBLIC_MODE and fernet is None:
                raise RuntimeError(f"公开模式必须配置 {CONFIG_KEY_ENV}。")
            if fernet is None:
                write_json(self.llm_cfg_file, llm_cfg)
            else:
                payload = json.dumps(llm_cfg, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
                write_json(self.llm_cfg_file, {"version": 1, "encrypted": True,
                                               "payload": fernet.encrypt(payload).decode("ascii")})
            logger.info(f"[{self.user_id}] LLM配置保存成功")
            return True, "LLM配置保存成功"
        except Exception as e:
            logger.error(f"[{self.user_id}] LLM配置保存失败: {str(e)}", exc_info=True)
            return False, f"保存失败:{str(e)}"

    def load_user_llm_config(self) -> Tuple[dict, str]:
        """读取当前用户llm配置文件"""
        if not os.path.exists(self.llm_cfg_file):
            # 文件缺失，使用模板生成空白配置
            blank_cfg = DEFAULT_LLM_TEMPLATE.copy()
            self.save_user_llm_config(blank_cfg)
            return blank_cfg, "llm配置文件缺失，已生成空白模板"
        try:
            with open(self.llm_cfg_file, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            if cfg.get("encrypted"):
                fernet = _config_fernet()
                if fernet is None:
                    return {}, f"读取配置失败：请配置 {CONFIG_KEY_ENV}"
                try:
                    cfg = json.loads(fernet.decrypt(cfg["payload"].encode("ascii")).decode("utf-8"))
                except (KeyError, TypeError, ValueError, InvalidToken):
                    return {}, "读取配置失败：配置密钥不匹配"
            return cfg, "读取成功"
        except Exception as e:
            logger.error(f"[{self.user_id}]读取llm配置异常:{str(e)}", exc_info=True)
            return {}, f"读取配置异常:{str(e)}"

    def create_new_session(self, session_name: Optional[str] = None) -> str:
        """创建全新会话文件夹，返回session_id"""
        import time
        session_id = session_name if session_name else f"session_{int(time.time())}_{uuid.uuid4().hex[:12]}"
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", session_id):
            raise ValueError("无效的会话名称")
        session_path = os.path.join(self.user_folder, session_id)
        os.makedirs(session_path, exist_ok=True)
        if session_id not in self.meta["session_list"]:
            self.meta["session_list"].append(session_id)
            self._save_meta()
        return session_id

    def remove_session(self, session_id: str):
        """从会话列表中移除指定会话（不删除磁盘目录，目录由调用方处理）"""
        if session_id in self.meta["session_list"]:
            self.meta["session_list"].remove(session_id)
            self._save_meta()

    def list_all_sessions(self) -> List[str]:
        """列出该用户全部会话id列表"""
        return self.meta.get("session_list", [])

    @staticmethod
    def check_user_exist(user_id: str) -> bool:
        """静态方法：判断用户是否已经存在"""
        check_path = os.path.join(USER_ROOT_DIR, validate_user_id(user_id), "user_meta.json")
        return os.path.isfile(check_path)

    @staticmethod
    def find_existing_user(user_id: str) -> Optional[str]:
        """遍历账户目录，按大小写不敏感方式查找已注册账户；命中返回已存在的用户名，否则 None"""
        target = validate_user_id(user_id).casefold()
        try:
            children = os.listdir(USER_ROOT_DIR)
        except OSError:
            return None
        for child in children:
            if child.casefold() == target and os.path.isfile(os.path.join(USER_ROOT_DIR, child, "user_meta.json")):
                return child
        return None

    @staticmethod
    def create_user(user_id: str, plain_pwd: str, *, allow_admin: bool = False) -> "UserAccount":
        """
        静态方法：新建用户
        - 创建用户目录
        - 设置密码哈希
        - 自动写入DEFAULT_LLM_TEMPLATE空白llm_config（api_key为空）
        """
        user_id = validate_user_id(user_id)
        if user_id.casefold() == "admin" and not allow_admin:
            raise ValueError("管理员账户只能通过 admin_init.py 初始化。")
        user = UserAccount(user_id)
        user.set_password(plain_pwd)
        # 新用户生成空白模板配置
        blank_cfg = DEFAULT_LLM_TEMPLATE.copy()
        user.save_user_llm_config(blank_cfg)
        logger.info(f"[UserAccount] 创建新用户:{user_id}")
        return user

    @staticmethod
    def login_user(user_id: str, plain_pwd: str) -> Optional["UserAccount"]:
        """
        用户登录校验
        返回 None: 用户不存在 / 密码错误
        返回 UserAccount实例：登录成功
        """
        uid = user_id.strip()
        if not UserAccount.check_user_exist(uid):
            return None
        user = UserAccount(uid)
        if user.verify_password(plain_pwd):
            return user
        return None
