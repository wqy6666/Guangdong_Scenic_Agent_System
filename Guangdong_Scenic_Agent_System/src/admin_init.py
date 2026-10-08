"""Interactive first-run administrator initialization."""
import getpass
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
DEPS = ROOT.parent / ".web-deps"
if DEPS.is_dir():
    sys.path.insert(0, str(DEPS))

from config import ADMIN_USERNAME
from user_account import UserAccount, validate_password


def main():
    existing = UserAccount.find_existing_user(ADMIN_USERNAME)
    password = getpass.getpass(f"为 {ADMIN_USERNAME} 设置密码（8-16位字母数字）：")
    try:
        validate_password(password)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2
    if existing:
        UserAccount(existing).set_password(password)
        print(f"管理员账户 {existing} 的密码已更新。")
    else:
        UserAccount.create_user(ADMIN_USERNAME, password, allow_admin=True)
        print(f"管理员账户 {ADMIN_USERNAME} 已创建。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
