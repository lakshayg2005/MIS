"""Create or reset a login.  python -m scripts.create_user <email> <company> <employee|admin> <password> [data_dir]"""
import sys
from pathlib import Path

from app.config import Settings
from app.users import UserStore

if __name__ == "__main__":
    email, company, role, password = sys.argv[1:5]
    data = sys.argv[5] if len(sys.argv) > 5 else Settings().data_dir
    Path(data).mkdir(parents=True, exist_ok=True)
    UserStore(str(Path(data) / "registry.db")).create(email, company, role, password)
    print(f"created {role} {email} for {company}")
