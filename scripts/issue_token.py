"""Mock identity provider CLI.  Usage: python -m scripts.issue_token <company> [employee|admin] [subject]"""
import sys

from app.auth import issue_token
from app.config import Settings

if __name__ == "__main__":
    company = sys.argv[1]
    role = sys.argv[2] if len(sys.argv) > 2 else "employee"
    sub = sys.argv[3] if len(sys.argv) > 3 else f"{role}@{company}.example"
    print(issue_token(Settings(), sub, company, role))
