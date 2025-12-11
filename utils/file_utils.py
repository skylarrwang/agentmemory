from pathlib import Path


def get_next_session_num(username: str) -> int:
    """Return the next session number for this user."""
    sessions_dir = Path("sessions")
    sessions_dir.mkdir(parents=True, exist_ok=True)
    
    # Find existing sessions for this user
    existing_sessions = []
    for session_dir in sessions_dir.iterdir():
        if session_dir.is_dir() and session_dir.name.startswith(f"{username}."):
            try:
                num = int(session_dir.name.split(".")[-1])
                existing_sessions.append(num)
            except ValueError:
                continue
    
    if not existing_sessions:
        return 1
    
    return max(existing_sessions) + 1