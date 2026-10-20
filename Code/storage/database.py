"""SQLite storage layer and DAO for AUTOSAR domain entities (Stage 11).

Provides persistence and query interfaces for extracted Software Components,
Ports, and Interfaces with unique constraints for deduplication.
"""

from pathlib import Path
import sqlite3
from typing import Any, Optional, Union
from config.settings import get_settings
from core.logger import get_logger

logger = get_logger(__name__)


class DatabaseManager:
    """Manages SQLite connection, schema creation, and entity DAO operations."""

    def __init__(self, db_path: Optional[Union[str, Path]] = None) -> None:
        """Initialize database manager.

        Args:
            db_path: Path to the SQLite database file. Defaults to settings.sqlite_db_path.
        """
        settings = get_settings()
        self.db_path = Path(db_path or settings.sqlite_db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Create a connection with row factory enabled."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        return conn

    def _init_db(self) -> None:
        """Create database tables if they do not exist."""
        with self._get_connection() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS software_components (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    component_type TEXT NOT NULL,
                    description TEXT,
                    document_id TEXT NOT NULL,
                    page_number INTEGER NOT NULL,
                    section TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(name, document_id)
                );

                CREATE TABLE IF NOT EXISTS ports (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    swc_parent TEXT,
                    interface_name TEXT,
                    description TEXT,
                    document_id TEXT NOT NULL,
                    page_number INTEGER NOT NULL,
                    section TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(name, swc_parent, document_id)
                );

                CREATE TABLE IF NOT EXISTS interfaces (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    interface_type TEXT NOT NULL,
                    description TEXT,
                    document_id TEXT NOT NULL,
                    page_number INTEGER NOT NULL,
                    section TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(name, document_id)
                );
                """
            )
            conn.commit()
        logger.info("Initialized database schema at '%s'", self.db_path)

    def save_software_components(self, components: list[Any]) -> int:
        """Insert or replace software components."""
        if not components:
            return 0
        with self._get_connection() as conn:
            for swc in components:
                conn.execute(
                    """
                    INSERT INTO software_components (name, component_type, description, document_id, page_number, section)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(name, document_id) DO UPDATE SET
                        component_type=excluded.component_type,
                        description=COALESCE(excluded.description, software_components.description),
                        page_number=excluded.page_number,
                        section=COALESCE(excluded.section, software_components.section)
                    """,
                    (
                        swc.name,
                        swc.component_type,
                        swc.description,
                        swc.document_id,
                        swc.page_number,
                        swc.section,
                    ),
                )
            conn.commit()
        return len(components)

    def save_ports(self, ports: list[Any]) -> int:
        """Insert or replace ports."""
        if not ports:
            return 0
        with self._get_connection() as conn:
            for p in ports:
                conn.execute(
                    """
                    INSERT INTO ports (name, direction, swc_parent, interface_name, description, document_id, page_number, section)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(name, swc_parent, document_id) DO UPDATE SET
                        direction=excluded.direction,
                        interface_name=COALESCE(excluded.interface_name, ports.interface_name),
                        description=COALESCE(excluded.description, ports.description),
                        page_number=excluded.page_number,
                        section=COALESCE(excluded.section, ports.section)
                    """,
                    (
                        p.name,
                        p.direction,
                        p.swc_parent or "",
                        p.interface_name,
                        p.description,
                        p.document_id,
                        p.page_number,
                        p.section,
                    ),
                )
            conn.commit()
        return len(ports)

    def save_interfaces(self, interfaces: list[Any]) -> int:
        """Insert or replace interfaces."""
        if not interfaces:
            return 0
        with self._get_connection() as conn:
            for iface in interfaces:
                conn.execute(
                    """
                    INSERT INTO interfaces (name, interface_type, description, document_id, page_number, section)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(name, document_id) DO UPDATE SET
                        interface_type=excluded.interface_type,
                        description=COALESCE(excluded.description, interfaces.description),
                        page_number=excluded.page_number,
                        section=COALESCE(excluded.section, interfaces.section)
                    """,
                    (
                        iface.name,
                        iface.interface_type,
                        iface.description,
                        iface.document_id,
                        iface.page_number,
                        iface.section,
                    ),
                )
            conn.commit()
        return len(interfaces)

    def get_software_components(self, document_id: Optional[str] = None) -> list[dict[str, Any]]:
        """Retrieve stored software components."""
        with self._get_connection() as conn:
            if document_id:
                rows = conn.execute(
                    "SELECT * FROM software_components WHERE document_id = ? ORDER BY id ASC",
                    (document_id,),
                ).fetchall()
            else:
                rows = conn.execute("SELECT * FROM software_components ORDER BY id ASC").fetchall()
            return [dict(r) for r in rows]

    def get_ports(self, document_id: Optional[str] = None) -> list[dict[str, Any]]:
        """Retrieve stored ports."""
        with self._get_connection() as conn:
            if document_id:
                rows = conn.execute(
                    "SELECT * FROM ports WHERE document_id = ? ORDER BY id ASC",
                    (document_id,),
                ).fetchall()
            else:
                rows = conn.execute("SELECT * FROM ports ORDER BY id ASC").fetchall()
            return [dict(r) for r in rows]

    def get_interfaces(self, document_id: Optional[str] = None) -> list[dict[str, Any]]:
        """Retrieve stored interfaces."""
        with self._get_connection() as conn:
            if document_id:
                rows = conn.execute(
                    "SELECT * FROM interfaces WHERE document_id = ? ORDER BY id ASC",
                    (document_id,),
                ).fetchall()
            else:
                rows = conn.execute("SELECT * FROM interfaces ORDER BY id ASC").fetchall()
            return [dict(r) for r in rows]

    def clear_entities(self, document_id: Optional[str] = None) -> None:
        """Clear entities from database."""
        with self._get_connection() as conn:
            if document_id:
                conn.execute("DELETE FROM software_components WHERE document_id = ?", (document_id,))
                conn.execute("DELETE FROM ports WHERE document_id = ?", (document_id,))
                conn.execute("DELETE FROM interfaces WHERE document_id = ?", (document_id,))
            else:
                conn.execute("DELETE FROM software_components")
                conn.execute("DELETE FROM ports")
                conn.execute("DELETE FROM interfaces")
            conn.commit()
