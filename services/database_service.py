"""
Base database service for CRUD operations.

This module provides a base class for database operations
with common CRUD functionality.
"""

import os
from abc import ABC, abstractmethod
from typing import List, Optional, TypeVar, Generic, Dict, Any
from uuid import UUID

try:
    import asyncpg
    HAS_ASYNCPG = True
except ImportError:
    HAS_ASYNCPG = False
from pydantic import BaseModel

from telemetrics.logger import logger

T = TypeVar('T', bound=BaseModel)


class DatabaseService(ABC, Generic[T]):
    """Base class for database CRUD operations."""

    def __init__(self, table_name: str):
        self.table_name = table_name
        self._pool: Optional[asyncpg.Pool] = None

    async def _get_pool(self):
        """Get database connection pool."""
        if not HAS_ASYNCPG:
            raise ImportError("asyncpg is required for database operations. Install it with: pip install asyncpg")

        if self._pool is None:
            # Get database URL from environment
            database_url = os.getenv("DATABASE_URL")
            if not database_url:
                raise ValueError("DATABASE_URL environment variable not set")

            self._pool = await asyncpg.create_pool(database_url)
        return self._pool

    async def create(self, data: Dict[str, Any]) -> UUID:
        """Create a new record and return its ID."""
        pool = await self._get_pool()

        # Build INSERT query dynamically
        columns = list(data.keys())
        placeholders = [f"${i+1}" for i in range(len(columns))]
        values = list(data.values())

        query = f"""
            INSERT INTO {self.table_name} ({', '.join(columns)})
            VALUES ({', '.join(placeholders)})
            RETURNING id
        """

        try:
            async with pool.acquire() as conn:
                result = await conn.fetchval(query, *values)
                logger.info(f"Created record in {self.table_name}: {result}")
                return result
        except Exception as e:
            logger.error(f"Error creating record in {self.table_name}: {e}")
            raise

    async def get_by_id(self, record_id: UUID) -> Optional[Dict[str, Any]]:
        """Get a record by ID."""
        pool = await self._get_pool()

        query = f"SELECT * FROM {self.table_name} WHERE id = $1"

        try:
            async with pool.acquire() as conn:
                result = await conn.fetchrow(query, record_id)
                return dict(result) if result else None
        except Exception as e:
            logger.error(f"Error fetching record from {self.table_name}: {e}")
            raise

    async def get_all_by_user(self, user_id: UUID, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """Get all records for a user with pagination."""
        pool = await self._get_pool()

        query = f"""
            SELECT * FROM {self.table_name}
            WHERE user_id = $1
            ORDER BY created_at DESC
            LIMIT $2 OFFSET $3
        """

        try:
            async with pool.acquire() as conn:
                results = await conn.fetch(query, user_id, limit, offset)
                return [dict(row) for row in results]
        except Exception as e:
            logger.error(f"Error fetching records from {self.table_name}: {e}")
            raise

    async def update(self, record_id: UUID, data: Dict[str, Any]) -> bool:
        """Update a record by ID."""
        pool = await self._get_pool()

        # Build UPDATE query dynamically
        set_clauses = [f"{key} = ${i+2}" for i, key in enumerate(data.keys())]
        values = list(data.values())

        query = f"""
            UPDATE {self.table_name}
            SET {', '.join(set_clauses)}, updated_at = NOW()
            WHERE id = $1
        """

        try:
            async with pool.acquire() as conn:
                result = await conn.execute(query, record_id, *values)
                updated = result.split()[-1] == "1"  # Check if 1 row was updated
                if updated:
                    logger.info(f"Updated record in {self.table_name}: {record_id}")
                else:
                    logger.warning(f"No record found to update in {self.table_name}: {record_id}")
                return updated
        except Exception as e:
            logger.error(f"Error updating record in {self.table_name}: {e}")
            raise

    async def delete(self, record_id: UUID) -> bool:
        """Delete a record by ID."""
        pool = await self._get_pool()

        query = f"DELETE FROM {self.table_name} WHERE id = $1"

        try:
            async with pool.acquire() as conn:
                result = await conn.execute(query, record_id)
                deleted = result.split()[-1] == "1"  # Check if 1 row was deleted
                if deleted:
                    logger.info(f"Deleted record from {self.table_name}: {record_id}")
                else:
                    logger.warning(f"No record found to delete in {self.table_name}: {record_id}")
                return deleted
        except Exception as e:
            logger.error(f"Error deleting record from {self.table_name}: {e}")
            raise

    async def count_by_user(self, user_id: UUID) -> int:
        """Count records for a user."""
        pool = await self._get_pool()

        query = f"SELECT COUNT(*) FROM {self.table_name} WHERE user_id = $1"

        try:
            async with pool.acquire() as conn:
                result = await conn.fetchval(query, user_id)
                return result or 0
        except Exception as e:
            logger.error(f"Error counting records in {self.table_name}: {e}")
            raise

    async def close(self):
        """Close database connections."""
        if self._pool:
            await self._pool.close()
            self._pool = None
