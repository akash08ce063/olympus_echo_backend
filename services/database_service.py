"""
Base database service for CRUD operations using Supabase.

This module provides a base class for database operations
with common CRUD functionality using Supabase client.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, TypeVar, Generic, Dict, Any
from uuid import UUID
from datetime import datetime

from pydantic import BaseModel
from data_layer.supabase_client import get_supabase_client
from telemetrics.logger import logger

T = TypeVar('T', bound=BaseModel)


class DatabaseService(ABC, Generic[T]):
    """Base class for database CRUD operations using Supabase."""

    def __init__(self, table_name: str):
        self.table_name = table_name
        self._client = None

    async def _get_client(self):
        """Get Supabase client instance."""
        if self._client is None:
            self._client = await get_supabase_client()
        return self._client

    async def create(self, data: Dict[str, Any]) -> UUID:
        """Create a new record and return its ID."""
        return await self._create_record(data, allow_id_override=False)

    async def create_with_id(self, data: Dict[str, Any]) -> str:
        """Create a new record with a specific ID and return the ID."""
        return await self._create_record(data, allow_id_override=True)

    async def _create_record(self, data: Dict[str, Any], allow_id_override: bool = False) -> str:
        """Internal method to create a record."""
        client = await self._get_client()

        # Convert UUID objects to strings for JSON serialization
        insert_data = {}
        for key, value in data.items():
            if isinstance(value, UUID):
                insert_data[key] = str(value)
            else:
                insert_data[key] = value

        # Handle 'id' field based on allow_id_override
        if not allow_id_override:
            # Remove 'id' field if present - let database generate it
            insert_data.pop('id', None)
        # If allow_id_override is True, keep the 'id' field if present

        # Add timestamps
        insert_data['created_at'] = datetime.utcnow().isoformat()
        insert_data['updated_at'] = datetime.utcnow().isoformat()

        try:
            result = await client.insert(self.table_name, insert_data)
            if result and len(result) > 0:
                record_id = result[0]['id']
                logger.info(f"Created record in {self.table_name}: {record_id}")
                return record_id
            else:
                raise Exception("No data returned from insert operation")
        except Exception as e:
            logger.error(f"Error creating record in {self.table_name}: {e}")
            raise

    async def get_by_id(self, record_id: UUID) -> Optional[Dict[str, Any]]:
        """Get a record by ID."""
        client = await self._get_client()

        try:
            result = await client.select(self.table_name, filters={'id': str(record_id)})
            return result[0] if result and len(result) > 0 else None
        except Exception as e:
            logger.error(f"Error getting record from {self.table_name}: {e}")
            raise

    async def get_all_by_user(self, user_id: UUID, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """Get all records for a user."""
        client = await self._get_client()

        try:
            result = await client.select(
                self.table_name,
                filters={'user_id': str(user_id)},
                limit=limit,
                offset=offset,
                order_by='created_at',
                order_desc=True
            )
            return result if result else []
        except Exception as e:
            logger.error(f"Error getting records from {self.table_name}: {e}")
            raise

    async def update(self, record_id: UUID, data: Dict[str, Any]) -> bool:
        """Update a record."""
        client = await self._get_client()

        # Convert UUID objects to strings for JSON serialization
        update_data = {}
        for key, value in data.items():
            if isinstance(value, UUID):
                update_data[key] = str(value)
            else:
                update_data[key] = value

        # Add updated timestamp
        update_data['updated_at'] = datetime.utcnow().isoformat()

        try:
            result = await client.update(self.table_name, {'id': str(record_id)}, update_data)
            updated = result is not None and len(result) > 0
            if updated:
                logger.info(f"Updated record in {self.table_name}: {record_id}")
            else:
                logger.warning(f"No record found to update in {self.table_name}: {record_id}")
            return updated
        except Exception as e:
            logger.error(f"Error updating record in {self.table_name}: {e}")
            raise

    async def delete(self, record_id: UUID) -> bool:
        """Delete a record."""
        client = await self._get_client()

        try:
            result = await client.delete(self.table_name, {'id': str(record_id)})
            if result:
                    logger.info(f"Deleted record from {self.table_name}: {record_id}")
            else:
                    logger.warning(f"No record found to delete in {self.table_name}: {record_id}")
            return result
        except Exception as e:
            logger.error(f"Error deleting record from {self.table_name}: {e}")
            raise

    async def count_by_user(self, user_id: UUID) -> int:
        """Count records for a user."""
        client = await self._get_client()

        try:
            result = await client.select(self.table_name, filters={'user_id': str(user_id)})
            return len(result) if result else 0
        except Exception as e:
            logger.error(f"Error counting records in {self.table_name}: {e}")
            raise

    async def close(self):
        """Close database connections (no-op for Supabase)."""
        pass
