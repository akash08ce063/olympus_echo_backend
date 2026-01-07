from supabase.client import AsyncClient, acreate_client
from static_memory_cache import StaticMemoryCache
from telemetrics.logger import logger
from typing import Optional, Dict, Any, List


class SupabaseClient:
    def __init__(self, async_client: AsyncClient):
        self.async_client = async_client

    @classmethod
    async def create(cls) -> "SupabaseClient":
        """Create Supabase client using static memory cache configuration."""
        db_config = StaticMemoryCache.get_database_config()
        supabase_url = db_config.get("supabase_url")
        supabase_key = db_config.get("supabase_key")

        if not supabase_url or not supabase_key:
            raise ValueError("Supabase URL and Key must be configured in static memory cache")

        async_client: AsyncClient = await acreate_client(supabase_url, supabase_key)
        logger.info("Supabase client created successfully")
        return cls(async_client)

    def get_client(self) -> AsyncClient:
        return self.async_client

    async def insert(self, table_name: str, data: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        """Insert data into a table."""
        try:
            logger.info(f"Supabase inserting into {table_name}: {data}")
            result = await self.async_client.table(table_name).insert(data).execute()
            logger.info(f"Supabase insert result: {result}")
            logger.info(f"Supabase insert result.data: {result.data}")
            return result.data if result.data else None
        except Exception as e:
            logger.error(f"Error inserting into {table_name}: {str(e)}")
            return None

    async def select(self, table_name: str, filters: Dict[str, Any] = None,
                    limit: int = None, offset: int = None,
                    order_by: str = None, order_desc: bool = False) -> Optional[List[Dict[str, Any]]]:
        """Select data from a table."""
        try:
            query = self.async_client.table(table_name).select("*")

            if filters:
                for key, value in filters.items():
                    query = query.eq(key, value)

            if order_by:
                if order_desc:
                    query = query.order(order_by, desc=True)
                else:
                    query = query.order(order_by)

            if limit:
                query = query.limit(limit)

            if offset:
                query = query.range(offset, offset + (limit or 1000) - 1)

            result = await query.execute()
            return result.data if result.data else None
        except Exception as e:
            logger.error(f"Error selecting from {table_name}: {str(e)}")
            return None

    async def update(self, table_name: str, filters: Dict[str, Any], data: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        """Update data in a table."""
        try:
            query = self.async_client.table(table_name).update(data)

            for key, value in filters.items():
                query = query.eq(key, value)

            result = await query.execute()
            return result.data if result.data else None
        except Exception as e:
            logger.error(f"Error updating {table_name}: {str(e)}")
            return None

    async def delete(self, table_name: str, filters: Dict[str, Any]) -> bool:
        """Delete data from a table."""
        try:
            # Start with delete query
            query = self.async_client.table(table_name).delete()

            # Apply filters
            for key, value in filters.items():
                query = query.eq(key, value)

            # Execute the delete
            result = await query.execute()
            return True
        except Exception as e:
            logger.error(f"Error deleting from {table_name}: {str(e)}")
            return False

    async def upload_file(self, bucket_name: str, file_path: str, file_content: bytes, content_type: str = "application/octet-stream") -> bool:
        """Upload a file to Supabase storage."""
        try:
            result = await self.async_client.storage.from_(bucket_name).upload(
                path=file_path,
                file=file_content,
                file_options={"content-type": content_type}
            )
            return True
        except Exception as e:
            logger.error(f"Error uploading file to {bucket_name}/{file_path}: {str(e)}")
            return False

    async def download_file(self, bucket_name: str, file_path: str) -> Optional[bytes]:
        """Download a file from Supabase storage."""
        try:
            result = await self.async_client.storage.from_(bucket_name).download(file_path)
            return result
        except Exception as e:
            logger.error(f"Error downloading file from {bucket_name}/{file_path}: {str(e)}")
            return None

    async def delete_file(self, bucket_name: str, file_path: str) -> bool:
        """Delete a file from Supabase storage."""
        try:
            result = await self.async_client.storage.from_(bucket_name).remove([file_path])
            return True
        except Exception as e:
            logger.error(f"Error deleting file from {bucket_name}/{file_path}: {str(e)}")
            return False

    async def create_signed_url(self, bucket_name: str, file_path: str, expires_in: int = 3600) -> Optional[str]:
        """Create a signed URL for file access."""
        try:
            result = await self.async_client.storage.from_(bucket_name).create_signed_url(
                path=file_path,
                expires_in=expires_in
            )
            if result and 'signedURL' in result:
                return result['signedURL']
            return None
        except Exception as e:
            logger.error(f"Error creating signed URL for {bucket_name}/{file_path}: {str(e)}")
            return None

    def get_public_url(self, bucket_name: str, file_path: str) -> str:
        """Get public URL for a file (bucket must be public)."""
        try:
            result = self.async_client.storage.from_(bucket_name).get_public_url(file_path)
            return result
        except Exception as e:
            logger.error(f"Error getting public URL for {bucket_name}/{file_path}: {str(e)}")
            return None

    async def execute_raw_query(self, query: str) -> Optional[List[Dict[str, Any]]]:
        """Execute a raw SQL query using PostgREST RPC."""
        try:
            # For now, we'll use a different approach since raw SQL execution is complex
            # We'll implement the cascade query using the existing select methods
            logger.warning("Raw query execution not fully implemented. Using alternative approach.")
            return None
        except Exception as e:
            logger.error(f"Error executing raw query: {str(e)}")
            return None

    async def call_rpc_function(self, function_name: str, params: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        """Call a PostgreSQL RPC function."""
        try:
            response = await self.async_client.rpc(function_name, params).execute()
            if response.data:
                logger.info(f"Successfully called RPC function {function_name}")
                return response.data
            else:
                logger.warning(f"No data returned from RPC function {function_name}")
                return []
        except Exception as e:
            logger.error(f"Error calling RPC function {function_name}: {str(e)}")
            return None


# Global instance for singleton pattern
_supabase_client: Optional[SupabaseClient] = None


async def get_supabase_client() -> SupabaseClient:
    """Get or create the global Supabase client instance."""
    global _supabase_client
    if _supabase_client is None:
        _supabase_client = await SupabaseClient.create()
    return _supabase_client
