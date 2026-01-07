"""
Recording file storage service for Supabase S3.

This module provides functionality to upload, download, and manage
recording files in Supabase storage under the "recording_files" folder.
"""

from typing import Optional, Dict, Any
from uuid import UUID, uuid4
import os

from data_layer.supabase_client import get_supabase_client
from telemetrics.logger import logger


class RecordingStorageService:
    """Service for managing recording files in Supabase storage."""

    RECORDING_BUCKET = "recording_files"

    def __init__(self):
        self.bucket_name = self.RECORDING_BUCKET

    async def upload_recording_file(
        self,
        file_content: bytes,
        file_name: str,
        content_type: str = "audio/wav"
    ) -> Optional[UUID]:
        """
        Upload a recording file to Supabase storage.

        Args:
            file_content: The binary content of the recording file
            file_name: Original filename (will be prefixed with UUID)
            content_type: MIME type of the file (default: audio/wav)

        File path format: recording_files/{uuid}_{filename}

        Returns:
            UUID of the uploaded file if successful, None otherwise
        """
        supabase_client = await get_supabase_client()

        try:
            # Generate a unique file ID
            file_id = uuid4()

            # Create the file path directly in recording_files bucket
            # Format: recording_files/{uuid}_{original_filename}
            file_path = f"{file_id}_{file_name}"

            # Upload the file to Supabase storage
            success = await supabase_client.upload_file(
                bucket_name=self.bucket_name,
                file_path=file_path,
                file_content=file_content,
                content_type=content_type
            )

            if success:
                logger.info(f"Successfully uploaded recording file: {file_path}")
                return file_id
            else:
                logger.error(f"Failed to upload recording file: {file_path}")
                return None

        except Exception as e:
            logger.error(f"Error uploading recording file {file_name}: {e}")
            return None

    async def download_recording_file(self, file_id: UUID, file_name: str) -> Optional[bytes]:
        """
        Download a recording file from Supabase storage.

        Args:
            file_id: UUID of the file to download
            file_name: Original filename

        Returns:
            File content as bytes if successful, None otherwise
        """
        supabase_client = await get_supabase_client()

        try:
            # Construct the file path
            file_path = f"{file_id}_{file_name}"

            # Download the file from Supabase storage
            file_content = await supabase_client.download_file(
                bucket_name=self.bucket_name,
                file_path=file_path
            )

            if file_content:
                logger.info(f"Successfully downloaded recording file: {file_path}")
                return file_content
            else:
                logger.error(f"Failed to download recording file: {file_path}")
                return None

        except Exception as e:
            logger.error(f"Error downloading recording file {file_id}/{file_name}: {e}")
            return None

    async def delete_recording_file(self, file_id: UUID, file_name: str) -> bool:
        """
        Delete a recording file from Supabase storage.

        Args:
            file_id: UUID of the file to delete
            file_name: Original filename

        Returns:
            True if successful, False otherwise
        """
        supabase_client = await get_supabase_client()

        try:
            # Construct the file path
            file_path = f"{file_id}_{file_name}"

            # Delete the file from Supabase storage
            success = await supabase_client.delete_file(
                bucket_name=self.bucket_name,
                file_path=file_path
            )

            if success:
                logger.info(f"Successfully deleted recording file: {file_path}")
                return True
            else:
                logger.error(f"Failed to delete recording file: {file_path}")
                return False

        except Exception as e:
            logger.error(f"Error deleting recording file {file_id}/{file_name}: {e}")
            return False

    async def get_recording_file_url(self, file_id: UUID, file_name: str, expires_in: int = 3600) -> Optional[str]:
        """
        Generate a signed URL for accessing a recording file.

        Args:
            file_id: UUID of the file
            file_name: Original filename
            expires_in: URL expiration time in seconds (default: 1 hour)

        Returns:
            Signed URL if successful, None otherwise
        """
        supabase_client = await get_supabase_client()

        try:
            # Construct the file path
            file_path = f"{file_id}_{file_name}"

            # Get the signed URL from Supabase storage
            # Note: This assumes the Supabase client has a method for creating signed URLs
            # If not available, this would need to be implemented differently
            signed_url = await self._create_signed_url(file_path, expires_in)

            if signed_url:
                logger.info(f"Generated signed URL for recording file: {file_path}")
                return signed_url
            else:
                logger.error(f"Failed to generate signed URL for recording file: {file_path}")
                return None

        except Exception as e:
            logger.error(f"Error generating signed URL for recording file {file_id}/{file_name}: {e}")
            return None

    async def _create_signed_url(self, file_path: str, expires_in: int) -> Optional[str]:
        """Create a signed URL for file access."""
        try:
            supabase_client = await get_supabase_client()
            return await supabase_client.create_signed_url(self.bucket_name, file_path, expires_in)
        except Exception as e:
            logger.error(f"Error creating signed URL: {e}")
            return None

    async def get_recording_url_by_file_id(self, file_id: str, test_case_id: str, call_number: int = 1) -> Optional[str]:
        """
        Get a signed URL for a recording file using file_id and test_case_id.
        Uses the correct format: {file_id}_test_case_{test_case_id}_call_recording.wav

        Args:
            file_id: UUID of the recording file
            test_case_id: UUID of the test case
            call_number: Call number (ignored, always uses call_1)

        Returns:
            Signed URL if successful, None otherwise
        """
        supabase_client = await get_supabase_client()

        # Only use the new correct format
        file_name = f"test_case_{test_case_id}_call_recording.wav"

        # Construct the file path
        file_path = f"{file_id}_{file_name}"

        try:
            signed_url = await supabase_client.create_signed_url(self.bucket_name, file_path, 3600)

            if signed_url:
                logger.info(f"Generated signed URL for recording: {file_path}")
                return signed_url
            else:
                logger.warning(f"Failed to generate signed URL for: {file_path}")
                return None

        except Exception as e:
            logger.warning(f"Recording file not found: {file_path}")
            return None

    def get_file_info(self, file_id: UUID, file_name: str) -> Dict[str, Any]:
        """
        Get file information for a recording.

        Args:
            file_id: UUID of the file
            file_name: Original filename

        Returns:
            Dictionary with file information
        """
        return {
            "file_id": str(file_id),
            "file_name": file_name,
            "bucket": self.bucket_name,
            "path": f"{file_id}_{file_name}"
        }
