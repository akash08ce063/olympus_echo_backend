#!/usr/bin/env python3
"""
Verify that the concurrent calls migration was successful.
Run this after executing the SQL migration in Supabase.
"""

import sys
import asyncio
sys.path.append('.')

from data_layer.supabase_client import get_supabase_client

async def verify_migration():
    try:
        print("🔍 Verifying concurrent calls migration...")
        
        supabase_client = await get_supabase_client()
        
        # Try to insert a test record with the new columns
        test_data = {
            'test_run_id': 'migration-test-123',
            'test_case_id': 'migration-test-456', 
            'test_suite_id': 'migration-test-789',
            'status': 'test',
            'concurrent_calls': 2,
            'wav_file_ids': ['file-id-1', 'file-id-2']
        }
        
        result = await supabase_client.insert('test_case_results', test_data)
        
        if result:
            print("✅ Migration successful! New columns work correctly.")
            
            # Clean up test record
            await supabase_client.delete('test_case_results', {'test_run_id': 'migration-test-123'})
            print("🧹 Cleaned up test record")
            
        else:
            print("❌ Migration failed - could not insert with new columns")
            
    except Exception as e:
        print(f"❌ Migration verification failed: {e}")

if __name__ == "__main__":
    asyncio.run(verify_migration())
